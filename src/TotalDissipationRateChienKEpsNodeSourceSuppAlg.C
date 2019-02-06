/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <TotalDissipationRateChienKEpsNodeSourceSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SupplementalAlgorithm.h>
#include <TimeIntegrator.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TotalDissipationRateChienKEpsNodeSourceSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TotalDissipationRateChienKEpsNodeSourceSuppAlg::TotalDissipationRateChienKEpsNodeSourceSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    cEpsOne_(realm.get_turb_model_constant(TM_cEpsOne)),
    cEpsTwo_(realm.get_turb_model_constant(TM_cEpsTwo)),
    fOne_(realm.get_turb_model_constant(TM_fOne)),
    tdrNp1_(NULL),
    tkeNp1_(NULL),
    densityNp1_(NULL),
    dplus_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    dudx_(NULL),
    minD_(NULL),
    dualNodalVolume_(NULL),
    nDim_(realm_.meta_data().spatial_dimension())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *tdr = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate");
  tdrNp1_ = &(tdr->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *tke = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  tkeNp1_ = &(tke->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  dplus_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dplus_wall_function");
  visc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  minD_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateChienKEpsNodeSourceSuppAlg::setup()
{
  // could extract user-based values
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateChienKEpsNodeSourceSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  const double tdr        = *stk::mesh::field_data(*tdrNp1_, node );
  const double tke        = *stk::mesh::field_data(*tkeNp1_, node );
  const double rho        = *stk::mesh::field_data(*densityNp1_, node );
  const double dplus      = *stk::mesh::field_data(*dplus_, node );
  const double visc       = *stk::mesh::field_data(*visc_, node );
  const double tvisc      = *stk::mesh::field_data(*tvisc_, node );
  const double *dudx      =  stk::mesh::field_data(*dudx_, node );
  const double minD       = *stk::mesh::field_data(*minD_, node );
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node );

  int nDim = nDim_;
  double Pk = 0.0;
  for ( int i = 0; i < nDim; ++i ) {
    const int offSet = nDim*i;
    for ( int j = 0; j < nDim; ++j ) {
      Pk += dudx[offSet+j]*(dudx[offSet+j] + dudx[nDim*j+i]);
    }
  }
  Pk *= tvisc;

  // Ftwo calc from Chien 1982 K-epsilon model
  const double Re_t = rho * tke * tke / visc / std::max(tdr, 1.0e-16);
  const double fTwo = 1.0 - 0.4/1.8 * std::exp(-Re_t*Re_t / 36.0);

  // Pe includes 1/k scaling; k may be zero at a dirichlet low Re approach (clip)
  const double PeFac = cEpsOne_ * fOne_ * Pk / std::max(tke, 1.0e-16);
  const double Pe = PeFac * tdr;
  // FIXME: Currently treating the epsilon in fTwo explicitly... 
  //        see LHS below ... assess if this matters
  const double DeFac = cEpsTwo_ * fTwo * rho * tdr / std::max(tke, 1.0e-16);
  const double De = DeFac * tdr;
  // Wall distance source term, rho's cancel...
  const double LeFac = 2.0 * visc * std::exp(-0.5*dplus) / minD / minD;
  const double Le = -LeFac * tdr;

  rhs[0] += (Pe - De + Le)*dualVolume;
  lhs[0] += (2.0*DeFac + LeFac)*dualVolume;
}

} // namespace nalu
} // namespace Sierra
