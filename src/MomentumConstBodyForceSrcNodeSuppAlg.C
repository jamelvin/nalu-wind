/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MomentumConstBodyForceSrcNodeSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <SupplementalAlgorithm.h>

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
// MomentumConstBodyForceSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumConstBodyForceSrcNodeSuppAlg::MomentumConstBodyForceSrcNodeSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    densityNp1_(NULL),
    dualNodalVolume_(NULL),
    nDim_(1),
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  nDim_ = meta_data.spatial_dimension();
  constBodyForce_.resize(nDim_);

  // extract user parameters from solution options
  constBodyForce_ = realm_.solutionOptions_->constBodyForce_;

}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumConstBodyForceSrcNodeSuppAlg::setup()
{
  // all set up in constructor
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumConstBodyForceSrcNodeSuppAlg::node_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity node)
{
  // rhs+=(rho-rhoRef)*gi
  // later, may choose to assemble buoyancy to scv ips: Nip_k*rho_k
  const double rhoNp1     = *stk::mesh::field_data(*densityNp1_, node );
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node );
  const double fac = -rhoNp1*dualVolume;
  const int nDim = nDim_;
  for ( int i = 0; i < nDim; ++i ) {
    rhs[i] += fac*constBodyForce_[i];
  }
}

} // namespace nalu
} // namespace Sierra
