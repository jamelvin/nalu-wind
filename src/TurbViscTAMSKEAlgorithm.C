/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <TurbViscTAMSKEAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbViscTAMSKEAlgorithm - compute tvisc for Chien K-epsilon model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbViscTAMSKEAlgorithm::TurbViscTAMSKEAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    cMu_(realm.get_turb_model_constant(TM_cMu)),
    fMuExp_(realm.get_turb_model_constant(TM_fMuExp)),
    density_(NULL),
    tke_(NULL),
    tdr_(NULL),
    dplus_(NULL),
    tvisc_(NULL),
    visc_(NULL),
    avgDudx_(NULL)
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_density");
  tke_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  tdr_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate");
  dplus_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dplus_wall_function");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  visc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  avgDudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "average_dudx");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbViscTAMSKEAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tvisc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *rho = stk::mesh::field_data(*density_, b);
    const double *tke = stk::mesh::field_data(*tke_, b);
    const double *tdr = stk::mesh::field_data(*tdr_, b);
    const double *dplus = stk::mesh::field_data(*dplus_, b);
    double *tvisc = stk::mesh::field_data(*tvisc_, b);
    double *visc = stk::mesh::field_data(*visc_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);

      // compute strain rate magnitude; pull pointer within the loop to make it managable
      double sijMag = 0.0;
      for ( int i = 0; i < nDim; ++i ) {
        const int offSet = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const double rateOfStrain = 0.5*(avgDudx[offSet+j] + avgDudx[nDim*j+i]);
          sijMag += rateOfStrain*rateOfStrain;
        }
      }
      sijMag = std::sqrt(2.0*sijMag);

      // some temps
      const double fMu = 1.0 - std::exp(fMuExp_*dplus[k]);
   
      double v2 = std::min(cMu_*fMu/0.22, 2.0/3.0)*std::max(tke[k],1.0e-16);
      v2 = std::max(v2, 2.0/3.0*1.0e-8);
      const double turbKE = std::max(tke[k], 1.0e-8);
      const double eps = std::max(tdr[k], 1.0e-8);
      double T = std::max(turbKE/eps, 6.0*std::sqrt(visc[k]/eps));
      T = std::min(T, 0.6*turbKE/std::max(std::sqrt(6.0)*0.22*v2*sijMag, 1.0e-12));
      T = std::min(T, 4.0);

      tvisc[k] = cMu_ * fMu * rho[k] * tke[k] * tke[k] / std::max(tdr[k],1.0e-16);
      // FIXME: Can't use this method until you change it so turbvisc is calculated
      //        during SST model and during low mach model, since T relies on both
      //        the K, epsilon scalars and on the velocity field
      //tvisc[k] = 0.22 * v2 * T * rho[k];
    }
  }
}

} // namespace nalu
} // namespace Sierra
