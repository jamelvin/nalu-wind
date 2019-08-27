/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSSSTKratioNodeAlgorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeTAMSSSTKratioNodeAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSSSTKratioNodeAlgorithm::ComputeTAMSSSTKratioNodeAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  alpha_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio");
  turbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  specDissRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  turbVisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  avgTkeRes_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"average_tke_resolved");
  avgTime_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"average_time");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSSSTKratioNodeAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // fill in nodal values
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*alpha_);

  stk::mesh::BucketVector const &node_buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // get field data
    double * tke = stk::mesh::field_data(*turbKineticEnergy_, b);
    double * alpha = stk::mesh::field_data(*alpha_, b);
    double * sdr = stk::mesh::field_data(*specDissRate_, b);
    double * avgTime = stk::mesh::field_data(*avgTime_, b);
    double * visc = stk::mesh::field_data(*viscosity_, b);
    double * tvisc = stk::mesh::field_data(*turbVisc_, b);
    double * tkeRes = stk::mesh::field_data(*avgTkeRes_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      if (tke[k] == 0.0)
         alpha[k] = 1.0;
      else {
         alpha[k] = 1.0 - tkeRes[k]/tke[k];
      
         // limiters
         alpha[k] = std::min(alpha[k],1.0);

         const double T_sst = avgTime[k]; //1.0 / (betaStar_ * std::max(sdr[k],1e-16));
	 //FIXME: I need a rho in here
	 const double v2 = 1.0/0.22 * (tvisc[k] / T_sst);
         const double epsilon = betaStar_*tke[k]*sdr[k];
         //const double v2 = 1.0/0.22 * (tvisc[k] * tdr[k]) / std::max(tke[k], 1.0e-16);
         // FIXME: What to do with a_kol in SST?
         const double a_kol = 0.01; //std::min(1.5*v2/tke[k]*std::sqrt(visc[k]*epsilon)/tke[k],1.0);
         
         if (T_sst > 0.0)
           alpha[k] = std::max(alpha[k], a_kol);
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
