/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSKEpsKratioNodeAlgorithm.h>

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
// ComputeTAMSKEpsKratioNodeAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSKEpsKratioNodeAlgorithm::ComputeTAMSKEpsKratioNodeAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  alpha_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio");
  turbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  totalDissRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  turbVisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  avgTkeRes_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"average_tke_resolved");
  avgTime_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"average_time");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSKEpsKratioNodeAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // select and loop through all nodes
  stk::mesh::Selector s_all_nodes = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &node_buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // get field data
    double * tke = stk::mesh::field_data(*turbKineticEnergy_, b);
    double * alpha = stk::mesh::field_data(*alpha_, b);
    double * tdr = stk::mesh::field_data(*totalDissRate_, b);
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

         const double T_ke = tke[k] / std::max(tdr[k],1e-16);
	 const double v2 = 1.0/0.22 * (tvisc[k] / T_ke);
         //const double v2 = 1.0/0.22 * (tvisc[k] * tdr[k]) / std::max(tke[k], 1.0e-16);
         const double a_kol = std::min(1.5*v2/tke[k]*std::sqrt(visc[k]*tdr[k])/tke[k],1.0);

         alpha[k] = std::max(alpha[k], a_kol);
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
