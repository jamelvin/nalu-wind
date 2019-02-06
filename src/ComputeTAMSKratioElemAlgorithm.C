/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSKratioElemAlgorithm.h>

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
// ComputeTAMSKratioElemAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSKratioElemAlgorithm::ComputeTAMSKratioElemAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  alpha_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio");
  avgTurbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_turbulent_ke");
  avgResolvedStress_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "average_resolved_stress");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSKratioElemAlgorithm::execute() {

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
    double * avgTke = stk::mesh::field_data(*avgTurbKineticEnergy_, b);
    double * alpha = stk::mesh::field_data(*alpha_, b);
      
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      double * avgResStress = stk::mesh::field_data(*avgResolvedStress_, b[k]);
      double kResolved = 0.0;
      for (int i = 0; i < nDim; ++i) {
        kResolved += avgResStress[i * nDim + i];
      }
      kResolved *= 0.5;

      if (avgTke[k] == 0.0)
         alpha[k] = 1.0;
      else
         alpha[k] = 1.0 - kResolved/avgTke[k];
    }
  }
}

} // namespace nalu
} // namespace sierra
