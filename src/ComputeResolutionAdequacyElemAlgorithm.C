/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <ComputeResolutionAdequacyElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeResolutionAdequacyElemAlgorithm - Resolution adequacy parameter
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeResolutionAdequacyElemAlgorithm::ComputeResolutionAdequacyElemAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
     sdr_(NULL),
    tvisc_(NULL),
    dudx_(NULL),
    Mij_(NULL),
    Ch_(realm.get_turb_model_constant(TM_Ch)),
    Chmu_(realm.get_turb_model_constant(TM_Chmu))
{
  // save off data
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  sdr_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate");
  tvisc_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  resolutionAdequacy_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter");
  dudx_ =
    meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  Mij_ =
    meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "metric_tensor");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeResolutionAdequacyElemAlgorithm::execute()
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // define some common selectors
  stk::mesh::Selector s_all_nodes = stk::mesh::selectUnion(partVec_);

  // initialize to one
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* resolutionAdequacy = stk::mesh::field_data(*resolutionAdequacy_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      resolutionAdequacy[k] = 1.0;
    }
  }

  // fill in nodal values
  stk::mesh::Selector s_locally_owned_union =
    meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    const double* sdr = stk::mesh::field_data(*sdr_, b);
    const double* tvisc = stk::mesh::field_data(*tvisc_, b);
    double* resolutionAdequacy = stk::mesh::field_data(*resolutionAdequacy_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      const double* dudx = stk::mesh::field_data(*dudx_, b[k]);
      const double* Mij = stk::mesh::field_data(*Mij_, b[k]);

      resolutionAdequacy[k] = 0.0;
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          double Pij = 0.0;
          for (int k = 0; k < nDim; ++k) {
            Pij +=
              dudx[nDim * j + k] * (dudx[nDim * i + k] + dudx[nDim * k + i]) +
              dudx[nDim * i + k] * (dudx[nDim * j + k] + dudx[nDim * k + j]);
          }
          resolutionAdequacy[k] += Mij[nDim * i + k] * Pij;
        }
      }

      const double v2 = tvisc[k] * sdr[k] / Chmu_;

      resolutionAdequacy[k] *= Ch_ / v2;
    }
  }

  // deal with periodicity
  if (realm_.hasPeriodic_) {
    realm_.periodic_field_update(resolutionAdequacy_, 1);
  }
}

} // namespace nalu
} // namespace sierra
