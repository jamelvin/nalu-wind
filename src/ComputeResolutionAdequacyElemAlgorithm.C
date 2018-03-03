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
    resolutionAdequacy_(NULL),
    dudx_(NULL),
    Mij_(NULL),
    Ch_(realm.get_turb_model_constant(TM_Ch)),
    Chmu_(realm.get_turb_model_constant(TM_Chmu))
{
  // save off data
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  sdr_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate");
  tvisc_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  resolutionAdequacy_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy");
  dudx_ = meta_data.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "dudx");
  Mij_ =  meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "metric_tensor");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeResolutionAdequacyElemAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // initialize nodal values to 0 
  zero_nodal_fields();

  // Vector for Mij at node
  ws_Mij.resize(nDim*nDim);

  // pointer for nodal Mij
  double *p_Mij = &ws_Mij[0];

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*resolutionAdequacy_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      //===============================================
      // gather element data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const *  elem_rels = b.begin_elements(k);
      int num_elements = b.num_elements(k);

      // zero out Mij
      for ( int j=0; j < nDim; ++j )        
        for ( int i=0; i < nDim; ++i )
          p_Mij[j*nDim + i] = 0.0;

      for ( int ne = 0; ne < num_elements; ++ne ) {
        stk::mesh::Entity elem = elem_rels[ne];

        // pointers to real data
        const double *Mij = stk::mesh::field_data(*Mij_, elem);
     
        // gather nodal Mij by averaging over connected elements
        for ( int j=0; j < nDim; ++j ) 
          for ( int i=0; i < nDim; ++i ) 
            p_Mij[j*nDim + i] += Mij[j*nDim + i]/num_elements;
      }

      // gather nodal scalars
      const double *sdr = stk::mesh::field_data(*sdr_, b[k]);
      const double *tvisc = stk::mesh::field_data(*tvisc_, b[k]);
      double *resAdeq = stk::mesh::field_data(*resolutionAdequacy_, b[k]);

      // gather nodal vector
      const double *dudx = stk::mesh::field_data(*dudx_, b[k]);

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          double Pij = 0.0;
          for (int k = 0; k < nDim; ++k) {
            Pij +=
              dudx[nDim * j + k] * (dudx[nDim * i + k] + dudx[nDim * k + i]) +
              dudx[nDim * i + k] * (dudx[nDim * j + k] + dudx[nDim * k + j]);
          }

          resAdeq[0] += p_Mij[nDim * i + j] * -1.0 * tvisc[0] * Pij;
        }
      }

      const double v2 = tvisc[0] * sdr[0] / Chmu_;
      resAdeq[0] *= Ch_ / v2;

    }
  }

  // deal with periodicity
  if (realm_.hasPeriodic_) {
    realm_.periodic_field_update(resolutionAdequacy_, 1);
  }
}

//--------------------------------------------------------------------------
//-------- zero_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeResolutionAdequacyElemAlgorithm::zero_nodal_fields()
{

  stk::mesh::Selector s_all_nodes = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    
    double* resolutionAdequacy = stk::mesh::field_data(*resolutionAdequacy_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      resolutionAdequacy[k] = 0.0;
    }
  }
}

} // namespace nalu
} // namespace sierra
