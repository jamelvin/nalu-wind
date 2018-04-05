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
    tke_(NULL),
    tvisc_(NULL),
    density_(NULL),
    viscosity_(NULL),
    resolutionAdequacy_(NULL),
    minDistance_(NULL),
    dudx_(NULL),
    Mij_(NULL),
    Ch_(realm.get_turb_model_constant(TM_Ch)),
    Chmu_(realm.get_turb_model_constant(TM_Chmu)),
    aOne_(realm.get_turb_model_constant(TM_aOne)),
    betaStar_(realm.get_turb_model_constant(TM_betaStar))
{
  // save off data
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  sdr_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate");
  tke_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  tvisc_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  density_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  viscosity_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "viscosity");
  resolutionAdequacy_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy");
  minDistance_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall");
  dudx_ = meta_data.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "dudx");
  Mij_ =  meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "metric_tensor");

  //FIXME: Debugging
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
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

    std::vector<double> ws_coordinates; // FIXME: DEBUGGING
    ws_coordinates.resize(nDim); //FIXME: DEBUGGING

    double *p_coords = &ws_coordinates[0]; // FIXME: DEBUGGING

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      //===============================================
      // gather element data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const *  elem_rels = b.begin_elements(k);
      int num_elements = b.num_elements(k);

      const double *coords = stk::mesh::field_data(*coordinates_, b[k]); //FIXME: DEBUGGING

      // gather coords vector FIXME: DEBUGGING
      for ( int j=0; j < nDim; ++j )
        p_coords[j] = coords[j];
 
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
      const double *rho = stk::mesh::field_data(*density_, b);
      const double *visc = stk::mesh::field_data(*viscosity_, b);
      const double *sdr = stk::mesh::field_data(*sdr_, b[k]);
      const double *tke = stk::mesh::field_data(*tke_, b[k]);
      const double *tvisc = stk::mesh::field_data(*tvisc_, b[k]);
      const double *minD = stk::mesh::field_data(*minDistance_, b);
      double *resAdeq = stk::mesh::field_data(*resolutionAdequacy_, b[k]);

      // gather nodal vector
      const double *dudx = stk::mesh::field_data(*dudx_, b[k]);

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          double Pij = 0.0;
          for (int k = 0; k < nDim; ++k) {
            Pij += 0.5 * (
              dudx[nDim * j + k] * (dudx[nDim * i + k] + dudx[nDim * k + i]) +
              dudx[nDim * i + k] * (dudx[nDim * j + k] + dudx[nDim * k + j]));
          }

          resAdeq[0] += p_Mij[nDim * i + j] * 1.0 * tvisc[0] * Pij;
        }
      }

      // Alternate denominator for nu_t switch in SST
      // compute strain rate magnitude; pull pointer within the loop to make it managable
      double sijMag = 0.0;
      for ( int i = 0; i < nDim; ++i ) {
        const int offSet = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const double rateOfStrain = 0.5*(dudx[offSet+j] + dudx[nDim*j+i]);
          sijMag += rateOfStrain*rateOfStrain;
        }
      }
      sijMag = std::sqrt(2.0*sijMag);
     
      // some temps
      const double minDSq = minD[k]*minD[k];
      const double trbDiss = std::sqrt(tke[k])/betaStar_/sdr[k]/minD[k];
      const double lamDiss = 500.0*visc[k]/rho[k]/sdr[k]/minDSq;
      const double fArgTwo = std::max(2.0*trbDiss, lamDiss);
      const double fTwo = std::tanh(fArgTwo*fArgTwo);

      // tvisc[k] = aOne_*tke[k]/std::max(aOne_*sdr[k], sijMag*fTwo);

 
      //const double v2 = tvisc[0] * sdr[0] / Chmu_;
      const double v2 = betaStar_ * sdr[0] * tvisc[0] / Chmu_;
      if (v2 == 0.0)
        resAdeq[0] = 1.0;
      else
        resAdeq[0] *= Ch_ / v2;
      num_elements = num_elements + 0.0;  //FIXME: Debugging
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
