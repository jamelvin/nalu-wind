/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSKEAveragesElemAlgorithm.h>

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
// ComputeTAMSKEAveragesElemAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSKEAveragesElemAlgorithm::ComputeTAMSKEAveragesElemAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    meshMotion_(realm_.does_mesh_move()) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // instantaneous quantities
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  turbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  totDissipationRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");

  // average quantities
  // FIXME: Do i need a if statement for mesh motion for the average too??
  avgVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "average_velocity");
  avgPress_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_pressure");
  avgDensity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_density");
  avgDudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "average_dudx");
  avgResolvedStress_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "average_resolved_stress");
}


//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSKEAveragesElemAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // time step
  const double dt = realm_.get_time_step();

  // deal with state FIXME: Do i need this?  Do i need this for other fields too???
  VectorFieldType &velocityNp1 = velocityRTM_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);   
  
  // select and loop through all nodes
  stk::mesh::Selector s_all_nodes = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &node_buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // get instantaneous field data
    const double * pres = stk::mesh::field_data(*pressure_, b);
    const double * rho = stk::mesh::field_data(densityNp1, b);
    const double * tke = stk::mesh::field_data(*turbKineticEnergy_, b);
    const double * tdr = stk::mesh::field_data(*totDissipationRate_, b);

    // get average field data
    double * avgPres = stk::mesh::field_data(*avgPress_, b);
    double * avgRho = stk::mesh::field_data(*avgDensity_, b);
      
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get velocity field data
      const double * vel = stk::mesh::field_data(velocityNp1, b[k]);
      const double * dudx = stk::mesh::field_data(*dudx_, b[k]);
      double * avgVel = stk::mesh::field_data(*avgVelocity_, b[k]);
      double * avgResStress = stk::mesh::field_data(*avgResolvedStress_, b[k]);
      double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);
      // FIXME: Verify this is correct for T_ave... this is from slides, 
      //        but CDP has something different

      // At the wall, tdr can be 0.0, so clip it
      const double T_ave = tke[k]/std::max(tdr[k],1.0e-16);

      const double weightAvg = std::max(1.0 - dt/T_ave, 0.0);
      const double weightInst = std::min(dt/T_ave, 1.0);

      for (int i = 0; i < nDim; ++i) {
        avgVel[i] = weightAvg * avgVel[i] + weightInst * vel[i];
        for (int j = 0; j < nDim; ++j) {
          avgResStress[i*nDim + j] = weightAvg * avgResStress[i*nDim + j] + weightInst * 
                (vel[i]*vel[j] - vel[i]*avgVel[j] - vel[j]*avgVel[i] + avgVel[i]*avgVel[j]);
          avgDudx[i*nDim + j] = weightAvg * avgDudx[i*nDim + j] + weightInst * dudx[i*nDim + j];
        }
      }
      
      // FIXME: Should I be doing Favre averaging?????
      avgPres[k] = weightAvg * avgPres[k] + weightInst * pres[k];
      avgRho[k]  = weightAvg * avgRho[k]  + weightInst * rho[k];
    }
  }
}

} // namespace nalu
} // namespace sierra
