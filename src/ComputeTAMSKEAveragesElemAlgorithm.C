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
  avgTkeRes_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_tke_resolved");
  avgProd_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_production");
  avgTime_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_time");

  // Other quantities
  visc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  alpha_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio");
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

    // get other field data
    const double * tvisc = stk::mesh::field_data(*tvisc_, b);
    const double * visc = stk::mesh::field_data(*visc_, b);
    const double * alpha = stk::mesh::field_data(*alpha_, b);

    // get average field data
    double * avgPres = stk::mesh::field_data(*avgPress_, b);
    double * avgRho = stk::mesh::field_data(*avgDensity_, b);
    double * avgProd = stk::mesh::field_data(*avgProd_, b);
    double * avgTkeRes = stk::mesh::field_data(*avgTkeRes_, b);
    double * avgTime = stk::mesh::field_data(*avgTime_,b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get velocity field data
      const double * vel = stk::mesh::field_data(velocityNp1, b[k]);
      const double * dudx = stk::mesh::field_data(*dudx_, b[k]);
      double * avgVel = stk::mesh::field_data(*avgVelocity_, b[k]);
      double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);
      // FIXME: Verify this is correct for T_ave... this is from slides, 
      //        but CDP has something different

      // At the wall, tdr can be 0.0, so clip it
      double T_ave = tke[k]/std::max(tdr[k],1.0e-16);

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

      const double v2 = 1.0/0.22 * (tvisc[k] * tdr[k]) / std::max(tke[k], 1.0e-16); 

      T_ave = std::max(T_ave, 6.0*std::sqrt(visc[k]/tdr[k]));
      T_ave = std::min(T_ave, 0.6*tke[k]/std::max(std::sqrt(6.0)*0.22*v2*sijMag,1.0e-12));

      const double weightAvg = std::max(1.0 - dt/T_ave, 0.0);
      const double weightInst = std::min(dt/T_ave, 1.0);

      for (int i = 0; i < nDim; ++i)
        avgVel[i] = weightAvg * avgVel[i] + weightInst * vel[i];

      double tkeRes = 0.0;
      for (int i = 0; i < nDim; ++i) {
        tkeRes += (vel[i] - avgVel[i])*(vel[i] - avgVel[i]);
        for (int j = 0; j < nDim; ++j) {
          avgDudx[i*nDim + j] = weightAvg * avgDudx[i*nDim + j] + weightInst * dudx[i*nDim + j];
          // Average strain rate tensor, used for production averaging
        }
      }
      
      // FIXME: Should I be doing Favre averaging?????
      avgPres[k] = weightAvg * avgPres[k] + weightInst * pres[k];
      avgRho[k]  = weightAvg * avgRho[k]  + weightInst * rho[k];
      avgTkeRes[k] = weightAvg * avgTkeRes[k] + weightInst * 0.5*tkeRes;
      avgTime[k] = T_ave;

      // Production averaging
      double tij[nDim][nDim];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          const double avgSij = 0.5*(avgDudx[i*nDim+j] + avgDudx[j*nDim+i]);
          tij[i][j] = 2.0*alpha[k]*tvisc[k]*avgSij;
        }
        tij[i][i] -= 2.0/3.0 * alpha[k] * tke[k];
      }

      double Pij[nDim][nDim];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          Pij[i][j] = 0.0;
          for (int m = 0; m < nDim; ++m) {
             Pij[i][j] += avgDudx[i*nDim + m] * tij[j][m] + avgDudx[j*nDim + m] * tij[i][m];
          }
          Pij[i][j] *= 0.5;
        }
      }

      double P_res = 0.0;
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          P_res += avgDudx[i*nDim + j] * ((avgVel[i] - vel[i])*(avgVel[j] - vel[j]));
        }
      }

      double instProd = 0.0;
      for (int i = 0; i < nDim; ++i) 
        instProd += Pij[i][i];

      instProd -= P_res;

      // FIXME: Need a different averaging timescale for production...
      avgProd[k] = weightAvg * avgProd[k] + weightInst * instProd;
    }
  }
}

} // namespace nalu
} // namespace sierra
