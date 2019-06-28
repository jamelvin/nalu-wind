/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSKEpsResAdequacyNodeAlgorithm.h>
#include <EigenDecomposition.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

#include "utils/TAMSUtils.h"

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeTAMSKEpsResAdequacyNodeAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSKEpsResAdequacyNodeAlgorithm::ComputeTAMSKEpsResAdequacyNodeAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part),
    nDim_(realm.meta_data().spatial_dimension()),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg))
{

  // save off data
  stk::mesh::MetaData &metaData = realm_.meta_data();

  coordinates_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, realm_.get_coordinates_name());
  turbVisc_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");

  // FIXME: Field of state stuff...
  densityNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  tkeNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  tdrNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "total_dissipation_rate");
  alphaNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "k_ratio");
  dudx_ = metaData.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "dudx");

  avgDensity_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_density");
  avgTime_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  avgDudx_ =  metaData.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "average_dudx");

  resAdeq_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter");
  avgResAdeq_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "avg_res_adequacy_parameter");
  Mij_ = metaData.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "metric_tensor");

  tmpFile.open("resAdeq.txt", std::fstream::app);
}

ComputeTAMSKEpsResAdequacyNodeAlgorithm::~ComputeTAMSKEpsResAdequacyNodeAlgorithm()
{
  tmpFile.close();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSKEpsResAdequacyNodeAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const double dt = realm_.get_time_step();

  // fill in elemental values
  //stk::mesh::Selector s_locally_owned_union =
  //    meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  // fill in elemental values
     stk::mesh::Selector s_locally_owned_union =
           stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &node_buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    tauSGET.resize(nDim_ * nDim_);
    tauSGRS.resize(nDim_ * nDim_);
    tau.resize(nDim_ * nDim_);
    Psgs.resize(nDim_ * nDim_);

    // pointers to the local storage vectors
    double *p_tauSGET = &tauSGET[0];
    double *p_tauSGRS = &tauSGRS[0];
    double *p_tau = &tau[0];
    double *p_Psgs = &Psgs[0];

    // Get scalars
    const double *mut     = stk::mesh::field_data(*turbVisc_, b);
    const double *rhoNp1  = stk::mesh::field_data(*densityNp1_, b);
    const double *tke     = stk::mesh::field_data(*tkeNp1_, b);
    const double *tdr     = stk::mesh::field_data(*tdrNp1_, b);
    const double *avgTime = stk::mesh::field_data(*avgTime_, b);
    const double *alpha   = stk::mesh::field_data(*alphaNp1_, b);
    const double *avgRho  = stk::mesh::field_data(*avgDensity_, b);

    // Get resolution adequacy field for filling
    double *resAdeq = stk::mesh::field_data(*resAdeq_, b);
    double *avgResAdeq = stk::mesh::field_data(*avgResAdeq_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // pointers to real data
      const double * coords = stk::mesh::field_data(*coordinates_, b[k]);
      const double * dudx = stk::mesh::field_data(*dudx_, b[k]);
      const double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);

      // get Mij field_data
      const double *p_Mij = stk::mesh::field_data(*Mij_, b[k]);

      double Mij[3][3];
      double PM[3][3];
      double Q[3][3];
      double D[3][3];

      for (unsigned i = 0; i < nDim_; i++) {
        const int iNdim = i*nDim_;
        for (unsigned j = 0; j < nDim_; j++) {
          Mij[i][j] = p_Mij[iNdim + j];
        }
      }

      // Eigenvalue decomposition of metric tensor
      EigenDecomposition::sym_diagonalize<double>(Mij, Q, D);

      // initialize M43 to 0
      double M43[3][3];
      for (unsigned i = 0; i < nDim_; ++i)
        for (unsigned j = 0; j < nDim_; ++j)
          M43[i][j] = 0.0;

      const double fourThirds = 4.0/3.0;

      for (unsigned l = 0; l < nDim_; l++) {
        const double D43 = stk::math::pow(D[l][l], fourThirds);
        for (unsigned i = 0; i < nDim_; i++) {
          for (unsigned j = 0; j < nDim_; j++) {
            M43[i][j] += Q[i][l] * Q[j][l] * D43;
          }
        }
      }

      // zeroing out tesnors
      for (unsigned i = 0; i < nDim_; ++i) {
        for (unsigned j = 0; j < nDim_; ++j) {
          p_tauSGRS[i*nDim_ + j] = 0.0;
          p_tauSGET[i*nDim_ + j] = 0.0;
          p_tau[i*nDim_ + j] = 0.0;
          p_Psgs[i*nDim_ + j] = 0.0;
        }
      }

      const double CM43 = tams_utils::get_M43_constant<double, 3>(D, CMdeg_);

      const double epsilon13 = stk::math::pow(tdr[k], 1.0/3.0);

      for (unsigned i = 0; i < nDim_; ++i) {
        for (unsigned j = 0; j < nDim_; ++j) {
          // Calculate tauSGRS_ij = 2*alpha*nu_t*<S_ij> where nu_t comes from
          // the SST model and <S_ij> is the strain rate tensor based on the
          // mean quantities... i.e this is (tauSGRS = alpha*tauSST)
          // The 2 in the coeff cancels with the 1/2 in the strain rate tensor
          const double coeffSGRS = alpha[k] * mut[k];
          p_tauSGRS[i*nDim_ + j] = avgDudx[i*nDim_ + j] + avgDudx[j*nDim_ + i];
          p_tauSGRS[i*nDim_ + j] *= coeffSGRS;

          for (unsigned l = 0; l < nDim_; ++l) {
            // Calculate tauSGET_ij = CM43*<eps>^(1/3)*(M43_ik*dkuj' + M43_jkdkui')
            // where <eps> is the mean dissipation backed out from the SST mean k and
            // mean omega and dkuj' is the fluctuating velocity gradients.
            const double coeffSGET = avgRho[k] * CM43 * epsilon13;
            const double fluctDudx_jl = dudx[j*nDim_ + l] - avgDudx[j*nDim_ + l];
            const double fluctDudx_il = dudx[i*nDim_ + l] - avgDudx[i*nDim_ + l];
            p_tauSGET[i*nDim_ + j] += coeffSGET * (M43[i][l] * fluctDudx_jl + 
                                                 M43[j][l] * fluctDudx_il);
          }
        }
      }

      // Calculate the full subgrid stress including the isotropic portion
      // FIXME: Do i need a rho in here?
      for (unsigned i = 0; i < nDim_; ++i) 
        for (unsigned j = 0; j < nDim_; ++j)
          p_tau[i*nDim_ + j] = p_tauSGRS[i*nDim_ + j] + p_tauSGET[i*nDim_ + j] - 
                      ((i==j) ? 2.0/3.0*alpha[k]*tke[k] : 0.0);

      // Calculate the SGS production PSGS_ij = 1/2(tau_ik*djuk + tau_jk*diuk)
      // where diuj is the instantaneous velocity gradients
      for (unsigned i = 0; i < nDim_; ++i) {
        for (unsigned j = 0; j < nDim_; ++j) {
          for (unsigned l = 0; l < nDim_; ++l) {
            p_Psgs[i*nDim_ + j] += p_tau[i*nDim_ + l] * dudx[l*nDim_ + j] + 
                                  p_tau[j*nDim_ + l] * dudx[l*nDim_ + i];
          }
          p_Psgs[i*nDim_+ j] *= 0.5;
        }
      }  

      for (unsigned i = 0; i < nDim_; ++i) { 
        for (unsigned j = 0; j < nDim_; ++j) {
          PM[i][j] = 0.0;
          for (unsigned l = 0; l < nDim_; ++l) 
            PM[i][j] += p_Psgs[i*nDim_ + l] * Mij[l][j];
        }
      }

      // Scale PM first
      const double T_ke = avgTime[k]; //tkeScs / tdrScs;
      const double v2 = 1.0/0.22 * (mut[k] / T_ke);
      const double PMscale = std::pow(1.5*alpha[k]*v2,-1.5);
      if (v2 == 0.0)
        throw std::runtime_error("TAMSKEResAdequacy: v2 is 0, will cause NaN");
      for (unsigned i = 0; i < nDim_; ++i)
        for (unsigned j = 0; j < nDim_; ++j)
          PM[i][j] = PM[i][j] * PMscale;

      //FIXME: PM is not symmetric....
      EigenDecomposition::unsym_matrix_force_sym<double>(PM, Q, D);

      const double maxPM = std::max(std::abs(D[0][0]), std::max(std::abs(D[1][1]), std::abs(D[2][2])));

      //tmpFile << coords[0] << " " << coords[1] << " " << coords[2] << " " << PM[0][0] << " " << PM[0][1] << " " << PM[0][2] << " " << PM[1][0] << " " << PM[1][1] << " " << PM[1][2] << " " << PM[2][0] << " " << PM[2][1] << " " << PM[2][2] << " " << Mij[0][0] << " " << Mij[1][1] << " " << Mij[2][2] << " " << Mij[1][2] << std::endl;        
        
      // Update the instantaneous resAdeq field
      resAdeq[k] = maxPM;
      // FIXME: Limiters as in CDP...
      resAdeq[k] = std::min(resAdeq[k],30.0);
      
      if (alpha[k] >= 1.0)
        resAdeq[k] = std::min(resAdeq[k],1.0);

      const double weightAvg = std::max(1.0 - dt/T_ke, 0.0);
      const double weightInst = std::min(dt/T_ke, 1.0);

      avgResAdeq[k] = weightAvg * avgResAdeq[k] + weightInst * resAdeq[k]; 
    }
  }
}

} // namespace nalu
} // namespace sierra
