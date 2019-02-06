/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeTAMSKEpsResAdequacyElemAlgorithm.h>
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

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeTAMSKEpsResAdequacyElemAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSKEpsResAdequacyElemAlgorithm::ComputeTAMSKEpsResAdequacyElemAlgorithm(
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
  velocityNp1_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  densityNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  tkeNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  tdrNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "total_dissipation_rate");
  alphaNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "k_ratio");
  massFlowRate_ = metaData.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK,"mass_flow_rate_scs");

  avgVelocity_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "average_velocity");
  avgDensity_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_density");
  avgTime_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  avgMdot_ = metaData.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK,"average_mass_flow_rate");

  resAdeq_ = metaData.get_field<ScalarFieldType>(
    stk::topology::ELEMENT_RANK, "resolution_adequacy_parameter");
  avgResAdeq_ = metaData.get_field<ScalarFieldType>(
    stk::topology::ELEMENT_RANK, "average_resolution_adequacy_parameter");
  Mij_ = metaData.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "metric_tensor");

  tmpFile.open("resAdeq.txt", std::fstream::app);
}

ComputeTAMSKEpsResAdequacyElemAlgorithm::~ComputeTAMSKEpsResAdequacyElemAlgorithm()
{
  tmpFile.close();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeTAMSKEpsResAdequacyElemAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const double dt = realm_.get_time_step();

  // fill in elemental values
  stk::mesh::Selector s_locally_owned_union =
      meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &elem_buckets =
      realm_.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(
            b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->num_integration_points();

    // resize std::vectors based on element type for local storage
    ws_coordinates.resize(nDim_ * nodesPerElement);
    ws_dndx.resize(nDim_ * numScsIp * nodesPerElement);
    ws_deriv.resize(nDim_ * numScsIp * nodesPerElement);
    ws_det_j.resize(numScsIp);
    ws_scs_areav.resize(numScsIp*nDim_);
    ws_shape_function.resize(numScsIp*nodesPerElement);

    ws_mut.resize(nodesPerElement);

    ws_uNp1.resize(nDim_ * nodesPerElement);
    ws_rhoNp1.resize(nodesPerElement);
    ws_tke.resize(nodesPerElement);
    ws_avgTime.resize(nodesPerElement);
    ws_tdr.resize(nodesPerElement);
    ws_alpha.resize(nodesPerElement);

    ws_avgU.resize(nDim_ * nodesPerElement);
    ws_avgRho.resize(nodesPerElement);

    fluctUjScs.resize(nDim_);
    avgUjScs.resize(nDim_);
    coordScs.resize(nDim_);
    fluctDudxScs.resize(nDim_ * nDim_);
    avgDudxScs.resize(nDim_ * nDim_);
    dudxScs.resize(nDim_ * nDim_);
    tauSGET.resize(nDim_ * nDim_);
    tauSGRS.resize(nDim_ * nDim_);
    tau.resize(nDim_ * nDim_);
    Psgs.resize(nDim_ * nDim_);

    // pointers to the local storage vectors
    double *p_coordinates = &ws_coordinates[0];
    double *p_dndx = &ws_dndx[0];
    double *p_deriv = &ws_deriv[0];
    double *p_det_j = &ws_det_j[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_shape_function = &ws_shape_function[0];

    double *p_mut = &ws_mut[0];

    double *p_uNp1 = &ws_uNp1[0];
    double *p_rhoNp1 = &ws_rhoNp1[0];
    double *p_avgTime = &ws_avgTime[0];
    double *p_tke = &ws_tke[0];
    double *p_tdr = &ws_tdr[0];
    double *p_alpha = &ws_alpha[0];

    double *p_avgU = &ws_avgU[0];
    double *p_avgRho = &ws_avgRho[0];

    double *p_fluctUjScs = &fluctUjScs[0];
    double *p_avgUjScs = &avgUjScs[0];
    double *p_coordScs = &coordScs[0];
    double *p_fluctDudxScs = &fluctDudxScs[0];
    double *p_avgDudxScs = &avgDudxScs[0];
    double *p_dudxScs = &dudxScs[0];

    double *p_tauSGET = &tauSGET[0];
    double *p_tauSGRS = &tauSGRS[0];
    double *p_tau = &tau[0];
    double *p_Psgs = &Psgs[0];

    // Get resolution adequacy field for filling
    double *resAdeq = stk::mesh::field_data(*resAdeq_, b);
    double *avgResAdeq = stk::mesh::field_data(*avgResAdeq_, b);

    meSCS->shape_fcn(&p_shape_function[0]);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // Will average mdot here as well, since it's defined on elements
      double * mdot = stk::mesh::field_data(*massFlowRate_, b, k );
      double * avgMdot = stk::mesh::field_data(*avgMdot_, b, k);

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
      double M43[nDim_][nDim_];
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

      const double CM43 = get_M43_constant(D);

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates_, node);
        const double * vel = stk::mesh::field_data(*velocityNp1_, node);   
        const double * avgVel = stk::mesh::field_data(*avgVelocity_, node);  

        // gather scalars
        p_mut[ni]     = *stk::mesh::field_data(*turbVisc_, node);
        p_rhoNp1[ni]  = *stk::mesh::field_data(*densityNp1_, node);
        p_tke[ni]     = *stk::mesh::field_data(*tkeNp1_, node); 
        p_tdr[ni]     = *stk::mesh::field_data(*tdrNp1_, node);
        p_avgTime[ni] = *stk::mesh::field_data(*avgTime_, node);
        p_alpha[ni]   = *stk::mesh::field_data(*alphaNp1_, node);

        p_avgRho[ni]  = *stk::mesh::field_data(*avgDensity_, node);

        // gather vectors
        const int niNdim = ni*nDim_;
        for ( int j=0; j < nDim_; ++j ) {
          p_coordinates[niNdim+j] = coords[j];
          p_uNp1[niNdim+j] = vel[j];
          p_avgU[niNdim+j] = avgVel[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // compute dndx: FIXME: shifted option?
      meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

      double resAdeqSum = 0.0;
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // zero out; instantaneous quantities
        double tkeScs = 0.0;
        double tdrScs = 0.0;
        double avgTime = 0.0;
        double alphaScs = 0.0;
        double mutScs = 0.0;

        // zero out; fluctuating quantities
        double fluctRhoScs = 0.0;
  
        // zero out; mean quantities
        double avgRhoScs = 0.0;

        // zero out; vectors and tensors
        for ( unsigned j = 0; j < nDim_; ++j ) {
          p_fluctUjScs[j] = 0.0;
          p_avgUjScs[j] = 0.0;
          p_coordScs[j] = 0.0;
          for ( unsigned l = 0; l < nDim_; ++l) {
            p_fluctDudxScs[j*nDim_ + l] = 0.0;
            p_avgDudxScs[j*nDim_ + l] = 0.0;
            p_dudxScs[j*nDim_ + l] = 0.0;
            p_tauSGRS[j*nDim_ + l] = 0.0;
            p_tauSGET[j*nDim_ + l] = 0.0;
            p_tau[j*nDim_ + l] = 0.0;
            p_Psgs[j*nDim_ + l] = 0.0;
          }
        }

        const int offSet = ip*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          // save off shape function
          const double r = p_shape_function[offSet+ic];

          // scalars 
          tkeScs += r*p_tke[ic];
          tdrScs += r*p_tdr[ic];
          avgTime += r*p_avgTime[ic];
          alphaScs += r*p_alpha[ic];
          mutScs += r*p_mut[ic];

          fluctRhoScs += r*(p_rhoNp1[ic] - p_avgRho[ic]); 

          avgRhoScs += r*p_avgRho[ic];

          const int offSetDnDx = nDim_*nodesPerElement*ip + ic*nDim_;
          for ( int j = 0; j < nDim_; ++j ) {
            const double fluctUj = p_uNp1[ic*nDim_+j] - p_avgU[ic*nDim_+j];
            const double avgUj = p_avgU[ic*nDim_+j];
            const double uj = p_uNp1[ic*nDim_+j];
            const double coordj = p_coordinates[ic*nDim_+j];

            p_fluctUjScs[j] += r*fluctUj;
            p_avgUjScs[j] += r*avgUj;
            p_coordScs[j] += r*coordj;

            for (unsigned k = 0; k < nDim_; ++k) {
              p_fluctDudxScs[nDim_*j + k] += p_dndx[offSetDnDx+k]*fluctUj;
              p_avgDudxScs[nDim_*j + k] += p_dndx[offSetDnDx+k]*avgUj;
              p_dudxScs[nDim_*j + k] += p_dndx[offSetDnDx+k]*uj;
            }
          }
        }

        // average mdot
        const double TaveScs = tkeScs / std::max(tdrScs, 1.e-16);
        const double weightAvgScs = std::max(1.0 - dt/TaveScs, 0.0);
        const double weightInstScs = std::min(dt/TaveScs, 1.0);

        //avgMdot[ip] = weightAvgScs * avgMdot[ip] + weightInstScs * mdot[ip];


        const double epsilon13 = stk::math::pow(tdrScs, 1.0/3.0);

        for (unsigned i = 0; i < nDim_; ++i) {

          for (unsigned j = 0; j < nDim_; ++j) {
            // Calculate tauSGRS_ij = 2*alpha*nu_t*<S_ij> where nu_t comes from
            // the SST model and <S_ij> is the strain rate tensor based on the
            // mean quantities... i.e this is (tauSGRS = alpha*tauSST)
            // The 2 in the coeff cancels with the 1/2 in the strain rate tensor
            const double coeffSGRS = alphaScs * mutScs;
            p_tauSGRS[i*nDim_ + j] += p_avgDudxScs[i*nDim_ + j] + p_avgDudxScs[j*nDim_ + i];
            p_tauSGRS[i*nDim_ + j] *= coeffSGRS;

            for (unsigned l = 0; l < nDim_; ++l) {
              // Calculate tauSGET_ij = CM43*<eps>^(1/3)*(M43_ik*dkuj' + M43_jkdkui')
              // where <eps> is the mean dissipation backed out from the SST mean k and
              // mean omega and dkuj' is the fluctuating velocity gradients.
              const double coeffSGET = avgRhoScs * CM43 * epsilon13;
              p_tauSGET[i*nDim_ + j] += coeffSGET * (M43[i][l] * p_fluctDudxScs[j*nDim_ + l] + 
                                                   M43[j][l] * p_fluctDudxScs[i*nDim_ + l]);
            }
          }
          //p_tauSGRS[i*nDim_ + i] += -alphaScs*2.0/3.0*tkeScs;
        }

        // Calculate the full subgrid stress including the isotropic portion
        // FIXME: Do i need a rho in here?
        for (unsigned i = 0; i < nDim_; ++i) 
          for (unsigned j = 0; j < nDim_; ++j)
            p_tau[i*nDim_ + j] = p_tauSGRS[i*nDim_ + j] + p_tauSGET[i*nDim_ + j] - 
                        ((i==j) ? 2.0/3.0*alphaScs*tkeScs : 0.0);

        // Calculate the SGS production PSGS_ij = 1/2(tau_ik*djuk + tau_jk*diuk)
        // where diuj is the instantaneous velocity gradients
        for (unsigned i = 0; i < nDim_; ++i) {
          for (unsigned j = 0; j < nDim_; ++j) {
            for (unsigned l = 0; l < nDim_; ++l)
              p_Psgs[i*nDim_ + j] += p_tau[i*nDim_ + l] * p_dudxScs[l*nDim_ + j] + 
                                    p_tau[j*nDim_ + l] * p_dudxScs[l*nDim_ + i];
            p_Psgs[i*nDim_+ j] *= 0.5;
          }
        }  

        for (unsigned i = 0; i < nDim_; ++i) 
          for (unsigned j = 0; j < nDim_; ++j) {
            PM[i][j] = 0.0;
            for (unsigned l = 0; l < nDim_; ++l) 
              PM[i][j] += p_Psgs[i*nDim_ + l] * Mij[l][j];
          }

        // Scale PM first
        const double T_ke = avgTime; //tkeScs / tdrScs;
        const double v2 = 1.0/0.22 * (mutScs / T_ke);
        const double PMscale = std::pow(1.5*alphaScs*v2,-1.5);
        if (v2 == 0.0)
          throw std::runtime_error("TAMSKEResAdequacy: v2 is 0, will cause NaN");
        for (unsigned i = 0; i < nDim_; ++i)
          for (unsigned j = 0; j < nDim_; ++j)
            PM[i][j] = PM[i][j] * PMscale;

        //FIXME: PM is not symmetric....
        EigenDecomposition::unsym_matrix_force_sym<double>(PM, Q, D);

        const double maxPM = std::max(std::abs(D[0][0]), std::max(std::abs(D[1][1]), std::abs(D[2][2])));

        //tmpFile << p_coordScs[0] << " " << p_coordScs[1] << " " << p_coordScs[2] << " " << maxPM << " "<< T_ke << " " << v2 << " " << PM[0][0] << " " << p_Psgs[0] << " " << p_tau[0] << " " << alphaScs << " " << mutScs << " " << epsilon13 << std::endl;        
        
        resAdeqSum += maxPM;
      }
      
      // Update the instantaneous resAdeq field
      resAdeq[k] = resAdeqSum/numScsIp;
      // FIXME: Limiters as in CDP...
      resAdeq[k] = std::min(resAdeq[k],30.0);
  
      // Update the average field here as well since it is an element quantity
      // and the averaging algorithm operates on the nodes
      double elemTke = 0.0;
      double elemTdr = 0.0;
      double elemAvgTime = 0.0;
      double elemAlpha = 0.0;
      for ( int ic = 0; ic < nodesPerElement; ++ic ) {
        elemTke += p_tke[ic];
	elemTdr += p_tdr[ic];
        elemAvgTime += p_avgTime[ic];
        elemAlpha += p_alpha[ic];
      }

      // elemAlpha is sum of nodal alpha, so we are checking alpha >= 1 here
      if (elemAlpha >= (double)nodesPerElement)
        resAdeq[k] = std::min(resAdeq[k],1.0);

      // The division by nodesPerElement cancels out here
      const double T_ave = elemTke/elemTdr;
      //const double T_ave = elemAvgTime/(double)nodesPerElement;
      
      const double weightAvg = std::max(1.0 - dt/T_ave, 0.0);
      const double weightInst = std::min(dt/T_ave, 1.0);

      avgResAdeq[k] = weightAvg * avgResAdeq[k] + weightInst * resAdeq[k]; 
    }
  }
}

double ComputeTAMSKEpsResAdequacyElemAlgorithm::get_M43_constant(double D[3][3])
{
  // Coefficients for the polynomial 
  double c[15] = {1.033749474513071,-0.154122686264488,-0.007737595743644,
                  0.177611732560139, 0.060868024017604, 0.162200630336440,
                 -0.041086757724764,-0.027380130027626, 0.005521188430182,
                  0.049139605169403, 0.002926283060215, 0.002672790587853,
                  0.000486437925728, 0.002136258066662, 0.005113058518679};

  if (nDim_ != 3)
     throw std::runtime_error("In ComputeTAMSKEpsResAdequacyElemAlgorithm, requires 3D");

  // FIXME: Can we find a more elegant way to sort the three eigenvalues...
  double smallestEV = stk::math::min(D[0][0], stk::math::min(D[1][1], D[2][2]));
  double largestEV = stk::math::max(D[0][0], stk::math::max(D[1][1], D[2][2]));
  double middleEV = (D[0][0] == smallestEV) ? stk::math::min(D[1][1], D[2][2]) :
                    ((D[1][1] == smallestEV) ? stk::math::min(D[0][0],D[2][2]) :
                                stk::math::min(D[0][0],D[1][1]));

  // Scale the EVs
  middleEV = middleEV/smallestEV;
  largestEV = largestEV/smallestEV;

  double r = stk::math::sqrt(stk::math::pow(middleEV,2) + stk::math::pow(largestEV,2));
  double theta = stk::math::acos(largestEV/r);

  double x = stk::math::log(r);
  double y = stk::math::log(stk::math::sin(2*theta));

  double poly = c[0] +
                c[1]*x + c[2]*y +
                c[3]*x*x + c[4]*x*y + c[5]*y*y +
                c[6]*x*x*x + c[7]*x*x*y + c[8]*x*y*y + c[9]*y*y*y +
                c[10]*x*x*x*x + c[11]*x*x*x*y + c[12]*x*x*y*y + c[13]*x*y*y*y + c[14]*y*y*y*y;

  return poly*CMdeg_;
}

} // namespace nalu
} // namespace sierra
