/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumTAMSDiffElemKernel.h"
#include "AlgTraits.h"
#include "EigenDecomposition.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MomentumTAMSDiffElemKernel<AlgTraits>::MomentumTAMSDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* viscosity,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    viscosity_(viscosity),
    includeDivU_(solnOpts.includeDivU_),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(
             AlgTraits::topo_)
             ->adjacentNodes()),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  densityNp1_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  tkeNp1_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  sdrNp1_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate");
  alphaNp1_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio");
  Mij_ = metaData.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "metric_tensor");

  avgVelocity_ =
    metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "average_velocity");
  avgDensity_ =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_density");
  avgTke_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_turbulent_ke");
  avgSdr_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_specific_dissipation_rate");

  //resAdeq_ = metaData.get_field<ScalarFieldType>(
  //  stk::topology::NODE_RANK, "resolution_adequacy_parameter");
  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(
      AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>(
    [&](double* ptr) { meSCS->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields
  dataPreReqs.add_coordinates_field(
    *coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*viscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*avgVelocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*avgDensity_, 1);
  dataPreReqs.add_gathered_nodal_field(*avgTke_, 1);
  dataPreReqs.add_gathered_nodal_field(*avgSdr_, 1);
  dataPreReqs.add_gathered_nodal_field(*alphaNp1_, 1);
  dataPreReqs.add_element_field(*Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

  // master element data
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
MomentumTAMSDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_mutijScs[AlgTraits::nDim_ * AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 =
    scratchViews.get_scratch_view_2D(*velocityNp1_);
  SharedMemView<DoubleType*>& v_rhoNp1 =
    scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(*tkeNp1_);
  SharedMemView<DoubleType*>& v_sdrNp1 =
    scratchViews.get_scratch_view_1D(*sdrNp1_);
  SharedMemView<DoubleType*>& v_viscosity = 
    scratchViews.get_scratch_view_1D(*viscosity_);
  SharedMemView<DoubleType**>& v_avgU =
    scratchViews.get_scratch_view_2D(*avgVelocity_);
  SharedMemView<DoubleType*>& v_avgRho =
    scratchViews.get_scratch_view_1D(*avgDensity_);
  SharedMemView<DoubleType*>& v_avgTke =
    scratchViews.get_scratch_view_1D(*avgTke_);
  SharedMemView<DoubleType*>& v_avgSdr =
    scratchViews.get_scratch_view_1D(*avgSdr_);
  SharedMemView<DoubleType*>& v_alphaNp1 =
    scratchViews.get_scratch_view_1D(*alphaNp1_);
  SharedMemView<DoubleType **> &v_Mij =
      scratchViews.get_scratch_view_2D(*Mij_);

  SharedMemView<DoubleType**>& v_scs_areav =
    scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx =
    shiftedGradOp_ ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
                   : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  // Mij, eigenvectors and eigenvalues
  DoubleType Mij[AlgTraits::nDim_][AlgTraits::nDim_];
  DoubleType Q[AlgTraits::nDim_][AlgTraits::nDim_];
  DoubleType D[AlgTraits::nDim_][AlgTraits::nDim_];
  for (unsigned i = 0; i < AlgTraits::nDim_; i++)
    for (unsigned j = 0; j < AlgTraits::nDim_; j++)
      Mij[i][j] = v_Mij(i, j);

  EigenDecomposition::sym_diagonalize<DoubleType>(Mij, Q, D);

  // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
  // so to create M43, we use Q D^(4/3) Q^T
  DoubleType M43[AlgTraits::nDim_][AlgTraits::nDim_];
  for (unsigned i = 0; i < AlgTraits::nDim_; i++)
    for (unsigned j = 0; j < AlgTraits::nDim_; j++)
      M43[i][j] = 0.0;

  const double fourThirds = 4. / 3.;
  for (unsigned k = 0; k < AlgTraits::nDim_; k++) {
    const DoubleType D43 = stk::math::pow(D[k][k], fourThirds);
    for (unsigned i = 0; i < AlgTraits::nDim_; i++) {
      for (unsigned j = 0; j < AlgTraits::nDim_; j++) {
        M43[i][j] += Q[i][k] * Q[j][k] * D43;
      }
    }
  }

  // Compute CM43
  DoubleType CM43 = get_M43_constant(D);

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {

    // left and right nodes for this ip
    const int il = lrscv_[2 * ip];
    const int ir = lrscv_[2 * ip + 1];

    // save off some offsets
    const int ilNdim = il * AlgTraits::nDim_;
    const int irNdim = ir * AlgTraits::nDim_;

    // zero out vector that prevail over all components
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      const int offset = i * AlgTraits::nDim_;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_mutijScs[offset + j] = 0.0;
      }
    }

    DoubleType muScs = 0.0;
    DoubleType fluctRhoScs = 0.0;
    DoubleType avgRhoScs = 0.0;
    DoubleType fluctTkeScs = 0.0;
    DoubleType avgTkeScs = 0.0;
    DoubleType fluctSdrScs = 0.0;
    DoubleType avgSdrScs = 0.0;
    DoubleType alphaScs = 0.0;
    DoubleType avgDivU = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function_(ip, ic);

      muScs       += r * v_viscosity(ic);
      fluctRhoScs += r * (v_rhoNp1(ic) - v_avgRho(ic));
      avgRhoScs   += r * v_avgRho(ic);
      fluctTkeScs += r * (v_tkeNp1(ic) - v_avgTke(ic));
      avgTkeScs   += r * v_avgTke(ic);
      fluctSdrScs += r * (v_sdrNp1(ic) - v_avgSdr(ic));
      avgSdrScs   += r * v_avgSdr(ic);
      alphaScs    += r * v_alphaNp1(ic);

      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        const DoubleType avgUj = v_avgU(ic, j);
        avgDivU += avgUj * v_dndx(ip,ic,j);
      }
    }

    // This is the divU term for the average quantities in the model for tau_ij^SGRS
    // Since we are letting SST calculate it's normal mu_t, we need to scale by alpha
    // here 
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      const DoubleType avgDivUstress = 2.0/3.0*alphaScs*muScs*avgDivU*v_scs_areav(ip,i)*includeDivU_;
      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;
      rhs(indexL) -= avgDivUstress;
      rhs(indexR) += avgDivUstress;
    }

    // FIXME: Does this need a rho in it?
    const DoubleType epsilon13Scs = stk::math::pow(betaStar_*avgTkeScs*avgSdrScs,1.0/3.0);

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const int icNdim = ic * AlgTraits::nDim_;

      for (int i = 0; i < AlgTraits::nDim_; ++i) {

        // FIXME: Don't believe we need these terms...
        // tke stress term
        //const DoubleType twoThirdRhoTke =
        //  2.0 / 3.0 * alphaScs * rhoScs * tkeScs * v_scs_areav(ip, i);

        // tke stress term
        //const DoubleType avgTwoThirdRhoTke =
        //   2.0 / 3.0 * alphaScs * avgRhoScs * avgTkeScs * v_scs_areav(ip, i);

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        // Hybrid turbulence diffusion term; -(mu^jk*dui/dxk + mu^ik*duj/dxk -
        // 2/3*rho*tke*del_ij)*Aj
        DoubleType lhs_riC_i = 0.0;
        DoubleType lhs_riCSGRS_i = 0.0;
        for (int j = 0; j < AlgTraits::nDim_; ++j) {

          const DoubleType axj = v_scs_areav(ip, j);
          const DoubleType fluctUj = v_uNp1(ic, j) - v_avgU(ic, j);
          const DoubleType avgUj = v_avgU(ic, j);

          // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
          DoubleType lhsfacDiff_i = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            //FIXME: I need to verify this form, fluctRho or avgRho, right indices on axj and dndx
            // ..., do I need a deviatoric part only...
            lhsfacDiff_i += -fluctRhoScs * CM43 * epsilon13Scs * M43[j][k] * v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_i = -alphaScs*muScs*v_dndx(ip,ic,j)*axj;

          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;
          lhs_riCSGRS_i += lhsfacDiffSGRS_i;

          // -mut^ik*duj/dxk*A_j
          DoubleType lhsfacDiff_j = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            //FIXME: See above notes...
            lhsfacDiff_j += -fluctRhoScs * CM43 * epsilon13Scs * M43[i][k] * v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_j = -alphaScs*muScs*v_dndx(ip,ic,i)*axj;

          // lhs; il then ir
          lhs(indexL, icNdim + j) += lhsfacDiff_j + lhsfacDiffSGRS_j;
          lhs(indexR, icNdim + j) -= lhsfacDiff_j + lhsfacDiffSGRS_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
          rhs(indexR) += lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
        }

        // deal with accumulated lhs and flux for -mut^jk*dui/dxk*Aj
        lhs(indexL, icNdim + i) += lhs_riC_i + lhs_riCSGRS_i;
        lhs(indexR, icNdim + i) -= lhs_riC_i + lhs_riCSGRS_i;
        const DoubleType fluctUi = v_uNp1(ic, i) - v_avgU(ic, i);
        const DoubleType avgUi = v_avgU(ic, i);

        //FIXME: Verify we shouldn't need these 2/3TKE terms...
        //rhs(indexL) -= lhs_riC_i * ui + twoThirdRhoTke + lhs_riCSGRS_i * avgUi + avgTwoThirdRhoTke;
        //rhs(indexR) += lhs_riC_i * ui + twoThirdRhoTke + lhs_riCSGRS_i * avgUi + avgTwoThirdRhoTke;
        rhs(indexL) -= lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
        rhs(indexR) += lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
      }
    }
  }
}

template <typename AlgTraits>
DoubleType
MomentumTAMSDiffElemKernel<AlgTraits>::get_M43_constant(
  DoubleType D[AlgTraits::nDim_][AlgTraits::nDim_])
{

  // Coefficients for the polynomial 
  double c[15] = {1.033749474513071,-0.154122686264488,-0.007737595743644, 
                  0.177611732560139, 0.060868024017604, 0.162200630336440,
                 -0.041086757724764,-0.027380130027626, 0.005521188430182, 
                  0.049139605169403, 0.002926283060215, 0.002672790587853,
	   	  0.000486437925728, 0.002136258066662, 0.005113058518679};
  

  if (AlgTraits::nDim_ != 3)
     throw std::runtime_error("In MomentumTAMSDiffElemKernel, requires 3D problem");

  // FIXME: Can we find a more elegant way to sort the three eigenvalues...
  DoubleType smallestEV = stk::math::min(D[0][0],stk::math::min(D[1][1],D[2][2]));
  DoubleType largestEV = stk::math::max(D[0][0],stk::math::max(D[1][1],D[2][2]));
  DoubleType middleEV = stk::math::if_then_else(D[0][0] == smallestEV,
			  stk::math::min(D[1][1], D[2][2]),
			  stk::math::if_then_else(D[1][1] == smallestEV,
				stk::math::min(D[0][0],D[2][2]),
				stk::math::min(D[0][0],D[1][1])));
  
  // Scale the EVs
  middleEV = middleEV/smallestEV;
  largestEV = largestEV/smallestEV;

  DoubleType r = stk::math::sqrt(stk::math::pow(middleEV,2) + stk::math::pow(largestEV,2));
  DoubleType theta = stk::math::acos(largestEV/r);

  DoubleType x = stk::math::log(r);
  DoubleType y = stk::math::log(stk::math::sin(2*theta));

  DoubleType poly = c[0] + 
                    c[1]*x + c[2]*y + 
                    c[3]*x*x + c[4]*x*y + c[5]*y*y +
		    c[6]*x*x*x + c[7]*x*x*y + c[8]*x*y*y + c[9]*y*y*y +
   		    c[10]*x*x*x*x + c[11]*x*x*x*y + c[12]*x*x*y*y + c[13]*x*y*y*y + c[14]*y*y*y*y;

  return poly*CMdeg_;

}

INSTANTIATE_KERNEL(MomentumTAMSDiffElemKernel);

} // namespace nalu
} // namespace sierra
