/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumTAMSKEpsDiffElemKernel.h"
#include "AlgTraits.h"
#include "EigenDecomposition.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

#include "utils/TAMSUtils.h"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MomentumTAMSKEpsDiffElemKernel<AlgTraits>::MomentumTAMSKEpsDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* viscosity,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    includeDivU_(solnOpts.includeDivU_),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(
             AlgTraits::topo_)
             ->adjacentNodes()),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  densityNp1_ = get_field_ordinal(metaData, "density");
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");
  tdrNp1_ = get_field_ordinal(metaData, "total_dissipation_rate");
  alphaNp1_ = get_field_ordinal(metaData, "k_ratio");
  Mij_ = get_field_ordinal(metaData, "metric_tensor");

  avgVelocity_ = get_field_ordinal(metaData, "average_velocity");
  avgDensity_ = get_field_ordinal(metaData, "average_density");

  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(
      AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>(
    [&](double* ptr) { meSCS->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(avgVelocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(avgDensity_, 1);
  dataPreReqs.add_gathered_nodal_field(alphaNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

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
MomentumTAMSKEpsDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType**>& v_uNp1 =
    scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_rhoNp1 =
    scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(tkeNp1_);
  SharedMemView<DoubleType*>& v_tdrNp1 =
    scratchViews.get_scratch_view_1D(tdrNp1_);
  SharedMemView<DoubleType*>& v_viscosity =
    scratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType**>& v_avgU =
    scratchViews.get_scratch_view_2D(avgVelocity_);
  SharedMemView<DoubleType*>& v_avgRho =
    scratchViews.get_scratch_view_1D(avgDensity_);
  SharedMemView<DoubleType*>& v_alphaNp1 =
    scratchViews.get_scratch_view_1D(alphaNp1_);
  SharedMemView<DoubleType***>& v_Mij = 
    scratchViews.get_scratch_view_3D(Mij_);

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
      Mij[i][j] = 0.0;

  // determine scs values of interest
  for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
    for (unsigned i = 0; i < AlgTraits::nDim_; i++)
      for (unsigned j = 0; j < AlgTraits::nDim_; j++)
        Mij[i][j] += v_Mij(ic, i, j)/AlgTraits::nodesPerElement_;
  }

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
  DoubleType CM43 = tams_utils::get_M43_constant<DoubleType, AlgTraits::nDim_>(D, CMdeg_);

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {

    // left and right nodes for this ip
    const int il = lrscv_[2 * ip];
    const int ir = lrscv_[2 * ip + 1];

    // save off some offsets
    const int ilNdim = il * AlgTraits::nDim_;
    const int irNdim = ir * AlgTraits::nDim_;

    DoubleType muScs = 0.0;
    DoubleType fluctRhoScs = 0.0;
    DoubleType avgRhoScs = 0.0;
    DoubleType tkeScs = 0.0;
    DoubleType tdrScs = 0.0;
    DoubleType alphaScs = 0.0;
    DoubleType avgDivU = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function_(ip, ic);

      muScs += r * v_viscosity(ic);
      fluctRhoScs += r * (v_rhoNp1(ic) - v_avgRho(ic));
      avgRhoScs += r * v_avgRho(ic);
      tkeScs += r * v_tkeNp1(ic);
      tdrScs += r * v_tdrNp1(ic);
      alphaScs += r * v_alphaNp1(ic);

      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        const DoubleType avgUj = v_avgU(ic, j);
        avgDivU += avgUj * v_dndx(ip, ic, j);
      }
    }

    // This is the divU term for the average quantities in the model for
    // tau_ij^SGRS Since we are letting SST calculate it's normal mu_t, we need
    // to scale by alpha here
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      const DoubleType avgDivUstress = 2.0 / 3.0 * alphaScs * muScs * avgDivU *
                                       v_scs_areav(ip, i) * includeDivU_;
      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;
      rhs(indexL) -= avgDivUstress;
      rhs(indexR) += avgDivUstress;
    }

    // FIXME: Does this need a rho in it?
    const DoubleType epsilon13Scs = stk::math::pow(tdrScs, 1.0 / 3.0);

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // Related to LHS, currently unused: FIXME: add some implicitness
      const int icNdim = ic * AlgTraits::nDim_;

      for (int i = 0; i < AlgTraits::nDim_; ++i) {

        // FIXME: Don't believe we need these terms...
        // tke stress term
        // const DoubleType twoThirdRhoTke =
        //  2.0 / 3.0 * alphaScs * rhoScs * tkeScs * v_scs_areav(ip, i);

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
            // FIXME: I need to verify this form, fluctRho or avgRho, right
            // indices on axj and dndx
            // ..., do I need a deviatoric part only...
            // fluctRho will be 0 for incompressible, so if that's the right
            // term, need a better way to handle it, probably up above...
            lhsfacDiff_i += -avgRhoScs * CM43 * epsilon13Scs * M43[j][k] *
                            v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_i =
            -alphaScs * muScs * v_dndx(ip, ic, j) * axj;

          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;
          lhs_riCSGRS_i += lhsfacDiffSGRS_i;

          // -mut^ik*duj/dxk*A_j
          DoubleType lhsfacDiff_j = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            // FIXME: See above notes...
            lhsfacDiff_j += -avgRhoScs * CM43 * epsilon13Scs * M43[i][k] *
                            v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_j =
            -alphaScs * muScs * v_dndx(ip, ic, i) * axj;

          // NOTE: lhs (implicit only from the fluctuating term, u' = u - <u>, so the lhs can function
          // as normal as it will only take the instantaneous part of the fluctuation u and the 
          // rhs can just stick with the fluctuating quantities
          lhs(indexL, icNdim + j) += lhsfacDiff_j;
          lhs(indexR, icNdim + j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
          rhs(indexR) += lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
        }

        // deal with accumulated lhs and flux for -mut^jk*dui/dxk*Aj
        // lhs handled only for fluctuating term (see NOTE above)
        lhs(indexL, icNdim + i) += lhs_riC_i;
        lhs(indexR, icNdim + i) -= lhs_riC_i;
        const DoubleType fluctUi = v_uNp1(ic, i) - v_avgU(ic, i);
        const DoubleType avgUi = v_avgU(ic, i);

        // FIXME: Verify we shouldn't need these 2/3TKE terms...
        // rhs(indexL) -= lhs_riC_i * ui + twoThirdRhoTke + lhs_riCSGRS_i *
        // avgUi + avgTwoThirdRhoTke; rhs(indexR) += lhs_riC_i * ui +
        // twoThirdRhoTke + lhs_riCSGRS_i * avgUi + avgTwoThirdRhoTke;
        rhs(indexL) -= lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
        rhs(indexR) += lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumTAMSKEpsDiffElemKernel)

} // namespace nalu
} // namespace sierra
