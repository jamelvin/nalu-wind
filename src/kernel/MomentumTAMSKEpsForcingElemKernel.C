/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumTAMSKEpsForcingElemKernel.h"
#include "AlgTraits.h"
#include "EigenDecomposition.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
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

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MomentumTAMSKEpsForcingElemKernel<AlgTraits>::MomentumTAMSKEpsForcingElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* viscosity,
  ScalarFieldType* turbViscosity,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    turbViscosity_(turbViscosity->mesh_meta_data_ordinal()),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(
                 AlgTraits::topo_)
                 ->ipNodeMap())
{
  pi_ = stk::math::acos(-1.0);
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  densityNp1_ = get_field_ordinal(metaData, "density");
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");
  tdrNp1_ = get_field_ordinal(metaData, "total_dissipation_rate");
  alphaNp1_ = get_field_ordinal(metaData, "k_ratio");
  Mij_ = get_field_ordinal(metaData, "metric_tensor");

  avgVelocity_ = get_field_ordinal(metaData, "average_velocity");
  avgDensity_ = get_field_ordinal(metaData, "average_density");
  avgTime_ = get_field_ordinal(metaData, "average_time");

  avgResAdeq_ = get_field_ordinal(metaData, "avg_res_adequacy_parameter");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  minDist_ = get_field_ordinal(metaData, "minimum_distance_to_wall");

  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(
      AlgTraits::topo_);

  get_scv_shape_fn_data<AlgTraits>(
    [&](double* ptr) { meSCV->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(turbViscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(avgVelocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(avgDensity_, 1);
  dataPreReqs.add_gathered_nodal_field(alphaNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(avgTime_, 1);
  dataPreReqs.add_gathered_nodal_field(minDist_, 1);
  dataPreReqs.add_gathered_nodal_field(avgResAdeq_, 1);
  dataPreReqs.add_gathered_nodal_field(Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

  // master element data
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);

  tmpFile.open("forcingField.txt", std::fstream::app);
}

template <typename AlgTraits>
MomentumTAMSKEpsForcingElemKernel<AlgTraits>::~MomentumTAMSKEpsForcingElemKernel()
{
  tmpFile.close();
}

template <typename AlgTraits>
void
MomentumTAMSKEpsForcingElemKernel<AlgTraits>::setup(
  const TimeIntegrator& timeIntegrator)
{
  // FIXME: Hack to match CDP time
  time_ = timeIntegrator.get_current_time() - 440.0;
  dt_ = timeIntegrator.get_time_step();
  step_ = timeIntegrator.get_time_step_count();
}

template <typename AlgTraits>
void
MomentumTAMSKEpsForcingElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_coordScs[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_avgUScs[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_fluctUScs[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_MijElem[AlgTraits::nDim_][AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_uNp1 =
    scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType**>& v_coordinates =
    scratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType*>& v_rhoNp1 =
    scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(tkeNp1_);
  SharedMemView<DoubleType*>& v_tdrNp1 =
    scratchViews.get_scratch_view_1D(tdrNp1_);
  SharedMemView<DoubleType*>& v_viscosity =
    scratchViews.get_scratch_view_1D(viscosity_);
  SharedMemView<DoubleType*>& v_turbViscosity =
    scratchViews.get_scratch_view_1D(turbViscosity_);
  SharedMemView<DoubleType**>& v_avgU =
    scratchViews.get_scratch_view_2D(avgVelocity_);
  SharedMemView<DoubleType*>& v_alphaNp1 =
    scratchViews.get_scratch_view_1D(alphaNp1_);
  SharedMemView<DoubleType*>& v_avgTime =
    scratchViews.get_scratch_view_1D(avgTime_);
  SharedMemView<DoubleType*>& v_minDist =
    scratchViews.get_scratch_view_1D(minDist_);
  SharedMemView<DoubleType*>& v_avgResAdeq =
    scratchViews.get_scratch_view_1D(avgResAdeq_);
  SharedMemView<DoubleType***>& v_Mij = 
    scratchViews.get_scratch_view_3D(Mij_);

  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {

    // nearest node for this ip
    const int nearestNode = ipNodeMap_[ip];

    // zero out values of interest for this ip
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      w_coordScs[j] = 0.0;
      w_avgUScs[j] = 0.0;
      w_fluctUScs[j] = 0.0;
      for (int k = 0; k < AlgTraits::nDim_; ++k) {
        w_MijElem[j][k] = 0.0;
      }
    }

    DoubleType muScs = 0.0;
    DoubleType mu_tScs = 0.0;
    DoubleType rhoScs = 0.0;
    DoubleType tkeScs = 0.0;
    DoubleType tdrScs = 0.0;
    DoubleType avgTimeScs = 0.0;
    DoubleType alphaScs = 0.0;
    DoubleType wallDistScs = 0.0;
    DoubleType avgResAdeqScs = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function_(ip, ic);

      muScs += r * v_viscosity(ic);
      mu_tScs += r * v_turbViscosity(ic);
      rhoScs += r * v_rhoNp1(ic);
      tkeScs += r * v_tkeNp1(ic);
      tdrScs += r * v_tdrNp1(ic);
      avgTimeScs += r * v_avgTime(ic);
      alphaScs += r * v_alphaNp1(ic);
      wallDistScs += r * v_minDist(ic);
      avgResAdeqScs += r * v_avgResAdeq(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        w_coordScs[i] += r * v_coordinates(ic, i);
        w_avgUScs[i] += r * v_avgU(ic, i);
        w_fluctUScs[i] += r * (v_uNp1(ic, i) - v_avgU(ic, i));

        // Don't allow Mij to vary over the element, so take it as a direct average over nodes
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_MijElem[i][j] += v_Mij(ic, i, j)/AlgTraits::nodesPerElement_;
        }

      }
    }

    // First we calculate the a_i's
    const double FORCING_CL = 4.0;
    const double Ceta = 70.0;
    const double Ct = 6.0;
    const double BL_T = 1.0;
    const double BL_KOL = 1.0;
    const double FORCING_FACTOR = 8.0;

    const DoubleType periodicForcingLengthX = pi_;
    const DoubleType periodicForcingLengthY = 0.25;
    const DoubleType periodicForcingLengthZ = 3.0 / 8.0 * pi_;

    DoubleType length =
      FORCING_CL * stk::math::pow(alphaScs * tkeScs, 1.5) / tdrScs;
    length = stk::math::max(length,
      Ceta * (stk::math::pow(muScs, 0.75) / stk::math::pow(tdrScs, 0.25)));
    // FIXME: For channel, only want to clip in wall normal direction with wallDist
    //        For other flows, will need a better approach...
    DoubleType lengthY = stk::math::min(length, wallDistScs);

    DoubleType T_alpha = alphaScs * tkeScs / tdrScs;
    T_alpha = stk::math::max(T_alpha, Ct * stk::math::sqrt(muScs / tdrScs));
    T_alpha = BL_T * T_alpha;

    const DoubleType ceilLengthX = stk::math::max(length, 2.0 * w_MijElem[0][0]);
    const DoubleType ceilLengthY = stk::math::max(lengthY, 2.0 * w_MijElem[1][1]);
    const DoubleType ceilLengthZ = stk::math::max(length, 2.0 * w_MijElem[2][2]);

    const DoubleType clipLengthX =
      stk::math::min(ceilLengthX, periodicForcingLengthX);
    const DoubleType clipLengthY =
      stk::math::min(ceilLengthY, periodicForcingLengthY);
    const DoubleType clipLengthZ =
      stk::math::min(ceilLengthZ, periodicForcingLengthZ);

    // FIXME: Hack to do a round/floor/ceil/mod operation since it isnt in
    // stk::math right now
    DoubleType ratioX;
    DoubleType ratioY;
    DoubleType ratioZ;
    for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
      double tmpD = stk::simd::get_data(clipLengthX, simdIndex);
      double tmpN = stk::simd::get_data(periodicForcingLengthX, simdIndex);
      double tmp = std::floor(tmpN / tmpD + 0.5);
      stk::simd::set_data(ratioX, simdIndex, tmp);

      tmpD = stk::simd::get_data(clipLengthY, simdIndex);
      tmpN = stk::simd::get_data(periodicForcingLengthY, simdIndex);
      tmp = std::floor(tmpN / tmpD + 0.5);
      stk::simd::set_data(ratioY, simdIndex, tmp);

      tmpD = stk::simd::get_data(clipLengthZ, simdIndex);
      tmpN = stk::simd::get_data(periodicForcingLengthZ, simdIndex);
      tmp = std::floor(tmpN / tmpD + 0.5);
      stk::simd::set_data(ratioZ, simdIndex, tmp);
    }
    // const DoubleType ratioX =
    // stk::math::floor(periodicForcingLengthX/clipLengthX + 0.5); const
    // DoubleType ratioY = stk::math::floor(periodicForcingLengthY/clipLengthY +
    // 0.5); const DoubleType ratioZ =
    // stk::math::floor(periodicForcingLengthZ/clipLengthZ + 0.5);

    const DoubleType denomX = periodicForcingLengthX / ratioX;
    const DoubleType denomY = periodicForcingLengthY / ratioY;
    const DoubleType denomZ = periodicForcingLengthZ / ratioZ;

    const DoubleType ax = pi_ / denomX;
    const DoubleType ay = pi_ / denomY;
    const DoubleType az = pi_ / denomZ;

    // Then we calculate the arguments for the Taylor-Green Vortex
    const DoubleType xarg = ax * (w_coordScs[0] + w_avgUScs[0] * time_);
    const DoubleType yarg = ay * (w_coordScs[1] + w_avgUScs[1] * time_);
    const DoubleType zarg = az * (w_coordScs[2] + w_avgUScs[2] * time_);

    // Now we calculate the initial Taylor-Green field
    DoubleType hX = 1.0 / 3.0 * stk::math::cos(xarg) * stk::math::sin(yarg) *
                    stk::math::sin(zarg);
    DoubleType hY =
      -1.0 * stk::math::sin(xarg) * stk::math::cos(yarg) * stk::math::sin(zarg);
    DoubleType hZ = 2.0 / 3.0 * stk::math::sin(xarg) * stk::math::sin(yarg) *
                    stk::math::cos(zarg);

    // Now we calculate the scaling of the initial field
    // FIXME: Pass the 0.22 as another turbulence constant (V2F_Cmu)
    const DoubleType v2Scs = mu_tScs / (0.22 * rhoScs * avgTimeScs);
    const DoubleType F_target =
      FORCING_FACTOR * stk::math::sqrt(alphaScs * v2Scs) / T_alpha;

    const DoubleType prod_r_temp =
      (F_target * dt_) *
      (hX * w_fluctUScs[0] + hY * w_fluctUScs[1] + hZ * w_fluctUScs[2]);
  
    const DoubleType prod_r_sgn = stk::math::if_then_else(prod_r_temp < 0.0, -1.0, 1.0);
    const DoubleType prod_r_abs = prod_r_sgn * prod_r_temp;

    const DoubleType prod_r = stk::math::if_then_else(prod_r_abs >= 1.0e-15, prod_r_temp, 0.0);

    const DoubleType arg1 = stk::math::sqrt(avgResAdeqScs) - 1.0;
    const DoubleType arg = stk::math::if_then_else(arg1 < 0.0, 1.0 - 1.0 / stk::math::sqrt(avgResAdeqScs),arg1);

    const DoubleType a_sign = stk::math::tanh(arg);

    DoubleType Sa = a_sign;

    const DoubleType a_kol = stk::math::min(
      BL_KOL * stk::math::sqrt(muScs * tdrScs / rhoScs) / tkeScs, 1.0);

    // FIXME: Can I do a compound if statement with if_then... it was not
    // working...
    for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
      double tmp_asign = stk::simd::get_data(a_sign, simdIndex);
      double tmp_akol = stk::simd::get_data(a_kol, simdIndex);
      double tmp_alpha = stk::simd::get_data(alphaScs, simdIndex);
      double tmp_Sa = stk::simd::get_data(Sa, simdIndex);

      if (tmp_asign < 0.0) {
        if (tmp_alpha <= tmp_akol)
          tmp_Sa = tmp_Sa - (1.0 + tmp_akol - tmp_alpha) * tmp_asign;
      } else {
        if (tmp_alpha >= 1.0)
          tmp_Sa = tmp_Sa - tmp_alpha * tmp_asign;
      }
      stk::simd::set_data(Sa, simdIndex, tmp_Sa);
    }

    // stk::math::if_then_else((a_sign < 0.0) && (alphaScs <= a_kol), Sa = Sa -
    // (1.0+a_kol-alphaScs)*a_sign, Sa = Sa*1.0);

    // stk::math::if_then_else((a_sign >= 0.0) && (alphaScs >= 1.0), Sa = Sa -
    // alphaScs*a_sign, Sa = Sa*1.0);

    const DoubleType fd_temp = avgResAdeqScs;

    DoubleType C_F;
    // FIXME: Can I do a compound if statement with if_then... it was not
    // working...
    for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
      double tmp_fd = stk::simd::get_data(fd_temp, simdIndex);
      double tmp_prodr = stk::simd::get_data(prod_r, simdIndex);
      double tmp_CF = stk::simd::get_data(C_F, simdIndex);
      double tmp_Ftarget = stk::simd::get_data(F_target, simdIndex);
      double tmp_Sa = stk::simd::get_data(Sa, simdIndex);

      if ((tmp_fd < 1.0) && (tmp_prodr >= 0.0))
        tmp_CF = -1.0 * tmp_Ftarget * tmp_Sa;
      else
        tmp_CF = 0.0;
      stk::simd::set_data(C_F, simdIndex, tmp_CF);
    }
    // stk::math::if_then_else((fd_temp < 1.0) && (prod_r >= 0.0), C_F = -1.0 *
    // F_target * Sa, C_F = 0.0);

    // Since we aren't projecting, ignore scaling by dt which is done before
    // projection and then removed...
    const DoubleType norm = C_F; //* dt_;

    // Now we determine the actual forcing field
    // FIXME: DEBUG HACK! Remove 100, only for testing
    DoubleType gX = norm * hX; //* 10.0;// / dt_;
    DoubleType gY = norm * hY; //* 10.0;// / dt_;
    DoubleType gZ = norm * hZ; //* 10.0;// / dt_;

    if (((step_ % 10000) == 9000) || ((step_ % 10000) == 4000))
    { 
      tmpFile << w_coordScs[0] << w_coordScs[1] << w_coordScs[2] << gX << gY << gZ << norm << alphaScs << v2Scs << tkeScs << tdrScs << fd_temp << Sa << a_kol << a_sign << std::endl;
    }

    //if ((step_ % 1) == 0)
    //{ 
    //  tmpFile << w_coordScs[0] << w_coordScs[1] << w_coordScs[2] << gX << gY << gZ << norm << alphaScs << v2Scs << tkeScs << tdrScs << fd_temp << hX << hY << hZ << std::endl;
    //}

    // g_i is not divergence free, so we must solve a Poisson equation
    // rhs = G * normal * area;
    // lhs = grad operator * normal * area;

    // TODO: Assess viability of first approach where we don't solve a poisson
    // problem and allow the field be divergent, which should get projected out
    // anyway This means we only have a contribution to the RHS here
    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;
    // for (int j = 0; j < AlgTraits::nDim_; j++_ {
    rhs(nnNdim + 0) += gX * scV;
    rhs(nnNdim + 1) += gY * scV;
    rhs(nnNdim + 2) += gZ * scV;
    //}
  }
}

INSTANTIATE_KERNEL(MomentumTAMSKEpsForcingElemKernel)

} // namespace nalu
} // namespace sierra
