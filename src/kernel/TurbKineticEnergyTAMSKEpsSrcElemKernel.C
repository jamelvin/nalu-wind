/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TurbKineticEnergyTAMSKEpsSrcElemKernel.h"
#include "FieldTypeDef.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

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
TurbKineticEnergyTAMSKEpsSrcElemKernel<AlgTraits>::
  TurbKineticEnergyTAMSKEpsSrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs,
    const bool lumpedMass)
  : Kernel(),
    lumpedMass_(lumpedMass),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity")),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    tkeProdLimitRatio_(solnOpts.get_turb_model_constant(TM_tkeProdLimitRatio)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(
                 AlgTraits::topo_)
                 ->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();

  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");
  tdrNp1_ = get_field_ordinal(metaData, "total_dissipation_rate");
  densityNp1_ = get_field_ordinal(metaData, "density");
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  visc_ = get_field_ordinal(metaData, "viscosity");
  tvisc_ = get_field_ordinal(metaData, "turbulent_viscosity");
  alpha_ = get_field_ordinal(metaData, "k_ratio");
  minD_ = get_field_ordinal(metaData, "minimum_distance_to_wall");
  prod_ = get_field_ordinal(metaData, "average_production");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(
      AlgTraits::topo_);

  // compute shape function
  if (lumpedMass_)
    get_scv_shape_fn_data<AlgTraits>(
      [&](double* ptr) { meSCV->shifted_shape_fcn(ptr); }, v_shape_function_);
  else
    get_scv_shape_fn_data<AlgTraits>(
      [&](double* ptr) { meSCV->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(visc_, 1);
  dataPreReqs.add_gathered_nodal_field(tvisc_, 1);
  dataPreReqs.add_gathered_nodal_field(alpha_, 1);
  dataPreReqs.add_gathered_nodal_field(minD_, 1);
  dataPreReqs.add_gathered_nodal_field(prod_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCV_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);

  tmpFile.open("tkeSource.txt", std::fstream::app);
}

template <typename AlgTraits>
TurbKineticEnergyTAMSKEpsSrcElemKernel<
  AlgTraits>::~TurbKineticEnergyTAMSKEpsSrcElemKernel()
{
  tmpFile.close();
}

template <typename AlgTraits>
void
TurbKineticEnergyTAMSKEpsSrcElemKernel<AlgTraits>::setup(
  const TimeIntegrator& timeIntegrator)
{
  // FIXME: Hack to match CDP time
  time_ = timeIntegrator.get_current_time() - 440.0;
  dt_ = timeIntegrator.get_time_step();
  step_ = timeIntegrator.get_time_step_count();
}

template <typename AlgTraits>
void
TurbKineticEnergyTAMSKEpsSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_coordScs[AlgTraits::nDim_];

  SharedMemView<DoubleType**>& v_coordinates =
    scratchViews.get_scratch_view_2D(coordinates_);
  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(tkeNp1_);
  SharedMemView<DoubleType*>& v_tdrNp1 =
    scratchViews.get_scratch_view_1D(tdrNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 =
    scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType**>& v_velocityNp1 =
    scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_visc = scratchViews.get_scratch_view_1D(visc_);
  SharedMemView<DoubleType*>& v_tvisc =
    scratchViews.get_scratch_view_1D(tvisc_);
  SharedMemView<DoubleType*>& v_alpha =
    scratchViews.get_scratch_view_1D(alpha_);
  SharedMemView<DoubleType*>& v_minD = scratchViews.get_scratch_view_1D(minD_);
  SharedMemView<DoubleType*>& v_prod = scratchViews.get_scratch_view_1D(prod_);
  SharedMemView<DoubleType***>& v_dndx =
    shiftedGradOp_
      ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv_shifted
      : scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv;
  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {

    // nearest node to ip
    const int nearestNode = ipNodeMap_[ip];

    // save off scvol
    const DoubleType scV = v_scv_volume(ip);

    // zero out values of interest for this ip
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      w_coordScs[j] = 0.0;
    }

    DoubleType rho = 0.0;
    DoubleType tke = 0.0;
    DoubleType tdr = 0.0;
    DoubleType visc = 0.0;
    DoubleType tvisc = 0.0;
    DoubleType alpha = 0.0;
    DoubleType minD = 0.0;
    DoubleType prod = 0.0;

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const DoubleType r = v_shape_function_(ip, ic);

      rho += r * v_densityNp1(ic);
      tke += r * v_tkeNp1(ic);
      tdr += r * v_tdrNp1(ic);
      visc += r * v_visc(ic);
      tvisc += r * v_tvisc(ic);
      alpha += r * v_alpha(ic);
      minD += r * v_minD(ic);
      prod += r * v_prod(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        w_coordScs[i] += r * v_coordinates(ic, i);
      }
    }

    // The changes to the standard KE RANS approach in TAMS result in two
    // changes: 1) improvements to the production based on the resolved
    // fluctuations 2) the addition of alpha to modify the production 3) the
    // averaging of the production, thus it's calculation has been moved to the
    //    averaging function
    const DoubleType Pk = prod;

    // dissipation
    const DoubleType Dk = rho * tdr;

    // wall distance source term (rho's cancel out...)
    const DoubleType lFac = 2.0 * visc / minD / minD;
    DoubleType Lk = -lFac * tke;

    // std::ofstream tmpFile;
    // tmpFile.open("TKEsrc.txt");

    // if ((step_ % 1) == 0)
    //{
    //  tmpFile << w_coordScs[0] << w_coordScs[1] << w_coordScs[2] << Pk << Dk
    //  << Lk << visc << minD << tke << tdr << std::endl;
    //}

    // assemble RHS and LHS
    rhs(nearestNode) += (Pk - Dk + Lk) * scV;
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) += v_shape_function_(ip, ic) * lFac * scV;
    }
  }
}

INSTANTIATE_KERNEL(TurbKineticEnergyTAMSKEpsSrcElemKernel)

} // namespace nalu
} // namespace sierra
