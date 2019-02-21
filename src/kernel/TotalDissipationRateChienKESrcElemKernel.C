/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TotalDissipationRateChienKESrcElemKernel.h"
#include "FieldTypeDef.h"
#include "SolutionOptions.h"

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
TotalDissipationRateChienKESrcElemKernel<AlgTraits>::
  TotalDissipationRateChienKESrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs,
    const bool lumpedMass)
  : Kernel(),
    lumpedMass_(lumpedMass),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity")),
    cEpsOne_(solnOpts.get_turb_model_constant(TM_cEpsOne)),
    cEpsTwo_(solnOpts.get_turb_model_constant(TM_cEpsTwo)),
    fOne_(solnOpts.get_turb_model_constant(TM_fOne)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(
                 AlgTraits::topo_)
                 ->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke", stk::mesh::StateNP1);
  tdrNp1_ = get_field_ordinal(metaData, "total_dissipation_rate", stk::mesh::StateNP1);
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  velocityNp1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNP1);
  visc_ = get_field_ordinal(metaData, "viscosity");
  tvisc_ = get_field_ordinal(metaData, "turbulent_viscosity");
  dplus_ = get_field_ordinal(metaData, "dplus_wall_function");
  minD_ = get_field_ordinal(metaData, "minimum_distance_to_wall");
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
  dataPreReqs.add_gathered_nodal_field(dplus_, 1);
  dataPreReqs.add_gathered_nodal_field(minD_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCV_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
TotalDissipationRateChienKESrcElemKernel<
  AlgTraits>::~TotalDissipationRateChienKESrcElemKernel()
{
}

template <typename AlgTraits>
void
TotalDissipationRateChienKESrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dudx[AlgTraits::nDim_][AlgTraits::nDim_];

  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(tkeNp1_);
  SharedMemView<DoubleType*>& v_tdrNp1 =
    scratchViews.get_scratch_view_1D(tdrNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 =
    scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType**>& v_velocityNp1 =
    scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_visc =
    scratchViews.get_scratch_view_1D(visc_);
  SharedMemView<DoubleType*>& v_tvisc =
    scratchViews.get_scratch_view_1D(tvisc_);
  SharedMemView<DoubleType*>& v_dplus =
    scratchViews.get_scratch_view_1D(dplus_);
  SharedMemView<DoubleType*>& v_minD =
    scratchViews.get_scratch_view_1D(minD_);
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

    DoubleType rho = 0.0;
    DoubleType tke = 0.0;
    DoubleType tdr = 0.0;
    DoubleType visc = 0.0;
    DoubleType tvisc = 0.0;
    DoubleType dplus = 0.0;
    DoubleType minD = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_dudx[i][j] = 0.0;
      }
    }

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const DoubleType r = v_shape_function_(ip, ic);

      rho += r * v_densityNp1(ic);
      tke += r * v_tkeNp1(ic);
      tdr += r * v_tdrNp1(ic);
      visc += r * v_visc(ic);
      tvisc += r * v_tvisc(ic);
      dplus += r * v_dplus(ic);
      minD += r * v_minD(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        const DoubleType dni = v_dndx(ip, ic, i);
        const DoubleType ui = v_velocityNp1(ic, i);
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_dudx[i][j] += v_dndx(ip, ic, j) * ui;
        }
      }
    }

    DoubleType Pk = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        Pk += w_dudx[i][j] * (w_dudx[i][j] + w_dudx[j][i]);
      }
    }
    Pk *= tvisc;

    // Ftwo calc from Chien 1982 K-epsilon model
    const DoubleType Re_t = rho * tke * tke / visc / stk::math::max(tdr, 1.0e-16);
    const DoubleType fTwo = 1.0 - 0.4/1.8 * stk::math::exp(-Re_t*Re_t / 36.0);

    // Pe includes 1/k scaling; k may be zero at a dirichlet low Re approach (clip)
    const DoubleType PeFac = cEpsOne_ * fOne_ * Pk / stk::math::max(tke, 1.0e-16);
    const DoubleType Pe = PeFac * tdr;
    // FIXME: Currently treating the epsilon in fTwo explicitly... 
    //        see LHS below ... assess if this matters
    const DoubleType DeFac = cEpsTwo_ * fTwo * rho * tdr / stk::math::max(tke, 1.0e-16);
    const DoubleType De = DeFac * tdr;
    // Wall distance source term, rho's cancel...
    const DoubleType LeFac = 2.0 * visc * stk::math::exp(-0.5*dplus) / minD / minD;
    const DoubleType Le = -LeFac * tdr;

    // assemble RHS and LHS
    rhs(nearestNode) += (Pe - De + Le) * scV;
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) +=
        v_shape_function_(ip, ic) * (-PeFac + 2.0*DeFac + LeFac) * scV;
    }
  }
}

INSTANTIATE_KERNEL(TotalDissipationRateChienKESrcElemKernel)

} // namespace nalu
} // namespace sierra
