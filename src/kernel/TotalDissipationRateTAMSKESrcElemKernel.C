/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TotalDissipationRateTAMSKESrcElemKernel.h"
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
TotalDissipationRateTAMSKESrcElemKernel<AlgTraits>::
  TotalDissipationRateTAMSKESrcElemKernel(
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
  ScalarFieldType* tke = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  tkeNp1_ = &tke->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType* tdr = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "total_dissipation_rate");
  tdrNp1_ = &tdr->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType* density =
    metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &density->field_of_state(stk::mesh::StateNP1);
  VectorFieldType* velocity =
    metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocityNp1_ = &(velocity->field_of_state(stk::mesh::StateNP1));
  visc_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "viscosity");
  tvisc_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  dplus_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dplus_wall_function");
  alpha_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "k_ratio");
  minD_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "minimum_distance_to_wall");
  prod_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_production");
  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

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
    *coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*tdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*visc_, 1);
  dataPreReqs.add_gathered_nodal_field(*tvisc_, 1);
  dataPreReqs.add_gathered_nodal_field(*alpha_, 1);
  dataPreReqs.add_gathered_nodal_field(*dplus_, 1);
  dataPreReqs.add_gathered_nodal_field(*minD_, 1);
  dataPreReqs.add_gathered_nodal_field(*prod_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCV_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
TotalDissipationRateTAMSKESrcElemKernel<
  AlgTraits>::~TotalDissipationRateTAMSKESrcElemKernel()
{
}

template <typename AlgTraits>
void
TotalDissipationRateTAMSKESrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dudx[AlgTraits::nDim_][AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_coords[AlgTraits::nDim_];

  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(*tkeNp1_);
  SharedMemView<DoubleType*>& v_tdrNp1 =
    scratchViews.get_scratch_view_1D(*tdrNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 =
    scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType**>& v_velocityNp1 =
    scratchViews.get_scratch_view_2D(*velocityNp1_);
  SharedMemView<DoubleType*>& v_visc =
    scratchViews.get_scratch_view_1D(*visc_);
  SharedMemView<DoubleType*>& v_tvisc =
    scratchViews.get_scratch_view_1D(*tvisc_);
  SharedMemView<DoubleType*>& v_alpha =
    scratchViews.get_scratch_view_1D(*alpha_);
  SharedMemView<DoubleType*>& v_dplus =
    scratchViews.get_scratch_view_1D(*dplus_);
  SharedMemView<DoubleType*>& v_minD =
    scratchViews.get_scratch_view_1D(*minD_);
  SharedMemView<DoubleType*>& v_prod =
    scratchViews.get_scratch_view_1D(*prod_);
  SharedMemView<DoubleType***>& v_dndx =
    shiftedGradOp_
      ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv_shifted
      : scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv;
  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;


  SharedMemView<DoubleType**>& v_coords = 
    scratchViews.get_scratch_view_2D(*coordinates_);

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
    DoubleType alpha = 0.0;
    DoubleType dplus = 0.0;
    DoubleType minD = 0.0;
    DoubleType prod = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      w_coords[i] = 0.0;
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
      alpha += r * v_alpha(ic);
      dplus += r * v_dplus(ic);
      minD += r * v_minD(ic);
      prod += r * v_prod(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        w_coords[i] += r*v_coords(ic,i);
        const DoubleType dni = v_dndx(ip, ic, i);
        const DoubleType ui = v_velocityNp1(ic, i);
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_dudx[i][j] += v_dndx(ip, ic, j) * ui;
        }
      }
    }

    // The changes to the standard KE RANS approach in TAMS result in two changes:
    // 1) improvements to the production based on the resolved fluctuations
    // 2) the addition of alpha to modify the production
    // 3) the averaging of the production, thus it's calculation has been moved to the
    //    averaging function
    const DoubleType Pk = prod;

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

    //std::ofstream tmpFile;
    //tmpFile.open("TDRsrc.txt");
    
    //tmpFile << " (" << w_coords[0] << ", "<< w_coords[1] << ", " << w_coords[2] << ") " << tdr <<" " << tke << " " << Pe << " " << De << " " << Le << " " << minD << " " << dplus << " " << tvisc << std::endl;
    
    //tmpFile.close();

    //const DoubleType extraFac = -cEpsTwo_ * stk::math::exp(-Re_t*Re_t / 36.0) * rho * rho * rho * tke * tke * tke / 81.0 / visc / visc / stk::math::max(tdr, 1.0e-16);

    // assemble RHS and LHS
    rhs(nearestNode) += (Pe - De + Le) * scV;
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) +=
        v_shape_function_(ip, ic) * (2.0*DeFac + LeFac) * scV;
    }
  }
}

INSTANTIATE_KERNEL(TotalDissipationRateTAMSKESrcElemKernel)

} // namespace nalu
} // namespace sierra
