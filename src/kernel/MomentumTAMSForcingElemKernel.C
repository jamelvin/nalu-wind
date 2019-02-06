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
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(
             AlgTraits::topo_)
             ->adjacentNodes()),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const pi_ = std::acos(-1.0);
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

  resAdeq_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter");
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

  // First we calculate the a_i's
  DoubleType length = FORCING_CL*pow(alpha*tke,1.5)/epsilon;
  length = max(length, Ceta*(pow(viscosity,0.75)/pow(epsilon,0.25)));
  length = min(length, wallDist_);
  
  DoubleType T_alpha = alpha*tke/epsilon;
  T_alpha = max(T_alpha,Ct*stk::math::sqrt(viscosity/epsilon));
  T_alpha = BL_T*T_alpha;

  const DoubleType T_kol = stk::math::sqrt(viscosity/epsilon);

  const DoubleType ceilLengthX = std::math::max(length,2.0d0*metric(1,1));
  const DoubleType ceilLengthY = stk::math::max(length,2.0d0*metric(1,1));
  const DoubleType ceilLengthZ = stk::math::max(length,2.0d0*metric(1,1));

  const DoubleType clipLengthX = stk::math::min(ceilLengthX, periodicForcingLengthX);
  const DoubleType clipLengthY = stk::math::min(ceilLengthY, periodicForcingLenghtY);
  const DoubleType clipLengthZ = stk::math::min(ceilLengthZ, periodicForcingLengthZ);

  const DoubleType ratioX = stk::math::floor(periodicForcingLengthX/clipLengthX + 0.5);
  const DoubleType ratioY = stk::math::floor(periodicForcingLengthY/clipLengthY + 0.5);
  const DoubleType ratioZ = stk::math::floor(periodicForcingLengthZ/clipLengthZ + 0.5);
  
  const DoubleType denomX = periodicForcingLengthX/ratioX;
  const DoubleType denomY = periodicForcingLengthY/ratioY;
  const DoubleType denomZ = periodicForcingLengthZ/ratioZ;

  const DoubleType ax = pi_/denomX;
  const DoubleType ay = pi_/denomY;
  const DoubleType az = pi_/denomZ;

  // Then we calculate the arguments for the Taylor-Green Vortex
  const DoubleType xarg = ax * (coordX + averageVelX * time_);
  const DoubleType yarg = ay * (coordY + averageVelY * time_);
  const DoubleType zarg = az * (coordZ + averageVelZ * time_);

  // Now we calculate the initial Taylor-Green field
  DoubleType hX = 1.0/3.0 * stk::math::cos(xarg) * stk::math::sin(yarg) * stk::math::sin(zarg);
  DoubleType hY =  -1.0   * stk::math::sin(xarg) * stk::math::cos(yarg) * stk::math::sin(zarg);
  DoubleType hZ = 2.0/3.0 * stk::math::sin(xarg) * stk::math::sin(yarg) * stk::math::cos(zarg);

  // Now we calculate the scaling of the initial field
  const DoubleType F_target = FORCING_FACTOR * math::stk::sqrt(alpha*v2)/T_alpha; 

  const prod_r = (F_target*dt_) * (hX*fluctVelX + hY*fluctVelY + hZ*fluctVelZ);

  DoubleType arg = stk::math::sqrt(avgResAdeq) - 1.0;
  stk::math::if_then(arg < 0.0, arg = 1.0 - 1.0/stk::math::sqrt(avgResAdeq));
  
  const DoubleType a_sign = stk::math::tanh(arg);

  DoubleType Sa = a_sign;

  DoubleType a_kol = stk::math::min(BL_KOL*stk::math::sqrt(viscosity*epsilon)/tke,1.0);

  stk::math::if_then_else(a_sign < 0.0, 
                            stk::math::if_then(alpha <= a_kol, Sa = Sa - (1.0+a_kol-alpha)*a_sign),
                            stk::math::if_then(alpha >= 1.0, Sa = Sa - alpha*a_sign));

  const DoubleType prod_temp = (alpha*tke/T_alpha)*Sa;
  const DoubleType fd_temp = avgResAdeq;

  DoubleType C_F;
  stk::math::if_then_else((fd_temp < 1.0) && (prod_r > 0.0), C_F = -1.0 * F_target * Sa, C_F = 0.0);

  const DoubleType norm = C_F * dt_;

  // Now we determine the actual forcing field
  DoubleType gX = norm * hX; 
  DoubleType gY = norm * hY;
  DoubleType gZ = norm * hZ;
 
  // g_i is not divergence free, so we must solve a Poisson equation
  rhs = G * normal * area;
  lhs = grad operator * normal * area;
}

INSTANTIATE_KERNEL(MomentumTAMSDiffElemKernel);

} // namespace nalu
} // namespace sierra
