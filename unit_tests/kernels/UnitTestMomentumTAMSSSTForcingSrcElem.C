/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumTAMSSSTForcingElemKernel.h"

namespace {
namespace hex8_golds {
namespace TAMS_SST_forcing {
static constexpr double rhs[24] = {3.6748245514994e-17, -0.0046182581299522, -1.6680666151408e-16, 0, 0, 0, 2.1305443842197e-16, -0.0065659125659498, -2.8636102207e-15, 4.1361147828464e-16, -0.0043047759736903, -1.8774542536571e-15, 0, 0,
 0, -5.1356662213437e-17, 0.019111066381996, -1.5105983901027e-16, 0, 0, 0, 0, 0, 0, };
} // namespace TAMS_SST_forcing
} // namespace hex8_golds
} // anonymous namespace

#ifndef KOKKOS_HAVE_CUDA

TEST_F(TAMSKernelHex8Mesh, TAMS_SST_forcing)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumTAMSSSTForcingElemKernel<
      sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, visc_, tvisc_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 1.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);
  
  namespace gold_values = ::hex8_golds::TAMS_SST_forcing;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
}

#endif
