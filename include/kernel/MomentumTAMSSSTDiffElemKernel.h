/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSSSTDIFFELEMKERNEL_H
#define MOMENTUMTAMSSSTDIFFELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Hybrid turbulence for momentum equation
 *
 */
template <typename AlgTraits>
class MomentumTAMSSSTDiffElemKernel : public Kernel
{
public:
  MomentumTAMSSSTDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumTAMSSSTDiffElemKernel() {}

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

  DoubleType get_M43_constant(DoubleType D[AlgTraits::nDim_][AlgTraits::nDim_]);

private:
  MomentumTAMSSSTDiffElemKernel() = delete;

  const double includeDivU_;

  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* sdrNp1_{nullptr};
  ScalarFieldType* alphaNp1_{nullptr};
  GenericFieldType* mutij_{nullptr};
  VectorFieldType* coordinates_{nullptr};
  GenericFieldType* Mij_{nullptr};

  VectorFieldType* avgVelocity_{nullptr};
  ScalarFieldType* avgDensity_{nullptr};

  ScalarFieldType *viscosity_{nullptr};

  // master element
  const int* lrscv_;

  const double betaStar_;
  const double CMdeg_;

  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMTAMSSSTDIFFELEMKERNEL_H */
