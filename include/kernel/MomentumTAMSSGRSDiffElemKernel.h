/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSSGRSDIFFELEMKERNEL_H
#define MOMENTUMTAMSSGRSDIFFELEMKERNEL_H

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

/** Advection diffusion term for momentum equation (velocity DOF)
 */
template <typename AlgTraits>
class MomentumTAMSSGRSDiffElemKernel : public Kernel
{
public:
  MomentumTAMSSGRSDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumTAMSSGRSDiffElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumTAMSSGRSDiffElemKernel() = delete;

  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* sdrNp1_{nullptr};
  ScalarFieldType* alphaNp1_{nullptr};

  ScalarFieldType* viscosity_{nullptr};

  const int* lrscv_;

  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMADVDIFFELEMKERNEL_H */
