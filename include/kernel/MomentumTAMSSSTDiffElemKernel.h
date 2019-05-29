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

private:
  MomentumTAMSSSTDiffElemKernel() = delete;

  const double includeDivU_;

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned alphaNp1_{stk::mesh::InvalidOrdinal};
  unsigned mutij_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};

  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgDensity_{stk::mesh::InvalidOrdinal};

  unsigned viscosity_{stk::mesh::InvalidOrdinal};

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
