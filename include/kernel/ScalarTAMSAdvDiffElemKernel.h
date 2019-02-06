/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARTAMSADVDIFFELEMKERNEL_H
#define SCALARTAMSADVDIFFELEMKERNEL_H

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

/** CVFEM scalar advection/diffusion kernel
 */
template <typename AlgTraits>
class ScalarTAMSAdvDiffElemKernel : public NGPKernel<ScalarTAMSAdvDiffElemKernel<AlgTraits>>
{
public:
  ScalarTAMSAdvDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  KOKKOS_FUNCTION ScalarTAMSAdvDiffElemKernel() = default;

  virtual ~ScalarTAMSAdvDiffElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned scalarQ_{stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};

  const bool shiftedGradOp_;
  const bool skewSymmetric_;

  MasterElement* meSCS_{nullptr};

};

} // namespace nalu
} // namespace sierra

#endif /* SCALARTAMSADVDIFFELEMKERNEL_H */
