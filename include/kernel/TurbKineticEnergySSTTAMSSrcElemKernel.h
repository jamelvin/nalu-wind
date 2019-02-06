/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBKINETICENERGYSSTTAMSSRCELEMKERNEL_H
#define TURBKINETICENERGYSSTTAMSSRCELEMKERNEL_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template <typename AlgTraits>
class TurbKineticEnergySSTTAMSSrcElemKernel : public Kernel
{
public:
  TurbKineticEnergySSTTAMSSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  virtual ~TurbKineticEnergySSTTAMSSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TurbKineticEnergySSTTAMSSrcElemKernel() = delete;

  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned resStressNp1_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned prod_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};

  const bool lumpedMass_;
  const bool shiftedGradOp_;
  const double betaStar_;
  double tkeProdLimitRatio_{0.0};

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* TURBKINETICENERGYSSTTAMSSRCELEMKERNEL_H */
