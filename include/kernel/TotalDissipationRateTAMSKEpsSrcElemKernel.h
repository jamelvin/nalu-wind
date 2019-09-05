/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TOTALDISSIPATIONRATETAMSKEPSSRCELEMKERNEL_H
#define TOTALDISSIPATIONRATETAMSKEPSSRCELEMKERNEL_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template <typename AlgTraits>
class TotalDissipationRateTAMSKEpsSrcElemKernel : public Kernel
{
public:
  TotalDissipationRateTAMSKEpsSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  virtual ~TotalDissipationRateTAMSKEpsSrcElemKernel();

  // Perform pre-timestep work for the computational kernel
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  double time_{0.0};
  double dt_{0.0};
  int step_{0};

  TotalDissipationRateTAMSKEpsSrcElemKernel() = delete;

  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned tdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned resStressNp1_{stk::mesh::InvalidOrdinal};
  unsigned visc_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned dplus_{stk::mesh::InvalidOrdinal};
  unsigned minD_{stk::mesh::InvalidOrdinal};
  unsigned prod_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};

  const bool lumpedMass_;
  const bool shiftedGradOp_;
  const double cEpsOne_;
  const double cEpsTwo_;
  const double fOne_;

  const int* ipNodeMap_;

  std::ofstream tmpFile;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* TOTALDISSIPATIONRATETAMSKEPSSRCELEMKERNEL_H */
