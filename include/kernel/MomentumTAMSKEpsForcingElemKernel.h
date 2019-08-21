/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSKEPSFORCINGELEMKERNEL_H
#define MOMENTUMTAMSKEPSFORCINGELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Hybrid turbulence for momentum equation
 *
 */
template <typename AlgTraits>
class MomentumTAMSKEpsForcingElemKernel : public Kernel
{
public:
  MomentumTAMSKEpsForcingElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumTAMSKEpsForcingElemKernel();

  // Perform pre-timestep work for the computational kernel
  virtual void setup(const TimeIntegrator&);

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  double time_{0.0};
  double dt_{0.0};
  int step_{0};

  DoubleType pi_;

  std::ofstream tmpFile;

  MomentumTAMSKEpsForcingElemKernel() = delete;

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned tdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned alphaNp1_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeq_{stk::mesh::InvalidOrdinal};
  unsigned minDist_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgDensity_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};

  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned turbViscosity_{stk::mesh::InvalidOrdinal};

  // master element
  const double betaStar_;
  const double forceFactor_;

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_func"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMTAMSKEPSFORCINGELEMKERNEL_H */
