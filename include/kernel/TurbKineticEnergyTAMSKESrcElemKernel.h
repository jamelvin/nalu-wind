/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBKINETICENERGYTAMSKESRCELEMKERNEL_H
#define TURBKINETICENERGYTAMSKESRCELEMKERNEL_H

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
class TurbKineticEnergyTAMSKESrcElemKernel : public Kernel
{
public:
  TurbKineticEnergyTAMSKESrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  virtual ~TurbKineticEnergyTAMSKESrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TurbKineticEnergyTAMSKESrcElemKernel() = delete;

  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* tdrNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  VectorFieldType* velocityNp1_{nullptr};
  GenericFieldType* resStressNp1_{nullptr};
  ScalarFieldType* visc_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  ScalarFieldType* alpha_{nullptr};
  ScalarFieldType* minD_{nullptr};
  VectorFieldType* coordinates_{nullptr};

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

#endif /* TURBKINETICENERGYTAMSKESRCELEMKERNEL_H */
