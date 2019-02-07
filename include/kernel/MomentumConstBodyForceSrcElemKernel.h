/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMCONSTBODYFORCESRCELEMKERNEL_H
#define MOMENTUMCONSTBODYFORCESRCELEMKERNEL_H

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

/** CMM body force term for momentum equation (velocity DOF)
 */
template<typename AlgTraits>
class MomentumConstBodyForceSrcElemKernel: public Kernel
{
public:
  MomentumConstBodyForceSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~MomentumConstBodyForceSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumConstBodyForceSrcElemKernel() = delete;

  ScalarFieldType *densityNp1_{nullptr};

  AlignedViewType<DoubleType[AlgTraits::nDim_]> constBodyForce_{"v_bodyForce"};

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_ { "v_shape_func" };
};

}  // nalu
}  // sierra

#endif /* MOMENTUMBUOYANCYSRCELEMKERNEL_H */
