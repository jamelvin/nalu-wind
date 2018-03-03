/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef AdaptivityParameterSFLESSrcElemKernel_H
#define AdaptivityParameterSFLESSrcElemKernel_H

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

/** Add Ksgs source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class AdaptivityParameterSFLESSrcElemKernel: public Kernel
{
public:
  AdaptivityParameterSFLESSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~AdaptivityParameterSFLESSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  AdaptivityParameterSFLESSrcElemKernel() = delete;

  VectorFieldType *coordinates_{nullptr};
  ScalarFieldType *alphaNp1_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *visc_{nullptr};
  ScalarFieldType *tkeNp1_{nullptr};
  ScalarFieldType *sdrNp1_{nullptr};
  ScalarFieldType *dualNodalVolume_{nullptr};
  ScalarFieldType *resolutionAdequacy_{nullptr};
  GenericFieldType *Mij_{nullptr};

  double cT_{0.0};
  double cNu_{0.0};
  
  /// Integration point to node mapping
  const int* ipNodeMap_;

  /// Shape functions
  Kokkos::View<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_ {"view_shape_func"};
};

}  // nalu
}  // sierra

#endif /* AdaptivityParameterSFLESSrcElemKernel_H */
