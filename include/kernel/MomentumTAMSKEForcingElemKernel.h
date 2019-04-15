/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSKEFORCINGELEMKERNEL_H
#define MOMENTUMTAMSKEFORCINGELEMKERNEL_H

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
class MomentumTAMSKEForcingElemKernel : public Kernel
{
public:
  MomentumTAMSKEForcingElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumTAMSKEForcingElemKernel();

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

  MomentumTAMSKEForcingElemKernel() = delete;

  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* tdrNp1_{nullptr};
  ScalarFieldType* alphaNp1_{nullptr};
  VectorFieldType* coordinates_{nullptr};
  GenericFieldType* Mij_{nullptr};
  ScalarFieldType* avgResAdeq_{nullptr};
  ScalarFieldType* minDist_{nullptr};
  VectorFieldType* avgVelocity_{nullptr};
  ScalarFieldType* avgDensity_{nullptr};
  ScalarFieldType *avgTime_{nullptr};

  ScalarFieldType *viscosity_{nullptr};
  ScalarFieldType *turbViscosity_{nullptr};

  // master element
  const double betaStar_;

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_]
        [AlgTraits::nodesPerElement_]> v_shape_function_ { "v_shape_func" };
  
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMTAMSKEFORCINGELEMKERNEL_H */
