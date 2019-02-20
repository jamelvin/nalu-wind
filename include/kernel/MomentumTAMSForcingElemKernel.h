/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSFORCINGELEMKERNEL_H
#define MOMENTUMTAMSFORCINGELEMKERNEL_H

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
class MomentumTAMSForcingElemKernel : public Kernel
{
public:
  MomentumTAMSForcingElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumTAMSForcingElemKernel() {}

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

  DoubleType pi_;

  MomentumTAMSForcingElemKernel() = delete;

  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* sdrNp1_{nullptr};
  ScalarFieldType* alphaNp1_{nullptr};
  VectorFieldType* coordinates_{nullptr};
  GenericFieldType* Mij_{nullptr};
  ScalarFieldType* avgResAdeq_{nullptr};
  ScalarFieldType* minDist_{nullptr};
  VectorFieldType* avgVelocity_{nullptr};
  ScalarFieldType* avgDensity_{nullptr};
  ScalarFieldType* avgTke_{nullptr};
  ScalarFieldType* avgSdr_{nullptr};

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

#endif /* MOMENTUMTAMSFORCINGELEMKERNEL_H */
