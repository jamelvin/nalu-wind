/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSKEPSDIFFEDGEKERNEL_H
#define MOMENTUMTAMSKEPSDIFFEDGEKERNEL_H

#include "edge_kernels/EdgeKernel.h"

//FIXME: For nDimMax_, is this necessary?
#include "AssembleEdgeSolverAlgorithm.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumTAMSKEpsDiffEdgeKernel : public NGPEdgeKernel<MomentumTAMSKEpsDiffEdgeKernel>
{
public:
  MomentumTAMSKEpsDiffEdgeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions& );

  KOKKOS_FUNCTION
  MomentumTAMSKEpsDiffEdgeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumTAMSKEpsDiffEdgeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    EdgeKernelTraits::LhsType&,
    EdgeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> edgeAreaVec_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> velocity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> density_;
  ngp::Field<double> tke_;
  ngp::Field<double> tdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> nodalMij_;
  ngp::Field<double> dudx_;
  ngp::Field<double> avgVelocity_;
  ngp::Field<double> avgDensity_;
  ngp::Field<double> avgDudx_;

  unsigned edgeAreaVecID_ {stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTMID_ {stk::mesh::InvalidOrdinal};
  unsigned turbViscID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned tdrNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned alphaID_ {stk::mesh::InvalidOrdinal};
  unsigned MijID_ {stk::mesh::InvalidOrdinal};
  unsigned dudxID_ {stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_ {stk::mesh::InvalidOrdinal};
  unsigned avgDensityID_ {stk::mesh::InvalidOrdinal};
  unsigned avgDudxID_ {stk::mesh::InvalidOrdinal};

  const double includeDivU_;

  const double CMdeg_;

  const int nDim_;
};

}  // nalu
}  // sierra



#endif /* MOMENTUMTAMSKEPSDIFFEDGEKERNEL_H */
