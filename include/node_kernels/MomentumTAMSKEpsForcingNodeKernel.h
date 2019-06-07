/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSKEPSFORCINGNODEKERNEL_H
#define MOMENTUMTAMSKEPSFORCINGNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumTAMSKEpsForcingNodeKernel : public NGPNodeKernel<MomentumTAMSKEpsForcingNodeKernel>
{
public:
  MomentumTAMSKEpsForcingNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions& );

  KOKKOS_FUNCTION
  MomentumTAMSKEpsForcingNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumTAMSKEpsForcingNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> velocity_;
  ngp::Field<double> viscosity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> density_;
  ngp::Field<double> tke_;
  ngp::Field<double> tdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> Mij_;
  ngp::Field<double> minDist_;
  ngp::Field<double> avgVelocity_;
  ngp::Field<double> avgDensity_;
  ngp::Field<double> avgTime_;
  ngp::Field<double> avgResAdeq_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTMID_ {stk::mesh::InvalidOrdinal};
  unsigned viscosityID_ {stk::mesh::InvalidOrdinal};
  unsigned turbViscID_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned tdrNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned alphaID_ {stk::mesh::InvalidOrdinal};
  unsigned MijID_ {stk::mesh::InvalidOrdinal};
  unsigned minDistID_ {stk::mesh::InvalidOrdinal};
  unsigned avgVelocityID_ {stk::mesh::InvalidOrdinal};
  unsigned avgDensityID_ {stk::mesh::InvalidOrdinal};
  unsigned avgTimeID_ {stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqID_ {stk::mesh::InvalidOrdinal};

  const int nDim_;
  double pi_;
  double time_;
  double dt_;
  int step_;
};

}  // nalu
}  // sierra



#endif /* MOMENTUMTAMSKEPSFORCINGNODEKERNEL_H */