/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBKINETICENERGYTAMSKEPSSRCNODEKERNEL_H
#define TURBKINETICENERGYTAMSKEPSSRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class TurbKineticEnergyTAMSKEpsSrcNodeKernel : public NGPNodeKernel<TurbKineticEnergyTAMSKEpsSrcNodeKernel>
{
public:
  TurbKineticEnergyTAMSKEpsSrcNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions& );

  KOKKOS_FUNCTION
  TurbKineticEnergyTAMSKEpsSrcNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~TurbKineticEnergyTAMSKEpsSrcNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> dualNodalVolume_;

  ngp::Field<double> coordinates_;
  ngp::Field<double> viscosity_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> rho_;
  ngp::Field<double> tke_;
  ngp::Field<double> tdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> minDist_;
  ngp::Field<double> prod_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned viscID_ {stk::mesh::InvalidOrdinal};
  unsigned tviscID_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned tdrNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned alphaID_ {stk::mesh::InvalidOrdinal};
  unsigned minDistID_ {stk::mesh::InvalidOrdinal};
  unsigned prodID_ {stk::mesh::InvalidOrdinal};
  unsigned densityID_ {stk::mesh::InvalidOrdinal};

  const int nDim_;
};

}  // nalu
}  // sierra



#endif /* TURBKINETICENERGYTAMSKEPSSRCNODEKERNEL_H */
