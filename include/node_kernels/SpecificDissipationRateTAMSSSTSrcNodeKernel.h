/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SPECIFICDISSIPATIONRATETAMSSSTSRCNODEKERNEL_H
#define SPECIFICDISSIPATIONRATETAMSSSTSRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class SpecificDissipationRateTAMSSSTSrcNodeKernel : public NGPNodeKernel<SpecificDissipationRateTAMSSSTSrcNodeKernel>
{
public:
  SpecificDissipationRateTAMSSSTSrcNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions & );

  KOKKOS_FUNCTION
  SpecificDissipationRateTAMSSSTSrcNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~SpecificDissipationRateTAMSSSTSrcNodeKernel() = default;

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
  ngp::Field<double> sdr_;
  ngp::Field<double> alpha_;
  ngp::Field<double> prod_;
  ngp::Field<double> fOneBlend_;
  ngp::Field<double> dkdx_;
  ngp::Field<double> dwdx_;

  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};
  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned tviscID_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned sdrNp1ID_ {stk::mesh::InvalidOrdinal};
  unsigned alphaID_ {stk::mesh::InvalidOrdinal};
  unsigned fOneBlendID_ {stk::mesh::InvalidOrdinal};
  unsigned dkdxID_ {stk::mesh::InvalidOrdinal};
  unsigned dwdxID_ {stk::mesh::InvalidOrdinal};
  unsigned prodID_ {stk::mesh::InvalidOrdinal};
  unsigned densityID_ {stk::mesh::InvalidOrdinal};

  const double betaStar_;
  const double sigmaWTwo_;
  const double betaOne_;
  const double betaTwo_;
  const double gammaOne_;
  const double gammaTwo_;
  const double tkeProdLimitRatio_;
  const int nDim_;
};

}  // nalu
}  // sierra



#endif /* SPECIFICDISSIPATIONRATETAMSSSTSRCNODEKERNEL_H */
