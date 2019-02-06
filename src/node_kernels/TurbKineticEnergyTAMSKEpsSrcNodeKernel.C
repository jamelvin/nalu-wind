/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TurbKineticEnergyTAMSKEpsSrcNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

TurbKineticEnergyTAMSKEpsSrcNodeKernel::TurbKineticEnergyTAMSKEpsSrcNodeKernel(
  const stk::mesh::BulkData& bulk,
  const SolutionOptions & solnOpts
) : NGPNodeKernel<TurbKineticEnergyTAMSKEpsSrcNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  viscID_ = get_field_ordinal(meta, "viscosity");
  tviscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  tdrNp1ID_ = get_field_ordinal(meta, "total_dissipation_rate", stk::mesh::StateNP1);
  alphaID_ = get_field_ordinal(meta, "k_ratio");
  minDistID_ = get_field_ordinal(meta, "minimum_distance_to_wall");

  // average quantities
  prodID_ = get_field_ordinal(meta, "average_production");
  densityID_ = get_field_ordinal(meta, "average_density");
}

void TurbKineticEnergyTAMSKEpsSrcNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  viscosity_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  rho_ = fieldMgr.get_field<double>(densityID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  tdr_ = fieldMgr.get_field<double>(tdrNp1ID_);
  alpha_ = fieldMgr.get_field<double>(alphaID_);
  minDist_ = fieldMgr.get_field<double>(minDistID_);
  prod_ = fieldMgr.get_field<double>(prodID_);
}

void
TurbKineticEnergyTAMSKEpsSrcNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType Pk = prod_.get(node, 0);
  const NodeKernelTraits::DblType Dk = rho_.get(node, 0) * tdr_.get(node, 0);
  const NodeKernelTraits::DblType minDist = minDist_.get(node, 0);
  const NodeKernelTraits::DblType lFac = 2.0 * viscosity_.get(node, 0) / minDist / minDist;
  const NodeKernelTraits::DblType Lk = -lFac * tke_.get(node, 0);

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  rhs(0) += (Pk - Dk + Lk) * dualVolume;
  
  //FIXME: Is this lhs(0) or lhs(0,0)? Or something else...
  lhs(0, 0) += lFac * dualVolume;
}

}  // nalu
}  // sierra
