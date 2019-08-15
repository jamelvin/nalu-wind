/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TurbKineticEnergyTAMSSSTSrcNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

TurbKineticEnergyTAMSSSTSrcNodeKernel::TurbKineticEnergyTAMSSSTSrcNodeKernel(
  const stk::mesh::BulkData& bulk,
  const SolutionOptions & solnOpts
) : NGPNodeKernel<TurbKineticEnergyTAMSSSTSrcNodeKernel>(),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    tkeProdLimitRatio_(solnOpts.get_turb_model_constant(TM_tkeProdLimitRatio)),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  viscID_ = get_field_ordinal(meta, "viscosity");
  tviscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1ID_ = get_field_ordinal(meta, "specific_dissipation_rate", stk::mesh::StateNP1);
  alphaID_ = get_field_ordinal(meta, "k_ratio");

  // average quantities
  prodID_ = get_field_ordinal(meta, "average_production");
  densityID_ = get_field_ordinal(meta, "average_density");
}

void TurbKineticEnergyTAMSSSTSrcNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  viscosity_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  rho_ = fieldMgr.get_field<double>(densityID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  alpha_ = fieldMgr.get_field<double>(alphaID_);
  prod_ = fieldMgr.get_field<double>(prodID_);
}

void
TurbKineticEnergyTAMSSSTSrcNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NodeKernelTraits::DblType Pk = prod_.get(node, 0);
  
  const NodeKernelTraits::DblType tkeFac = betaStar_ * rho_.get(node, 0) * sdr_.get(node, 0);
  NodeKernelTraits::DblType Dk = tkeFac * stk::math::max(tke_.get(node, 0), 1.0e-12); 

  Pk = stk::math::min(Pk, tkeProdLimitRatio_ * Dk);

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  rhs(0) += (Pk - Dk) * dualVolume;
  
  //FIXME: Is this lhs(0) or lhs(0,0)? Or something else...
  lhs(0, 0) += tkeFac * dualVolume;
}

}  // nalu
}  // sierra
