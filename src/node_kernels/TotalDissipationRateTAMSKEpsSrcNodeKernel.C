/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TotalDissipationRateTAMSKEpsSrcNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

TotalDissipationRateTAMSKEpsSrcNodeKernel::TotalDissipationRateTAMSKEpsSrcNodeKernel(
  const stk::mesh::BulkData& bulk,
  const SolutionOptions& solnOpts
) : NGPNodeKernel<TotalDissipationRateTAMSKEpsSrcNodeKernel>(),
    cEpsOne_(solnOpts.get_turb_model_constant(TM_cEpsOne)),
    cEpsTwo_(solnOpts.get_turb_model_constant(TM_cEpsTwo)),
    fOne_(solnOpts.get_turb_model_constant(TM_fOne)),
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
  dplusID_ = get_field_ordinal(meta, "dplus_wall_function");

  // average quantities
  prodID_ = get_field_ordinal(meta, "average_production");
  densityID_ = get_field_ordinal(meta, "average_density");
  avgTimeID_ = get_field_ordinal(meta, "average_time");
}

void TotalDissipationRateTAMSKEpsSrcNodeKernel::setup(Realm& realm)
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
  dplus_ = fieldMgr.get_field<double>(dplusID_);
  time_ = fieldMgr.get_field<double>(avgTimeID_);
}

void
TotalDissipationRateTAMSKEpsSrcNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType rho = rho_.get(node, 0);
  const NodeKernelTraits::DblType tdr = tdr_.get(node, 0);
  const NodeKernelTraits::DblType tke = tke_.get(node, 0);
  const NodeKernelTraits::DblType visc = viscosity_.get(node, 0);
  const NodeKernelTraits::DblType time = time_.get(node, 0);
  const NodeKernelTraits::DblType minD = minDist_.get(node, 0);
  const NodeKernelTraits::DblType dplus = dplus_.get(node, 0);


  const NodeKernelTraits::DblType Pk = prod_.get(node, 0);

  const NodeKernelTraits::DblType Re_t = rho * tke * tke / visc / stk::math::max(tdr, 1.0e-16);
  const NodeKernelTraits::DblType fTwo = 1.0 - 0.4 / 1.8 * stk::math::exp(-Re_t * Re_t / 36.0);

  const NodeKernelTraits::DblType Pe = cEpsOne_ * fOne_ * Pk / time;

  const NodeKernelTraits::DblType DeFac = cEpsTwo_ * fTwo * rho / time;
  const NodeKernelTraits::DblType De = DeFac * tdr;

  // Wall distance source term, rho's cancel...
  const NodeKernelTraits::DblType lFac = 2.0 * visc * stk::math::exp(-0.5 * dplus) / minD / minD;
  const NodeKernelTraits::DblType Le = -lFac * tdr;


  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  rhs(0) += (Pe - De + Le) * dualVolume;
  
  //FIXME: Is this lhs(0) or lhs(0,0)? Or something else...
  lhs(0,0) += (2.0 * DeFac + lFac) * dualVolume;
}

}  // nalu
}  // sierra
