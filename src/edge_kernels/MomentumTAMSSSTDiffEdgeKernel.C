/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "edge_kernels/MomentumTAMSSSTDiffEdgeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "EigenDecomposition.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include "utils/TAMSUtils.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

MomentumTAMSSSTDiffEdgeKernel::MomentumTAMSSSTDiffEdgeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPEdgeKernel<MomentumTAMSSSTDiffEdgeKernel>(),
    includeDivU_(solnOpts.includeDivU_),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  edgeAreaVecID_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);

  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  const std::string velField =
    solnOpts.does_mesh_move() ? "velocity_rtm" : "velocity";
  velocityRTMID_ = get_field_ordinal(meta, velField);
  turbViscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1ID_ =
    get_field_ordinal(meta, "specific_dissipation_rate", stk::mesh::StateNP1);
  alphaID_ = get_field_ordinal(meta, "k_ratio");
  MijID_ = get_field_ordinal(meta, "metric_tensor");
  dudxID_ = get_field_ordinal(meta, "dudx");

  // average quantities
  avgVelocityID_ = get_field_ordinal(meta, "average_velocity");
  avgDensityID_ = get_field_ordinal(meta, "average_density");
  avgDudxID_ = get_field_ordinal(meta, "average_dudx");
}

void
MomentumTAMSSSTDiffEdgeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  edgeAreaVec_ = fieldMgr.get_field<double>(edgeAreaVecID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  velocity_ = fieldMgr.get_field<double>(velocityRTMID_);
  tvisc_ = fieldMgr.get_field<double>(turbViscID_);
  density_ = fieldMgr.get_field<double>(densityNp1ID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  alpha_ = fieldMgr.get_field<double>(alphaID_);
  nodalMij_ = fieldMgr.get_field<double>(MijID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  avgVelocity_ = fieldMgr.get_field<double>(avgVelocityID_);
  avgDensity_ = fieldMgr.get_field<double>(avgDensityID_);
  avgDudx_ = fieldMgr.get_field<double>(avgDudxID_);
}

void
MomentumTAMSSSTDiffEdgeKernel::execute(
  EdgeKernelTraits::ShmemDataType& smdata,
  const stk::mesh::FastMeshIndex& edge,
  const stk::mesh::FastMeshIndex& nodeL,
  const stk::mesh::FastMeshIndex& nodeR)
{
  const int ndim = nDim_;

  // Scratch work arrays
  // Make this variable?? TAMS only works in 3D...
  NALU_ALIGNED EdgeKernelTraits::DblType av[3];

  for (int d = 0; d < nDim_; d++) {
    av[d] = edgeAreaVec_.get(edge, d);
  }

  // Mij, eigenvectors and eigenvalues
  // FIXME: make the 3 -> ndim
  EdgeKernelTraits::DblType Mij[3][3];
  EdgeKernelTraits::DblType Q[3][3];
  EdgeKernelTraits::DblType D[3][3];
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      // FIXME: Is this right for accessing 2D array of nodal Mij, is it 1D or
      // 2D??
      Mij[i][j] = 0.5 * (nodalMij_.get(nodeL, i * ndim + j) +
                         nodalMij_.get(nodeR, i * ndim + j));

  EigenDecomposition::sym_diagonalize<EdgeKernelTraits::DblType>(Mij, Q, D);

  // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
  // so to create M43, we use Q D^(4/3) Q^T
  EdgeKernelTraits::DblType M43[3][3];
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      M43[i][j] = 0.0;

  const double fourThirds = 4. / 3.;
  for (int k = 0; k < ndim; k++) {
    const EdgeKernelTraits::DblType D43 = stk::math::pow(D[k][k], fourThirds);
    for (int i = 0; i < ndim; i++) {
      for (int j = 0; j < ndim; j++) {
        M43[i][j] += Q[i][k] * Q[j][k] * D43;
      }
    }
  }

  // Compute CM43
  EdgeKernelTraits::DblType CM43 = tams_utils::get_M43_constant<EdgeKernelTraits::DblType, 3>(D, CMdeg_);

  const EdgeKernelTraits::DblType muIp =
    0.5 * (tvisc_.get(nodeL, 0) + tvisc_.get(nodeR, 0));
  const EdgeKernelTraits::DblType avgRhoIp =
    0.5 * (avgDensity_.get(nodeL, 0) + avgDensity_.get(nodeR, 0));
  const EdgeKernelTraits::DblType fluctRhoIp =
    0.5 * (density_.get(nodeL, 0) + density_.get(nodeR, 0)) - avgRhoIp;
  const EdgeKernelTraits::DblType tkeIp = 0.5 * (stk::math::max(tke_.get(nodeL, 0), 1.0e-12) + 
                                                 stk::math::max(tke_.get(nodeR, 0), 1.0e-12));
  const EdgeKernelTraits::DblType sdrIp = 0.5 * (stk::math::max(sdr_.get(nodeL, 0), 1.0e-12) + 
                                                 stk::math::max(sdr_.get(nodeR, 0), 1.0e-12));
  const EdgeKernelTraits::DblType alphaIp =
    0.5 * (alpha_.get(nodeL, 0) + alpha_.get(nodeR, 0));

  EdgeKernelTraits::DblType avgdUidxj[3][3];
  EdgeKernelTraits::DblType fluctdUidxj[3][3];

  EdgeKernelTraits::DblType axdx = 0.0;
  EdgeKernelTraits::DblType asq = 0.0;
  for (int d = 0; d < ndim; ++d) {
    const EdgeKernelTraits::DblType dxj =
      coordinates_.get(nodeR, d) - coordinates_.get(nodeL, d);
    asq += av[d] * av[d];
    axdx += av[d] * dxj;
  }
  const EdgeKernelTraits::DblType inv_axdx = 1.0 / axdx;

  // Compute average divU
  for (int i = 0; i < ndim; ++i) {

    // difference between R and L nodes for component i
    const EdgeKernelTraits::DblType avgUidiff =
      avgVelocity_.get(nodeR, i) - avgVelocity_.get(nodeL, i);
    const EdgeKernelTraits::DblType fluctUidiff =
      (velocity_.get(nodeR, i) - velocity_.get(nodeL, i)) - avgUidiff;

    const int offSetI = ndim * i;

    // start sum for NOC contribution
    EdgeKernelTraits::DblType GlavgUidxl = 0.0;
    EdgeKernelTraits::DblType GlfluctUidxl = 0.0;
    for (int l = 0; l < ndim; ++l) {
      const int offSetIL = offSetI + l;
      const EdgeKernelTraits::DblType dxl =
        coordinates_.get(nodeR, l) - coordinates_.get(nodeL, l);
      const EdgeKernelTraits::DblType GlavgUi =
        0.5 * (avgDudx_.get(nodeL, offSetIL) + avgDudx_.get(nodeR, offSetIL));
      const EdgeKernelTraits::DblType GlfluctUi =
        0.5 * (dudx_.get(nodeL, offSetIL) + dudx_.get(nodeR, offSetIL)) -
        GlavgUi;
      GlavgUidxl += GlavgUi * dxl;
      GlfluctUidxl += GlfluctUi * dxl;
    }

    // form full tensor dui/dxj with NOC
    for (int j = 0; j < ndim; ++j) {
      const int offSetIJ = offSetI + j;
      const EdgeKernelTraits::DblType GjavgUi =
        0.5 * (avgDudx_.get(nodeL, offSetIJ) + avgDudx_.get(nodeR, offSetIJ));
      const EdgeKernelTraits::DblType GjfluctUi =
        0.5 * (dudx_.get(nodeL, offSetIJ) + dudx_.get(nodeR, offSetIJ)) -
        GjavgUi;
      avgdUidxj[i][j] = GjavgUi + (avgUidiff - GlavgUidxl) * av[j] * inv_axdx;
      fluctdUidxj[i][j] =
        GjfluctUi + (fluctUidiff - GlfluctUidxl) * av[j] * inv_axdx;
    }
  }

  EdgeKernelTraits::DblType avgDivU = 0.0;
  for (int i = 0; i < ndim; ++i) {
    avgDivU += avgdUidxj[i][i];
  }

  // FIXME: Does this need a rho in it?
  const EdgeKernelTraits::DblType epsilon13Ip = stk::math::pow(betaStar_ * tkeIp * sdrIp, 1.0 / 3.0);

  for (int i = 0; i < ndim; ++i) {

    // This is the divU term for the average quantities in the model for
    // tau_ij^SGRS Since we are letting SST calculate it's normal mu_t, we
    // need to scale by alpha here
    const EdgeKernelTraits::DblType avgDivUstress =
      2.0 / 3.0 * alphaIp * muIp * avgDivU * av[i] * includeDivU_;
    smdata.rhs(0 + i) -= avgDivUstress;
    smdata.rhs(3 + i) += avgDivUstress;

    // Hybrid turbulence diffusion term; -(mu^jk*dui/dxk + mu^ik*duj/dxk -
    // 2/3*rho*tke*del_ij)*Aj
    for (int j = 0; j < ndim; ++j) {
      const EdgeKernelTraits::DblType avgUjIp =
        0.5 * (avgVelocity_.get(nodeL, j) + avgVelocity_.get(nodeR, j));
      const EdgeKernelTraits::DblType fluctUjIp =
        0.5 * (velocity_.get(nodeL, j) + velocity_.get(nodeR, j)) - avgUjIp;

      // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
      EdgeKernelTraits::DblType rhsfacDiff_i = 0.0;
      for (int k = 0; k < ndim; ++k) {
        // FIXME: I need to verify this form, fluctRho or avgRho
        // ..., do I need a deviatoric part only...
        // fluctRho will be 0 for incompressible, so if that's the right
        // term, need a better way to handle it, probably up above...
        rhsfacDiff_i += -avgRhoIp * CM43 * epsilon13Ip * M43[j][k] *
                        fluctdUidxj[i][k] * av[j];
      }

      // SGRS (average) term, scaled by alpha
      const EdgeKernelTraits::DblType rhsSGRCfacDiff_i =
        -alphaIp * muIp * avgdUidxj[i][j] * av[j];

      smdata.rhs(0 + i) -= rhsfacDiff_i + rhsSGRCfacDiff_i;
      smdata.rhs(3 + i) += rhsfacDiff_i + rhsSGRCfacDiff_i;

      // -mut^ik*duj/dxk*A_j
      EdgeKernelTraits::DblType rhsfacDiff_j = 0.0;
      for (int k = 0; k < ndim; ++k) {
        // FIXME: See above notes...
        rhsfacDiff_j += -avgRhoIp * CM43 * epsilon13Ip * M43[i][k] *
                        fluctdUidxj[j][k] * av[j];
      }

      // SGRS (average) term, scaled by alpha
      const EdgeKernelTraits::DblType rhsSGRCfacDiff_j =
        -alphaIp * muIp * avgdUidxj[j][i] * av[j];

      smdata.rhs(0 + i) -= rhsfacDiff_j + rhsSGRCfacDiff_j;
      smdata.rhs(3 + i) += rhsfacDiff_j + rhsSGRCfacDiff_j;
    }
  }
}

} // namespace nalu
} // namespace sierra
