/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "edge_kernels/MomentumTAMSKEpsEdgeDiffSolverAlg.h"
#include "EigenDecomposition.h"
#include "utils/StkHelpers.h"
#include "utils/TAMSUtils.h"

namespace sierra {
namespace nalu {

MomentumTAMSKEpsEdgeDiffSolverAlg::MomentumTAMSKEpsEdgeDiffSolverAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm.get_divU()),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg))
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string velField =
    realm.does_mesh_move() ? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(meta, velField);
  dudx_ = get_field_ordinal(meta, "dudx");
  densityNp1_ = get_field_ordinal(meta, "density");
  tkeNp1_ = get_field_ordinal(meta, "turbulent_ke");
  tdrNp1_ = get_field_ordinal(meta, "total_dissipation_rate");
  alphaNp1_ = get_field_ordinal(meta, "k_ratio");
  tvisc_ = get_field_ordinal(meta, "turbulent_viscosity");
  Mij_ = get_field_ordinal(meta, "metric_tensor");

  avgVelocity_ = get_field_ordinal(meta, "average_velocity");
  avgDudx_ = get_field_ordinal(meta, "avgDudx");
  avgDensity_ = get_field_ordinal(meta, "average_density");
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
}

void
MomentumTAMSKEpsEdgeDiffSolverAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  // STK ngp::Field instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto velocity = fieldMgr.get_field<double>(velocityRTM_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  const auto density = fieldMgr.get_field<double>(densityNp1_);
  const auto tke = fieldMgr.get_field<double>(tkeNp1_);
  const auto tdr = fieldMgr.get_field<double>(tdrNp1_);
  const auto alpha = fieldMgr.get_field<double>(alphaNp1_);
  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  const auto avgVelocity = fieldMgr.get_field<double>(avgVelocity_);
  const auto avgDudx = fieldMgr.get_field<double>(avgDudx_);
  const auto avgDensity = fieldMgr.get_field<double>(avgDensity_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto nodalMij = fieldMgr.get_field<double>(Mij_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[nDimMax_];
      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      // Mij, eigenvectors and eigenvalues
      // FIXME make the nDimMax_ -> ndim
      DblType Mij[nDimMax_][nDimMax_];
      DblType Q[nDimMax_][nDimMax_];
      DblType D[nDimMax_][nDimMax_];
      for (unsigned i = 0; i < ndim; i++)
        for (unsigned j = 0; j < ndim; j++)
           // FIXME: Is this right for accessing 2D array of nodal Mij, is it 1D or 2D??
           Mij[i][j] = 0.5 * (nodalMij.get(nodeL, i*ndim + j) + nodalMij.get(nodeR, i*ndim + j));

      EigenDecomposition::sym_diagonalize<DblType>(Mij, Q, D);

      // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
      // so to create M43, we use Q D^(4/3) Q^T
      DblType M43[nDimMax_][nDimMax_];
      for (unsigned i = 0; i < ndim; i++)
        for (unsigned j = 0; j < ndim; j++)
          M43[i][j] = 0.0;

      const double fourThirds = 4. / 3.;
      for (unsigned k = 0; k < ndim; k++) {
        const DblType D43 = stk::math::pow(D[k][k], fourThirds);
        for (unsigned i = 0; i < ndim; i++) {
          for (unsigned j = 0; j < ndim; j++) {
            M43[i][j] += Q[i][k] * Q[j][k] * D43;
          }
        }
      }

      // Compute CM43
      DblType CM43 = tams_utils::get_M43_constant(D, CMdeg_);

      const DblType muIp = 0.5 * (tvisc.get(nodeL, 0) + tvisc.get(nodeR, 0));
      const DblType avgRhoIp =
        0.5 * (avgDensity.get(nodeL, 0) + avgDensity.get(nodeR, 0));
      const DblType fluctRhoIp =
        0.5 * (density.get(nodeL, 0) + density.get(nodeR, 0)) - avgRhoIp;
      const DblType tkeIp = 0.5 * (tke.get(nodeL, 0) + tke.get(nodeR, 0));
      const DblType tdrIp = 0.5 * (tdr.get(nodeL, 0) + tdr.get(nodeR, 0));
      const DblType alphaIp = 0.5 * (alpha.get(nodeL, 0) + alpha.get(nodeR, 0));
      DblType avgdUidxj[nDimMax_][nDimMax_];
      DblType fluctdUidxj[nDimMax_][nDimMax_];

      DblType axdx = 0.0;
      DblType asq = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      // Compute average divU
      for (int i = 0; i < ndim; ++i) {

        // difference between R and L nodes for component i
        const DblType avgUidiff =
          avgVelocity.get(nodeR, i) - avgVelocity.get(nodeL, i);
        const DblType fluctUidiff =
          (velocity.get(nodeR, i) - velocity.get(nodeL, i)) - avgUidiff;

        const int offSetI = ndim * i;

        // start sum for NOC contribution
        DblType GlavgUidxl = 0.0;
        DblType GlfluctUidxl = 0.0;
        for (int l = 0; l < ndim; ++l) {
          const int offSetIL = offSetI + l;
          const DblType dxl =
            coordinates.get(nodeR, l) - coordinates.get(nodeL, l);
          const DblType GlavgUi =
            0.5 * (avgDudx.get(nodeL, offSetIL) + avgDudx.get(nodeR, offSetIL));
          const DblType GlfluctUi =
            0.5 * (dudx.get(nodeL, offSetIL) + dudx.get(nodeR, offSetIL)) -
            GlavgUi;
          GlavgUidxl += GlavgUi * dxl;
          GlfluctUidxl += GlfluctUi * dxl;
        }

        // form full tensor dui/dxj with NOC
        for (int j = 0; j < ndim; ++j) {
          const int offSetIJ = offSetI + j;
          const DblType GjavgUi =
            0.5 * (avgDudx.get(nodeL, offSetIJ) + avgDudx.get(nodeR, offSetIJ));
          const DblType GjfluctUi =
            0.5 * (dudx.get(nodeL, offSetIJ) + dudx.get(nodeR, offSetIJ)) -
            GjavgUi;
          avgdUidxj[i][j] =
            GjavgUi + (avgUidiff - GlavgUidxl) * av[j] * inv_axdx;
          fluctdUidxj[i][j] =
            GjfluctUi + (fluctUidiff - GlfluctUidxl) * av[j] * inv_axdx;
        }
      }

      DblType avgDivU = 0.0;
      for (int i = 0; i < ndim; ++i) {
        avgDivU += avgdUidxj[i][i];
      }

      // FIXME: Does this need a rho in it?
      const DblType epsilon13Ip = stk::math::pow(tdrIp, 1.0 / 3.0);

      for (int i = 0; i < ndim; ++i) {

        // This is the divU term for the average quantities in the model for
        // tau_ij^SGRS Since we are letting SST calculate it's normal mu_t, we
        // need to scale by alpha here
        const DblType avgDivUstress =
          2.0 / 3.0 * alphaIp * muIp * avgDivU * av[i] * includeDivU_;
        smdata.rhs(0 + i) -= avgDivUstress;
        smdata.rhs(3 + i) += avgDivUstress;

        // Hybrid turbulence diffusion term; -(mu^jk*dui/dxk + mu^ik*duj/dxk -
        // 2/3*rho*tke*del_ij)*Aj
        for (int j = 0; j < ndim; ++j) {
          const DblType avgUjIp =
            0.5 * (avgVelocity.get(nodeL, j) + avgVelocity.get(nodeR, j));
          const DblType fluctUjIp =
            0.5 * (velocity.get(nodeL, j) + velocity.get(nodeR, j)) - avgUjIp;

          // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
          DblType rhsfacDiff_i = 0.0;
          for (int k = 0; k < ndim; ++k) {
            // FIXME: I need to verify this form, fluctRho or avgRho
            // ..., do I need a deviatoric part only...
            // fluctRho will be 0 for incompressible, so if that's the right
            // term, need a better way to handle it, probably up above...
            rhsfacDiff_i += -avgRhoIp * CM43 * epsilon13Ip * M43[j][k] *
                            fluctdUidxj[i][k] * av[j];
          }

          // SGRS (average) term, scaled by alpha
          const DblType rhsSGRCfacDiff_i =
            -alphaIp * muIp * avgdUidxj[i][j] * av[i];

          smdata.rhs(0 + i) -= rhsfacDiff_i + rhsSGRCfacDiff_i;
          smdata.rhs(3 + i) += rhsfacDiff_i + rhsSGRCfacDiff_i;

          // -mut^ik*duj/dxk*A_j
          DblType rhsfacDiff_j = 0.0;
          for (int k = 0; k < ndim; ++k) {
            // FIXME: See above notes...
            rhsfacDiff_j += -avgRhoIp * CM43 * epsilon13Ip * M43[i][k] *
                            fluctdUidxj[j][k] * av[j];
          }

          // SGRS (average) term, scaled by alpha
          const DblType rhsSGRCfacDiff_j =
            -alphaIp * muIp * avgdUidxj[j][i] * av[i];

          smdata.rhs(0 + i) -= rhsfacDiff_j + rhsSGRCfacDiff_j;
          smdata.rhs(3 + i) += rhsfacDiff_j + rhsSGRCfacDiff_j;
        }
      }
    });
}

} // namespace nalu
} // namespace sierra
