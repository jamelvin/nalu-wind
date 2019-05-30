/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "edge_kernels/MomentumTAMSKEpsForcingEdgeSolverAlg.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MomentumTAMSKEpsForcingEdgeSolverAlg::MomentumTAMSKEpsForcingEdgeSolverAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem
) : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();
  pi_ = stk::math::acos(-1.0); 
  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string velField = realm.does_mesh_move()? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(meta, velField);
  viscosity_ = get_field_ordinal(meta, "viscosity");
  turbVisc_ = get_field_ordinal(meta, "turbulent_viscosity");
  densityNp1_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  tkeNp1_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  tdrNp1_ = get_field_ordinal(meta, "total_dissipation_rate", stk::mesh::StateNP1);
  alpha_ = get_field_ordinal(meta, "k_ratio");
  Mij_ = get_field_ordinal(meta, "metric_tensor");
  minDist_ = get_field_ordinal(meta, "minimum_distance_to_wall");

  // average quantities
  avgVelocity_ = get_field_ordinal(meta, "average_velocity");
  avgDensity_ = get_field_ordinal(meta, "average_density"); 
  avgTime_ = get_field_ordinal(meta, "average_time"); 
  avgResAdeq_ =  get_field_ordinal(meta, "average_resolution_adequacy_parameter");

  edgeAreaVec_ = get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
}

void
MomentumTAMSKEpsForcingEdgeSolverAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  // Time information
  const DblType dt = realm_.get_time_step();
  const DblType time = realm_.get_current_time();
  const DblType step = realm_.get_time_step_count();

  // STK ngp::Field instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto velocity = fieldMgr.get_field<double>(velocityRTM_);
  const auto viscosity = fieldMgr.get_field<double>(viscosity_);
  const auto tvisc = fieldMgr.get_field<double>(turbVisc_);
  const auto density = fieldMgr.get_field<double>(densityNp1_);
  const auto tke = fieldMgr.get_field<double>(tkeNp1_);
  const auto tdr = fieldMgr.get_field<double>(tdrNp1_);
  const auto alpha = fieldMgr.get_field<double>(alpha_);
  const auto Mij = fieldMgr.get_field<double>(Mij_);
  const auto minDist = fieldMgr.get_field<double>(minDist_);
  const auto avgVelocity = fieldMgr.get_field<double>(avgVelocity_);
  const auto avgDensity = fieldMgr.get_field<double>(avgDensity_);
  const auto avgTime = fieldMgr.get_field<double>(avgTime_);
  const auto avgResAdeq = fieldMgr.get_field<double>(avgResAdeq_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType& smdata,
      const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR)
    {
      // Scratch work arrays 
      NALU_ALIGNED DblType av[nDimMax_];      // edgeAreaVector
      NALU_ALIGNED DblType coords[nDimMax_];  // coordinates
      NALU_ALIGNED DblType avgU[nDimMax_];    // averageVelocity
      NALU_ALIGNED DblType fluctU[nDimMax_];  // fluctuatingVelocity

      // Populate area vector work array
      for (int d=0; d < ndim; ++d) 
        av[d] = edgeAreaVec.get(edge, d);

      const DblType muIp = 0.5 * (viscosity.get(nodeL,0) + viscosity.get(nodeR, 0));
      const DblType tviscIp = 0.5 * (tvisc.get(nodeL, 0) + tvisc.get(nodeR, 0));
      const DblType rhoIp = 0.5 * (density.get(nodeL, 0) + density.get(nodeR, 0));
      const DblType tkeIp = 0.5 * (tke.get(nodeL, 0) + tke.get(nodeR, 0));
      const DblType tdrIp = 0.5 * (tdr.get(nodeL, 0) + tdr.get(nodeR, 0));
      const DblType alphaIp = 0.5 * (alpha.get(nodeL, 0) + alpha.get(nodeR, 0));
      const DblType wallDistIp = 0.5 * (minDist.get(nodeL, 0) + minDist.get(nodeR, 0));
      const DblType avgRhoIp = 0.5 * (avgDensity.get(nodeL, 0) + avgDensity.get(nodeR, 0));
      const DblType avgTimeIp = 0.5 * (avgTime.get(nodeL, 0) + avgTime.get(nodeR, 0));
      const DblType avgResAdeqIp = 0.5 * (avgResAdeq.get(nodeL, 0) + avgResAdeq.get(nodeR, 0));
      // FIXME: How do I properly access array elements from Mij??
      const DblType Mij_00 = 0.5 * (Mij.get(nodeL, 0)) + (Mij.get(nodeR, 0));
      const DblType Mij_11 = 0.5 * (Mij.get(nodeL, 5)) + (Mij.get(nodeR, 5));
      const DblType Mij_22 = 0.5 * (Mij.get(nodeL, 9)) + (Mij.get(nodeR, 9));

      DblType asq = 0.0;
      DblType axdx = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj = coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
        fluctU[d] = 0.5 * (velocity.get(nodeL, d) - avgVelocity.get(nodeL, d) + 
                           velocity.get(nodeR, d) - avgVelocity.get(nodeR, d));
        avgU[d] = 0.5 * (avgVelocity.get(nodeL, d) + avgVelocity.get(nodeR, d));
        coords[d] = 0.5 * (coordinates.get(nodeL, d) + coordinates.get(nodeR, d));
      }
      const DblType inv_axdx = 1.0 / axdx;

      // First we calculate the a_i's
      const double FORCING_CL = 4.0;
      const double Ceta = 70.0;
      const double Ct = 6.0;
      const double BL_T = 1.0;
      const double BL_KOL = 1.0;
      const double FORCING_FACTOR = 8.0;
  
      const DblType periodicForcingLengthX = pi_;
      const DblType periodicForcingLengthY = 0.25;
      const DblType periodicForcingLengthZ = 3.0 / 8.0 * pi_;
  
      DblType length = FORCING_CL * stk::math::pow(alphaIp * tkeIp, 1.5) / tdrIp;
      length = stk::math::max(length, 
                  Ceta*(stk::math::pow(muIp, 0.75)/stk::math::pow(tdrIp, 0.25)));
      // FIXME: For channel, only want to clip in wall normal direction with wallDist
      //        For other flows, will need a better approach?
      DblType lengthY = stk::math::min(length, wallDistIp);
  
      DblType T_alpha = alphaIp * tkeIp / tdrIp;
      T_alpha = stk::math::max(T_alpha, Ct * stk::math::sqrt(muIp / tdrIp));
      T_alpha = BL_T * T_alpha;
  
      // FIXME: Deal with Mij
      const DblType ceilLengthX = stk::math::max(length,  2.0 * Mij_00);
      const DblType ceilLengthY = stk::math::max(lengthY, 2.0 * Mij_11);
      const DblType ceilLengthZ = stk::math::max(length,  2.0 * Mij_22);
  
      const DblType clipLengthX = stk::math::min(ceilLengthX, periodicForcingLengthX);
      const DblType clipLengthY = stk::math::min(ceilLengthY, periodicForcingLengthY);
      const DblType clipLengthZ = stk::math::min(ceilLengthZ, periodicForcingLengthZ);

      // FIXME: Hack to do a round/floor/ceil/mod operation since it isnt in
      // stk::math right now
      DblType ratioX;
      DblType ratioY;
      DblType ratioZ;
      for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
        double tmpD = stk::simd::get_data(clipLengthX, simdIndex);
        double tmpN = stk::simd::get_data(periodicForcingLengthX, simdIndex);
        double tmp = std::floor(tmpN / tmpD + 0.5);
        stk::simd::set_data(ratioX, simdIndex, tmp);
  
        tmpD = stk::simd::get_data(clipLengthY, simdIndex);
        tmpN = stk::simd::get_data(periodicForcingLengthY, simdIndex);
        tmp = std::floor(tmpN / tmpD + 0.5);
        stk::simd::set_data(ratioY, simdIndex, tmp);
  
        tmpD = stk::simd::get_data(clipLengthZ, simdIndex);
        tmpN = stk::simd::get_data(periodicForcingLengthZ, simdIndex);
        tmp = std::floor(tmpN / tmpD + 0.5);
        stk::simd::set_data(ratioZ, simdIndex, tmp);
      }
  
      const DblType denomX = periodicForcingLengthX / ratioX;
      const DblType denomY = periodicForcingLengthY / ratioY;
      const DblType denomZ = periodicForcingLengthZ / ratioZ;
  
      const DblType ax = pi_ / denomX;
      const DblType ay = pi_ / denomY;
      const DblType az = pi_ / denomZ;
  
      // Then we calculate the arguments for the Taylor-Green Vortex
      const DblType xarg = ax * (coords[0] + avgU[0] * time);
      const DblType yarg = ay * (coords[1] + avgU[1] * time);
      const DblType zarg = az * (coords[2] + avgU[2] * time);
  
      // Now we calculate the initial Taylor-Green field
      DblType hX =  1.0/3.0 * stk::math::cos(xarg)*stk::math::sin(yarg)*stk::math::sin(zarg);
      DblType hY = -1.0     * stk::math::sin(xarg)*stk::math::cos(yarg)*stk::math::sin(zarg);
      DblType hZ =  2.0/3.0 * stk::math::sin(xarg)*stk::math::sin(yarg)*stk::math::cos(zarg);
  
      // Now we calculate the scaling of the initial field
      // FIXME: Pass the 0.22 as another turbulence constant (V2F_Cmu)
      const DblType v2Ip = tviscIp / (0.22 * rhoIp * avgTimeIp);
      const DblType F_target = FORCING_FACTOR * stk::math::sqrt(alphaIp * v2Ip) / T_alpha;
  
      const DblType prod_r = (F_target*dt)*(hX*fluctU[0] + hY*fluctU[1] + hZ*fluctU[2]);
  
      const DblType arg1 = stk::math::sqrt(avgResAdeqIp) - 1.0;
      const DblType arg = stk::math::if_then_else(arg1 < 0.0, 1.0 - 1.0 /
                                       stk::math::sqrt(avgResAdeqIp),arg1);
  
      const DblType a_sign = stk::math::tanh(arg);
  
      DblType Sa = a_sign;
  
      const DblType a_kol = stk::math::min(
        BL_KOL * stk::math::sqrt(muIp * tdrIp / rhoIp) / tkeIp, 1.0);
  
      // FIXME: Can I do a compound if statement with if_then... it was not
      // working...
      for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
        double tmp_asign = stk::simd::get_data(a_sign, simdIndex);
        double tmp_akol = stk::simd::get_data(a_kol, simdIndex);
        double tmp_alpha = stk::simd::get_data(alphaIp, simdIndex);
        double tmp_Sa = stk::simd::get_data(Sa, simdIndex);
  
        if (tmp_asign < 0.0) {
          if (tmp_alpha <= tmp_akol)
            tmp_Sa = tmp_Sa - (1.0 + tmp_akol - tmp_alpha) * tmp_asign;
        } else {
          if (tmp_alpha >= 1.0)
            tmp_Sa = tmp_Sa - tmp_alpha * tmp_asign;
        }
        stk::simd::set_data(Sa, simdIndex, tmp_Sa);
      }

      const DblType fd_temp = avgResAdeqIp;
  
      DblType C_F;
      // FIXME: Can I do a compound if statement with if_then... it was not
      // working...
      for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
        double tmp_fd = stk::simd::get_data(fd_temp, simdIndex);
        double tmp_prodr = stk::simd::get_data(prod_r, simdIndex);
        double tmp_CF = stk::simd::get_data(C_F, simdIndex);
        double tmp_Ftarget = stk::simd::get_data(F_target, simdIndex);
        double tmp_Sa = stk::simd::get_data(Sa, simdIndex);
  
        if ((tmp_fd < 1.0) && (tmp_prodr >= 0.0))
          tmp_CF = -1.0 * tmp_Ftarget * tmp_Sa;
        else
          tmp_CF = 0.0;
        stk::simd::set_data(C_F, simdIndex, tmp_CF);
      }
  
      // Since we aren't projecting, ignore scaling by dt which is done before
      // projection and then removed...
      const DblType norm = C_F;
  
      // Now we determine the actual forcing field
      DblType gX = norm * hX; 
      DblType gY = norm * hY;
      DblType gZ = norm * hZ;
  
      // TODO: Assess viability of first approach where we don't solve a poisson
      // problem and allow the field be divergent, which should get projected out
      // anyway. This means we only have a contribution to the RHS here
      const DblType areaFac = -asq * inv_axdx;

      // Left node entries
      smdata.rhs(0) = -gX * areaFac;
      smdata.rhs(1) = -gY * areaFac;
      smdata.rhs(2) = -gZ * areaFac;

      // Right node entries
      smdata.rhs(3) = gX * areaFac;
      smdata.rhs(4) = gY * areaFac;
      smdata.rhs(5) = gZ * areaFac;
    });
}

}  // nalu
}  // sierra
