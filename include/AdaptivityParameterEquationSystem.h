/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AdaptivityParameterEquationSystem_h
#define AdaptivityParameterEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class AssembleNodalGradAlgorithmDriver;
class LinearSystem;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;

class AdaptivityParameterEquationSystem : public EquationSystem {

public:

  AdaptivityParameterEquationSystem(
    EquationSystems& equationSystems);
  virtual ~AdaptivityParameterEquationSystem();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  void register_interior_algorithm(
    stk::mesh::Part *part);
  
  void register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const InflowBoundaryConditionData &inflowBCData);
  
  void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const OpenBoundaryConditionData &openBCData);

  void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const WallBoundaryConditionData &wallBCData);
  
  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const SymmetryBoundaryConditionData &symmetryBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();
  
  void predict_state();
  
  void solve_and_update();
  void initial_work();

  void compute_effective_diff_flux_coeff();
  void compute_resolution_adequacy_parameters();
  void compute_metric_tensor();
  void update_and_clip();

  void manage_projected_nodal_gradient(
    EquationSystems& eqSystems);
  void compute_projected_nodal_gradient();
  
  const bool managePNG_;

  ScalarFieldType *alpha_;
  VectorFieldType *dadx_;
  ScalarFieldType *aTmp_;
  ScalarFieldType *evisc_;
  ScalarFieldType *resAdeq_;
  GenericFieldType *metric_;
 
  AssembleNodalGradAlgorithmDriver *assembleNodalGradAlgDriver_;
  AlgorithmDriver *diffFluxCoeffAlgDriver_;
  AlgorithmDriver *resolutionAdequacyAlgDriver_;
  AlgorithmDriver *metricTensorAlgDriver_;
  const TurbulenceModel turbulenceModel_;

  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;

  bool isInit_;

};


} // namespace nalu
} // namespace Sierra

#endif
