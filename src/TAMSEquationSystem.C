/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <AssembleScalarElemSolverAlgorithm.h>
#include <AssembleScalarElemOpenSolverAlgorithm.h>
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <AssembleNodeSolverAlgorithm.h>
#include <AssembleNodalGradAlgorithmDriver.h>
#include <AssembleNodalGradEdgeAlgorithm.h>
#include <AssembleNodalGradElemAlgorithm.h>
#include <AssembleNodalGradBoundaryAlgorithm.h>
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <AuxFunctionAlgorithm.h>
#include <ConstantAuxFunction.h>
#include <CopyFieldAlgorithm.h>
#include <ComputeTAMSResAdequacyElemAlgorithm.h>
#include <ComputeMetricTensorElemAlgorithm.h>
#include <ComputeTAMSAveragesElemAlgorithm.h>
#include <ComputeTAMSKratioElemAlgorithm.h>
#include <DirichletBC.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <ProjectedNodalGradientEquationSystem.h>
#include <Realm.h>
#include <Realms.h>
#include <ScalarGclNodeSuppAlg.h>
#include <ScalarMassBackwardEulerNodeSuppAlg.h>
#include <ScalarMassBDF2NodeSuppAlg.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TAMSEquationSystem.h>
#include <TimeIntegrator.h>

#include <SolverAlgorithmDriver.h>


// template for supp algs
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// consolidated
#include <AssembleElemSolverAlgorithm.h>
#include <kernel/ScalarMassElemKernel.h>
#include <kernel/ScalarAdvDiffElemKernel.h>
#include <kernel/ScalarUpwAdvDiffElemKernel.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>

// deprecated
#include <ScalarMassElemSuppAlgDep.h>
#include <nso/ScalarNSOKeElemSuppAlg.h>
#include <nso/ScalarNSOElemSuppAlgDep.h>

#include <overset/UpdateOversetFringeAlgorithmDriver.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TAMSEquationSystem - manages alpha pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TAMSEquationSystem::TAMSEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "TAMSEQS","time_averaged_model_split"),
    managePNG_(realm_.get_consistent_mass_matrix_png("adaptivity_parameter")),
    avgVelocity_(NULL),
    avgPressure_(NULL),
    avgDensity_(NULL),
    avgResolvedStress_(NULL),
    avgDudx_(NULL),
    metric_(NULL),
    alpha_(NULL),
    resAdequacy_(NULL),
    avgResAdequacy_(NULL),
    gTmp_(NULL),
    metricTensorAlgDriver_(new AlgorithmDriver(realm_)),
    resolutionAdequacyAlgDriver_(new AlgorithmDriver(realm_)),
    averagingAlgDriver_(new AlgorithmDriver(realm_)),
    alphaAlgDriver_(new AlgorithmDriver(realm_)),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    isInit_(true)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("time_averaged_model_split");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TAMS);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  if ( turbulenceModel_ != TAMS ) {
    throw std::runtime_error("User has requested TAMSEqs, however, turbulence model has not been set to tams, the only one supported by this equation system currently.");
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TAMSEquationSystem::~TAMSEquationSystem()
{
  if (NULL != metricTensorAlgDriver_)
    delete metricTensorAlgDriver_;
  if (NULL != averagingAlgDriver_)
    delete averagingAlgDriver_;
  if (NULL != alphaAlgDriver_)
    delete alphaAlgDriver_;
  if (NULL != resolutionAdequacyAlgDriver_)
    delete resolutionAdequacyAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  alpha_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio"));
  stk::mesh::put_field_on_mesh(*alpha_, *part, nullptr);

  avgVelocity_ = &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "average_velocity"));
  stk::mesh::put_field_on_mesh(*avgVelocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("average_velocity");

  avgPressure_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_pressure"));
  stk::mesh::put_field_on_mesh(*avgPressure_, *part, nullptr);
  realm_.augment_restart_variable_list("average_pressure");

  avgDensity_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_density"));
  stk::mesh::put_field_on_mesh(*avgDensity_, *part, nullptr);
  realm_.augment_restart_variable_list("average_density");

  avgDudx_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "average_dudx"));
  stk::mesh::put_field_on_mesh(*avgDudx_, *part, nDim*nDim, nullptr);
  realm_.augment_restart_variable_list("average_dudx");

  avgResolvedStress_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "average_resolved_stress"));
  stk::mesh::put_field_on_mesh(*avgResolvedStress_, *part, nDim*nDim, nullptr);
  realm_.augment_restart_variable_list("average_resolved_stress");

  metric_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "metric_tensor"));
  stk::mesh::put_field_on_mesh(*metric_, *part, nDim*nDim, nullptr);

  resAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, "resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*resAdequacy_, *part, nullptr);

  avgResAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, "average_resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*avgResAdequacy_, *part, nullptr);
  realm_.augment_restart_variable_list("average_resolution_adequacy_parameter");

  // delta solution for linear solver; share delta with other split systems
  gTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "gTmp"));
  stk::mesh::put_field_on_mesh(*gTmp_, *part, nDim, nullptr);

}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // resolution adequacy algorithm
  if ( NULL == resolutionAdequacyAlgDriver_ )
    resolutionAdequacyAlgDriver_ = new AlgorithmDriver(realm_);
  
  std::map<AlgorithmType, Algorithm *>::iterator it = 
    resolutionAdequacyAlgDriver_->algMap_.find(algType);

  if (it == resolutionAdequacyAlgDriver_->algMap_.end() ) {
    ComputeTAMSResAdequacyElemAlgorithm *resAdeqAlg =
      new ComputeTAMSResAdequacyElemAlgorithm(realm_, part);
    resolutionAdequacyAlgDriver_->algMap_[algType] = resAdeqAlg;
  }
  else {
    it->second->partVec_.push_back(part);
  }

  // metric tensor algorithm
  if ( NULL == metricTensorAlgDriver_ )
    metricTensorAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm *>::iterator itmt =
    metricTensorAlgDriver_->algMap_.find(algType);

  if (itmt == metricTensorAlgDriver_->algMap_.end() ) {
    ComputeMetricTensorElemAlgorithm *metricTensorAlg =
      new ComputeMetricTensorElemAlgorithm(realm_, part);
    metricTensorAlgDriver_->algMap_[algType] = metricTensorAlg;
  }
  else {
    itmt->second->partVec_.push_back(part);
  }

  // averaging algorithm
  if ( NULL == averagingAlgDriver_ )
    averagingAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm *>::iterator itavg =
    averagingAlgDriver_->algMap_.find(algType);

  if (itavg == averagingAlgDriver_->algMap_.end() ) {
    ComputeTAMSAveragesElemAlgorithm *averagingAlg =
      new ComputeTAMSAveragesElemAlgorithm(realm_, part);
    averagingAlgDriver_->algMap_[algType] = averagingAlg;
  }
  else {
    itmt->second->partVec_.push_back(part);
  }

  // alpha algorithm
  if ( NULL == alphaAlgDriver_ )
    alphaAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm *>::iterator itkr =
    alphaAlgDriver_->algMap_.find(algType);

  if (itkr == alphaAlgDriver_->algMap_.end() ) {
    ComputeTAMSKratioElemAlgorithm *alphaAlg =
      new ComputeTAMSKratioElemAlgorithm(realm_, part);
    alphaAlgDriver_->algMap_[algType] = alphaAlg;
  }
  else {
    itmt->second->partVec_.push_back(part);
  }

  //KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, realm_.using_tensor_product_kernels());

  //kb.build_topo_kernel_if_requested<TAMSForcingElemKernel>
  //("forcing",
  //  realm_.bulk_data(), *realm_.solutionOptions_, kb.data_prereqs());

  //kb.report();

}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const OpenBoundaryConditionData &openBCData)
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &symmetryBCData)
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_overset_bc()
{
  // Nothing to do here...
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::initialize()
{
  //solverAlgDriver_->initialize_connectivity();
  //linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::reinitialize_linear_system()
{
  /*
  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_ADAPT_PARAM;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName = realm_.equationSystems_.get_solver_block_name("adaptivity_parameter");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_ADAPT_PARAM);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
  */
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::solve_and_update()
{

  if ( isInit_ ) {
    //compute_projected_nodal_gradient();
    //compute_metric_tensor();
    isInit_ = false;
  }

  compute_averages();
 
  compute_alpha();

  compute_resolution_adequacy_parameters();

  // TODO: Add recalculation of metric tensor if mesh changes

  // Forcing poisson solve
  // start the iteration loop
/* 
    for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // continuity assemble, load_complete and solve
    assemble_and_solve(gTmp_);

    // update... (Can use this if I don't need to clip, other use update_and_clip...
    timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *gTmp_,
      1.0, *forcing_,
      realm_.get_activate_aura());
    timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);
  }
*/
}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::initial_work()
{
  compute_metric_tensor();

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // FIXME: Hack since setting an element field to a constant using Aux doesn't seem to work...
  // required fields

  // define some common selectors
  stk::mesh::Selector s_all_elem
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*avgResAdequacy_);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_all_elem );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    double *avgResAdeq = stk::mesh::field_data(*avgResAdequacy_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
       avgResAdeq[k] = 1.0;
    }
  }

  const int nDim = meta_data.spatial_dimension();

  GenericFieldType *dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*avgDudx_);

  // FIXME: I need to initialize the computed quantities... avg_dudx
  // since they will be weighted
  s_all_nodes = (meta_data.locally_owned_part() | meta_data.globally_shared_part()) 
    &stk::mesh::selectField(*avgResolvedStress_);

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = buckets.begin();
       ib != buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get velocity field data
      const double * dudx = stk::mesh::field_data(*dudx_, b[k]);
      double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);

      for (int i = 0; i < nDim; ++i) 
        for (int j = 0; j < nDim; ++j) 
          avgDudx[i*nDim + j] = dudx[i*nDim + j];
    }
  }

  compute_averages();
  compute_resolution_adequacy_parameters();
  compute_alpha(); 
}

//--------------------------------------------------------------------------
//-------- compute_resolution_adequacy_parameters() ------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::compute_resolution_adequacy_parameters()
{
  if ( NULL != resolutionAdequacyAlgDriver_)
    resolutionAdequacyAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_metric_tensor() -----------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::compute_metric_tensor()
{
  if ( NULL != metricTensorAlgDriver_)
    metricTensorAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_averages() ----------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::compute_averages()
{
  if ( NULL != averagingAlgDriver_)
    averagingAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_alpha() -------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::compute_alpha()
{
  if ( NULL != alphaAlgDriver_)
    alphaAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::update_and_clip()
{
  // nothing to do here...
}

} // namespace nalu
} // namespace Sierra
