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
#include <ComputeTAMSAvgMdotEdgeAlgorithm.h>
#include <ComputeTAMSAvgMdotElemAlgorithm.h>
#include <ComputeMetricTensorNodeAlgorithm.h>
#include <ComputeTAMSKEpsAveragesNodeAlgorithm.h>
#include <ComputeTAMSKEpsResAdequacyNodeAlgorithm.h>
#include <ComputeTAMSKEpsKratioNodeAlgorithm.h>
#include <ComputeTAMSSSTAveragesNodeAlgorithm.h>
#include <ComputeTAMSSSTKratioNodeAlgorithm.h>
#include <ComputeTAMSSSTResAdequacyNodeAlgorithm.h>
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
#include <TurbViscChienKEpsAlgorithm.h>
#include <TurbViscKsgsAlgorithm.h>
#include <TurbViscSmagorinskyAlgorithm.h>
#include <TurbViscSSTAlgorithm.h>
#include <TurbViscTAMSSSTAlgorithm.h>
#include <TurbViscTAMSKEpsAlgorithm.h>
#include <TurbViscWaleAlgorithm.h>

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
    avgTkeResolved_(NULL),
    avgDudx_(NULL),
    metric_(NULL),
    alpha_(NULL),
    resAdequacy_(NULL),
    avgResAdequacy_(NULL),
    avgProduction_(NULL),
    avgTime_(NULL),
    avgMdotScs_(NULL),
    avgMdot_(NULL),
    gTmp_(NULL),
    metricTensorAlgDriver_(new AlgorithmDriver(realm_)),
    resolutionAdequacyAlgDriver_(new AlgorithmDriver(realm_)),
    averagingAlgDriver_(new AlgorithmDriver(realm_)),
    alphaAlgDriver_(new AlgorithmDriver(realm_)),
    avgMdotAlgDriver_(new AlgorithmDriver(realm_)),
    tviscAlgDriver_(new AlgorithmDriver(realm_)),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    isInit_(true)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("time_averaged_model_split");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_TAMS);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  if ( turbulenceModel_ != TAMS_SST && turbulenceModel_ != TAMS_KEPS ) {
    throw std::runtime_error("User has requested TAMSEqs, however, turbulence model has not been set to tams_sst or tams_keps, the only ones supported by this equation system currently.");
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
  if (NULL != avgMdotAlgDriver_)
    delete avgMdotAlgDriver_;
  if (NULL != tviscAlgDriver_)
    delete tviscAlgDriver_;
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

  avgProduction_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_production"));
  stk::mesh::put_field_on_mesh(*avgProduction_, *part, nullptr);
  realm_.augment_restart_variable_list("average_production");

  NaluEnv::self().naluOutputP0() << "Declaring avgDudx in TAMS" << std::endl;

  avgDudx_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "average_dudx"));
  stk::mesh::put_field_on_mesh(*avgDudx_, *part, nDim*nDim, nullptr);
  realm_.augment_restart_variable_list("average_dudx");

  avgTkeResolved_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "average_tke_resolved"));
  stk::mesh::put_field_on_mesh(*avgTkeResolved_, *part, nullptr);
  realm_.augment_restart_variable_list("average_tke_resolved");

  avgTime_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,"average_time"));
  stk::mesh::put_field_on_mesh(*avgTime_, *part, nullptr);

  metric_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "metric_tensor"));
  stk::mesh::put_field_on_mesh(*metric_, *part, nDim*nDim, nullptr);

  resAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*resAdequacy_, *part, nullptr);

  avgResAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "avg_res_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*avgResAdequacy_, *part, nullptr);
  realm_.augment_restart_variable_list("avg_res_adequacy_parameter");

//  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
//  const int numScsIp = meSCS->numIntPoints_;

//  avgMdot_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "average_mass_flow_rate"));
//  stk::mesh::put_field_on_mesh(*avgMdot_, *part, numScsIp, nullptr);
//  realm_.augment_restart_variable_list("average_mass_flow_rate");

  // delta solution for linear solver; share delta with other split systems
  gTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "gTmp"));
  stk::mesh::put_field_on_mesh(*gTmp_, *part, nDim, nullptr);

}

//--------------------------------------------------------------------------
//-------- register_element_fields -----------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_element_fields(
  stk::mesh::Part *part,
  const stk::topology &theTopo)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  //metric_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK,"metric_tensor"));
  //stk::mesh::put_field_on_mesh(*metric_, *part, nDim*nDim, nullptr);

  //resAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::ELEMENT_RANK,"resolution_adequacy_parameter"));
  //stk::mesh::put_field_on_mesh(*resAdequacy_, *part, nullptr);

  //avgResAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::ELEMENT_RANK,"avg_res_adequacy_parameter"));
  //stk::mesh::put_field_on_mesh(*avgResAdequacy_, *part, nullptr);
  //realm_.augment_restart_variable_list("avg_res_adequacy_parameter");

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meSCS->num_integration_points();

  NaluEnv::self().naluOutputP0() << "Mdot average added in TAMS " << std::endl;

  avgMdotScs_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK,"average_mass_flow_rate_scs"));
  stk::mesh::put_field_on_mesh(*avgMdotScs_, *part, numScsIp, nullptr);
  realm_.augment_restart_variable_list("average_mass_flow_rate_scs");
}

//--------------------------------------------------------------------------
//-------- register_edge_fields -------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::register_edge_fields(
  stk::mesh::Part *part)
{
  NaluEnv::self().naluOutputP0() << "Edge fields added in TAMS " << std::endl;
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  avgMdot_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::EDGE_RANK, "average_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*avgMdot_, *part, nullptr);
  realm_.augment_restart_variable_list("average_mass_flow_rate");
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
    Algorithm * theAlg = NULL;
    switch (turbulenceModel_ ) {
      case TAMS_SST:
        theAlg = new ComputeTAMSSSTResAdequacyNodeAlgorithm(realm_, part);
        break;
      case TAMS_KEPS:
        theAlg = new ComputeTAMSKEpsResAdequacyNodeAlgorithm(realm_, part);
        break;
      default:
        throw std::runtime_error("TAMSEquationSystem: non-supported turb model");
    }
    resolutionAdequacyAlgDriver_->algMap_[algType] = theAlg;    
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
    ComputeMetricTensorNodeAlgorithm *metricTensorAlg =
      new ComputeMetricTensorNodeAlgorithm(realm_, part);
    metricTensorAlgDriver_->algMap_[algType] = metricTensorAlg;
  }
  else {
    itmt->second->partVec_.push_back(part);
  }

  // averaging algorithm
  if ( NULL == averagingAlgDriver_ )
    averagingAlgDriver_ = new AlgorithmDriver(realm_);
 
  std::map<AlgorithmType, Algorithm *>::iterator itav =
    averagingAlgDriver_->algMap_.find(algType);

  if (itav == averagingAlgDriver_->algMap_.end() ) {      
    Algorithm * theAlg = NULL;
    switch (turbulenceModel_ ) {
      case TAMS_SST:
        theAlg = new ComputeTAMSSSTAveragesNodeAlgorithm(realm_, part);
        break;
      case TAMS_KEPS:
        theAlg = new ComputeTAMSKEpsAveragesNodeAlgorithm(realm_, part);
        break;
      default:
        throw std::runtime_error("TAMSEquationSystem: non-supported turb model");
    }
    averagingAlgDriver_->algMap_[algType] = theAlg;    
  }
  else {
    itav->second->partVec_.push_back(part);
  }

  // alpha algorithm
  if ( NULL == alphaAlgDriver_ )
    alphaAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm *>::iterator itkr =
    alphaAlgDriver_->algMap_.find(algType);

  if (itkr == alphaAlgDriver_->algMap_.end() ) {
    Algorithm * theAlg = NULL;
    switch (turbulenceModel_ ) {
      case TAMS_SST:
        theAlg = new ComputeTAMSSSTKratioNodeAlgorithm(realm_, part);
        break;
      case TAMS_KEPS:
        theAlg = new ComputeTAMSKEpsKratioNodeAlgorithm(realm_, part);
        break;
      default:
        throw std::runtime_error("TAMSEquationSystem: non-supported turb model");
    }
    alphaAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    itkr->second->partVec_.push_back(part);
  }

  // avgMdot algorithm
  if ( NULL == avgMdotAlgDriver_ )
    avgMdotAlgDriver_ = new AlgorithmDriver(realm_);

  if (realm_.realmUsesEdges_) {
    std::map<AlgorithmType, Algorithm *>::iterator itmd =
      avgMdotAlgDriver_->algMap_.find(algType);

    if (itmd == avgMdotAlgDriver_->algMap_.end() ) {
      ComputeTAMSAvgMdotEdgeAlgorithm *avgMdotEdgeAlg =
        new ComputeTAMSAvgMdotEdgeAlgorithm(realm_, part);
      avgMdotAlgDriver_->algMap_[algType] = avgMdotEdgeAlg;
    }
    else {
      itmd->second->partVec_.push_back(part);
    }
  }
  else {
    std::map<AlgorithmType, Algorithm *>::iterator itmd =
      avgMdotAlgDriver_->algMap_.find(algType);

    if (itmd == avgMdotAlgDriver_->algMap_.end() ) {
      ComputeTAMSAvgMdotElemAlgorithm *avgMdotAlg =
        new ComputeTAMSAvgMdotElemAlgorithm(realm_, part);
      avgMdotAlgDriver_->algMap_[algType] = avgMdotAlg;
    }
    else {
      itmd->second->partVec_.push_back(part);
    }
  }

  // FIXME: tvisc needed for initialization only as TAMS goes before LowMach
  //        but relies on tvisc.  Perhaps there is a way to call tvisc from LowMach here? 
  std::map<AlgorithmType, Algorithm *>::iterator it_tv =
    tviscAlgDriver_->algMap_.find(algType);
  if ( it_tv == tviscAlgDriver_->algMap_.end() ) {
    Algorithm * theAlg = NULL;
    switch (realm_.solutionOptions_->turbulenceModel_ ) {
      case KSGS:
        theAlg = new TurbViscKsgsAlgorithm(realm_, part);
        break;
      case SMAGORINSKY:
        theAlg = new TurbViscSmagorinskyAlgorithm(realm_, part);
        break;
      case WALE:
        theAlg = new TurbViscWaleAlgorithm(realm_, part);
        break;
      case SST: case SST_DES:
        theAlg = new TurbViscSSTAlgorithm(realm_, part);
        break;
      case KEPS:
        theAlg = new TurbViscChienKEpsAlgorithm(realm_, part);
        break;
      case TAMS_SST: 
        theAlg = new TurbViscTAMSSSTAlgorithm(realm_,part);
        break;
      case TAMS_KEPS:
        theAlg = new TurbViscTAMSKEpsAlgorithm(realm_,part);
        break;
      default:
        throw std::runtime_error("non-supported turb model");
    }
    tviscAlgDriver_->algMap_[algType] = theAlg;
  }
  else {
    it_tv->second->partVec_.push_back(part);
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
    initialize_mdot();
    isInit_ = false;
  }

  // FIXME: Need this to be part of TAMS so that the order of operations is right, 
  //        is there a way to turn it off in LowMach, so it's not called twice?
  //tviscAlgDriver_->execute();

  //compute_averages();
 
  //compute_alpha();

  //compute_resolution_adequacy_parameters();

  //compute_avgMdot();

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
  // Need to calculate tvisc in initial work in cases where TAMS executes before
  // LowMach to prevent NaNs in initial avgResAdeq calculation
  //tviscAlgDriver_->execute();

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // FIXME: Hack since setting an element field to a constant using Aux doesn't seem to work...
  // required fields... Update: resAdeq has now been moved to a nodal quantity

  // define some common selectors
  //stk::mesh::Selector s_all_elem
  //  = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
  //  &stk::mesh::selectField(*avgResAdequacy_);

  //stk::mesh::BucketVector const& node_buckets =
  //  realm_.get_buckets( stk::topology::ELEMENT_RANK, s_all_elem );
  //for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
  //      ib != elem_buckets.end() ; ++ib ) {
  //  stk::mesh::Bucket & b = **ib ;
  //  const stk::mesh::Bucket::size_type length = b.size();

  //  double *avgResAdeq = stk::mesh::field_data(*avgResAdequacy_, b);

  //  for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
  //     avgResAdeq[k] = 1.0;
  //  }
  //}

  const int nDim = meta_data.spatial_dimension();

  GenericFieldType *dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  ScalarFieldType *turbKinEne_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  ScalarFieldType *tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  
  ScalarFieldType &tkeNp1 = turbKinEne_->field_of_state(stk::mesh::StateNP1);

  NaluEnv::self().naluOutputP0() << "About to initialize avgDudx in TAMS" << std::endl;

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*avgDudx_);

  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = buckets.begin();
       ib != buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double * tke = stk::mesh::field_data(tkeNp1, b);
    double * tvisc = stk::mesh::field_data(*tvisc_, b);
    double * avgProd = stk::mesh::field_data(*avgProduction_, b);
 
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get velocity field data
      const double * dudx = stk::mesh::field_data(*dudx_, b[k]);
      double * avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);

      // FIXME: Want to turn this off if restarting...
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          avgDudx[i*nDim + j] = dudx[i*nDim + j];
    
      // Initialize average production to mean production
      // FIXME: Want to turn this off if restarting...
      double *tij = new double[nDim*nDim];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          const double avgSij = 0.5*(avgDudx[i*nDim+j] + avgDudx[j*nDim+i]);
          tij[i*nDim + j] = 2.0 * tvisc[k] * avgSij;
        }
        tij[i*nDim + i] -= 2.0/3.0 * tke[k];
      }

      double *Pij = new double[nDim*nDim];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          Pij[i*nDim + j] = 0.0;
          for (int m = 0; m < nDim; ++m) {
             Pij[i*nDim + j] += avgDudx[i*nDim + m] * tij[j*nDim + m] + avgDudx[j*nDim + m] * tij[i*nDim + m];
          }
          Pij[i*nDim + j] *= 0.5;
        }
      }

      double instProd = 0.0;
      for (int i = 0; i < nDim; ++i)
        instProd += Pij[i*nDim + i];

      avgProd[k] = instProd;

      delete [] tij;
      delete [] Pij;
    }
  }

  compute_averages();
  initialize_mdot();
  compute_alpha();
  compute_resolution_adequacy_parameters();
  compute_avgMdot();
}

//--------------------------------------------------------------------------
//-------- post_converged_work ---------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::post_converged_work()
{
  tviscAlgDriver_->execute();

  compute_averages();

  compute_alpha();

  compute_resolution_adequacy_parameters();

  compute_avgMdot();
}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::initialize_mdot()
{
  //FIXME: Don't do this if it's a restart and average_mdot has been defined...

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if (realm_.realmUsesEdges_) {
    ScalarFieldType *massFlowRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate");

    // FIXME: Is this selector right?
    stk::mesh::Selector s_locally_owned_union = (meta_data.locally_owned_part()
      | meta_data.globally_shared_part())
      &stk::mesh::selectField(*avgMdot_);

    stk::mesh::BucketVector const& edge_buckets =
      realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );
    for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
          ib != edge_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      const stk::mesh::Bucket::size_type length   = b.size();

      const double * mdot = stk::mesh::field_data(*massFlowRate_, b);
      double * avgMdot    = stk::mesh::field_data(*avgMdot_, b);
  
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k )
        avgMdot[k] = mdot[k];
    }
  }
  else {
    GenericFieldType *massFlowRateScs_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");
  
    // FIXME: Hack since setting an element field to a constant using Aux doesn't seem to work...
    // required fields
  
    // define some common selectors
    stk::mesh::Selector s_all_elem
      = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
      &stk::mesh::selectField(*avgMdotScs_);
  
    stk::mesh::BucketVector const& elem_buckets =
      realm_.get_buckets( stk::topology::ELEMENT_RANK, s_all_elem );
    for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
          ib != elem_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      const stk::mesh::Bucket::size_type length = b.size();
  
      // extract master element
      MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
  
      // extract master element specifics
      const int numScsIp = meSCS->num_integration_points();
  
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
         double *avgMdotScs = stk::mesh::field_data(*avgMdotScs_, b, k);
         const double *mdotScs = stk::mesh::field_data(*massFlowRateScs_, b, k);
  
         for (int ip = 0; ip < numScsIp; ip++)
           avgMdotScs[ip] = mdotScs[ip];
      }
    }
  }
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
//-------- compute_avgMdot() -----------------------------------------------
//--------------------------------------------------------------------------
void
TAMSEquationSystem::compute_avgMdot()
{
  if ( NULL != avgMdotAlgDriver_)
    avgMdotAlgDriver_->execute();
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
