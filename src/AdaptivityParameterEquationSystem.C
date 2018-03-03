/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <AdaptivityParameterEquationSystem.h>
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
#include <ComputeResolutionAdequacyElemAlgorithm.h>
#include <ComputeMetricTensorElemAlgorithm.h>
#include <DirichletBC.h>
#include <EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm.h>
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
#include <kernel/AdaptivityParameterSFLESSrcElemKernel.h>

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
// AdaptivityParameterEquationSystem - manages alpha pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AdaptivityParameterEquationSystem::AdaptivityParameterEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "AdaptivityParameterEQS","adaptivity_parameter"),
    managePNG_(realm_.get_consistent_mass_matrix_png("adaptivity_parameter")),
    alpha_(NULL),
    dadx_(NULL),
    aTmp_(NULL),
    evisc_(NULL),
    resAdeq_(NULL),
    metric_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(realm_, "adaptivity_parameter", "dadx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_)),
    resolutionAdequacyAlgDriver_(new AlgorithmDriver(realm_)),
    metricTensorAlgDriver_(new AlgorithmDriver(realm_)),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    projectedNodalGradEqs_(NULL),
    isInit_(true)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name("adaptivity_parameter");
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, EQ_ADAPT_PARAM);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("adaptivity_parameter");
  NaluEnv::self().naluOutputP0() << "Edge projected nodal gradient for adaptivity_parameter: " << edgeNodalGradient_ <<std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  if ( turbulenceModel_ != SFLES ) {
    throw std::runtime_error("User has requested AdaptParamEqs, however, turbulence model has not been set to alpha_SFLES, the only one supported by this equation system currently.");
  }

  // create projected nodal gradient equation system
  if ( managePNG_ ) {
    manage_projected_nodal_gradient(eqSystems);
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AdaptivityParameterEquationSystem::~AdaptivityParameterEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  if (NULL != resolutionAdequacyAlgDriver_)
    delete resolutionAdequacyAlgDriver_;
  if (NULL != metricTensorAlgDriver_)
    delete metricTensorAlgDriver_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  alpha_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "adaptivity_parameter", numStates));
  stk::mesh::put_field(*alpha_, *part);
  realm_.augment_restart_variable_list("adaptivity_parameter");

  dadx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dadx"));
  stk::mesh::put_field(*dadx_, *part, nDim);

  // delta solution for linear solver; share delta since this is a split system
  aTmp_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "aTmp"));
  stk::mesh::put_field(*aTmp_, *part);

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "effective_viscosity_alpha"));
  stk::mesh::put_field(*evisc_, *part);

  resAdeq_ = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "resolution_adequacy"));
  stk::mesh::put_field(*resAdeq_, *part);

  metric_ = &(meta_data.declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "metric_tensor"));
  stk::mesh::put_field(*metric_, *part, nDim*nDim);

  // make sure all states are properly populated (restart can handle this)
  if ( numStates > 2 && (!realm_.restarted_simulation() || realm_.support_inconsistent_restart()) ) {
    ScalarFieldType &alphaN = alpha_->field_of_state(stk::mesh::StateN);
    ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm *theCopyAlg
      = new CopyFieldAlgorithm(realm_, part,
                               &alphaNp1, &alphaN,
                               0, 1,
                               stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dadx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = NULL;
      if ( edgeNodalGradient_ && realm_.realmUsesEdges_ ) {
        theAlg = new AssembleNodalGradEdgeAlgorithm(realm_, part, &alphaNp1, &dadxNone);
      }
      else {
        theAlg = new AssembleNodalGradElemAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
      }
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; interior contribution (advection + diffusion)
  if ( !realm_.solutionOptions_->useConsolidatedSolverAlg_ ) {

    throw std::runtime_error("AdaptivityParameterEquationSystem::Error: This is not set up to use non kernel-based source terms.");
   /* 
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
      SolverAlgorithm *theAlg = NULL;
      if ( realm_.realmUsesEdges_ ) {
        theAlg = new AssembleScalarEdgeSolverAlgorithm(realm_, part, this, alpha_, dadx_, evisc_);
      }
      else {
        theAlg = new AssembleScalarElemSolverAlgorithm(realm_, part, this, alpha_, dadx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
      
      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->elemSrcTermsMap_.find("adaptivity_parameter");
      if ( isrc != realm_.solutionOptions_->elemSrcTermsMap_.end() ) {
        
        if ( realm_.realmUsesEdges_ )
          throw std::runtime_error("AdaptivityParameterElemSrcTerms::Error can not use element source terms for an edge-based scheme");
        
        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if (sourceName == "adaptivity_parameter_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, alpha_, false);
          }
          else if (sourceName == "lumped_adaptivity_parameter_time_derivative" ) {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, alpha_, true);
          }
          else {
            throw std::runtime_error("AdaptivityParameterElemSrcTerms::Error Source term is not supported: " + sourceName);
          }     
          NaluEnv::self().naluOutputP0() << "AdaptivityParameterElemSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg); 
        }
      }
    }
    else {
      itsi->second->partVec_.push_back(part);
    }
   

    // time term; (Pk-Dk); both nodally lumped
    const AlgorithmType algMass = MASS;
    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {"adaptivity_parameter_time_derivative",
                                              "lumped_adaptivity_parameter_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);
    std::map<AlgorithmType, SolverAlgorithm *>::iterator itsm =
      solverAlgDriver_->solverAlgMap_.find(algMass);
    if ( itsm == solverAlgDriver_->solverAlgMap_.end() ) {
      // create the solver alg
      AssembleNodeSolverAlgorithm *theAlg
        = new AssembleNodeSolverAlgorithm(realm_, part, this);
      solverAlgDriver_->solverAlgMap_[algMass] = theAlg;
      
      // now create the supplemental alg for mass term
      if ( !elementMassAlg ) {
        if ( realm_.number_of_states() == 2 ) {
          ScalarMassBackwardEulerNodeSuppAlg *theMass
            = new ScalarMassBackwardEulerNodeSuppAlg(realm_, alpha_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
        else {
          ScalarMassBDF2NodeSuppAlg *theMass
            = new ScalarMassBDF2NodeSuppAlg(realm_, alpha_);
          theAlg->supplementalAlg_.push_back(theMass);
        }
      }
      
      // now create the src alg for alpha source
      SupplementalAlgorithm *theSrc = NULL;
      switch(turbulenceModel_) {
      case SFLES:
        {
          //FIXME: Do I want a nodal source SuppAlg... I think we are only
          //       supporting the kernel implementation?
          //theSrc = new AdaptivityParameterSFLESNodeSourceSuppAlg(realm_);
	  throw std::runtime_error("Error in AdpatParam: only homogenous kernel implemntation for SFLES supported");
        }
        break;
      default:
        throw std::runtime_error("Unsupported turbulence model in AdaptParam: only SFLES supported");
      }
      theAlg->supplementalAlg_.push_back(theSrc);
     
      // TODO: Do I actually have any nodal source terms here??? FIXME 
      // Add nodal src term supp alg...; limited number supported
      std::map<std::string, std::vector<std::string> >::iterator isrc 
        = realm_.solutionOptions_->srcTermsMap_.find("adaptivity_parameter");
      if ( isrc != realm_.solutionOptions_->srcTermsMap_.end() ) {
        std::vector<std::string> mapNameVec = isrc->second;   
        for (size_t k = 0; k < mapNameVec.size(); ++k ) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm *suppAlg = NULL;
          if ( sourceName == "PLACEHOLDER" ) {
            suppAlg = new ScalarGclNodeSuppAlg(alpha_,realm_);
          }
          else {
            throw std::runtime_error("AdaptivityParameterNodalSrcTerms::Error Source term is not supported: " + sourceName);
          }
          NaluEnv::self().naluOutputP0() << "AdaptivityParameterNodalSrcTerms::added() " << sourceName << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    }
    else {
      itsm->second->partVec_.push_back(part);
    }
   */
  }
  else {
    // Homogeneous kernel implementation
    if ( realm_.realmUsesEdges_ )
      throw std::runtime_error("AdaptivityParameterEquationSystem::Error can not use element source terms for an edge-based scheme");
    
    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;
    
    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;
    
    std::tie(solverAlg, solverAlgWasBuilt) = build_or_add_part_to_solver_alg
      (*this, *part, solverAlgMap);
    
    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels, "adaptivity_parameter_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, alpha_, dataPreReqs, false);
      
      build_topo_kernel_if_requested<ScalarMassElemKernel>
        (partTopo, *this, activeKernels, "lumped_adaptivity_parameter_time_derivative",
         realm_.bulk_data(), *realm_.solutionOptions_, alpha_, dataPreReqs, true);
      
      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, alpha_, evisc_, dataPreReqs);
      
      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>
        (partTopo, *this, activeKernels, "upw_advection_diffusion",
         realm_.bulk_data(), *realm_.solutionOptions_, this, alpha_, dadx_, evisc_, dataPreReqs);

      build_topo_kernel_if_requested<AdaptivityParameterSFLESSrcElemKernel>
        (partTopo, *this, activeKernels, "alpha_sfles",
         realm_.bulk_data(), *realm_.solutionOptions_, dataPreReqs);

      report_invalid_supp_alg_names();
      report_built_supp_alg_names();
    }
  }

  // effective viscosity alg FIXME: Right now this is just needed to set evisc_ to 0.0, there must
  // be an easier way to do this? 
  std::map<AlgorithmType, Algorithm *>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if ( itev == diffFluxCoeffAlgDriver_->algMap_.end() ) {
    Algorithm *effDiffAlg = NULL;
    effDiffAlg = new EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm(realm_, part, evisc_);
    diffFluxCoeffAlgDriver_->algMap_[algType] = effDiffAlg;
  }
  else {
    itev->second->partVec_.push_back(part);
  }

  // resolution adequacy algorithm
  if ( NULL == resolutionAdequacyAlgDriver_ )
    resolutionAdequacyAlgDriver_ = new AlgorithmDriver(realm_);
  
  std::map<AlgorithmType, Algorithm *>::iterator it = 
    resolutionAdequacyAlgDriver_->algMap_.find(algType);

  if (it == resolutionAdequacyAlgDriver_->algMap_.end() ) {
    ComputeResolutionAdequacyElemAlgorithm *resAdeqAlg =
      new ComputeResolutionAdequacyElemAlgorithm(realm_, part);
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
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; alpha_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "alpha_bc"));
  stk::mesh::put_field(*theBcField, *part);

  // extract the value for user specified alpha and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  AdaptParam alpha = userData.alpha_;
  std::vector<double> userSpec(1);
  userSpec[0] = alpha.adaptParam_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);

  // how to populate the field?
  if ( userData.externalData_ ) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  }
  else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // copy alpha_bc to adaptivity_parameter np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &alphaNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dadx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd
    = solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg
      = new DirichletBC(realm_, this, part, &alphaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const OpenBoundaryConditionData &openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; alpha_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "open_alpha_bc"));
  stk::mesh::put_field(*theBcField, *part);

  // extract the value for user specified alpha and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  AdaptParam alpha = userData.alpha_;
  std::vector<double> userSpec(1);
  userSpec[0] = alpha.adaptParam_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dadx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    } 
  }

  // solver open; lhs
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi = solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    SolverAlgorithm *theAlg = NULL;
    if ( realm_.realmUsesEdges_ ) {
      theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(realm_, part, this, alpha_, theBcField, &dadxNone, evisc_);
    }
    else {
      theAlg = new AssembleScalarElemOpenSolverAlgorithm(realm_, part, this, alpha_, theBcField, &dadxNone, evisc_);
    }
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }

}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  // register boundary data; alpha_bc
  ScalarFieldType *theBcField = &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "alpha_bc"));
  stk::mesh::put_field(*theBcField, *part);

  // extract the value for user specified alpha and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string alphaName = "adaptivity_parameter";
  const bool alphaSpecified = bc_data_specified(userData, alphaName);
  bool wallFunctionApproach = userData.wallFunctionApproach_;
  if ( alphaSpecified && wallFunctionApproach ) {
    NaluEnv::self().naluOutputP0() << "Both wall function and alpha specified; will go with dirichlet" << std::endl;
    wallFunctionApproach = false;
  }

  // FIXME: Generalize for constant vs function

  // extract data
  std::vector<double> userSpec(1);
  AdaptParam alpha = userData.alpha_;
  userSpec[0] = alpha.adaptParam_;

  // new it
  ConstantAuxFunction *theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm *auxAlg
    = new AuxFunctionAlgorithm(realm_, part,
                               theBcField, theAuxFunc,
                               stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // copy alpha_bc to alpha np1...
  CopyFieldAlgorithm *theCopyAlg
    = new CopyFieldAlgorithm(realm_, part,
                             theBcField, &alphaNp1,
                             0, 1,
                             stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itd =
      solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if ( itd == solverAlgDriver_->solverDirichAlgMap_.end() ) {
    DirichletBC *theAlg =
        new DirichletBC(realm_, this, part, &alphaNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  }
  else {
    itd->second->partVec_.push_back(part);
  }

  // non-solver; dadx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &symmetryBCData)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dadx; allow for element-based shifted
  if ( !managePNG_ ) {
    std::map<AlgorithmType, Algorithm *>::iterator it
      = assembleNodalGradAlgDriver_->algMap_.find(algType);
    if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
      Algorithm *theAlg 
        = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    }
    else {
      it->second->partVec_.push_back(part);
    }
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType &dadxNone = dadx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dadx; DG algorithm decides on locations for integration points
  if ( !managePNG_ ) {
    if ( edgeNodalGradient_ ) {    
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        Algorithm *theAlg 
          = new AssembleNodalGradBoundaryAlgorithm(realm_, part, &alphaNp1, &dadxNone, edgeNodalGradient_);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
    else {
      // proceed with DG
      std::map<AlgorithmType, Algorithm *>::iterator it
        = assembleNodalGradAlgDriver_->algMap_.find(algType);
      if ( it == assembleNodalGradAlgDriver_->algMap_.end() ) {
        AssembleNodalGradNonConformalAlgorithm *theAlg 
          = new AssembleNodalGradNonConformalAlgorithm(realm_, part, &alphaNp1, &dadxNone);
        assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
      }
      else {
        it->second->partVec_.push_back(part);
      }
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm *>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( itsi == solverAlgDriver_->solverAlgMap_.end() ) {
    AssembleScalarNonConformalSolverAlgorithm *theAlg
      = new AssembleScalarNonConformalSolverAlgorithm(realm_, part, this, alpha_, evisc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    itsi->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(alpha_);

  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(alpha_,1,1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::reinitialize_linear_system()
{

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
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::solve_and_update()
{

  // compute dk/dx
  if ( isInit_ ) {
    compute_projected_nodal_gradient();
    compute_metric_tensor();
    isInit_ = false;
  }

  compute_effective_diff_flux_coeff();

  compute_resolution_adequacy_parameters();

  // TODO: Add recalculation of metric tensor if mesh changes

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << userSuppliedName_ << std::endl;

    // alpha assemble, load_complete and solve
    assemble_and_solve(aTmp_);

    // update
    double timeA = NaluEnv::self().nalu_time();
    update_and_clip();
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);

    // projected nodal gradient
    compute_projected_nodal_gradient();
  }

}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::initial_work()
{
  // TODO: Do we need clipping... I believe we are not clipping alpha FIXME
  // do not let the user specify a negative field
  //const double clipValue = 1.0e-16;

  //stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  //stk::mesh::Selector s_all_nodes
  //  = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
  //  &stk::mesh::selectField(*alpha_);

  //stk::mesh::BucketVector const& node_buckets =
  //  realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  //for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
  //      ib != node_buckets.end() ; ++ib ) {
    //stk::mesh::Bucket & b = **ib ;
    //const stk::mesh::Bucket::size_type length   = b.size();

    //double *alpha = stk::mesh::field_data(*alpha_, b);

    //for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      //const double alphaNp1 = alpha[k];
      //FIXME: No clipping for now...
      //if ( alphaNp1 < 0.0 ) {
      //  alpha[k] = clipValue;
      //}
  //  }
  //}
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = NaluEnv::self().nalu_time();
  diffFluxCoeffAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() - timeA);
}

//--------------------------------------------------------------------------
//-------- compute_resolution_adequacy_parameters() ------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::compute_resolution_adequacy_parameters()
{
  if ( NULL != resolutionAdequacyAlgDriver_)
    resolutionAdequacyAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- compute_metric_tensor() -----------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::compute_metric_tensor()
{
  if ( NULL != metricTensorAlgDriver_)
    metricTensorAlgDriver_->execute();
}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::update_and_clip()
{
  // FIXME: No clipping for alpha as of now...
  //const double clipValue = 1.0e-16;
  //size_t numClip = 0;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*alpha_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *alpha = stk::mesh::field_data(*alpha_, b);
    double *aTmp = stk::mesh::field_data(*aTmp_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double alphaNp1 = alpha[k] + aTmp[k];
      // FIXME: No clipping for alph as of now...
      //if ( alphaNp1 < 0.0 ) {
      //  alpha[k] = clipValue;
      //  numClip++;
      //}
      //else {
        alpha[k] = alphaNp1;
      //}
    }
  }

  // FIXME: No clipping for alpha as of now...
  // parallel assemble clipped value
  //if (realm_.debug()) {
  //  size_t g_numClip = 0;
  //  stk::ParallelMachine comm =  NaluEnv::self().parallel_comm();
  //  stk::all_reduce_sum(comm, &numClip, &g_numClip, 1);

  //  if ( g_numClip > 0 ) {
  //    NaluEnv::self().naluOutputP0() << "alpha clipped " << g_numClip << " times " << std::endl;
  //  }

  //}
}

void
AdaptivityParameterEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType &alphaN = alpha_->field_of_state(stk::mesh::StateN);
  ScalarFieldType &alphaNp1 = alpha_->field_of_state(stk::mesh::StateNP1);
  field_copy(realm_.meta_data(), realm_.bulk_data(), alphaN, alphaNp1, realm_.get_activate_aura());
}

//--------------------------------------------------------------------------
//-------- manage_projected_nodal_gradient ---------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::manage_projected_nodal_gradient(
  EquationSystems& eqSystems)
{
  if ( NULL == projectedNodalGradEqs_ ) {
    projectedNodalGradEqs_ 
      = new ProjectedNodalGradientEquationSystem(eqSystems, EQ_PNG_ALPHA, "dadx", "qTmp", "adaptivity_parameter", "PNGradAlphaEQS");
  }
  // fill the map for expected boundary condition names; can be more complex...
  projectedNodalGradEqs_->set_data_map(INFLOW_BC, "adaptivity_parameter");
  projectedNodalGradEqs_->set_data_map(WALL_BC, "adaptivity_parameter"); // wall function...
  projectedNodalGradEqs_->set_data_map(OPEN_BC, "adaptivity_parameter");
  projectedNodalGradEqs_->set_data_map(SYMMETRY_BC, "adaptivity_parameter");
}

//--------------------------------------------------------------------------
//-------- compute_projected_nodal_gradient() ---------------------------------------
//--------------------------------------------------------------------------
void
AdaptivityParameterEquationSystem::compute_projected_nodal_gradient()
{
  if ( !managePNG_ ) {
    const double timeA = -NaluEnv::self().nalu_time();
    assembleNodalGradAlgDriver_->execute();
    timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
  }
  else {
    projectedNodalGradEqs_->solve_and_update_external();
  }
}

} // namespace nalu
} // namespace Sierra
