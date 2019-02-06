/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <TotalDissipationRateEquationSystem.h>
#include <AlgorithmDriver.h>
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
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
#include <DirichletBC.h>
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <Realms.h>
#include <ScalarMassElemSuppAlgDep.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TotalDissipationRateChienKEpsNodeSourceSuppAlg.h>
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
#include <kernel/TotalDissipationRateChienKEpsSrcElemKernel.h>

// edge kernels
#include <edge_kernels/ScalarEdgeSolverAlg.h>

// node kernels
#include <node_kernels/NodeKernelUtils.h>
#include <node_kernels/ScalarMassBDFNodeKernel.h>

// UT Austin Hybird TAMS kernel
#include <kernel/TotalDissipationRateTAMSKEpsSrcElemKernel.h>
#include <node_kernels/TDRTAMSKEpsNodeKernel.h>

// nso
#include <nso/ScalarNSOElemKernel.h>
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

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// TotalDissipationRateEquationSystem - manages tdr pde system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TotalDissipationRateEquationSystem::TotalDissipationRateEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "TotDissRateEQS", "total_dissipation_rate"),
    managePNG_(realm_.get_consistent_mass_matrix_png("total_dissipation_rate")),
    tdr_(NULL),
    dedx_(NULL),
    eTmp_(NULL),
    visc_(NULL),
    tvisc_(NULL),
    evisc_(NULL),
    assembleNodalGradAlgDriver_(new AssembleNodalGradAlgorithmDriver(
      realm_, "total_dissipation_rate", "dedx")),
    diffFluxCoeffAlgDriver_(new AlgorithmDriver(realm_))
{
  // extract solver name and solver object
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("total_dissipation_rate");
  LinearSolver* solver =
    realm_.root()->linearSolvers_->create_solver(solverName, EQ_TOT_DISS_RATE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // determine nodal gradient form
  set_nodal_gradient("total_dissipation_rate");
  NaluEnv::self().naluOutputP0()
    << "Edge projected nodal gradient for total_dissipation_rate: "
    << edgeNodalGradient_ << std::endl;

  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create projected nodal gradient equation system
  if (managePNG_)
    throw std::runtime_error(
      "TotalDissipationRateEquationSystem::Error managePNG is not complete");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TotalDissipationRateEquationSystem::~TotalDissipationRateEquationSystem()
{
  delete assembleNodalGradAlgDriver_;
  delete diffFluxCoeffAlgDriver_;
  std::vector<Algorithm*>::iterator ii;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const int numStates = realm_.number_of_states();

  // register dof; set it as a restart variable
  tdr_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "total_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*tdr_, *part, nullptr);
  realm_.augment_restart_variable_list("total_dissipation_rate");

  dedx_ = &(
    meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dedx"));
  stk::mesh::put_field_on_mesh(*dedx_, *part, nDim, nullptr);

  // delta solution for linear solver; share delta since this is a split system
  eTmp_ = &(
    meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "eTmp"));
  stk::mesh::put_field_on_mesh(*eTmp_, *part, nullptr);

  visc_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "viscosity"));
  stk::mesh::put_field_on_mesh(*visc_, *part, nullptr);

  tvisc_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity"));
  stk::mesh::put_field_on_mesh(*tvisc_, *part, nullptr);

  evisc_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "effective_viscosity_tdr"));
  stk::mesh::put_field_on_mesh(*evisc_, *part, nullptr);

  // make sure all states are properly populated (restart can handle this)
  if (
    numStates > 2 &&
    (!realm_.restarted_simulation() || realm_.support_inconsistent_restart())) {
    ScalarFieldType& tdrN = tdr_->field_of_state(stk::mesh::StateN);
    ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);

    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part, &tdrNp1, &tdrN, 0, 1, stk::topology::NODE_RANK);
    copyStateAlg_.push_back(theCopyAlg);
  }
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_interior_algorithm(
  stk::mesh::Part* part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  // non-solver, dedx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm*>::iterator it =
    assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = NULL;
    if (edgeNodalGradient_ && realm_.realmUsesEdges_) {
      theAlg =
        new AssembleNodalGradEdgeAlgorithm(realm_, part, &tdrNp1, &dedxNone);
    } else {
      theAlg = new AssembleNodalGradElemAlgorithm(
        realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
    }
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }

  // solver; interior contribution (advection + diffusion)
  if (!realm_.solutionOptions_->useConsolidatedSolverAlg_) {

    std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
      solverAlgDriver_->solverAlgMap_.find(algType);
    if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
      SolverAlgorithm* theAlg = NULL;
      if (realm_.realmUsesEdges_) {
        const bool useAvgMdot =
          (realm_.solutionOptions_->turbulenceModel_ == TAMS_KEPS) ? true
                                                                   : false;
        theAlg = new ScalarEdgeSolverAlg(
          realm_, part, this, tdr_, dedx_, evisc_, useAvgMdot);
      } else {
        theAlg = new AssembleScalarElemSolverAlgorithm(
          realm_, part, this, tdr_, dedx_, evisc_);
      }
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;

      // look for fully integrated source terms
      std::map<std::string, std::vector<std::string>>::iterator isrc =
        realm_.solutionOptions_->elemSrcTermsMap_.find(
          "total_dissipation_rate");
      if (isrc != realm_.solutionOptions_->elemSrcTermsMap_.end()) {

        if (realm_.realmUsesEdges_)
          throw std::runtime_error(
            "TotalDissipationElemSrcTerms::Error can not use element source "
            "terms for an edge-based scheme");

        std::vector<std::string> mapNameVec = isrc->second;
        for (size_t k = 0; k < mapNameVec.size(); ++k) {
          std::string sourceName = mapNameVec[k];
          SupplementalAlgorithm* suppAlg = NULL;
          if (sourceName == "NSO_2ND_ALT") {
            suppAlg = new ScalarNSOElemSuppAlgDep(
              realm_, tdr_, dedx_, evisc_, 0.0, 1.0);
          } else if (sourceName == "NSO_4TH_ALT") {
            suppAlg = new ScalarNSOElemSuppAlgDep(
              realm_, tdr_, dedx_, evisc_, 1.0, 1.0);
          } else if (sourceName == "NSO_2ND_KE") {
            const double turbSc = realm_.get_turb_schmidt(tdr_->name());
            suppAlg =
              new ScalarNSOKeElemSuppAlg(realm_, tdr_, dedx_, turbSc, 0.0);
          } else if (sourceName == "NSO_4TH_KE") {
            const double turbSc = realm_.get_turb_schmidt(tdr_->name());
            suppAlg =
              new ScalarNSOKeElemSuppAlg(realm_, tdr_, dedx_, turbSc, 1.0);
          } else if (sourceName == "total_dissipation_rate_time_derivative") {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, tdr_, false);
          } else if (
            sourceName == "lumped_total_dissipation_rate_time_derivative") {
            suppAlg = new ScalarMassElemSuppAlgDep(realm_, tdr_, true);
          } else {
            throw std::runtime_error(
              "TotalDissipationElemSrcTerms::Error Source term is not "
              "supported: " +
              sourceName);
          }
          NaluEnv::self().naluOutputP0()
            << "TotalDissipationElemSrcTerms::added() " << sourceName
            << std::endl;
          theAlg->supplementalAlg_.push_back(suppAlg);
        }
      }
    } else {
      itsi->second->partVec_.push_back(part);
    }

    // time term; src; both nodally lumped
    const AlgorithmType algMass = SRC;
    // Check if the user has requested CMM or LMM algorithms; if so, do not
    // include Nodal Mass algorithms
    std::vector<std::string> checkAlgNames = {
      "total_dissipation_rate_time_derivative",
      "lumped_total_dissipation_rate_time_derivative"};
    bool elementMassAlg = supp_alg_is_requested(checkAlgNames);

    auto& solverAlgMap = solverAlgDriver_->solverAlgMap_;
    process_ngp_node_kernels(
      solverAlgMap, realm_, part, this,
      [&](AssembleNGPNodeSolverAlgorithm& nodeAlg) {
        if (!elementMassAlg)
          nodeAlg.add_kernel<ScalarMassBDFNodeKernel>(realm_.bulk_data(), tdr_);

        if (realm_.solutionOptions_->turbulenceModel_ == TAMS_KEPS)
          nodeAlg.add_kernel<TDRTAMSKEpsNodeKernel>(
            realm_.bulk_data(), *realm_.solutionOptions_);
      },
      [&](
        AssembleNGPNodeSolverAlgorithm& /* nodeAlg */,
        std::string& /* srcName */) {
        // No source terms available yet
      });
  } else {
    // Homogeneous kernel implementation
    if (realm_.realmUsesEdges_)
      throw std::runtime_error(
        "TotalDissipationRateEquationSystem::Error can not use element source "
        "terms for an edge-based scheme");

    stk::topology partTopo = part->topology();
    auto& solverAlgMap = solverAlgDriver_->solverAlgorithmMap_;

    AssembleElemSolverAlgorithm* solverAlg = nullptr;
    bool solverAlgWasBuilt = false;

    std::tie(solverAlg, solverAlgWasBuilt) =
      build_or_add_part_to_solver_alg(*this, *part, solverAlgMap);

    ElemDataRequests& dataPreReqs = solverAlg->dataNeededByKernels_;
    auto& activeKernels = solverAlg->activeKernels_;

    if (solverAlgWasBuilt) {
      build_topo_kernel_if_requested<ScalarMassElemKernel>(
        partTopo, *this, activeKernels,
        "total_dissipation_rate_time_derivative", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dataPreReqs, false);

      build_topo_kernel_if_requested<ScalarMassElemKernel>(
        partTopo, *this, activeKernels,
        "lumped_total_dissipation_rate_time_derivative", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>(
        partTopo, *this, activeKernels, "advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, tdr_, evisc_,
        dataPreReqs);

      build_topo_kernel_if_requested<ScalarAdvDiffElemKernel>(
        partTopo, *this, activeKernels, "TAMS_advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, tdr_, evisc_, dataPreReqs,
        true);

      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>(
        partTopo, *this, activeKernels, "upw_advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, this, tdr_, dedx_, evisc_,
        dataPreReqs);

      build_topo_kernel_if_requested<ScalarUpwAdvDiffElemKernel>(
        partTopo, *this, activeKernels, "TAMS_upw_advection_diffusion",
        realm_.bulk_data(), *realm_.solutionOptions_, this, tdr_, dedx_, evisc_,
        dataPreReqs, true);

      build_topo_kernel_if_requested<
        TotalDissipationRateChienKEpsSrcElemKernel>(
        partTopo, *this, activeKernels, "keps", realm_.bulk_data(),
        *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<
        TotalDissipationRateChienKEpsSrcElemKernel>(
        partTopo, *this, activeKernels, "lumped_keps", realm_.bulk_data(),
        *realm_.solutionOptions_, dataPreReqs, true);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>(
        partTopo, *this, activeKernels, "NSO_2ND", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dedx_, evisc_, 0.0, 0.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>(
        partTopo, *this, activeKernels, "NSO_2ND_ALT", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dedx_, evisc_, 0.0, 1.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>(
        partTopo, *this, activeKernels, "NSO_4TH", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dedx_, evisc_, 1.0, 0.0, dataPreReqs);

      build_topo_kernel_if_requested<ScalarNSOElemKernel>(
        partTopo, *this, activeKernels, "NSO_4TH_ALT", realm_.bulk_data(),
        *realm_.solutionOptions_, tdr_, dedx_, evisc_, 1.0, 1.0, dataPreReqs);

      // UT Austin Hybrid TAMS model implementations for TDR source terms
      build_topo_kernel_if_requested<TotalDissipationRateTAMSKEpsSrcElemKernel>(
        partTopo, *this, activeKernels, "tams_keps", realm_.bulk_data(),
        *realm_.solutionOptions_, dataPreReqs, false);

      build_topo_kernel_if_requested<TotalDissipationRateTAMSKEpsSrcElemKernel>(
        partTopo, *this, activeKernels, "lumped_tams_keps", realm_.bulk_data(),
        *realm_.solutionOptions_, dataPreReqs, true);

      report_invalid_supp_alg_names();
      report_built_supp_alg_names();
    }
  }

  // effective diffusive flux coefficient alg for SST
  std::map<AlgorithmType, Algorithm*>::iterator itev =
    diffFluxCoeffAlgDriver_->algMap_.find(algType);
  if (itev == diffFluxCoeffAlgDriver_->algMap_.end()) {
    const double sigmaEps = realm_.get_turb_model_constant(TM_sigmaEps);
    EffectiveDiffFluxCoeffAlgorithm* effDiffAlg =
      new EffectiveDiffFluxCoeffAlgorithm(
        realm_, part, visc_, tvisc_, evisc_, 1.0, sigmaEps);
    diffFluxCoeffAlgDriver_->algMap_[algType] = effDiffAlg;
  } else {
    itev->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_inflow_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const InflowBoundaryConditionData& inflowBCData)
{

  // algorithm type
  const AlgorithmType algType = INFLOW;

  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tdr_bc
  ScalarFieldType* theBcField = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "tdr_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tdr and save off the AuxFunction
  InflowUserData userData = inflowBCData.userData_;
  TotDissRate tdr = userData.tdr_;
  std::vector<double> userSpec(1);
  userSpec[0] = tdr.totDissRate_;

  // new it
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);

  // how to populate the field?
  if (userData.externalData_) {
    // xfer will handle population; only need to populate the initial value
    realm_.initCondAlg_.push_back(auxAlg);
  } else {
    // put it on bcData
    bcDataAlg_.push_back(auxAlg);
  }

  // copy tdr_bc to total_dissipation_rate np1...
  CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
    realm_, part, theBcField, &tdrNp1, 0, 1, stk::topology::NODE_RANK);
  bcDataMapAlg_.push_back(theCopyAlg);

  // non-solver; dedx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm*>::iterator it =
    assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = new AssembleNodalGradBoundaryAlgorithm(
      realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &tdrNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_open_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const OpenBoundaryConditionData& openBCData)
{

  // algorithm type
  const AlgorithmType algType = OPEN;

  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tdr_bc
  ScalarFieldType* theBcField = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "open_tdr_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  OpenUserData userData = openBCData.userData_;
  TotDissRate tdr = userData.tdr_;
  std::vector<double> userSpec(1);
  userSpec[0] = tdr.totDissRate_;

  // new it
  ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

  // bc data alg
  AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
    realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
  bcDataAlg_.push_back(auxAlg);

  // non-solver; dedx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm*>::iterator it =
    assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = new AssembleNodalGradBoundaryAlgorithm(
      realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }

  // solver open; lhs
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    SolverAlgorithm* theAlg = NULL;
    if (realm_.realmUsesEdges_) {
      theAlg = new AssembleScalarEdgeOpenSolverAlgorithm(
        realm_, part, this, tdr_, theBcField, &dedxNone, evisc_);
    } else {
      theAlg = new AssembleScalarElemOpenSolverAlgorithm(
        realm_, part, this, tdr_, theBcField, &dedxNone, evisc_);
    }
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    itsi->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_wall_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const WallBoundaryConditionData& wallBCData)
{

  // algorithm type
  const AlgorithmType algType = WALL;

  // np1
  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // register boundary data; tke_bc
  ScalarFieldType* theBcField = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "tdr_bc"));
  stk::mesh::put_field_on_mesh(*theBcField, *part, nullptr);

  // extract the value for user specified tke and save off the AuxFunction
  WallUserData userData = wallBCData.userData_;
  std::string tdrName = "total_dissipation_rate";
  const bool tdrSpecified = bc_data_specified(userData, tdrName);
  bool wallFunctionApproach = userData.wallFunctionApproach_;
  if (tdrSpecified && wallFunctionApproach) {
    NaluEnv::self().naluOutputP0()
      << "Both wall function and tdr specified; will go with dirichlet"
      << std::endl;
    wallFunctionApproach = false;
  }

  if (wallFunctionApproach) {
    throw std::runtime_error(
      "Total dissipation rate is not set up with wall functions yet.");
  } else if (tdrSpecified) {

    // FIXME: Generalize for constant vs function

    // extract data
    std::vector<double> userSpec(1);
    TotDissRate tdr = userData.tdr_;
    userSpec[0] = tdr.totDissRate_;

    // new it
    ConstantAuxFunction* theAuxFunc = new ConstantAuxFunction(0, 1, userSpec);

    // bc data alg
    AuxFunctionAlgorithm* auxAlg = new AuxFunctionAlgorithm(
      realm_, part, theBcField, theAuxFunc, stk::topology::NODE_RANK);
    bcDataAlg_.push_back(auxAlg);

    // copy tke_bc to tke np1...
    CopyFieldAlgorithm* theCopyAlg = new CopyFieldAlgorithm(
      realm_, part, theBcField, &tdrNp1, 0, 1, stk::topology::NODE_RANK);
    bcDataMapAlg_.push_back(theCopyAlg);

  } else {
    throw std::runtime_error("TDR active with wall bc, however, no value of "
                             "total_dissipation_rate specified");
  }

  // Dirichlet bc
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itd =
    solverAlgDriver_->solverDirichAlgMap_.find(algType);
  if (itd == solverAlgDriver_->solverDirichAlgMap_.end()) {
    DirichletBC* theAlg =
      new DirichletBC(realm_, this, part, &tdrNp1, theBcField, 0, 1);
    solverAlgDriver_->solverDirichAlgMap_[algType] = theAlg;
  } else {
    itd->second->partVec_.push_back(part);
  }

  // non-solver; dedx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm*>::iterator it =
    assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = new AssembleNodalGradBoundaryAlgorithm(
      realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_symmetry_bc(
  stk::mesh::Part* part,
  const stk::topology& /*theTopo*/,
  const SymmetryBoundaryConditionData& /* symmetryBCData */)
{

  // algorithm type
  const AlgorithmType algType = SYMMETRY;

  // np1
  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  // non-solver; dwdx; allow for element-based shifted
  std::map<AlgorithmType, Algorithm*>::iterator it =
    assembleNodalGradAlgDriver_->algMap_.find(algType);
  if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = new AssembleNodalGradBoundaryAlgorithm(
      realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
    assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_non_conformal_bc(
  stk::mesh::Part* part, const stk::topology& /*theTopo*/)
{

  const AlgorithmType algType = NON_CONFORMAL;

  // np1
  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType& dedxNone = dedx_->field_of_state(stk::mesh::StateNone);

  // non-solver; contribution to dwdx; DG algorithm decides on locations for
  // integration points
  if (edgeNodalGradient_) {
    std::map<AlgorithmType, Algorithm*>::iterator it =
      assembleNodalGradAlgDriver_->algMap_.find(algType);
    if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
      Algorithm* theAlg = new AssembleNodalGradBoundaryAlgorithm(
        realm_, part, &tdrNp1, &dedxNone, edgeNodalGradient_);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  } else {
    // proceed with DG
    std::map<AlgorithmType, Algorithm*>::iterator it =
      assembleNodalGradAlgDriver_->algMap_.find(algType);
    if (it == assembleNodalGradAlgDriver_->algMap_.end()) {
      AssembleNodalGradNonConformalAlgorithm* theAlg =
        new AssembleNodalGradNonConformalAlgorithm(
          realm_, part, &tdrNp1, &dedxNone);
      assembleNodalGradAlgDriver_->algMap_[algType] = theAlg;
    } else {
      it->second->partVec_.push_back(part);
    }
  }

  // solver; lhs; same for edge and element-based scheme
  std::map<AlgorithmType, SolverAlgorithm*>::iterator itsi =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if (itsi == solverAlgDriver_->solverAlgMap_.end()) {
    AssembleScalarNonConformalSolverAlgorithm* theAlg =
      new AssembleScalarNonConformalSolverAlgorithm(
        realm_, part, this, tdr_, evisc_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  } else {
    itsi->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(tdr_);

  UpdateOversetFringeAlgorithmDriver* theAlg =
    new UpdateOversetFringeAlgorithmDriver(realm_);
  // Perform fringe updates before all equation system solves
  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(tdr_, 1, 1)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::initialize()
{
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::reinitialize_linear_system()
{

  // delete linsys
  delete linsys_;

  // delete old solver
  const EquationType theEqID = EQ_TOT_DISS_RATE;
  LinearSolver* theSolver = NULL;
  std::map<EquationType, LinearSolver*>::const_iterator iter =
    realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver
  std::string solverName =
    realm_.equationSystems_.get_solver_block_name("total_dissipation_rate");
  LinearSolver* solver =
    realm_.root()->linearSolvers_->create_solver(solverName, EQ_TOT_DISS_RATE);
  linsys_ = LinearSystem::create(realm_, 1, this, solver);

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- assemble_nodal_gradient() ---------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::assemble_nodal_gradient()
{
  const double timeA = -NaluEnv::self().nalu_time();
  assembleNodalGradAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- compute_effective_flux_coeff() ----------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::compute_effective_diff_flux_coeff()
{
  const double timeA = -NaluEnv::self().nalu_time();
  diffFluxCoeffAlgDriver_->execute();
  timerMisc_ += (NaluEnv::self().nalu_time() + timeA);
}

//--------------------------------------------------------------------------
//-------- predict_state() -------------------------------------------------
//--------------------------------------------------------------------------
void
TotalDissipationRateEquationSystem::predict_state()
{
  // copy state n to state np1
  ScalarFieldType& tdrN = tdr_->field_of_state(stk::mesh::StateN);
  ScalarFieldType& tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  field_copy(
    realm_.meta_data(), realm_.bulk_data(), tdrN, tdrNp1,
    realm_.get_activate_aura());
}

} // namespace nalu
} // namespace sierra
