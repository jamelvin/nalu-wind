/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ChienKEpsilonEquationSystem.h>
#include <AlgorithmDriver.h>
#include <FieldFunctions.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>
#include <TotalDissipationRateEquationSystem.h>
#include <SolutionOptions.h>
#include <TurbKineticEnergyEquationSystem.h>
#include <Realm.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// basic c++
#include <cmath>
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ChienKEpsilonEquationSystem - manage K-Epsilon 
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ChienKEpsilonEquationSystem::ChienKEpsilonEquationSystem(
  EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "ChienKEpsilonWrap"),
    tkeEqSys_(NULL),
    tdrEqSys_(NULL),
    tke_(NULL),
    tdr_(NULL),
    minDistanceToWall_(NULL),
    dplus_(NULL),
    isInit_(true)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  // create momentum and pressure
  tkeEqSys_= new TurbKineticEnergyEquationSystem(eqSystems);
  tdrEqSys_ = new TotalDissipationRateEquationSystem(eqSystems);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ChienKEpsilonEquationSystem::~ChienKEpsilonEquationSystem()
{
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::initialize()
{
  // let equation systems that are owned some information
  tkeEqSys_->convergenceTolerance_ = convergenceTolerance_;
  tdrEqSys_->convergenceTolerance_ = convergenceTolerance_;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{

  stk::mesh::MetaData &meta_data = realm_.meta_data();
  const int numStates = realm_.number_of_states();

  // re-register tke and tdr for convenience
  tke_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke", numStates));
  stk::mesh::put_field_on_mesh(*tke_, *part, nullptr);
  tdr_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate", numStates));
  stk::mesh::put_field_on_mesh(*tdr_, *part, nullptr);

  // KE parameters that everyone needs
  minDistanceToWall_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall"));
  stk::mesh::put_field_on_mesh(*minDistanceToWall_, *part, nullptr);
  dplus_ =  &(meta_data.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dplus_wall_function"));
  stk::mesh::put_field_on_mesh(*dplus_, *part, nullptr);
  
  // add to restart field
  realm_.augment_restart_variable_list("minimum_distance_to_wall");
  realm_.augment_restart_variable_list("dplus_wall_function");
}


//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{
  // nothing to do here...
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &/*wallBCData*/)
{
  // push mesh part
  wallBcPart_.push_back(part);
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::solve_and_update()
{
  // wrap timing
  // KEps_FIXME: deal with timers; all on misc for KEpsEqs double timeA, timeB;
  if ( isInit_ ) {
    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    tdrEqSys_->assemble_nodal_gradient();
    // FIXME: This needs to be done every timestep dynamically once utau issue
    //        is resolved...
    compute_dplus_function();

    // FIXME: Figure out a way to do this properly in TAMS Eqn System, so that 
    //        they are decoupled
    if (realm_.solutionOptions_->turbulenceModel_ == TAMS_KEPS) 
        initialize_mdot();
    
    isInit_ = false;
  } else if (realm_.has_mesh_motion()) {
    if (realm_.currentNonlinearIteration_ == 1) {
      //FIXME: This needs to be done every timestep dynamically once utau issue
      //       is resolved...
      compute_dplus_function();
    }
  }

  // KEps effective viscosity for k and epsilon 
  tkeEqSys_->compute_effective_diff_flux_coeff();
  tdrEqSys_->compute_effective_diff_flux_coeff();

  // start the iteration loop
  for ( int k = 0; k < maxIterations_; ++k ) {

    NaluEnv::self().naluOutputP0() << " " << k+1 << "/" << maxIterations_
                    << std::setw(15) << std::right << name_ << std::endl;

    // tke and tdr assemble, load_complete and solve; Jacobi iteration
    tkeEqSys_->assemble_and_solve(tkeEqSys_->kTmp_);
    tdrEqSys_->assemble_and_solve(tdrEqSys_->eTmp_);

    // update each
    update_and_clip();

    // compute projected nodal gradients
    tkeEqSys_->compute_projected_nodal_gradient();
    tdrEqSys_->assemble_nodal_gradient();
  }

}

//--------------------------------------------------------------------------
//-------- initial_work ----------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::initial_work()
{
  // do not lett he user specify a negative field
  const double clipValue = 1.0e-8;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // required fields with state
  ScalarFieldType &tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tdr_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    double *tke = stk::mesh::field_data(tkeNp1, b);
    double *tdr = stk::mesh::field_data(tdrNp1, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      const double tkeNew = tke[k];
      const double tdrNew = tdr[k];
      
      if ( (tkeNew >= 0.0) && (tdrNew >= 0.0) ) {
        // nothing
      }
      else if ( (tkeNew < 0.0) && (tdrNew < 0.0) ) {
        // both negative;
        tke[k] = clipValue;
        tdr[k] = clipValue;
      }
      else if ( tkeNew < 0.0 ) {
        tke[k] = clipValue;
        tdr[k] = tdrNew;
      }
      else {
        tdr[k] = clipValue;
        tke[k] = tkeNew;
      }
    }
  }

  if (isInit_)
    compute_dplus_function();
}

//--------------------------------------------------------------------------
//-------- post_adapt_work -------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::post_adapt_work()
{
  if ( realm_.process_adaptivity() ) {
    NaluEnv::self().naluOutputP0() << "--ChienKEpsilonEquationSystem::post_adapt_work()" << std::endl;
  }

}

//--------------------------------------------------------------------------
//-------- update_and_clip() -----------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::update_and_clip()
{
  const double clipValue = 1.0e-8;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // model parameters
  const double cMu = realm_.get_turb_model_constant(TM_cMu);
  const double fMuExp = realm_.get_turb_model_constant(TM_fMuExp);

  // required fields
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *turbViscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");

  // required fields with state
  ScalarFieldType &tdrNp1 = tdr_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &tkeNp1 = tke_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*turbViscosity);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *rho = stk::mesh::field_data(*density, b);
    const double *kTmp = stk::mesh::field_data(*tkeEqSys_->kTmp_, b);
    const double *eTmp = stk::mesh::field_data(*tdrEqSys_->eTmp_, b);
    double *tke = stk::mesh::field_data(tkeNp1, b);
    double *tdr = stk::mesh::field_data(tdrNp1, b);
    double *tvisc = stk::mesh::field_data(*turbViscosity, b);
    double *dp = stk::mesh::field_data(*dplus_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      const double tkeNew = tke[k] + kTmp[k];
      const double tdrNew = tdr[k] + eTmp[k];
      
      if ( (tkeNew >= 0.0) && (tdrNew >= 0.0) ) {
        // if all is well
        tke[k] = tkeNew;
        tdr[k] = tdrNew;
      }
      else if ( (tkeNew < 0.0) && (tdrNew < 0.0) ) {
        // both negative; set k to small, tvisc to molecular visc and use Prandtl/Kolm for tdr
        tke[k] = clipValue;
        tdr[k] = clipValue;
        const double fMu = 1.0 - std::exp(fMuExp*dp[k]);
        tvisc[k] = cMu*fMu*rho[k]*clipValue;
      }
      else if ( tkeNew < 0.0 ) {
        // only tke is off; reset tvisc to molecular visc and compute new tke appropriately
        tke[k] = clipValue;
        tdr[k] = tdrNew;
        const double fMu = 1.0 - std::exp(fMuExp*dp[k]);
        tvisc[k] = cMu*fMu*rho[k]*clipValue*clipValue/tdrNew; 
      }
      else {
        // only tdr if off; reset tvisc to molecular visc and compute new tdr appropriately
        tdr[k] = clipValue;
        tke[k] = tkeNew;
        const double fMu = 1.0 - std::exp(fMuExp*dp[k]);
        tvisc[k] = cMu*fMu*rho[k]*tkeNew*tkeNew/clipValue;
      }
    }
  }

  // parallel assemble clipped value
  if (realm_.debug()) {
    NaluEnv::self().naluOutputP0() << "Add KEps clipping diagnostic" << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- compute_dplus_function ------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::compute_dplus_function()
{
  // compute dplus with parameters appropriate for Chien 1982 K-epsilon model
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // model paramters FIXME: This shouldn't be an input option, needs to be calculated dynamically!
  const double utau = realm_.get_turb_model_constant(TM_utau);

  // fields not saved off
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ScalarFieldType *viscosity = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");

  //select all nodes (locally and shared)
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*dplus_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // fields; supplemental and non-const fOne and ftwo
    const double * minD = stk::mesh::field_data(*minDistanceToWall_, b);
    const double * rho = stk::mesh::field_data(*density, b);
    const double * mu = stk::mesh::field_data(*viscosity, b);
    double * dp = stk::mesh::field_data(*dplus_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // FIXME: This uses an input option for utau...utau needs to be calculated
      // dynamically for non-channel cases and not just at the wall, but available
      // everywhere based on which point is the closest to the wall from that one...
      dp[k] = minD[k]*rho[k]*utau/mu[k];

    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize_mdot -------------------------------------------------
//--------------------------------------------------------------------------
void
ChienKEpsilonEquationSystem::initialize_mdot()
{
  //FIXME: Don't do this if it's a restart and average_mdot has been defined...

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if (realm_.realmUsesEdges_) {
    ScalarFieldType *massFlowRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate");
    ScalarFieldType *avgMdot_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "average_mass_flow_rate");

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
    GenericFieldType *avgMdotScs_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "average_mass_flow_rate_scs");

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


} // namespace nalu
} // namespace Sierra
