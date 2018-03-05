/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/AdaptivityParameterSFLESSrcElemKernel.h"
#include "AlgTraits.h"
#include "Enums.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits> 
AdaptivityParameterSFLESSrcElemKernel<AlgTraits>::AdaptivityParameterSFLESSrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    cT_(solnOpts.get_turb_model_constant(TM_cT)),
    cNu_(solnOpts.get_turb_model_constant(TM_cNu)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  alphaNp1_ = metaData.get_field<ScalarFieldType>(      // FIXME: Don't need this for current model, but may need 
    stk::topology::NODE_RANK, "adaptivity_parameter");  //        for later iterations, so leaving here for now
  densityNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  velocityNp1_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  visc_ = metaData.get_field<ScalarFieldType>(  // TODO: This should be molecular viscosity here right?
    stk::topology::NODE_RANK, "viscosity");
  tkeNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  sdrNp1_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate");
  resolutionAdequacy_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy");
  Mij_ = metaData.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "metric_tensor");

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);
  
  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*alphaNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*visc_, 1);
  dataPreReqs.add_gathered_nodal_field(*tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*resolutionAdequacy_, 1);
  dataPreReqs.add_element_field(*Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

  // Do I needed a shifted_grad_op check here?
  dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  // Removing the ip based in favor of element based for now
  //dataPreReqs.add_master_element_call(SCV_MIJ, CURRENT_COORDINATES);
}

template<typename AlgTraits>
AdaptivityParameterSFLESSrcElemKernel<AlgTraits>::~AdaptivityParameterSFLESSrcElemKernel()
{}

template<typename AlgTraits>
void
AdaptivityParameterSFLESSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>&lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  // FIXME: DEBUGGING
  SharedMemView<DoubleType**>& v_coords = scratchViews.get_scratch_view_2D(*coordinates_);

  SharedMemView<DoubleType*>& v_alphaNp1 = scratchViews.get_scratch_view_1D(
    *alphaNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 = scratchViews.get_scratch_view_1D(
    *densityNp1_);
  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(
    *velocityNp1_);
  SharedMemView<DoubleType*>& v_visc = scratchViews.get_scratch_view_1D(
    *visc_);
  SharedMemView<DoubleType*>& v_tkeNp1 = scratchViews.get_scratch_view_1D(
    *tkeNp1_);
  SharedMemView<DoubleType*>& v_sdrNp1 = scratchViews.get_scratch_view_1D(
    *sdrNp1_);
  SharedMemView<DoubleType*>& v_resAdeq = scratchViews.get_scratch_view_1D(
    *resolutionAdequacy_);
  SharedMemView<DoubleType***>& v_Mij = scratchViews.get_scratch_view_3D(
    *Mij_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  //SharedMemView<DoubleType***>& v_Mij = scratchViews.get_me_views(CURRENT_COORDINATES).metric;

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap_[ip];

    // zero out; scalar
    DoubleType alphaNp1Scv = 0.0;
    DoubleType rhoNp1Scv = 0.0;
    DoubleType tkeNp1Scv = 0.0;
    DoubleType sdrNp1Scv = 0.0;
    DoubleType viscScv = 0.0;
    DoubleType resAdeqScv = 0.0;
    DoubleType coords[AlgTraits::nDim_]; // FIXME: DEBUGGING
    
    DoubleType uNp1Scv[AlgTraits::nDim_]; 
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) { 
      uNp1Scv[i] = 0.0;
      coords[i] = 0.0; // FIXME: DEBUGGING
    }

    // First we interpolate the nodal quantities to the integration points
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      //FIXME: FOR DEBUGGING, REMOVE
      for (int j = 0; j < AlgTraits::nDim_; ++j)
        coords[j] = v_coords(ic, j);

      alphaNp1Scv += r * v_alphaNp1(ic);
      rhoNp1Scv   += r * v_densityNp1(ic);
      tkeNp1Scv   += r * v_tkeNp1(ic);
      sdrNp1Scv   += r * v_sdrNp1(ic);
      viscScv     += r * v_visc(ic);
      resAdeqScv  += r * v_resAdeq(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j)
        uNp1Scv[j] += v_uNp1(ic, j);
    }

    // Define length and timescales
    const DoubleType T = stk::math::max(1.0/sdrNp1Scv, cT_*stk::math::sqrt(viscScv/(tkeNp1Scv*sdrNp1Scv)));

    //const DoubleType L = stk::math::max(stk::math::sqrt(tkeNp1Scv/sdrNp1Scv), cNu_*stk::math::pow(stk::math::pow(viscScv,3)/(tkeNp1Scv*sdrNp1Scv),0.75));

    const DoubleType Sr = stk::math::if_then_else(resAdeqScv < 1.0, stk::math::tanh(1.0 - 1.0/resAdeqScv), stk::math::tanh(resAdeqScv - 1.0));

    // FIXME: I need to take a derivative in space of the metric tensor (defined at IPs), how would I do this?
    //const DoubleType Tc = L / 1.0;  // stk::math::max(ujdjMmn)

    // FIXME: Is there a better way to do a compound if statement?  Does this way even work?
    //const DoubleType Sc = stk::math::if_then_else(Sr >= 0.0, stk::math::if_then_else(Tc >= 0.0, 1.0, 0.0), 0.0);

    // rhs assembly
    const DoubleType scvol = v_scv_volume(ip);
    // The Sc/Tc term has been removed for the time being as its form is undergoing modifications
    rhs(nearestNode) += (Sr/T)*scvol; // (Sr/T + Sc/Tc)*scvol

    // FIXME: Do I have a LHS contribution here??? I think this would only be the case if I had an alpha term in the source
    //        and thus there would be a LHS contribution from the implicitness, but since alpha doesn't appear in the source
    //        I believe the LHS contribution is 0 
    //lhs(nearestNode,nearestNode) += 1.5*tkeFac*scvol;
    
  }
}

INSTANTIATE_KERNEL(AdaptivityParameterSFLESSrcElemKernel);

}  // nalu
}  // sierra
