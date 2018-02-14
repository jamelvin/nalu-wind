/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "AdaptivityParameterSFLESSrcElemKernel.h"
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
  dualNodalVolume_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);
  
  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_gathered_nodal_field(*alphaNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*visc_, 1);
  dataPreReqs.add_gathered_nodal_field(*tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*dualNodalVolume_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_MIJ, CURRENT_COORDINATES);
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
  SharedMemView<DoubleType*>& v_dualNodalVolume = scratchViews.get_scratch_view_1D(
    *dualNodalVolume_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  SharedMemView<DoubleType***>& v_Mij = scratchViews.get_me_views(CURRENT_COORDINATES).metric;

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap_[ip];

    // zero out; scalar
    DoubleType alphaNp1Scs = 0.0;
    DoubleType rhoNp1Scs = 0.0;
    DoubleType tkeNp1Scs = 0.0;
    DoubleType sdrNp1Scs = 0.0;
    DoubleType viscScs = 0.0;
    
    DoubleType uNp1Scs[AlgTraits::nDim_]; 
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) 
      uNp1Scs[i] = 0.0;
    
    // First we interpolate the nodal quantities to the integration points
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      alphaNp1Scs += r * v_alphaNp1(ic);
      rhoNp1Scs   += r * v_densityNp1(ic);
      tkeNp1Scs   += r * v_tkeNp1(ic);
      sdrNp1Scs   += r * v_sdrNp1(ic);
      viscScs     += r * v_visc(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j)
        uNp1Scs[j] += v_uNp1(ic, j);
    }

    // FIXME: Need to link in resolution adequacy parameter
    DoubleType resAdeqScs = 1.0;

    // Define length and timescales
    const DoubleType T = stk::math::max(1.0/sdrNp1Scs, cT_*stk::math::sqrt(viscScs/(tkeNp1Scs*sdrNp1Scs)));

    const DoubleType L = stk::math::max(stk::math::sqrt(tkeNp1Scs/sdrNp1Scs), cNu_*stk::math::pow(stk::math::pow(viscScs,3)/(tkeNp1Scs*sdrNp1Scs),0.75));

    const DoubleType Sr = stk::math::if_then_else(resAdeqScs < 1.0, stk::math::tanh(1.0 - 1.0/resAdeqScs), stk::math::tanh(resAdeqScs - 1.0));

    // FIXME: I need to take a derivative in space of the metric tensor (defined at IPs), how would I do this?
    const DoubleType Tc = L / 1.0;  // stk::math::max(ujdjMmn)

    // FIXME: Is there a better way to do a compound if statement?  Does this way even work?
    const DoubleType Sc = stk::math::if_then_else(Sr >= 0.0, stk::math::if_then_else(Tc >= 0.0, 1.0, 0.0), 0.0);

    // rhs assembly
    const DoubleType scvol = v_scv_volume(ip);
    rhs(nearestNode) += (Sr/T + Sc/Tc)*scvol;

    // FIXME: Do I have a LHS contribution here??? I think this would only be the case if I had an alpha term in the source
    //        and thus there would be a LHS contribution from the implicitness, but since alpha doesn't appear in the source
    //        I believe the LHS contribution is 0 
    //lhs(nearestNode,nearestNode) += 1.5*tkeFac*scvol;
    
  }
}

INSTANTIATE_KERNEL(AdaptivityParameterSFLESSrcElemKernel);

}  // nalu
}  // sierra
