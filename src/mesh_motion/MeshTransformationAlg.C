
#include "mesh_motion/MeshTransformationAlg.h"

#include "NaluParsing.h"

#include <cassert>
#include <iostream>

#include "../../include/mesh_motion/FrameReference.h"

namespace sierra{
namespace nalu{

MeshTransformationAlg::MeshTransformationAlg(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node)
  : bulk_(bulk)
{
  load(bulk, node);
}

void MeshTransformationAlg::load(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node)
{
  // get motion information for entire mesh
  const int num_groups = node.size();
  referenceFrameVec_.resize(num_groups);

  for (int i=0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = node[i];

    referenceFrameVec_[i].reset(new FrameReference(bulk, ginfo));
  }
}

void MeshTransformationAlg::initialize( const double time )
{
  if(isInit_)
    throw std::runtime_error("MeshTransformationAlg::initialize(): Re-initialization of MeshTransformationAlg not valid");

  for (size_t i=0; i < referenceFrameVec_.size(); i++)
  {
    referenceFrameVec_[i]->setup();

    // update coordinates
    referenceFrameVec_[i]->update_coordinates(time);
  }

  // TODO: NGP Transition
  // Manually synchronize fields to device
  {
    auto* coords = bulk_.mesh_meta_data().get_field(
        stk::topology::NODE_RANK, "coordinates");
    if (coords != nullptr) {
      coords->modify_on_host();
      coords->sync_to_device();
    }
  }
  isInit_ = true;
}

} // nalu
} // sierra
