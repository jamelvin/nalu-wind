/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSKEPSFORCINGEDGESOLVERALG_H
#define MOMENTUMTAMSKEPSFORCINGEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class MomentumTAMSKEpsForcingEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  MomentumTAMSKEpsForcingEdgeSolverAlg(
    Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~MomentumTAMSKEpsForcingEdgeSolverAlg() = default;

  virtual void execute();

private:
  DblType pi_;
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned turbVisc_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned tdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};
  unsigned minDist_{stk::mesh::InvalidOrdinal};

  // average quantities
  unsigned avgVelocity_ = {stk::mesh::InvalidOrdinal};
  unsigned avgDensity_ = {stk::mesh::InvalidOrdinal};
  unsigned avgTime_ = {stk::mesh::InvalidOrdinal};
  unsigned avgResAdeq_ = {stk::mesh::InvalidOrdinal};

  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMTAMSKEPSFORCINGEDGESOLVERALG_H */
