/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMTAMSSSTEDGEDIFFSOLVERALG_H
#define MOMENTUMTAMSSSTEDGEDIFFSOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class MomentumTAMSSSTEdgeDiffSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  MomentumTAMSSSTEdgeDiffSolverAlg(Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~MomentumTAMSSSTEdgeDiffSolverAlg() = default;

  virtual void execute();

private:
  const double includeDivU_;
  const double betaStar_;
  const double CMdeg_;

  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned alphaNp1_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned Mij_ {stk::mesh::InvalidOrdinal};
  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgDudx_{stk::mesh::InvalidOrdinal};
  unsigned avgDensity_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMTAMSSSTEDGEDIFFSOLVERALG_H */
