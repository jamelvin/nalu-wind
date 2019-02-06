/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSAVERAGESELEMALGORITHM_H
#define COMPUTETAMSAVERAGESELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSAveragesElemAlgorithm : public Algorithm {
public:
  ComputeTAMSAveragesElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSAveragesElemAlgorithm() {}

  virtual void execute();

  const bool meshMotion_;

  VectorFieldType *velocityRTM_{nullptr};
  ScalarFieldType *pressure_{nullptr};
  ScalarFieldType *density_{nullptr};
  ScalarFieldType *specDissipationRate_{nullptr};
  ScalarFieldType *turbKineticEnergy_{nullptr};
  VectorFieldType *avgVelocity_{nullptr};
  ScalarFieldType *avgPress_{nullptr};
  ScalarFieldType *avgDensity_{nullptr};
  ScalarFieldType *avgTurbKineticEnergy_{nullptr};
  ScalarFieldType *avgSpecDissipationRate_{nullptr};
  GenericFieldType *avgResolvedStress_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
