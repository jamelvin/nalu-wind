/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSSSTAVERAGESELEMALGORITHM_H
#define COMPUTETAMSSSTAVERAGESELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSSSTAveragesElemAlgorithm : public Algorithm {
public:
  ComputeTAMSSSTAveragesElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSSSTAveragesElemAlgorithm() {}

  virtual void execute();

  const double betaStar_;
  const bool meshMotion_;

  VectorFieldType *velocityRTM_{nullptr};
  ScalarFieldType *pressure_{nullptr};
  ScalarFieldType *density_{nullptr};
  ScalarFieldType *specDissipationRate_{nullptr};
  ScalarFieldType *turbKineticEnergy_{nullptr};
  GenericFieldType *dudx_{nullptr};
  VectorFieldType *avgVelocity_{nullptr};
  ScalarFieldType *avgPress_{nullptr};
  ScalarFieldType *avgDensity_{nullptr};
  GenericFieldType *avgResolvedStress_{nullptr};
  GenericFieldType *avgDudx_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
