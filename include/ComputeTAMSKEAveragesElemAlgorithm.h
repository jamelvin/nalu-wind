/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSKEAVERAGESELEMALGORITHM_H
#define COMPUTETAMSKEAVERAGESELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSKEAveragesElemAlgorithm : public Algorithm {
public:
  ComputeTAMSKEAveragesElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSKEAveragesElemAlgorithm() {}

  virtual void execute();

  const double betaStar_;
  const bool meshMotion_;

  VectorFieldType *velocityRTM_{nullptr};
  ScalarFieldType *pressure_{nullptr};
  ScalarFieldType *density_{nullptr};
  ScalarFieldType *totDissipationRate_{nullptr};
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
