/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSSSTKRATIOELEMALGORITHM_H
#define COMPUTETAMSSSTKRATIOELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSSSTKratioElemAlgorithm : public Algorithm {
public:
  ComputeTAMSSSTKratioElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSSSTKratioElemAlgorithm() {}

  virtual void execute();

  const double betaStar_;

  ScalarFieldType *alpha_{nullptr};
  ScalarFieldType *turbKineticEnergy_{nullptr};
  ScalarFieldType *specDissRate_{nullptr};
  ScalarFieldType *viscosity_{nullptr};
  ScalarFieldType *turbVisc_{nullptr};
  ScalarFieldType *avgTkeRes_{nullptr};
  ScalarFieldType *avgTime_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
