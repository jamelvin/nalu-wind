/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSKRATIOELEMALGORITHM_H
#define COMPUTETAMSKRATIOELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSKratioElemAlgorithm : public Algorithm {
public:
  ComputeTAMSKratioElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSKratioElemAlgorithm() {}

  virtual void execute();

  ScalarFieldType *alpha_{nullptr};
  ScalarFieldType *avgTurbKineticEnergy_{nullptr};
  GenericFieldType *avgResolvedStress_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
