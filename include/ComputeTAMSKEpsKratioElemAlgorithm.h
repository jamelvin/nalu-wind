/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSKEPSKRATIOELEMALGORITHM_H
#define COMPUTETAMSKEPSKRATIOELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSKEpsKratioElemAlgorithm : public Algorithm {
public:
  ComputeTAMSKEpsKratioElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSKEpsKratioElemAlgorithm() {}

  virtual void execute();

  ScalarFieldType *alpha_{nullptr};
  ScalarFieldType *turbKineticEnergy_{nullptr};
  ScalarFieldType *totalDissRate_{nullptr};
  ScalarFieldType *viscosity_{nullptr};
  ScalarFieldType *turbVisc_{nullptr};
  ScalarFieldType *avgTkeRes_{nullptr};
  ScalarFieldType *avgTime_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
