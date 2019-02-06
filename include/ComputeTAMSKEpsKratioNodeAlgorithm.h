/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSKEPSKRATIONODEALGORITHM_H
#define COMPUTETAMSKEPSKRATIONODEALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSKEpsKratioNodeAlgorithm : public Algorithm
{
public:
  ComputeTAMSKEpsKratioNodeAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~ComputeTAMSKEpsKratioNodeAlgorithm() {}

  virtual void execute();

  ScalarFieldType* alpha_{nullptr};
  ScalarFieldType* turbKineticEnergy_{nullptr};
  ScalarFieldType* totalDissRate_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  ScalarFieldType* turbVisc_{nullptr};
  ScalarFieldType* avgTkeRes_{nullptr};
  ScalarFieldType* avgTime_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
