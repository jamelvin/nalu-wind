/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTERESOLUTIONADEQUACYELEMALGORITHM_H
#define COMPUTERESOLUTIONADEQUACYELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeResolutionAdequacyElemAlgorithm : public Algorithm
{
public:
  ComputeResolutionAdequacyElemAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~ComputeResolutionAdequacyElemAlgorithm() {}

  virtual void execute();

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* resolutionAdequacy_{nullptr};
  ScalarFieldType* sdr_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  GenericFieldType* dudx_{nullptr};
  GenericFieldType* Mij_{nullptr};
  const double Ch_;
  const double Chmu_;
};

} // namespace nalu
} // namespace sierra

#endif
