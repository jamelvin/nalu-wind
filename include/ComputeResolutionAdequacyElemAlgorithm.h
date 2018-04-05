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

  void zero_nodal_fields();

  ScalarFieldType* sdr_{nullptr};
  ScalarFieldType* tke_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  ScalarFieldType* density_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  ScalarFieldType* resolutionAdequacy_{nullptr};
  ScalarFieldType* minDistance_{nullptr};
  GenericFieldType* dudx_{nullptr};
  GenericFieldType* Mij_{nullptr};
  VectorFieldType *coordinates_{nullptr}; //FIXME: DEBUGGING

  const double Ch_;
  const double Chmu_;
  const double aOne_;
  const double betaStar_;

  std::vector<double> ws_Mij;
};

} // namespace nalu
} // namespace sierra

#endif
