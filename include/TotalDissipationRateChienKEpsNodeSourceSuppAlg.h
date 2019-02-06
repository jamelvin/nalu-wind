/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TotalDissipationRateChienKEpsNodeSourceSuppAlg_h
#define TotalDissipationRateChienKEpsNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

class Realm;

class TotalDissipationRateChienKEpsNodeSourceSuppAlg
  : public SupplementalAlgorithm
{
public:
  TotalDissipationRateChienKEpsNodeSourceSuppAlg(Realm& realm);

  virtual ~TotalDissipationRateChienKEpsNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(double* lhs, double* rhs, stk::mesh::Entity node);

  const double cEpsOne_, cEpsTwo_, fOne_;
  ScalarFieldType* tdrNp1_;
  ScalarFieldType* tkeNp1_;
  ScalarFieldType* densityNp1_;
  ScalarFieldType* dplus_;
  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  GenericFieldType* dudx_;
  ScalarFieldType* minD_;
  ScalarFieldType* dualNodalVolume_;
  double tkeProdLimitRatio_;
  int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif
