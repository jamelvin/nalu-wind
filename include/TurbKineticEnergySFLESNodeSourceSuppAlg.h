/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergySFLESNodeSourceSuppAlg_h
#define TurbKineticEnergySFLESNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TurbKineticEnergySFLESNodeSourceSuppAlg : public SupplementalAlgorithm
{
public:
  TurbKineticEnergySFLESNodeSourceSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergySFLESNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  const double betaStar_;
  ScalarFieldType *tkeNp1_;
  ScalarFieldType *sdrNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *tvisc_;
  GenericFieldType *dudx_;
  ScalarFieldType *alpha_;
  ScalarFieldType *dualNodalVolume_;
  double tkeProdLimitRatio_;
  int nDim_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
