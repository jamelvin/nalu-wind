/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef EffectiveSFLESDiffFluxCoeffAlgorithm_h
#define EffectiveSFLESDiffFluxCoeffAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffectiveSFLESDiffFluxCoeffAlgorithm : public Algorithm
{
public:

  EffectiveSFLESDiffFluxCoeffAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *visc,
    ScalarFieldType *tvisc,
    ScalarFieldType *evisc,
    const double sigmaLam,
    const double sigmaTurb);
  virtual ~EffectiveSFLESDiffFluxCoeffAlgorithm() {}
  virtual void execute();

  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType *alpha_;

  const double sigmaLam_;
  const double sigmaTurb_;  
  const bool isTurbulent_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
