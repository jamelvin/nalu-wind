/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm_h
#define EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm : public Algorithm
{
public:

  EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *evisc);
  virtual ~EffectiveAdaptivityParameterDiffFluxCoeffAlgorithm() {}
  virtual void execute();

  ScalarFieldType *evisc_;

};

} // namespace nalu
} // namespace Sierra

#endif
