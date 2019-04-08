/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscTAMSKEAlgorithm_h
#define TurbViscTAMSKEAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscTAMSKEAlgorithm : public Algorithm
{
public:
  
  TurbViscTAMSKEAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscTAMSKEAlgorithm() {}
  virtual void execute();

  const double cMu_;
  const double fMuExp_;

  ScalarFieldType *density_;
  ScalarFieldType *tke_;
  ScalarFieldType *tdr_;
  ScalarFieldType *dplus_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *visc_;
  GenericFieldType *avgDudx_;  
};

} // namespace nalu
} // namespace Sierra

#endif
