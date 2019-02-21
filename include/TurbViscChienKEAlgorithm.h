/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscChienKEAlgorithm_h
#define TurbViscChienKEAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscChienKEAlgorithm : public Algorithm
{
public:
  
  TurbViscChienKEAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscChienKEAlgorithm() {}
  virtual void execute();

  const double cMu_;
  const double fMuExp_;

  ScalarFieldType *density_;
  ScalarFieldType *tke_;
  ScalarFieldType *tdr_;
  ScalarFieldType *dplus_;
  ScalarFieldType *tvisc_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
