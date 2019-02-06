/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscChienKEpsAlgorithm_h
#define TurbViscChienKEpsAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscChienKEpsAlgorithm : public Algorithm
{
public:
  
  TurbViscChienKEpsAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscChienKEpsAlgorithm() {}
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
