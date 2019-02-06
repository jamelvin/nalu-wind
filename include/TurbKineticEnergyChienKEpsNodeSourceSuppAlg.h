/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyChienKEpsNodeSourceSuppAlg_h
#define TurbKineticEnergyChienKEpsNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TurbKineticEnergyChienKEpsNodeSourceSuppAlg : public SupplementalAlgorithm
{
public:
  TurbKineticEnergyChienKEpsNodeSourceSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergyChienKEpsNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  ScalarFieldType *tkeNp1_;
  ScalarFieldType *tdrNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  GenericFieldType *dudx_;
  ScalarFieldType *minD_;
  ScalarFieldType *dualNodalVolume_;
  int nDim_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
