/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSKEPSRESADEQUACYELEMALGORITHM_H
#define COMPUTETAMSKEPSRESADEQUACYELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSKEpsResAdequacyElemAlgorithm : public Algorithm {
public:
  ComputeTAMSKEpsResAdequacyElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSKEpsResAdequacyElemAlgorithm();

  virtual void execute();

  const unsigned nDim_{0};
  const double CMdeg_;

  std::ofstream tmpFile;

  VectorFieldType *coordinates_{nullptr};
  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *turbVisc_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *tdrNp1_{nullptr};
  ScalarFieldType *tkeNp1_{nullptr};
  ScalarFieldType *alphaNp1_{nullptr};
  GenericFieldType *dudx_{nullptr};
  VectorFieldType *avgVelocity_{nullptr};
  ScalarFieldType *avgDensity_{nullptr};
  ScalarFieldType *avgTime_{nullptr};
  GenericFieldType *avgDudx_{nullptr};
  ScalarFieldType *resAdeq_{nullptr};
  ScalarFieldType *avgResAdeq_{nullptr};
  GenericFieldType *Mij_{nullptr};

  std::vector<double> tauSGET;
  std::vector<double> tauSGRS;
  std::vector<double> tau;
  std::vector<double> Psgs;

};

} // namespace nalu
} // namespace sierra

#endif
