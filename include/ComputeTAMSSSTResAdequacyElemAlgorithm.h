/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSSSTRESADEQUACYELEMALGORITHM_H
#define COMPUTETAMSSSTRESADEQUACYELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSSSTResAdequacyElemAlgorithm : public Algorithm {
public:
  ComputeTAMSSSTResAdequacyElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSSSTResAdequacyElemAlgorithm() {}

  virtual void execute();
  double get_M43_constant(double D[3][3]);

  const unsigned nDim_{0};
  const double betaStar_;
  const double CMdeg_;

  VectorFieldType *coordinates_{nullptr};
  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *viscosity_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *sdrNp1_{nullptr};
  ScalarFieldType *tkeNp1_{nullptr};
  ScalarFieldType *alphaNp1_{nullptr};
  VectorFieldType *avgVelocity_{nullptr};
  ScalarFieldType *avgDensity_{nullptr};
  ScalarFieldType *resAdeq_{nullptr};
  ScalarFieldType *avgResAdeq_{nullptr};
  GenericFieldType *Mij_{nullptr};

  std::vector<double> ws_coordinates;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_shape_function;
  std::vector<double> ws_mu;

  std::vector<double> ws_uNp1;
  std::vector<double> ws_rhoNp1;
  std::vector<double> ws_tke;
  std::vector<double> ws_sdr;
  std::vector<double> ws_alpha;

  std::vector<double> ws_avgU;
  std::vector<double> ws_avgRho;

  std::vector<double> fluctUjScs;
  std::vector<double> avgUjScs;
  std::vector<double> coordScs;
  std::vector<double> fluctDudxScs;
  std::vector<double> avgDudxScs;
  std::vector<double> dudxScs;
  std::vector<double> tauSGET;
  std::vector<double> tauSGRS;
  std::vector<double> tau;
  std::vector<double> Psgs;

};

} // namespace nalu
} // namespace sierra

#endif
