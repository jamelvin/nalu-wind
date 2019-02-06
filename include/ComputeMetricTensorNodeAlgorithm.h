/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTEMETRICTENSORNODEALGORITHM_H
#define COMPUTEMETRICTENSORNODEALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeMetricTensorNodeAlgorithm : public Algorithm {
public:
  ComputeMetricTensorNodeAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeMetricTensorNodeAlgorithm();

  virtual void execute();

  void post_work();

  std::ofstream tmpFile;

  VectorFieldType *coordinates_{nullptr};
  GenericFieldType *nodalMij_{nullptr};
  ScalarFieldType *dualNodalVolume_{nullptr};

  std::vector<double> ws_coordinates;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_scv_volume;
  std::vector<double> ws_det_j;
  std::vector<double> ws_Mij;
};

} // namespace nalu
} // namespace sierra

#endif
