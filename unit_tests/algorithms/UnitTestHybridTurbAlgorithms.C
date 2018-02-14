/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "UnitTestAlgorithm.h"
#include "UnitTestKokkosUtils.h"
#include "UnitTestFieldUtils.h"
#include "UnitTestAlgorithmUtils.h"

#include "ComputeResolutionAdequacyElemAlgorithm.h"

TEST_F(TestTurbulenceAlgorithm, computeresolutionadequacyelemalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::ComputeResolutionAdequacyElemAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*resolutionAdequacy_);
  const double gold_norm = 0.77621545071780862;
  EXPECT_NEAR(norm, gold_norm, tol);
}
