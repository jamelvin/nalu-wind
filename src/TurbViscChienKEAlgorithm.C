/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <TurbViscChienKEAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbViscChienKEAlgorithm - compute tvisc for Chien K-epsilon model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbViscChienKEAlgorithm::TurbViscChienKEAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    cMu_(realm.get_turb_model_constant(TM_cMu)),
    fMuExp_(realm.get_turb_model_constant(TM_fMuExp)),
    density_(NULL),
    tke_(NULL),
    tdr_(NULL),
    dplus_(NULL),
    tvisc_(NULL)
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  tke_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  tdr_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "total_dissipation_rate");
  dplus_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dplus_wall_function");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbViscChienKEAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tvisc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *rho = stk::mesh::field_data(*density_, b);
    const double *tke = stk::mesh::field_data(*tke_, b);
    const double *tdr = stk::mesh::field_data(*tdr_, b);
    const double *dplus = stk::mesh::field_data(*dplus_, b);
    double *tvisc = stk::mesh::field_data(*tvisc_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // some temps
      const double fMu = 1.0 - std::exp(fMuExp_*dplus[k]);

      tvisc[k] = cMu_ * fMu * rho[k] * tke[k] * tke[k] / std::max(tdr[k],1.0e-16);

    }
  }
}

} // namespace nalu
} // namespace Sierra
