target_sources(nalu PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOElemSuppAlgDep.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOGradElemSuppAlg.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOKeElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOKeElemSuppAlgDep.C
   ${CMAKE_CURRENT_SOURCE_DIR}/MomentumNSOSijElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOElemSuppAlgDep.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOKeElemKernel.C
   ${CMAKE_CURRENT_SOURCE_DIR}/ScalarNSOKeElemSuppAlg.C
)
