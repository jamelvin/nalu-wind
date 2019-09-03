add_sources(GlobalSourceList
  # Edge kernels
  AssembleEdgeKernelAlg.C
  AssembleTAMSEdgeKernelAlg.C
  ContinuityEdgeSolverAlg.C
  MomentumEdgeSolverAlg.C
  MomentumTAMSKEpsDiffEdgeSolverAlg.C
  MomentumTAMSKEpsDiffEdgeKernel.C
  MomentumTAMSSSTDiffEdgeKernel.C
  MomentumTAMSSSTDiffEdgeSolverAlg.C
  ScalarEdgeSolverAlg.C
  WallDistEdgeSolverAlg.C
  # Face/edge BC kernels
  MomentumABLWallFuncEdgeKernel.C
  MomentumSymmetryEdgeKernel.C
  ScalarEdgeOpenSolverAlg.C
  )
