add_sources(GlobalSourceList
  # Edge Kernel wrapper
  AssembleEdgeKernelAlg.C
  AssembleTAMSEdgeKernelAlg.C

  # Edge kernels
  ContinuityEdgeSolverAlg.C
  MomentumEdgeSolverAlg.C
  MomentumTAMSKEpsDiffEdgeKernel.C
  ScalarEdgeSolverAlg.C
  WallDistEdgeSolverAlg.C

  # Face/edge BC kernels
  MomentumABLWallFuncEdgeKernel.C
  MomentumSymmetryEdgeKernel.C
  )
