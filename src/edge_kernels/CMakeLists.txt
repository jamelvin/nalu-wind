add_sources(GlobalSourceList
  # Edge Kernel wrapper
  AssembleEdgeKernelAlg.C

  # Edge kernels
  ContinuityEdgeSolverAlg.C
  MomentumEdgeSolverAlg.C
  ScalarEdgeSolverAlg.C
  WallDistEdgeSolverAlg.C

  # Face/edge BC kernels
  MomentumABLWallFuncEdgeKernel.C
  MomentumSymmetryEdgeKernel.C
  )
