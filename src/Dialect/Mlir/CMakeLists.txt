# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMMlirDialects
  IndexExpr.cpp
  IndexExprDetail.cpp
  IndexExprBuilder.cpp
  DialectBuilder.cpp
  VectorMachineSupport.cpp
  ParallelLoopTiling.cpp
  DependencyGraph.cpp
  IrReorganization.cpp
  KernelParallelizationPass.cpp
  TopoLogicalSort.cpp
  KernelCostEstimator.cpp
  KernelExecutionOptimizer.cpp
  TileSizeOptimizer.cpp
  ConvertONNXToculibs.cpp
  EliminateRedundantPtrConversion.cpp
  UseHandleStreamPool.cpp
  FP16ThresholdPass.cpp
  EnhancedDependencyGraph.cpp
  EnhancedIrReorganization.cpp
  StreamUnificationPass.cpp
  HierarchicalParallelScheduler.cpp
  EliminateRedundantStreamSync.cpp
  InsertMainFunctionPass.cpp
  AffineLoopFlattenPass.cpp
  SNNBatchFusionPass.cpp
  SNNStructureAnnotation.cpp
  Lower4DMatMul.cpp

  DEPENDS
  OMKrnlIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  OMCompilerOptions
  MLIRMathDialect
  MLIRLLVMDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  MLIRShapeDialect
  )
