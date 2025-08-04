// #ifndef DEPENDENCY_GRAPH_H
// #define DEPENDENCY_GRAPH_H

// #include "mlir/IR/Operation.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/ADT/SetVector.h"
// #include <vector>
// #include <memory>

// using namespace mlir;

// namespace onnx_mlir {

// // Node types in the dependency graph
// enum class NodeType {
//   Kernel,   // GPU kernel launch
//   Loop,     // SCF loop nest  
//   CuLibs    // CuLibs wrapper function call
// };

// // Represents a node in the dependency graph
// struct DependencyNode {
//   NodeType type;
//   Operation* op;  // The main operation (gpu.launch_func, scf.for, or culibs call)
  
//   // For kernel nodes
//   StringRef kernelModuleName;
//   StringRef kernelName;
  
//   // For culibs nodes
//   StringRef culibsFunctionName;
//   llvm::SmallVector<Operation*, 4> culibsOps;  // All related ops (create, call, sync, destroy)
  
//   // Dependencies
//   llvm::SetVector<Value> inputs;   // Input memrefs
//   llvm::SetVector<Value> outputs;  // Output memrefs
  
//   // Topological sort level (0 = unassigned)
//   unsigned topologicalLevel = 0;
// };

// // The dependency graph structure
// struct DependencyGraph {
//   // All nodes in the graph
//   llvm::SmallVector<std::unique_ptr<DependencyNode>, 16> nodes;
  
//   // Edge lists
//   llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> outEdges;
//   llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> inEdges;
  
//   // Mapping from operations to nodes
//   llvm::DenseMap<Operation*, DependencyNode*> opToNodeMap;
  
//   // Helper methods
//   DependencyNode* addNode(std::unique_ptr<DependencyNode> node);
//   void addEdge(DependencyNode* from, DependencyNode* to);
// };

// // Function declarations
// std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp);
// void dumpDependencyGraph(DependencyGraph &graph);

// // Helper functions
// bool isKernelLaunch(Operation* op);
// bool isLoopNest(Operation* op);
// bool isCuLibsCall(Operation* op);
// bool isCuLibsStreamCreate(Operation* op);
// bool isCuLibsStreamSync(Operation* op);
// bool isCuLibsStreamDestroy(Operation* op);

// } // namespace onnx_mlir

// #endif // DEPENDENCY_GRAPH_H

#ifndef DEPENDENCY_GRAPH_H
#define DEPENDENCY_GRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <memory>

using namespace mlir;

namespace onnx_mlir {

// Node types in the dependency graph
enum class NodeType {
  Kernel,   // GPU kernel launch
  Loop,     // SCF loop nest  
  CuLibs    // CuLibs wrapper function call
};

// Represents a node in the dependency graph
struct DependencyNode {
  NodeType type;
  Operation* op;  // The main operation (gpu.launch_func, scf.for, or culibs call)

  // For kernel nodes
  StringRef kernelModuleName;
  StringRef kernelName;
  llvm::SmallVector<Operation*, 4> kernelOps;  // NEW: All related ops (reinterpret_cast, kernel_launch, etc.)

  // For culibs nodes
  StringRef culibsFunctionName;
  llvm::SmallVector<Operation*, 4> culibsOps;  // All related ops (create, call, sync, destroy)

  // Dependencies
  llvm::SetVector<Value> inputs;   // Input memrefs
  llvm::SetVector<Value> outputs;  // Output memrefs

  // Topological sort level (0 = unassigned)
  unsigned topologicalLevel = 0;
};

// The dependency graph structure
struct DependencyGraph {
  // All nodes in the graph
  llvm::SmallVector<std::unique_ptr<DependencyNode>, 16> nodes;

  // Edge lists
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> outEdges;
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> inEdges;

  // Mapping from operations to nodes
  llvm::DenseMap<Operation*, DependencyNode*> opToNodeMap;

  // Helper methods
  DependencyNode* addNode(std::unique_ptr<DependencyNode> node);
  void addEdge(DependencyNode* from, DependencyNode* to);
};

// Function declarations
std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp);
void dumpDependencyGraph(DependencyGraph &graph);

// Helper functions
bool isKernelLaunch(Operation* op);
bool isLoopNest(Operation* op);
bool isCuLibsCall(Operation* op);
bool isCuLibsStreamCreate(Operation* op);
bool isCuLibsStreamSync(Operation* op);
bool isCuLibsStreamDestroy(Operation* op);

// NEW: Helper functions for kernel sequence analysis
bool isMemrefReinterpretCast(Operation* op);
bool shouldIncludeInKernelSequence(Operation* op);
llvm::SmallVector<Operation*, 4> findExtendedKernelSequence(Operation* kernelLaunchOp);

} // namespace onnx_mlir

#endif // DEPENDENCY_GRAPH_H