#ifndef ENHANCED_DEPENDENCY_GRAPH_H
#define ENHANCED_DEPENDENCY_GRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <set>
#include <queue>
#include <memory>

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// // Node types for dependency graph
// enum class NodeType {
//   Kernel,   // GPU kernel launch
//   Loop,     // Loop nest (outermost loop)
//   CuLibs    // CuLibs function call sequence
// };

// Forward declarations
struct EnhancedDependencyNode;
struct Subgraph;
class EnhancedDependencyGraph;

// Enhanced node structure with triple numbering (x,y,z)
struct EnhancedDependencyNode {
  // Original fields
  NodeType type;
  Operation* op;
  llvm::SetVector<Value> inputs;
  llvm::SetVector<Value> outputs;
  
  // For different node types
  std::string kernelModuleName;
  std::string kernelName;
  llvm::SmallVector<Operation*, 4> culibsOps;
  std::string culibsFunctionName;
  
  // Enhanced numbering system
  int parallelGroupId = -1;    // x: parallel group (subgraph topological level)
  int subgraphId = -1;         // y: subgraph id
  int intraSubgraphLevel = -1; // z: topological level within subgraph
  
  // Helper fields for algorithms
  bool visited = false;
  bool inCurrentSubgraph = false;
  
  EnhancedDependencyNode() = default;
};

// Subgraph representation
struct Subgraph {
  int id;
  llvm::SmallVector<EnhancedDependencyNode*, 8> nodes;
  llvm::SetVector<Subgraph*> predecessors;
  llvm::SetVector<Subgraph*> successors;
  int topologicalLevel = -1;  // For subgraph-level topological sorting
  
  Subgraph(int subgraphId) : id(subgraphId) {}
};

class EnhancedDependencyGraph {
public:
  std::vector<std::unique_ptr<EnhancedDependencyNode>> nodes;
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  llvm::DenseMap<Operation*, EnhancedDependencyNode*> opToNodeMap;
  llvm::DenseMap<EnhancedDependencyNode*, llvm::SmallVector<EnhancedDependencyNode*, 4>> outEdges;
  llvm::DenseMap<EnhancedDependencyNode*, llvm::SmallVector<EnhancedDependencyNode*, 4>> inEdges;
  
  // Add node to graph
  EnhancedDependencyNode* addNode(std::unique_ptr<EnhancedDependencyNode> node);
  
  // Add edge between nodes
  void addEdge(EnhancedDependencyNode* from, EnhancedDependencyNode* to);
  
  // Get out degree of a node
  int getOutDegree(EnhancedDependencyNode* node);
  
  // Get in degree of a node
  int getInDegree(EnhancedDependencyNode* node);
  
  // Partition graph into subgraphs
  void partitionIntoSubgraphs();
  
  // Perform topological sorting on subgraphs
  void topologicalSortSubgraphs();
  
  // Perform topological sorting within each subgraph
  void topologicalSortWithinSubgraphs();
  
  // Complete enhanced scheduling
  void performEnhancedScheduling();
  
  // Debug printing functions
  void dumpEnhancedGraph();
  void dumpSubgraphInfo();
  void dumpFinalScheduling();
  
private:
  int nextSubgraphId = 0;
  
  // Helper function to create new subgraph
  Subgraph* createSubgraph();
  
  // Helper function to add node to subgraph
  void addNodeToSubgraph(EnhancedDependencyNode* node, Subgraph* subgraph);
  
  // Helper function to find nodes with zero in-degree
  llvm::SmallVector<EnhancedDependencyNode*, 8> findZeroInDegreeNodes();
  
  // Helper function to build subgraph edges
  void buildSubgraphEdges();
  
  // Helper function for DFS traversal in subgraph partitioning
  void dfsPartition(EnhancedDependencyNode* node, Subgraph* currentSubgraph);
};

// Helper function declarations (dependency analysis functions from original code)
bool isKernelLaunch_plus(Operation* op);
bool isLoopNest_plus(Operation* op);
bool isCuLibsCall_plus(Operation* op);
bool isCuLibsStreamCreate_plus(Operation* op);
bool isCuLibsStreamSync_plus(Operation* op);
bool isCuLibsStreamDestroy_plus(Operation* op);
gpu::GPUFuncOp findKernelFunc_plus(gpu::LaunchFuncOp kernelOp);
void extractKernelDependencies_plus(gpu::LaunchFuncOp kernelOp, 
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs);
void extractLoopDependencies_plus(scf::ForOp loopOp,
                           llvm::SetVector<Value> &inputs,
                           llvm::SetVector<Value> &outputs);
void extractCuLibsDependencies_plus(const llvm::SmallVector<Operation*, 4> &culibsOps,
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs);
llvm::SmallVector<Operation*, 4> findCuLibsSequence_plus(Operation* streamCreateOp);

// Main function to build enhanced dependency graph
std::unique_ptr<EnhancedDependencyGraph> buildEnhancedDependencyGraph(func::FuncOp funcOp);

// Analysis and utility functions
void analyzeSchedulingResults(const EnhancedDependencyGraph& graph);
void generateStreamAssignmentPlan(const EnhancedDependencyGraph& graph);
void exportSchedulingInfo(const EnhancedDependencyGraph& graph, const std::string& functionName);
void performEnhancedSchedulingWorkflow(func::FuncOp funcOp);
void integrateEnhancedScheduling(func::FuncOp funcOp);

} // namespace onnx_mlir

#endif // ENHANCED_DEPENDENCY_GRAPH_H