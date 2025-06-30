#ifndef ENHANCED_IR_REORGANIZATION_H
#define ENHANCED_IR_REORGANIZATION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <map>
#include <vector>
#include <memory>

#include "EnhancedDependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// ===== Data Structures =====

struct ParallelGroupResources {
  std::map<int, Value> subgraphStreams;
  std::map<int, std::vector<Value>> subgraphTokens;
  std::map<int, std::vector<EnhancedDependencyNode*>> subgraphNodes;
};

struct SubgraphAnalysis {
  int kernelCount = 0;
  int culibsCount = 0;
  int loopCount = 0;
  bool hasSequentialKernels = false;
  bool hasParallelSegments = false;
};

enum class SegmentType {
  ParallelSegment,
  SequentialSegment
};

struct ExecutionSegment {
  SegmentType type;
  std::vector<EnhancedDependencyNode*> nodes;
  int startLevel;
  int endLevel;
};

struct ExtendedParallelGroupResources {
  std::map<int, std::vector<Value>> subgraphStreams;
  std::map<int, std::vector<Value>> subgraphTokens;
  std::map<int, std::vector<EnhancedDependencyNode*>> subgraphNodes;
};

struct SeparatedParallelGroupResources {

  std::map<int, std::vector<Value>> subgraphGpuTokens;
  std::map<int, int> subgraphGpuTokenCount;
  
  std::map<int, std::vector<Value>> subgraphCulibsStreams;
  std::map<int, int> subgraphCulibsStreamCount;
  
  std::map<int, std::vector<EnhancedDependencyNode*>> subgraphNodes;
};

struct SubgraphResourceAnalysis {
  int maxGpuParallelism = 0;
  int maxCulibsParallelism = 0;
  bool hasGpuKernels = false;
  bool hasCulibsCalls = false;
  bool hasLoops = false;
  int totalLevels = 0;
};

struct SingleNodeChain {
  int startLevel;
  int endLevel;
  std::vector<EnhancedDependencyNode*> nodes;
  bool hasKernels = false;
  bool hasCulibs = false;
  bool isMixed = false;
  std::vector<int> kernelPositions;
};

struct ChainSegment {
  int startLevel;
  int endLevel;
  std::vector<EnhancedDependencyNode*> nodes;
  bool isKernelSegment = false;
  bool needsSyncBefore = false;
  bool needsSyncAfter = false;
};

struct SubgraphChainAnalysis {
  std::vector<SingleNodeChain> singleNodeChains;
  std::vector<std::pair<int, std::vector<EnhancedDependencyNode*>>> parallelLevels;
  bool hasComplexMixing = false;
};

struct SeparatedResources {
  std::vector<Value> gpuTokens;
  std::vector<Value> culibsStreams;
  int gpuTokenCount = 0;
  int culibsStreamCount = 0;
};

struct ExplicitSyncResources {
  std::vector<Value> gpuTokens;
  std::vector<Value> culibsStreams;
  std::vector<EnhancedDependencyNode*> nodes;
  
  int gpuTokenCount = 0;
  int culibsStreamCount = 0;
};

enum class SegmentNodeType {
  CuLibsSegment,
  KernelSegment,
  MixedSingleLevel,
  LoopSegment
};

struct ExecutionSegmentEnhanced {
  SegmentNodeType type;
  std::vector<int> levels;
  std::vector<EnhancedDependencyNode*> nodes;
  int maxParallelism = 1;
  bool needsSyncBefore = false;
  bool needsSyncAfter = false;
};

struct SegmentBasedSubgraphAnalysis {
  int maxCulibsParallelism = 0;
  int maxKernelParallelism = 0;
  bool hasOnlySegments = true;
  std::vector<ExecutionSegmentEnhanced> segments;
};

struct SegmentBasedResources {
  std::vector<Value> culibsStreams;
  std::vector<Value> kernelTokens;
  std::vector<EnhancedDependencyNode*> nodes;
  SegmentBasedSubgraphAnalysis analysis;
};

struct ParallelGroupSegmentResources {
  std::map<int, SegmentBasedResources> subgraphResources;
  std::vector<Value> allFinalTokens;
};

// ===== Main Functions =====

void reorganizeIRWithEnhancedScheduling_plus(func::FuncOp funcOp, EnhancedDependencyGraph &graph);

void integrateEnhancedIRReorganization_plus(func::FuncOp funcOp);

void reorganizeGPUModules_plus(ModuleOp moduleOp, EnhancedDependencyGraph& graph);

// ===== Resource Management Functions =====

SubgraphAnalysis analyzeSubgraphExecutionPattern_plus(
    const std::vector<EnhancedDependencyNode*>& nodes);

// ===== Subgraph Processing Functions =====

std::vector<Value> processSubgraphNodes_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    Value subgraphStream,
    std::vector<Value>& availableTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

std::vector<ExecutionSegment> identifyExecutionSegments_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& nodesByLevel);

std::vector<Value> processParallelSegment_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    std::vector<Value>& availableTokens,
    int& tokenIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

Value processSequentialSegment_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    Value sharedStream,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

// ===== Node Processing Functions =====

Value processKernelNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                    IRMapping& mapper, Value waitToken, 
                                    llvm::DenseSet<Operation*>& processedOps);

Value processKernelNodeSequential_plus(EnhancedDependencyNode* node, OpBuilder& builder,
                                      IRMapping& mapper, Value dependentToken,
                                      llvm::DenseSet<Operation*>& processedOps);

void processLoopNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                 IRMapping& mapper, llvm::SmallVector<Operation*, 16>& allocaOps,
                                 llvm::DenseSet<Operation*>& processedOps);

void processCuLibsNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                   IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps);

void processCuLibsNodeWithStreamEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder,
                                             IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps,
                                             Value stream);

// ===== Helper Functions =====

void collectDependentOpsEnhanced_plus(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
                                     const llvm::DenseSet<Operation*>& processedOps);

void analyzeGPUWaitOperations_plus(func::FuncOp funcOp, EnhancedDependencyGraph& graph,
                                  llvm::DenseMap<Operation*, bool>& waitOpShouldKeep);

void copyPrefixOperations_plus(Block* oldBlock, Block* newBlock, OpBuilder& builder,
                              IRMapping& mapper, EnhancedDependencyGraph& graph,
                              llvm::DenseSet<Operation*>& processedOps,
                              const llvm::DenseMap<Operation*, bool>& waitOpShouldKeep,
                              llvm::SmallVector<Operation*, 16>& allocaOps);

void finalizeIRReorganization_plus(
    Block* oldBlock, Block* newBlock, OpBuilder& builder,
    IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps,
    const llvm::DenseMap<Operation*, bool>& waitOpShouldKeep,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    func::FuncOp funcOp, func::FuncOp descriptorReturnFunc,
    func::FuncOp workspaceReturnFunc,
    const std::map<int, std::map<int, std::vector<EnhancedDependencyNode*>>>& nodesByParallelGroupAndSubgraph);

void replaceFunctionBody_plus(func::FuncOp funcOp, Block* oldBlock, Block* newBlock, IRMapping& mapper);

void printEnhancedSchedulingSummary_plus(const EnhancedDependencyGraph& graph);

// ===== Function Declarations from Original Code (reused) =====

func::FuncOp ensureDescriptorReturnFuncDecl(ModuleOp moduleOp, OpBuilder& builder);

func::FuncOp ensureWorkspaceReturnFuncDecl(ModuleOp moduleOp, OpBuilder& builder);

void insertDescriptorReturnCall(OpBuilder& builder, Location loc, 
                               func::FuncOp descriptorReturnFunc);

void insertWorkspaceReturnCall(OpBuilder& builder, Location loc, 
                              func::FuncOp workspaceReturnFunc);

bool shouldMoveWithCuLibs(Operation* op);

void finalizeSeparatedParallelGroup_plus(
    SeparatedParallelGroupResources& resources,
    const std::vector<Value>& groupLevelGpuTokens,
    OpBuilder& builder, Block* newBlock,
    func::FuncOp descriptorReturnFunc,
    func::FuncOp workspaceReturnFunc,
    Location loc,
    int parallelGroupId,
    size_t totalGroups);

void processParallelGroupWithSeparatedResources_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& subgraphs,
    SeparatedParallelGroupResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    std::vector<Value>& groupLevelGpuTokens,
    Location loc);

std::vector<Value> processSubgraphWithSeparatedResources_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    const std::vector<Value>& gpuTokens,
    const std::vector<Value>& culibsStreams,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

SeparatedParallelGroupResources createSeparatedResources_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& subgraphs,
    OpBuilder& builder, Block* newBlock, Location loc);

SubgraphResourceAnalysis analyzeSubgraphResourceNeeds_plus(const std::vector<EnhancedDependencyNode*>& nodes);

void finalizeSubgraphResources_plus(SeparatedResources& resources, OpBuilder& builder, Location loc);

SeparatedResources createSeparatedResourcesSimple_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    OpBuilder& builder, Location loc);

void processParallelGroupWithExplicitSync_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& subgraphs,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    std::vector<Value>& groupLevelTokens,
    Location loc);

SubgraphChainAnalysis analyzeSubgraphChains_plus(const std::vector<EnhancedDependencyNode*>& nodes);

std::vector<Value> processSubgraphWithExplicitSync_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    SeparatedResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

std::vector<Value> processMixedSingleNodeChain_plus(
    const SingleNodeChain& chain,
    SeparatedResources& resources,
    int& gpuTokenIndex, int& culibsStreamIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

std::vector<Value> processPureSingleNodeChain_plus(
    const SingleNodeChain& chain,
    SeparatedResources& resources,
    int& gpuTokenIndex, int& culibsStreamIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

std::vector<ChainSegment> analyzeAndSplitChains_plus(
    const std::vector<EnhancedDependencyNode*>& nodes);

std::vector<ChainSegment> splitChainAtKernels_plus(const SingleNodeChain& chain);

ExplicitSyncResources createExplicitSyncResources_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    OpBuilder& builder, Block* newBlock, Location loc);

std::vector<Value> processGpuSegment_plus(
    const ChainSegment& segment,
    const std::vector<Value>& gpuTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc);

void processCuLibsSegment_plus(
    const ExecutionSegmentEnhanced& segment,
    const std::vector<Value>& culibsStreams,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc);

std::vector<ExecutionSegmentEnhanced> identifyExecutionSegmentsEnhanced_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& nodesByLevel);

SegmentBasedSubgraphAnalysis analyzeSubgraphSegments_plus(const std::vector<EnhancedDependencyNode*>& nodes);

SegmentBasedResources createSegmentBasedResources_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    OpBuilder& builder, Block* newBlock, Location loc);

std::vector<Value> processSubgraphWithSegments_plus(
    SegmentBasedResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

std::vector<Value> processKernelSegment_plus(
    const ExecutionSegmentEnhanced& segment,
    const std::vector<Value>& kernelTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc);

std::vector<Value> processMixedLevelWithExplicitSync_plus(
    const ExecutionSegmentEnhanced& segment,
    SegmentBasedResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc);

void synchronizeAllActiveResources_plus(SegmentBasedResources& resources, 
                                       const std::vector<Value>& activeTokens,
                                       OpBuilder& builder, Location loc);

void synchronizeCulibsStreams_plus(const ExecutionSegmentEnhanced& segment,
                                 const std::vector<Value>& culibsStreams,
                                 OpBuilder& builder, Location loc);

} // namespace onnx_mlir

#endif // ENHANCED_IR_REORGANIZATION_H