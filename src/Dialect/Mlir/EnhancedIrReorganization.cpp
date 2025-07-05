#include "EnhancedIrReorganization.h"
#include "StreamUnificationPass.cpp"

using namespace mlir;

namespace onnx_mlir {

// ===== Main Functions Implementation =====

void reorganizeIRWithEnhancedScheduling_plus(func::FuncOp funcOp, EnhancedDependencyGraph &graph) {
  OpBuilder builder(funcOp.getContext());
  
  if (funcOp.getBody().empty()) {
    llvm::errs() << "Warning: Function body is empty, skipping reorganization\n";
    return;
  }
  
  // Get the module to ensure function declarations
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    llvm::errs() << "Error: Cannot find parent module for function\n";
    return;
  }
  
  // Ensure function declarations
  func::FuncOp descriptorReturnFunc = ensureDescriptorReturnFuncDecl(moduleOp, builder);
  func::FuncOp workspaceReturnFunc = ensureWorkspaceReturnFuncDecl(moduleOp, builder);
  
  IRMapping mapper;
  
  std::map<int, std::map<int, std::vector<EnhancedDependencyNode*>>> nodesByParallelGroupAndSubgraph;
  
  for (auto& nodePtr : graph.nodes) {
    EnhancedDependencyNode* node = nodePtr.get();
    nodesByParallelGroupAndSubgraph[node->parallelGroupId][node->subgraphId].push_back(node);
  }
  
  for (auto& [parallelGroupId, subgraphs] : nodesByParallelGroupAndSubgraph) {
    for (auto& [subgraphId, nodes] : subgraphs) {
      std::sort(nodes.begin(), nodes.end(), 
          [](EnhancedDependencyNode* a, EnhancedDependencyNode* b) {
              return a->intraSubgraphLevel < b->intraSubgraphLevel;
          });
    }
  }
  
  if (nodesByParallelGroupAndSubgraph.empty()) {
    llvm::errs() << "Warning: No nodes to reorganize, skipping\n";
    return;
  }
  
  Block* oldBlock = &funcOp.getBody().front();
  Block* newBlock = new Block();
  
  for (auto &blockArg : oldBlock->getArguments()) {
    auto newArg = newBlock->addArgument(blockArg.getType(), blockArg.getLoc());
    mapper.map(blockArg, newArg);
  }
  
  llvm::DenseSet<Operation*> processedOps;

  llvm::SmallVector<Operation*, 16> allocaOps;
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    allocaOps.push_back(allocaOp);
  });
  
  llvm::DenseMap<Operation*, bool> waitOpShouldKeep;
  analyzeGPUWaitOperations_plus(funcOp, graph, waitOpShouldKeep);
  
  copyPrefixOperations_plus(oldBlock, newBlock, builder, mapper, graph, processedOps, waitOpShouldKeep, allocaOps);

  for (auto& [parallelGroupId, subgraphs] : nodesByParallelGroupAndSubgraph) {
    // llvm::errs() << "\n=== Processing Parallel Group " << parallelGroupId 
    //              << " with " << subgraphs.size() << " subgraphs ===\n";
    
    builder.setInsertionPointToEnd(newBlock);
    // insertParallelGroupStartMarker(builder, funcOp.getLoc(), parallelGroupId, subgraphs.size());
    
    auto groupResources = createSeparatedResources_plus(subgraphs, builder, newBlock, funcOp.getLoc());
    
    std::vector<Value> groupLevelGpuTokens;
    processParallelGroupWithSeparatedResources_plus(subgraphs, groupResources, builder, mapper, 
                                                    processedOps, allocaOps, groupLevelGpuTokens, funcOp.getLoc());
    
    finalizeSeparatedParallelGroup_plus(groupResources, groupLevelGpuTokens, builder, newBlock, 
                                       descriptorReturnFunc, workspaceReturnFunc, funcOp.getLoc(),
                                       parallelGroupId, nodesByParallelGroupAndSubgraph.size());
  }

  finalizeIRReorganization_plus(oldBlock, newBlock, builder, mapper, processedOps, 
                               waitOpShouldKeep, allocaOps, funcOp, descriptorReturnFunc, 
                               workspaceReturnFunc, nodesByParallelGroupAndSubgraph);
}

void integrateEnhancedIRReorganization_plus(func::FuncOp funcOp) {
  llvm::errs() << "\n" << std::string(70, '=') << "\n";
  llvm::errs() << "ENHANCED DEPENDENCY GRAPH IR REORGANIZATION\n";
  llvm::errs() << "Function: " << funcOp.getName() << "\n";
  llvm::errs() << std::string(70, '=') << "\n";

  try {
    // Step 1: Build enhanced dependency graph
    // llvm::errs() << "\n[STEP 1] Building Enhanced Dependency Graph...\n";
    auto enhancedGraph = buildEnhancedDependencyGraph(funcOp);
    
    if (enhancedGraph->nodes.empty()) {
      llvm::errs() << "No nodes found in function, skipping IR reorganization.\n";
      return;
    }

    // Step 2: Reorganize IR based on enhanced scheduling
    llvm::errs() << "\n[STEP 2] Reorganizing IR with Enhanced Scheduling...\n";
    reorganizeIRWithEnhancedScheduling_plus(funcOp, *enhancedGraph);

    // Step 3: Reorganize GPU modules for optimized kernel organization
    llvm::errs() << "\n[STEP 3] Reorganizing GPU Modules...\n";
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (moduleOp) {
      reorganizeGPUModules_plus(moduleOp, *enhancedGraph);
    }

    // Step 4: Print summary
    // llvm::errs() << "\n[STEP 4] Printing Enhanced Scheduling Summary...\n";
    // printEnhancedSchedulingSummary_plus(*enhancedGraph);

    llvm::errs() << "\n" << std::string(70, '=') << "\n";
    llvm::errs() << "ENHANCED IR REORGANIZATION COMPLETED SUCCESSFULLY\n";
    llvm::errs() << std::string(70, '=') << "\n\n";

  } catch (const std::exception& e) {
    llvm::errs() << "Error during enhanced IR reorganization: " << e.what() << "\n\n";
  } catch (...) {
    llvm::errs() << "Unknown error during enhanced IR reorganization\n\n";
  }
}

void reorganizeGPUModules_plus(ModuleOp moduleOp, EnhancedDependencyGraph& graph) {
  OpBuilder builder(moduleOp.getContext());
  
  // Scan all existing modules and functions
  llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
  std::map<std::pair<std::string, std::string>, gpu::GPUFuncOp> funcMap;
  
  moduleOp.walk([&](gpu::GPUModuleOp op) {
    allModules.push_back(op);
    
    // Collect all functions in this module
    std::string moduleName = op.getName().str();
    op.walk([&](gpu::GPUFuncOp funcOp) {
      std::string funcName = funcOp.getName().str();
      funcMap[{moduleName, funcName}] = funcOp;
    });
  });
  
  // Exit if no modules
  if (allModules.empty())
    return;
  
  // Collect all kernel nodes from the enhanced dependency graph
  llvm::SmallVector<EnhancedDependencyNode*, 16> allKernels;
  for (const auto& nodePtr : graph.nodes) {
    EnhancedDependencyNode* node = nodePtr.get();
    if (node->type == NodeType::Kernel) {
      allKernels.push_back(node);
    }
  }
  
  // Exit if no kernels to process
  if (allKernels.empty()) {
    // Still need to clean up old modules even if no kernels
    for (auto moduleOp : allModules) {
      moduleOp.erase();
    }
    return;
  }
  
  // Create renaming map: <old module name, old function name> -> <new module name, new function name>
  using ModuleFuncKey = std::pair<std::string, std::string>;
  std::map<ModuleFuncKey, ModuleFuncKey> renameMap;
  
  // Create a single combined module for all kernels with enhanced naming
  std::string combinedModuleName = "enhanced_scheduled_kernels";
  builder.setInsertionPointToStart(moduleOp.getBody());
  
  auto combinedModule = builder.create<gpu::GPUModuleOp>(
      moduleOp.getLoc(),
      builder.getStringAttr(combinedModuleName));
  
  builder.setInsertionPointToStart(combinedModule.getBody());
  
  // Function counter to ensure uniqueness across all kernels
  int globalFuncCounter = 0;
  
  // Copy all kernel functions to the combined module with enhanced naming
  for (EnhancedDependencyNode* kernel : allKernels) {
    std::string oldModuleName = kernel->kernelModuleName;
    std::string oldFuncName = kernel->kernelName;
    
    // Find the original function
    auto funcKey = std::make_pair(oldModuleName, oldFuncName);
    auto funcIt = funcMap.find(funcKey);
    
    if (funcIt != funcMap.end()) {
      // Create a new function name that includes scheduling information
      std::string newFuncName = "sched_P" + std::to_string(kernel->parallelGroupId) + 
                                "_S" + std::to_string(kernel->subgraphId) +
                                "_L" + std::to_string(kernel->intraSubgraphLevel) +
                                "_" + std::to_string(globalFuncCounter++);
      
      // Clone the function to the combined module
      auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*funcIt->second));
      
      // Set the new function name
      clonedFunc.setName(newFuncName);
      
      // Save mapping relationship
      renameMap[funcKey] = {combinedModuleName, newFuncName};
      
    //   llvm::errs() << "  Renamed kernel: " << oldFuncName << " -> " << newFuncName 
    //                << " (P" << kernel->parallelGroupId 
    //                << ",S" << kernel->subgraphId 
    //                << ",L" << kernel->intraSubgraphLevel << ")\n";
    }
  }
  
  // Update all kernel launch references to point to the combined module
  moduleOp.walk([&](gpu::LaunchFuncOp op) {
    std::string oldModuleName = op.getKernelModuleName().str();
    std::string oldFuncName = op.getKernelName().str();
    
    auto funcKey = std::make_pair(oldModuleName, oldFuncName);
    auto renameIt = renameMap.find(funcKey);
    
    if (renameIt != renameMap.end()) {
      std::string newModuleName = renameIt->second.first;
      std::string newFuncName = renameIt->second.second;
      
      // Create new symbol reference pointing to the combined module
      auto newKernel = SymbolRefAttr::get(
          builder.getContext(),
          StringAttr::get(builder.getContext(), newModuleName),
          {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
      // Update kernel reference
      op->setAttr("kernel", newKernel);
    }
  });
  
  // Delete all old modules
  for (auto moduleOp : allModules) {
    moduleOp.erase();
  }
  
  llvm::errs() << "Enhanced GPU module reorganization completed. Created combined module: " 
               << combinedModuleName << "\n";
}

// ===== Resource Management Functions Implementation =====

std::vector<ExecutionSegmentEnhanced> identifyExecutionSegmentsEnhanced_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& nodesByLevel) {
  
  std::vector<ExecutionSegmentEnhanced> segments;
  
  auto levelIter = nodesByLevel.begin();
  while (levelIter != nodesByLevel.end()) {
    int currentLevel = levelIter->first;
    const auto& currentNodes = levelIter->second;
    
    int culibsCount = 0, kernelCount = 0, loopCount = 0;
    for (auto* node : currentNodes) {
      switch (node->type) {
        case NodeType::CuLibs: culibsCount++; break;
        case NodeType::Kernel: kernelCount++; break;
        case NodeType::Loop: loopCount++; break;
      }
    }
    
    bool isMixedLevel = (culibsCount > 0 && kernelCount > 0);
    bool isPureCuLibs = (culibsCount > 0 && kernelCount == 0);
    bool isPureKernel = (kernelCount > 0 && culibsCount == 0);
    
    if (isMixedLevel) {
      ExecutionSegmentEnhanced segment;
      segment.type = SegmentNodeType::MixedSingleLevel;
      segment.levels.push_back(currentLevel);
      segment.nodes = currentNodes;
      segment.maxParallelism = currentNodes.size();
      segment.needsSyncBefore = true;
      segment.needsSyncAfter = true;
      segments.push_back(segment);
      
    //   llvm::errs() << "        Segment: Mixed level " << currentLevel 
    //                << " (" << culibsCount << " CuLibs + " << kernelCount << " Kernels)\n";
      
      ++levelIter;
      
    } else if (isPureCuLibs) {
      ExecutionSegmentEnhanced segment;
      segment.type = SegmentNodeType::CuLibsSegment;
      segment.levels.push_back(currentLevel);
      segment.nodes.insert(segment.nodes.end(), currentNodes.begin(), currentNodes.end());
      segment.maxParallelism = currentNodes.size();
      
      ++levelIter;
      
      while (levelIter != nodesByLevel.end()) {
        const auto& nextNodes = levelIter->second;
        int nextCulibs = 0, nextKernels = 0;
        for (auto* node : nextNodes) {
          if (node->type == NodeType::CuLibs) nextCulibs++;
          else if (node->type == NodeType::Kernel) nextKernels++;
        }
        
        if (nextCulibs > 0 && nextKernels == 0) {
          segment.levels.push_back(levelIter->first);
          segment.nodes.insert(segment.nodes.end(), nextNodes.begin(), nextNodes.end());
          segment.maxParallelism = std::max(segment.maxParallelism, (int)nextNodes.size());
          ++levelIter;
        } else {
          break;
        }
      }
      
      segments.push_back(segment);
      
    //   llvm::errs() << "        Segment: CuLibs levels " << segment.levels.front() 
    //                << "-" << segment.levels.back() << " (" << segment.nodes.size() << " nodes)\n";
      
    } else if (isPureKernel) {
      ExecutionSegmentEnhanced segment;
      segment.type = SegmentNodeType::KernelSegment;
      segment.levels.push_back(currentLevel);
      segment.nodes.insert(segment.nodes.end(), currentNodes.begin(), currentNodes.end());
      segment.maxParallelism = currentNodes.size();
      
      ++levelIter;
      
      while (levelIter != nodesByLevel.end()) {
        const auto& nextNodes = levelIter->second;
        int nextCulibs = 0, nextKernels = 0;
        for (auto* node : nextNodes) {
          if (node->type == NodeType::CuLibs) nextCulibs++;
          else if (node->type == NodeType::Kernel) nextKernels++;
        }
        
        if (nextKernels > 0 && nextCulibs == 0) {
          segment.levels.push_back(levelIter->first);
          segment.nodes.insert(segment.nodes.end(), nextNodes.begin(), nextNodes.end());
          segment.maxParallelism = std::max(segment.maxParallelism, (int)nextNodes.size());
          ++levelIter;
        } else {
          break;
        }
      }
      
      segments.push_back(segment);
      
    //   llvm::errs() << "        Segment: Kernel levels " << segment.levels.front() 
    //                << "-" << segment.levels.back() << " (" << segment.nodes.size() << " nodes)\n";
    } else {
      ++levelIter;
    }
  }
  
  for (int i = 0; i < segments.size(); ++i) {
    if (i > 0) {
      auto& prevSegment = segments[i-1];
      auto& currSegment = segments[i];
      
      if (prevSegment.type == SegmentNodeType::CuLibsSegment && 
          currSegment.type == SegmentNodeType::KernelSegment) {
        prevSegment.needsSyncAfter = true;
      }
      
      if (prevSegment.type == SegmentNodeType::KernelSegment && 
          currSegment.type == SegmentNodeType::CuLibsSegment) {
        prevSegment.needsSyncAfter = true;
      }
      
      if (currSegment.type == SegmentNodeType::MixedSingleLevel) {
        if (i > 0) prevSegment.needsSyncAfter = true;
        currSegment.needsSyncBefore = true;
      }
    }
  }
  
  return segments;
}

SegmentBasedSubgraphAnalysis analyzeSubgraphSegments_plus(const std::vector<EnhancedDependencyNode*>& nodes) {
  SegmentBasedSubgraphAnalysis analysis;
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
  analysis.segments = identifyExecutionSegmentsEnhanced_plus(nodesByLevel);
  
  for (const auto& segment : analysis.segments) {
    switch (segment.type) {
      case SegmentNodeType::CuLibsSegment:
        analysis.maxCulibsParallelism = std::max(analysis.maxCulibsParallelism, segment.maxParallelism);
        break;
      case SegmentNodeType::KernelSegment:
        analysis.maxKernelParallelism = std::max(analysis.maxKernelParallelism, segment.maxParallelism);
        break;
      case SegmentNodeType::MixedSingleLevel:
        analysis.hasOnlySegments = false;
        int culibsInMixed = 0, kernelsInMixed = 0;
        for (auto* node : segment.nodes) {
          if (node->type == NodeType::CuLibs) culibsInMixed++;
          else if (node->type == NodeType::Kernel) kernelsInMixed++;
        }
        analysis.maxCulibsParallelism = std::max(analysis.maxCulibsParallelism, culibsInMixed);
        analysis.maxKernelParallelism = std::max(analysis.maxKernelParallelism, kernelsInMixed);
        break;
    }
  }
  
//   llvm::errs() << "    Segment analysis: CuLibs max=" << analysis.maxCulibsParallelism
//                << ", Kernel max=" << analysis.maxKernelParallelism
//                << ", segments=" << analysis.segments.size()
//                << ", pure segments=" << analysis.hasOnlySegments << "\n";
  
  return analysis;
}

SegmentBasedResources createSegmentBasedResources_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    OpBuilder& builder, Block* newBlock, Location loc) {
  
  SegmentBasedResources resources;
  resources.nodes = nodes;
  
  builder.setInsertionPointToEnd(newBlock);
  
  resources.analysis = analyzeSubgraphSegments_plus(nodes);
  
  if (resources.analysis.maxCulibsParallelism > 0) {
    for (int i = 0; i < resources.analysis.maxCulibsParallelism; ++i) {
      auto streamCreateOp = builder.create<func::CallOp>(
          loc, "mgpuStreamCreate",
          TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
          ValueRange{});
      
      builder.create<func::CallOp>(
          loc, "mgpuCreateHandlesForStream",
          TypeRange{}, ValueRange{streamCreateOp.getResult(0)});
      
      resources.culibsStreams.push_back(streamCreateOp.getResult(0));
    }
    
    // llvm::errs() << "  Created " << resources.analysis.maxCulibsParallelism << " CuLibs streams\n";
  }
  
  if (resources.analysis.maxKernelParallelism > 0) {
    for (int i = 0; i < resources.analysis.maxKernelParallelism; ++i) {
      auto waitOp = builder.create<gpu::WaitOp>(
          loc, builder.getType<gpu::AsyncTokenType>(), ValueRange{});
      resources.kernelTokens.push_back(waitOp.getAsyncToken());
    }
    
    // llvm::errs() << "  Created " << resources.analysis.maxKernelParallelism << " GPU tokens\n";
  }
  
  return resources;
}

std::vector<Value> processSubgraphWithSegments_plus(
    SegmentBasedResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
  builder.setInsertionPointToEnd(builder.getBlock());
  
//   llvm::errs() << "      Processing subgraph with " << resources.analysis.segments.size() << " segments\n";
  
  std::vector<Value> finalTokens;
  
  for (int segIdx = 0; segIdx < resources.analysis.segments.size(); ++segIdx) {
    const auto& segment = resources.analysis.segments[segIdx];
    
    // llvm::errs() << "        Processing segment " << segIdx << " (type: " 
    //              << (int)segment.type << ", levels: " << segment.levels.size() << ")\n";
    
    if (segment.needsSyncBefore) {
    //   insertSegmentSyncMarker_plus(builder, loc, segIdx, "BEFORE", segment.type);
      synchronizeAllActiveResources_plus(resources, finalTokens, builder, loc);
      finalTokens.clear();
    }
    
    std::vector<Value> segmentTokens;
    
    switch (segment.type) {
      case SegmentNodeType::CuLibsSegment:
        processCuLibsSegment_plus(segment, resources.culibsStreams, builder, mapper, processedOps, loc);
        break;
        
      case SegmentNodeType::KernelSegment:
        segmentTokens = processKernelSegment_plus(segment, resources.kernelTokens, builder, mapper, processedOps, loc);
        break;
        
      case SegmentNodeType::MixedSingleLevel:
        segmentTokens = processMixedLevelWithExplicitSync_plus(segment, resources, builder, mapper, processedOps, allocaOps, loc);
        break;
    }
    
    if (segment.needsSyncAfter) {
    //   insertSegmentSyncMarker_plus(builder, loc, segIdx, "AFTER", segment.type);
      
      if (segment.type == SegmentNodeType::CuLibsSegment) {
        synchronizeCulibsStreams_plus(segment, resources.culibsStreams, builder, loc);
      } else if (segment.type == SegmentNodeType::KernelSegment) {
        if (!segmentTokens.empty()) {
          builder.create<gpu::WaitOp>(loc, TypeRange{}, segmentTokens);
        }
      }
    } else {
      finalTokens.insert(finalTokens.end(), segmentTokens.begin(), segmentTokens.end());
    }
  }
  
  return finalTokens;
}

std::vector<Value> processKernelSegment_plus(
    const ExecutionSegmentEnhanced& segment,
    const std::vector<Value>& kernelTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc) {
  
//   llvm::errs() << "          Kernel segment: " << segment.nodes.size() << " nodes\n";
  
  std::vector<Value> segmentTokens;
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : segment.nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
  for (const auto& [level, levelNodes] : nodesByLevel) {
    // insertLevelStartMarker(builder, loc, level, levelNodes.size());
    
    std::vector<Value> levelTokens;
    int tokenIndex = 0;
    
    for (auto* node : levelNodes) {
      if (node->type == NodeType::Kernel && tokenIndex < kernelTokens.size()) {
        // insertNodeExecutionMarker(builder, loc, node, tokenIndex);
        
        Value waitToken = kernelTokens[tokenIndex++];
        Value kernelToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
        
        if (kernelToken) {
          levelTokens.push_back(kernelToken);
        }
        
        // llvm::errs() << "            Kernel " << node->kernelName 
        //              << " -> token[" << (tokenIndex-1) << "]\n";
      }
    }
    
    bool isLastLevel = (level == nodesByLevel.rbegin()->first);
    if (levelNodes.size() > 1 && !isLastLevel) {
      if (!levelTokens.empty()) {
        builder.create<gpu::WaitOp>(loc, TypeRange{}, levelTokens);
      }
    } else {
      segmentTokens.insert(segmentTokens.end(), levelTokens.begin(), levelTokens.end());
    }
  }
  
  return segmentTokens;
}

std::vector<Value> processMixedLevelWithExplicitSync_plus(
    const ExecutionSegmentEnhanced& segment,
    SegmentBasedResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
//   llvm::errs() << "          Mixed level: " << segment.nodes.size() << " nodes (explicit sync)\n";
  
  std::vector<EnhancedDependencyNode*> culibsNodes, kernelNodes;
  for (auto* node : segment.nodes) {
    if (node->type == NodeType::CuLibs) {
      culibsNodes.push_back(node);
    } else if (node->type == NodeType::Kernel) {
      kernelNodes.push_back(node);
    }
  }
  
  std::vector<Value> mixedLevelTokens;
  
  if (!culibsNodes.empty()) {
    // llvm::errs() << "            Processing " << culibsNodes.size() << " CuLibs nodes first\n";
    
    int streamIndex = 0;
    for (auto* node : culibsNodes) {
      if (streamIndex < resources.culibsStreams.size()) {
        // insertNodeExecutionMarker(builder, loc, node, streamIndex);
        
        Value assignedStream = resources.culibsStreams[streamIndex++];
        processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, assignedStream);
      }
    }
    
    // insertSynchronizationMarker(builder, loc, segment.levels[0], culibsNodes.size());
    for (int i = 0; i < culibsNodes.size() && i < resources.culibsStreams.size(); ++i) {
      builder.create<func::CallOp>(
          loc, "mgpuStreamSynchronize",
          TypeRange{}, ValueRange{resources.culibsStreams[i]});
    }
  }
  
  if (!kernelNodes.empty()) {
    // llvm::errs() << "            Processing " << kernelNodes.size() << " Kernel nodes after CuLibs sync\n";
    
    int tokenIndex = 0;
    for (auto* node : kernelNodes) {
      if (tokenIndex < resources.kernelTokens.size()) {
        // insertNodeExecutionMarker(builder, loc, node, tokenIndex);
        
        Value waitToken = resources.kernelTokens[tokenIndex++];
        Value kernelToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
        
        if (kernelToken) {
          mixedLevelTokens.push_back(kernelToken);
        }
      }
    }
  }
  
  return mixedLevelTokens;
}

void synchronizeAllActiveResources_plus(SegmentBasedResources& resources, 
                                       const std::vector<Value>& activeTokens,
                                       OpBuilder& builder, Location loc) {
  if (!activeTokens.empty()) {
    builder.create<gpu::WaitOp>(loc, TypeRange{}, activeTokens);
  }
  
  for (auto stream : resources.culibsStreams) {
    builder.create<func::CallOp>(
        loc, "mgpuStreamSynchronize",
        TypeRange{}, ValueRange{stream});
  }
}

void synchronizeCulibsStreams_plus(const ExecutionSegmentEnhanced& segment,
                                 const std::vector<Value>& culibsStreams,
                                 OpBuilder& builder, Location loc) {
  int streamsUsed = std::min(segment.maxParallelism, (int)culibsStreams.size());
  
  for (int i = 0; i < streamsUsed; ++i) {
    builder.create<func::CallOp>(
        loc, "mgpuStreamSynchronize",
        TypeRange{}, ValueRange{culibsStreams[i]});
  }
}

std::vector<ChainSegment> splitChainAtKernels_plus(const SingleNodeChain& chain) {
  std::vector<ChainSegment> segments;
  
  if (!chain.hasKernels) {
    ChainSegment segment;
    segment.startLevel = chain.startLevel;
    segment.endLevel = chain.endLevel;
    segment.nodes = chain.nodes;
    segment.isKernelSegment = false;
    segments.push_back(segment);
    
    // llvm::errs() << "        Pure CuLibs chain: levels " << chain.startLevel 
    //              << "-" << chain.endLevel << "\n";
    return segments;
  }
  
  int segmentStart = 0;
  
  for (int kernelPos : chain.kernelPositions) {
    if (kernelPos > segmentStart) {
      ChainSegment culibsSegment;
      culibsSegment.startLevel = chain.startLevel + segmentStart;
      culibsSegment.endLevel = chain.startLevel + kernelPos - 1;
      culibsSegment.nodes.assign(chain.nodes.begin() + segmentStart, 
                                chain.nodes.begin() + kernelPos);
      culibsSegment.isKernelSegment = false;
      culibsSegment.needsSyncAfter = true;
      segments.push_back(culibsSegment);
      
    //   llvm::errs() << "        CuLibs segment before kernel: levels " 
    //                << culibsSegment.startLevel << "-" << culibsSegment.endLevel << "\n";
    }
    
    ChainSegment kernelSegment;
    kernelSegment.startLevel = kernelSegment.endLevel = chain.startLevel + kernelPos;
    kernelSegment.nodes.push_back(chain.nodes[kernelPos]);
    kernelSegment.isKernelSegment = true;
    kernelSegment.needsSyncBefore = (segmentStart < kernelPos);
    kernelSegment.needsSyncAfter = (kernelPos < chain.nodes.size() - 1);
    segments.push_back(kernelSegment);
    
    // llvm::errs() << "        Kernel segment: level " << kernelSegment.startLevel << "\n";
    
    segmentStart = kernelPos + 1;
  }
  
  if (segmentStart < chain.nodes.size()) {
    ChainSegment culibsSegment;
    culibsSegment.startLevel = chain.startLevel + segmentStart;
    culibsSegment.endLevel = chain.endLevel;
    culibsSegment.nodes.assign(chain.nodes.begin() + segmentStart, chain.nodes.end());
    culibsSegment.isKernelSegment = false;
    culibsSegment.needsSyncBefore = true;
    segments.push_back(culibsSegment);
    
    // llvm::errs() << "        CuLibs segment after kernel: levels " 
    //              << culibsSegment.startLevel << "-" << culibsSegment.endLevel << "\n";
  }
  
  return segments;
}

SeparatedResources createSeparatedResourcesSimple_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    OpBuilder& builder, Location loc) {
  
  SeparatedResources resources;
  builder.setInsertionPointToEnd(builder.getBlock());
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
  int maxGpuParallel = 0, maxCulibsParallel = 0;
  bool hasGpu = false, hasCulibs = false;
  
  for (const auto& [level, levelNodes] : nodesByLevel) {
    int gpuCount = 0, culibsCount = 0;
    for (auto* node : levelNodes) {
      if (node->type == NodeType::Kernel) {
        gpuCount++;
        hasGpu = true;
      } else if (node->type == NodeType::CuLibs) {
        culibsCount++;
        hasCulibs = true;
      }
    }
    maxGpuParallel = std::max(maxGpuParallel, gpuCount);
    maxCulibsParallel = std::max(maxCulibsParallel, culibsCount);
  }
  
  if (hasGpu && maxGpuParallel > 0) {
    for (int i = 0; i < maxGpuParallel; ++i) {
      auto waitOp = builder.create<gpu::WaitOp>(
          loc, builder.getType<gpu::AsyncTokenType>(), ValueRange{});
      resources.gpuTokens.push_back(waitOp.getAsyncToken());
    }
    resources.gpuTokenCount = maxGpuParallel;
  }
  
  if (hasCulibs && maxCulibsParallel > 0) {
    for (int i = 0; i < maxCulibsParallel; ++i) {
      auto streamCreateOp = builder.create<func::CallOp>(
          loc, "mgpuStreamCreate",
          TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
          ValueRange{});
      
      builder.create<func::CallOp>(
          loc, "mgpuCreateHandlesForStream",
          TypeRange{}, ValueRange{streamCreateOp.getResult(0)});
      
      resources.culibsStreams.push_back(streamCreateOp.getResult(0));
    }
    resources.culibsStreamCount = maxCulibsParallel;
  }
  
//   llvm::errs() << "      Created " << resources.gpuTokenCount << " GPU tokens, "
//                << resources.culibsStreamCount << " CuLibs streams for current subgraph\n";
  
  return resources;
}

std::vector<Value> processGpuSegment_plus(
    const ChainSegment& segment,
    const std::vector<Value>& gpuTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc) {
  
  std::vector<Value> segmentTokens;
  int tokenIndex = 0;
  
  for (auto* node : segment.nodes) {
    if (node->type == NodeType::Kernel && tokenIndex < gpuTokens.size()) {
    //   insertNodeExecutionMarker(builder, loc, node, tokenIndex);
      
      Value waitToken = gpuTokens[tokenIndex++];
      Value kernelToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
      
      if (kernelToken) {
        segmentTokens.push_back(kernelToken);
      }
      
    //   llvm::errs() << "          GPU kernel " << node->kernelName 
    //                << " using token[" << (tokenIndex-1) << "]\n";
    }
  }
  
  return segmentTokens;
}

void processCuLibsSegment_plus(
    const ExecutionSegmentEnhanced& segment,
    const std::vector<Value>& culibsStreams,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    Location loc) {
  
//   llvm::errs() << "          CuLibs segment: " << segment.nodes.size() << " nodes\n";
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : segment.nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
  for (const auto& [level, levelNodes] : nodesByLevel) {
    // insertLevelStartMarker(builder, loc, level, levelNodes.size());
    
    int streamIndex = 0;
    for (auto* node : levelNodes) {
      if (node->type == NodeType::CuLibs && streamIndex < culibsStreams.size()) {
        // insertNodeExecutionMarker(builder, loc, node, streamIndex);
        
        Value assignedStream = culibsStreams[streamIndex++];
        processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, assignedStream);
        
        // llvm::errs() << "            CuLibs " << node->culibsFunctionName 
        //              << " -> stream[" << (streamIndex-1) << "]\n";
      }
    }
    
    if (levelNodes.size() > 1) {
      for (int i = 0; i < levelNodes.size() && i < culibsStreams.size(); ++i) {
        builder.create<func::CallOp>(
            loc, "mgpuStreamSynchronize",
            TypeRange{}, ValueRange{culibsStreams[i]});
      }
    }
  }
}

std::vector<Value> processPureSingleNodeChain_plus(
    const SingleNodeChain& chain,
    SeparatedResources& resources,
    int& gpuTokenIndex, int& culibsStreamIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
//   llvm::errs() << "        Processing pure single-node chain: " << chain.nodes.size() << " nodes\n";
  
  std::vector<Value> chainTokens;
  Value lastKernelToken;
  
  for (int i = 0; i < chain.nodes.size(); ++i) {
    auto* node = chain.nodes[i];
    
    // insertNodeExecutionMarker(builder, loc, node, 
    //                          node->type == NodeType::Kernel ? gpuTokenIndex : culibsStreamIndex);
    
    if (node->type == NodeType::Kernel && gpuTokenIndex < resources.gpuTokens.size()) {
      ValueRange dependencies = lastKernelToken ? ValueRange{lastKernelToken} : ValueRange{};
      if (dependencies.empty() && gpuTokenIndex < resources.gpuTokens.size()) {
        dependencies = ValueRange{resources.gpuTokens[gpuTokenIndex++]};
      }
      
      lastKernelToken = processKernelNodeSequential_plus(node, builder, mapper, 
                                                        dependencies.empty() ? Value() : dependencies[0], 
                                                        processedOps);
      if (lastKernelToken) chainTokens.push_back(lastKernelToken);
      
    } else if (node->type == NodeType::CuLibs) {
      Value stream = resources.culibsStreams[culibsStreamIndex % resources.culibsStreams.size()];
      processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, stream);
      
    } else if (node->type == NodeType::Loop) {
      processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
    }
  }
  
  return chainTokens;
}

void finalizeSubgraphResources_plus(SeparatedResources& resources, OpBuilder& builder, Location loc) {
  builder.setInsertionPointToEnd(builder.getBlock());
  
  for (Value stream : resources.culibsStreams) {
    builder.create<func::CallOp>(
        loc, "mgpuStreamSynchronize",
        TypeRange{}, ValueRange{stream});
    
    builder.create<func::CallOp>(
        loc, "mgpuStreamDestroy",
        TypeRange{}, ValueRange{stream});
  }
  
  
//   llvm::errs() << "      Cleaned up " << resources.culibsStreams.size() << " CuLibs streams\n";
}

std::vector<Value> processMixedSingleNodeChain_plus(
    const SingleNodeChain& chain,
    SeparatedResources& resources,
    int& gpuTokenIndex, int& culibsStreamIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
//   llvm::errs() << "        Processing mixed single-node chain: " << chain.nodes.size() 
//                << " nodes with " << chain.kernelPositions.size() << " kernels\n";
  
  std::vector<Value> chainTokens;
  
  
  for (int i = 0; i < chain.nodes.size(); ++i) {
    auto* node = chain.nodes[i];
    bool isKernel = (node->type == NodeType::Kernel);
    
    if (isKernel && i > 0) {
      bool hasPrecedingCulibs = false;
      for (int j = 0; j < i; ++j) {
        if (chain.nodes[j]->type == NodeType::CuLibs) {
          hasPrecedingCulibs = true;
          break;
        }
      }
      
      if (hasPrecedingCulibs) {
        // llvm::errs() << "          Syncing CuLibs before kernel at position " << i << "\n";
        // insertSynchronizationMarker(builder, loc, chain.startLevel + i, 1);
        
        for (int s = std::max(0, culibsStreamIndex - 1); s < culibsStreamIndex && s < resources.culibsStreams.size(); ++s) {
          builder.create<func::CallOp>(
              loc, "mgpuStreamSynchronize",
              TypeRange{}, ValueRange{resources.culibsStreams[s]});
        }
      }
    }
    
    // insertNodeExecutionMarker(builder, loc, node, isKernel ? gpuTokenIndex : culibsStreamIndex);
    
    if (isKernel && gpuTokenIndex < resources.gpuTokens.size()) {
      Value waitToken = resources.gpuTokens[gpuTokenIndex++];
      Value kernelToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
      if (kernelToken) chainTokens.push_back(kernelToken);
      
      if (i < chain.nodes.size() - 1) {
        bool hasFollowingCulibs = false;
        for (int j = i + 1; j < chain.nodes.size(); ++j) {
          if (chain.nodes[j]->type == NodeType::CuLibs) {
            hasFollowingCulibs = true;
            break;
          }
        }
        
        if (hasFollowingCulibs) {
        //   llvm::errs() << "          Syncing kernel at position " << i << " before following CuLibs\n";
          builder.create<gpu::WaitOp>(loc, TypeRange{}, ValueRange{kernelToken});
          chainTokens.clear();
        }
      }
      
    } else if (node->type == NodeType::CuLibs && culibsStreamIndex < resources.culibsStreams.size()) {
      Value stream = resources.culibsStreams[culibsStreamIndex % resources.culibsStreams.size()];
      processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, stream);
      
      if (culibsStreamIndex < resources.culibsStreams.size() - 1) {
        culibsStreamIndex++;
      }
      
    } else if (node->type == NodeType::Loop) {
      processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
    }
  }
  
  return chainTokens;
}

SubgraphResourceAnalysis analyzeSubgraphResourceNeeds_plus(const std::vector<EnhancedDependencyNode*>& nodes) {
  SubgraphResourceAnalysis analysis;
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
  analysis.totalLevels = nodesByLevel.size();
  
  for (const auto& [level, levelNodes] : nodesByLevel) {
    int levelGpuNodes = 0;
    int levelCulibsNodes = 0;
    
    for (auto* node : levelNodes) {
      switch (node->type) {
        case NodeType::Kernel:
          levelGpuNodes++;
          analysis.hasGpuKernels = true;
          break;
        case NodeType::CuLibs:
          levelCulibsNodes++;
          analysis.hasCulibsCalls = true;
          break;
        case NodeType::Loop:
          analysis.hasLoops = true;
          break;
      }
    }
    
    analysis.maxGpuParallelism = std::max(analysis.maxGpuParallelism, levelGpuNodes);
    analysis.maxCulibsParallelism = std::max(analysis.maxCulibsParallelism, levelCulibsNodes);
  }
  
//   llvm::errs() << "    Resource analysis: GPU max=" << analysis.maxGpuParallelism
//                << ", CuLibs max=" << analysis.maxCulibsParallelism
//                << ", levels=" << analysis.totalLevels << "\n";
  
  return analysis;
}

SeparatedParallelGroupResources createSeparatedResources_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& subgraphs,
    OpBuilder& builder, Block* newBlock, Location loc) {
  
  SeparatedParallelGroupResources resources;
  builder.setInsertionPointToEnd(newBlock);
  
  for (const auto& [subgraphId, nodes] : subgraphs) {
    resources.subgraphNodes[subgraphId] = nodes;
    
    auto analysis = analyzeSubgraphResourceNeeds_plus(nodes);
    
    if (analysis.hasGpuKernels && analysis.maxGpuParallelism > 0) {
      std::vector<Value> gpuTokens;
      for (int i = 0; i < analysis.maxGpuParallelism; ++i) {
        auto waitOp = builder.create<gpu::WaitOp>(
            loc, builder.getType<gpu::AsyncTokenType>(), ValueRange{});
        gpuTokens.push_back(waitOp.getAsyncToken());
      }
      resources.subgraphGpuTokens[subgraphId] = gpuTokens;
      resources.subgraphGpuTokenCount[subgraphId] = analysis.maxGpuParallelism;
      
    //   llvm::errs() << "  Created " << analysis.maxGpuParallelism << " GPU tokens for subgraph " << subgraphId << "\n";
    }
    
    if (analysis.hasCulibsCalls && analysis.maxCulibsParallelism > 0) {
      std::vector<Value> culibsStreams;
      for (int i = 0; i < analysis.maxCulibsParallelism; ++i) {
        auto streamCreateOp = builder.create<func::CallOp>(
            loc, "mgpuStreamCreate",
            TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
            ValueRange{});
        
        builder.create<func::CallOp>(
            loc, "mgpuCreateHandlesForStream",
            TypeRange{}, ValueRange{streamCreateOp.getResult(0)});
        
        culibsStreams.push_back(streamCreateOp.getResult(0));
      }
      resources.subgraphCulibsStreams[subgraphId] = culibsStreams;
      resources.subgraphCulibsStreamCount[subgraphId] = analysis.maxCulibsParallelism;
      
    //   llvm::errs() << "  Created " << analysis.maxCulibsParallelism << " CuLibs streams for subgraph " << subgraphId << "\n";
    }
  }
  
  return resources;
}

std::vector<Value> processSubgraphWithSeparatedResources_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    const std::vector<Value>& gpuTokens,
    const std::vector<Value>& culibsStreams,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
  builder.setInsertionPointToEnd(builder.getBlock());
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
//   llvm::errs() << "      Processing subgraph with " << nodesByLevel.size() << " levels, "
//                << gpuTokens.size() << " GPU tokens, " << culibsStreams.size() << " CuLibs streams\n";
  
  std::vector<Value> finalGpuTokens;
  
  for (auto levelIter = nodesByLevel.begin(); levelIter != nodesByLevel.end(); ++levelIter) {
    int currentLevel = levelIter->first;
    const auto& levelNodes = levelIter->second;
    
    // llvm::errs() << "        Processing level " << currentLevel << " with " << levelNodes.size() << " nodes\n";
    
    // insertLevelStartMarker(builder, loc, currentLevel, levelNodes.size());
    
    std::vector<Value> currentLevelGpuTokens;
    
    int gpuTokenIndex = 0;
    int culibsStreamIndex = 0;
    
    for (auto* node : levelNodes) {
    //   insertNodeExecutionMarker(builder, loc, node, 
    //                            node->type == NodeType::Kernel ? gpuTokenIndex : culibsStreamIndex);
      
      if (node->type == NodeType::Kernel && gpuTokenIndex < gpuTokens.size()) {
        Value waitToken = gpuTokens[gpuTokenIndex++];
        Value kernelToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
        
        if (kernelToken) {
          currentLevelGpuTokens.push_back(kernelToken);
        }
        
        // llvm::errs() << "          GPU kernel " << node->kernelName 
        //              << " using token[" << (gpuTokenIndex-1) << "]\n";
        
      } else if (node->type == NodeType::CuLibs && culibsStreamIndex < culibsStreams.size()) {
        Value assignedStream = culibsStreams[culibsStreamIndex++];
        processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, assignedStream);
        
        // llvm::errs() << "          CuLibs " << node->culibsFunctionName 
        //              << " using stream[" << (culibsStreamIndex-1) << "]\n";
        
      } else if (node->type == NodeType::Loop) {
        processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
      }
    }
    
    bool isLastLevel = (std::next(levelIter) == nodesByLevel.end());
    
    if (levelNodes.size() > 1) {
      if (!isLastLevel) {
        // insertSynchronizationMarker(builder, loc, currentLevel, levelNodes.size());
        
        if (!currentLevelGpuTokens.empty()) {
        //   llvm::errs() << "        Synchronizing " << currentLevelGpuTokens.size() << " GPU tokens in level " << currentLevel << "\n";
          builder.create<gpu::WaitOp>(loc, TypeRange{}, currentLevelGpuTokens);
        }
        
        if (culibsStreamIndex > 0) {
        //   llvm::errs() << "        Synchronizing " << culibsStreamIndex << " CuLibs streams in level " << currentLevel << "\n";
          for (int i = 0; i < culibsStreamIndex && i < culibsStreams.size(); ++i) {
            builder.create<func::CallOp>(
                loc, "mgpuStreamSynchronize",
                TypeRange{}, ValueRange{culibsStreams[i]});
          }
        }
        
      } else {
        // llvm::errs() << "        Level " << currentLevel << " is final, keeping " 
        //              << currentLevelGpuTokens.size() << " GPU tokens\n";
        finalGpuTokens.insert(finalGpuTokens.end(), currentLevelGpuTokens.begin(), currentLevelGpuTokens.end());
      }
      
    } else {
      if (isLastLevel && !currentLevelGpuTokens.empty()) {
        finalGpuTokens.insert(finalGpuTokens.end(), currentLevelGpuTokens.begin(), currentLevelGpuTokens.end());
      }
    }
  }
  
  return finalGpuTokens;
}

void processParallelGroupWithSeparatedResources_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& subgraphs,
    SeparatedParallelGroupResources& resources,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    std::vector<Value>& groupLevelGpuTokens,
    Location loc) {
  
  for (const auto& [subgraphId, nodes] : subgraphs) {
    // llvm::errs() << "    Processing subgraph " << subgraphId 
    //              << " with " << nodes.size() << " nodes\n";
    
    builder.setInsertionPointToEnd(builder.getBlock());
    // insertSubgraphStartMarker(builder, loc, nodes[0]->parallelGroupId, subgraphId, nodes.size());
    
    std::vector<Value> gpuTokens, culibsStreams;
    
    auto gpuTokensIt = resources.subgraphGpuTokens.find(subgraphId);
    if (gpuTokensIt != resources.subgraphGpuTokens.end()) {
      gpuTokens = gpuTokensIt->second;
    }
    
    auto culibsStreamsIt = resources.subgraphCulibsStreams.find(subgraphId);
    if (culibsStreamsIt != resources.subgraphCulibsStreams.end()) {
      culibsStreams = culibsStreamsIt->second;
    }
    
    auto subgraphFinalTokens = processSubgraphWithSeparatedResources_plus(
        nodes, gpuTokens, culibsStreams, 
        builder, mapper, processedOps, allocaOps, loc);
    
    groupLevelGpuTokens.insert(groupLevelGpuTokens.end(), 
                              subgraphFinalTokens.begin(), subgraphFinalTokens.end());
  }
}

void finalizeSeparatedParallelGroup_plus(
    SeparatedParallelGroupResources& resources,
    const std::vector<Value>& groupLevelGpuTokens,
    OpBuilder& builder, Block* newBlock,
    func::FuncOp descriptorReturnFunc,
    func::FuncOp workspaceReturnFunc,
    Location loc,
    int parallelGroupId,
    size_t totalGroups) {
  
  builder.setInsertionPointToEnd(newBlock);
  
  if (!groupLevelGpuTokens.empty()) {
    // llvm::errs() << "Final sync: " << groupLevelGpuTokens.size() << " GPU tokens for parallel group " << parallelGroupId << "\n";
    builder.create<gpu::WaitOp>(loc, TypeRange{}, groupLevelGpuTokens);
  }
  
  for (auto& [subgraphId, streams] : resources.subgraphCulibsStreams) {
    // llvm::errs() << "Final sync: " << streams.size() << " CuLibs streams for subgraph " << subgraphId << "\n";
    
    for (Value stream : streams) {
      builder.create<func::CallOp>(
          loc, "mgpuStreamSynchronize",
          TypeRange{}, ValueRange{stream});
       
      builder.create<func::CallOp>(
          loc, "mgpuStreamDestroy",
          TypeRange{}, ValueRange{stream});
    }
  }
  
  
  if (parallelGroupId < static_cast<int>(totalGroups) - 1) {
    insertDescriptorReturnCall(builder, loc, descriptorReturnFunc);
    insertWorkspaceReturnCall(builder, loc, workspaceReturnFunc);
  }
  
//   llvm::errs() << "=== Completed Parallel Group " << parallelGroupId << " (separated resources) ===\n";
}

SubgraphAnalysis analyzeSubgraphExecutionPattern_plus(const std::vector<EnhancedDependencyNode*>& nodes) {
  SubgraphAnalysis analysis;
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
    
    if (node->type == NodeType::Kernel) analysis.kernelCount++;
    else if (node->type == NodeType::CuLibs) analysis.culibsCount++;
    else if (node->type == NodeType::Loop) analysis.loopCount++;
  }
  
  int maxParallelNodesInAnyLevel = 0;
  for (const auto& [level, levelNodes] : nodesByLevel) {
    int levelParallelism = levelNodes.size();
    maxParallelNodesInAnyLevel = std::max(maxParallelNodesInAnyLevel, levelParallelism);
    
    if (levelParallelism > 1) {
      analysis.hasParallelSegments = true;
    }
  }
  
  analysis.kernelCount = maxParallelNodesInAnyLevel;
  analysis.hasSequentialKernels = nodesByLevel.size() > 1;
  
  return analysis;
}

std::vector<Value> processSubgraphNodesWithCorrectStreams_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    const std::vector<Value>& subgraphStreams,
    std::vector<Value>& availableTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
  builder.setInsertionPointToEnd(builder.getBlock());
  

  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
//   llvm::errs() << "      Processing subgraph with " << nodesByLevel.size() << " levels, "
//                << subgraphStreams.size() << " available streams\n";
  
  std::vector<Value> finalTokens;
  
  for (auto levelIter = nodesByLevel.begin(); levelIter != nodesByLevel.end(); ++levelIter) {
    int currentLevel = levelIter->first;
    const auto& levelNodes = levelIter->second;
    
    // llvm::errs() << "        Processing level " << currentLevel << " with " << levelNodes.size() << " nodes\n";
    
    // insertLevelStartMarker(builder, loc, currentLevel, levelNodes.size());
    
    std::vector<Value> currentLevelStreamTokens;
    
    for (int i = 0; i < levelNodes.size(); ++i) {
      auto* node = levelNodes[i];
      
      int streamIndex = i % subgraphStreams.size();
      Value assignedStream = subgraphStreams[streamIndex];
      
    //   insertNodeExecutionMarker(builder, loc, node, streamIndex);
      
      if (node->type == NodeType::Kernel && i < availableTokens.size()) {
        Value waitToken = availableTokens[i];
        Value nodeToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
        
        if (nodeToken) {
          currentLevelStreamTokens.push_back(nodeToken);
        }
        
      } else if (node->type == NodeType::Loop) {
        processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
        
      } else if (node->type == NodeType::CuLibs) {
        processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, assignedStream);
        
        // llvm::errs() << "          Assigned " << node->culibsFunctionName 
        //              << " to stream[" << streamIndex << "]\n";
      }
    }
    
    bool isLastLevel = (std::next(levelIter) == nodesByLevel.end());
    
    if (levelNodes.size() > 1) {
      if (!isLastLevel) {
        // insertSynchronizationMarker(builder, loc, currentLevel, subgraphStreams.size());
        
        // llvm::errs() << "        Synchronizing " << levelNodes.size() << " parallel nodes in level " << currentLevel << "\n";
        
        for (int i = 0; i < levelNodes.size() && i < subgraphStreams.size(); ++i) {
          Value stream = subgraphStreams[i];
          builder.create<func::CallOp>(
              loc, "mgpuStreamSynchronize",
              TypeRange{}, ValueRange{stream});
        }
        
        if (!currentLevelStreamTokens.empty()) {
          builder.create<gpu::WaitOp>(loc, TypeRange{}, currentLevelStreamTokens);
        }
        
      } else {
        // llvm::errs() << "        Level " << currentLevel << " is final level with " 
        //              << levelNodes.size() << " parallel nodes\n";
        
        finalTokens.insert(finalTokens.end(), currentLevelStreamTokens.begin(), currentLevelStreamTokens.end());
      }
      
    } else {
      if (isLastLevel) {
        finalTokens.insert(finalTokens.end(), currentLevelStreamTokens.begin(), currentLevelStreamTokens.end());
      }
    }
  }
  
  return finalTokens;
}

std::vector<Value> processSubgraphNodes_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    Value subgraphStream,
    std::vector<Value>& availableTokens,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
  builder.setInsertionPointToEnd(builder.getBlock());
  
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesByLevel;
  for (auto* node : nodes) {
    nodesByLevel[node->intraSubgraphLevel].push_back(node);
  }
  
//   llvm::errs() << "      Processing subgraph with " << nodesByLevel.size() << " levels\n";
  
  std::vector<Value> finalTokens;
  int tokenIndex = 0;
  
  for (auto levelIter = nodesByLevel.begin(); levelIter != nodesByLevel.end(); ++levelIter) {
    int currentLevel = levelIter->first;
    const auto& levelNodes = levelIter->second;
    
    // llvm::errs() << "        Processing level " << currentLevel << " with " << levelNodes.size() << " nodes\n";
    
    std::vector<Value> currentLevelTokens;
    
    for (auto* node : levelNodes) {
      Value nodeToken;
      
      if (node->type == NodeType::Kernel && tokenIndex < availableTokens.size()) {
        Value waitToken = availableTokens[tokenIndex++];
        nodeToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
        
      } else if (node->type == NodeType::Loop) {
        processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
        
      } else if (node->type == NodeType::CuLibs) {

        processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, subgraphStream);
      }
      
      if (nodeToken) {
        currentLevelTokens.push_back(nodeToken);
      }
    }
    
    bool isLastLevel = (std::next(levelIter) == nodesByLevel.end());
    
    if (!currentLevelTokens.empty()) {
      if (!isLastLevel) {
        // llvm::errs() << "        Synchronizing level " << currentLevel << " (has " << currentLevelTokens.size() << " async ops)\n";
        builder.create<gpu::WaitOp>(loc, TypeRange{}, currentLevelTokens);
      } else {
        // llvm::errs() << "        Level " << currentLevel << " is final level, keeping " << currentLevelTokens.size() << " tokens\n";
        finalTokens.insert(finalTokens.end(), currentLevelTokens.begin(), currentLevelTokens.end());
      }
    }
  }
  
  return finalTokens;
}

std::vector<ExecutionSegment> identifyExecutionSegments_plus(
    const std::map<int, std::vector<EnhancedDependencyNode*>>& nodesByLevel) {
  
  std::vector<ExecutionSegment> segments;
  
  auto levelIter = nodesByLevel.begin();
  while (levelIter != nodesByLevel.end()) {
    int currentLevel = levelIter->first;
    const auto& currentNodes = levelIter->second;
    
    if (currentNodes.size() > 1) {
      ExecutionSegment segment;
      segment.type = SegmentType::ParallelSegment;
      segment.nodes = currentNodes;
      segment.startLevel = segment.endLevel = currentLevel;
      segments.push_back(segment);
      
      ++levelIter;
      
    } else {
      ExecutionSegment segment;
      segment.type = SegmentType::SequentialSegment;
      segment.startLevel = currentLevel;
      
      while (levelIter != nodesByLevel.end() && levelIter->second.size() == 1) {
        segment.nodes.push_back(levelIter->second[0]);
        segment.endLevel = levelIter->first;
        ++levelIter;
      }
      
      segments.push_back(segment);
    }
  }
  
  return segments;
}

std::vector<Value> processParallelSegment_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    std::vector<Value>& availableTokens,
    int& tokenIndex,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
  std::vector<Value> segmentTokens;
  
//   llvm::errs() << "      Processing parallel segment with " << nodes.size() << " nodes\n";
  
  for (auto* node : nodes) {
    Value nodeToken;
    
    if (node->type == NodeType::Kernel && tokenIndex < availableTokens.size()) {
      Value waitToken = availableTokens[tokenIndex++];
      nodeToken = processKernelNodeEnhanced_plus(node, builder, mapper, waitToken, processedOps);
      
    } else if (node->type == NodeType::Loop) {
      processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
      
    } else if (node->type == NodeType::CuLibs) {
      processCuLibsNodeEnhanced_plus(node, builder, mapper, processedOps);
    }
    
    if (nodeToken) {
      segmentTokens.push_back(nodeToken);
    }
  }
  
  if (!segmentTokens.empty()) {
    builder.create<gpu::WaitOp>(loc, TypeRange{}, segmentTokens);
  }
  
  return segmentTokens;
}

Value processSequentialSegment_plus(
    const std::vector<EnhancedDependencyNode*>& nodes,
    Value sharedStream,
    OpBuilder& builder, IRMapping& mapper,
    llvm::DenseSet<Operation*>& processedOps,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    Location loc) {
  
//   llvm::errs() << "      Processing sequential segment with " << nodes.size() << " nodes\n";
  
  Value lastToken;
  
  for (auto* node : nodes) {
    if (node->type == NodeType::Kernel) {
      lastToken = processKernelNodeSequential_plus(node, builder, mapper, lastToken, processedOps);
      
    } else if (node->type == NodeType::Loop) {
      processLoopNodeEnhanced_plus(node, builder, mapper, allocaOps, processedOps);
      
    } else if (node->type == NodeType::CuLibs) {
      processCuLibsNodeWithStreamEnhanced_plus(node, builder, mapper, processedOps, sharedStream);
    }
  }
  
  return lastToken;
}

// ===== Node Processing Functions Implementation =====

//p
// Value processKernelNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
//                                     IRMapping& mapper, Value waitToken, 
//                                     llvm::DenseSet<Operation*>& processedOps) {
//   auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
  
//   auto kernelSymbol = SymbolRefAttr::get(
//       builder.getContext(),
//       kernelOp.getKernelModuleName(),
//       {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
  
//   SmallVector<Value, 8> remappedOperands;
//   for (Value operand : kernelOp.getKernelOperands()) {
//     remappedOperands.push_back(mapper.lookupOrDefault(operand));
//   }
  
//   auto gridSize = kernelOp.getGridSizeOperandValues();
//   auto blockSize = kernelOp.getBlockSizeOperandValues();
  
//   mlir::gpu::KernelDim3 mappedGridSize = {
//     mapper.lookupOrDefault(gridSize.x),
//     mapper.lookupOrDefault(gridSize.y),
//     mapper.lookupOrDefault(gridSize.z)
//   };
  
//   mlir::gpu::KernelDim3 mappedBlockSize = {
//     mapper.lookupOrDefault(blockSize.x),
//     mapper.lookupOrDefault(blockSize.y),
//     mapper.lookupOrDefault(blockSize.z)
//   };
  
//   auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
//       kernelOp.getLoc(),
//       kernelSymbol,
//       mappedGridSize,
//       mappedBlockSize,
//       Value(),
//       remappedOperands,
//       builder.getType<gpu::AsyncTokenType>(),
//       ValueRange{waitToken},
//       std::nullopt);

//     addSchedulingAttributes(newLaunchOp, node->parallelGroupId, node->subgraphId, node->intraSubgraphLevel, builder);


//   if (kernelOp->getNumResults() > 0) {
//     mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
//   }
  
//   processedOps.insert(node->op);
//   return newLaunchOp.getAsyncToken();
// }

Value processKernelNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                    IRMapping& mapper, Value waitToken, 
                                    llvm::DenseSet<Operation*>& processedOps) {
  auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
  
  // 收集所有依赖操作，包括 memref 转换操作
  llvm::SetVector<Operation*> requiredOps;
  collectDependentOpsEnhanced_plus(kernelOp, requiredOps, processedOps);
  
  // 处理依赖操作
  for (Operation* requiredOp : requiredOps) {
    if (!processedOps.count(requiredOp)) {
      Operation* newOp = builder.clone(*requiredOp, mapper);
      for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
        mapper.map(requiredOp->getResult(i), newOp->getResult(i));
      }
      processedOps.insert(requiredOp);
      
      if (isMemRefTransformOp(requiredOp)) {
        // llvm::errs() << "        Cloned memref transform op for kernel: " << requiredOp->getName() << "\n";
      }
    }
  }
  
  auto kernelSymbol = SymbolRefAttr::get(
      builder.getContext(),
      kernelOp.getKernelModuleName(),
      {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
  
  SmallVector<Value, 8> remappedOperands;
  for (Value operand : kernelOp.getKernelOperands()) {
    remappedOperands.push_back(mapper.lookupOrDefault(operand));
  }
  
  auto gridSize = kernelOp.getGridSizeOperandValues();
  auto blockSize = kernelOp.getBlockSizeOperandValues();
  
  mlir::gpu::KernelDim3 mappedGridSize = {
    mapper.lookupOrDefault(gridSize.x),
    mapper.lookupOrDefault(gridSize.y),
    mapper.lookupOrDefault(gridSize.z)
  };
  
  mlir::gpu::KernelDim3 mappedBlockSize = {
    mapper.lookupOrDefault(blockSize.x),
    mapper.lookupOrDefault(blockSize.y),
    mapper.lookupOrDefault(blockSize.z)
  };
  
  auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
      kernelOp.getLoc(),
      kernelSymbol,
      mappedGridSize,
      mappedBlockSize,
      Value(),
      remappedOperands,
      builder.getType<gpu::AsyncTokenType>(),
      ValueRange{waitToken},
      std::nullopt);

  addSchedulingAttributes(newLaunchOp, node->parallelGroupId, node->subgraphId, node->intraSubgraphLevel, builder);

  if (kernelOp->getNumResults() > 0) {
    mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
  }
  
  processedOps.insert(node->op);
  return newLaunchOp.getAsyncToken();
}

Value processKernelNodeSequential_plus(EnhancedDependencyNode* node, OpBuilder& builder,
                                      IRMapping& mapper, Value dependentToken,
                                      llvm::DenseSet<Operation*>& processedOps) {
  auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
  
  ValueRange dependencies = dependentToken ? ValueRange{dependentToken} : ValueRange{};
  
  auto kernelSymbol = SymbolRefAttr::get(
      builder.getContext(),
      kernelOp.getKernelModuleName(),
      {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
  
  SmallVector<Value, 8> remappedOperands;
  for (Value operand : kernelOp.getKernelOperands()) {
    remappedOperands.push_back(mapper.lookupOrDefault(operand));
  }
  
  auto gridSize = kernelOp.getGridSizeOperandValues();
  auto blockSize = kernelOp.getBlockSizeOperandValues();
  
  mlir::gpu::KernelDim3 mappedGridSize = {
    mapper.lookupOrDefault(gridSize.x),
    mapper.lookupOrDefault(gridSize.y),
    mapper.lookupOrDefault(gridSize.z)
  };
  
  mlir::gpu::KernelDim3 mappedBlockSize = {
    mapper.lookupOrDefault(blockSize.x),
    mapper.lookupOrDefault(blockSize.y),
    mapper.lookupOrDefault(blockSize.z)
  };
  
  auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
      kernelOp.getLoc(),
      kernelSymbol,
      mappedGridSize,
      mappedBlockSize,
      Value(),
      remappedOperands,
      builder.getType<gpu::AsyncTokenType>(),
      dependencies,
      std::nullopt);
      
  if (kernelOp->getNumResults() > 0) {
    mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
  }
  
  processedOps.insert(node->op);
  return newLaunchOp.getAsyncToken();
}

void processLoopNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                 IRMapping& mapper, llvm::SmallVector<Operation*, 16>& allocaOps,
                                 llvm::DenseSet<Operation*>& processedOps) {
  llvm::SmallVector<Operation*, 8> loopLocalAllocas;
  for (auto allocaOp : allocaOps) {
    bool used = false;
    Value allocaResult = allocaOp->getResult(0);
    node->op->walk([&](Operation *user) {
      for (Value operand : user->getOperands()) {
        if (operand == allocaResult) {
          used = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    if (used) {
      loopLocalAllocas.push_back(allocaOp);
      processedOps.insert(allocaOp);
    }
  }
  
  for (auto allocaOp : loopLocalAllocas) {
    auto newAllocaOp = builder.clone(*allocaOp, mapper);
    
    for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
      mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
    }
  }
  
  Operation *newOp = builder.clone(*node->op, mapper);
  
  addSchedulingAttributes(newOp, node->parallelGroupId, node->subgraphId, node->intraSubgraphLevel, builder);

  for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
    mapper.map(node->op->getResult(i), newOp->getResult(i));
  }
  
  processedOps.insert(node->op);
}

void processCuLibsNodeEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder, 
                                   IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps) {
  llvm::SetVector<Operation*> requiredOps;
  
  Operation* mainCall = nullptr;
  for (Operation* culibsOp : node->culibsOps) {
    if (isCuLibsCall_plus(culibsOp)) {
      mainCall = culibsOp;
      break;
    }
  }
  
  if (!mainCall) {
    return;
  }
  
  collectDependentOpsEnhanced_plus(mainCall, requiredOps, processedOps);
  
  for (Operation* requiredOp : requiredOps) {
    if (!processedOps.count(requiredOp)) {
      Operation* newOp = builder.clone(*requiredOp, mapper);
      
      for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
        mapper.map(requiredOp->getResult(i), newOp->getResult(i));
      }
      
      processedOps.insert(requiredOp);
    }
  }
  
  for (Operation* culibsOp : node->culibsOps) {
    if (!processedOps.count(culibsOp)) {
      Operation* newOp = builder.clone(*culibsOp, mapper);
      
      for (unsigned i = 0; i < culibsOp->getNumResults(); ++i) {
        mapper.map(culibsOp->getResult(i), newOp->getResult(i));
      }
      
      processedOps.insert(culibsOp);
    }
  }
}
// p
// void processCuLibsNodeWithStreamEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder,
//                                              IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps,
//                                              Value stream) {

//   llvm::SetVector<Operation*> requiredOps;
  
//   Operation* mainCall = nullptr;
//   for (Operation* culibsOp : node->culibsOps) {
//     if (isCuLibsCall_plus(culibsOp)) {
//       mainCall = culibsOp;
//       break;
//     }
//   }
  
//   if (!mainCall) {
//     llvm::errs() << "Warning: No main CuLibs call found in sequence\n";
//     return;
//   }
  

//   collectDependentOpsEnhanced_plus(mainCall, requiredOps, processedOps);
  
//   for (Operation* requiredOp : requiredOps) {
//     if (!processedOps.count(requiredOp)) {
//       Operation* newOp = builder.clone(*requiredOp, mapper);
//       for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
//         mapper.map(requiredOp->getResult(i), newOp->getResult(i));
//       }
//       processedOps.insert(requiredOp);
//     }
//   }
  
//   if (!processedOps.count(mainCall)) {
//     auto callOp = cast<func::CallOp>(mainCall);
    
//     llvm::SmallVector<Value, 8> newOperands;
//     for (unsigned i = 0; i < callOp.getNumOperands() - 1; ++i) {
//       newOperands.push_back(mapper.lookupOrDefault(callOp.getOperand(i)));
//     }
//     newOperands.push_back(stream);
    
//     auto newCallOp = builder.create<func::CallOp>(
//         callOp.getLoc(),
//         callOp.getCallee(),
//         callOp.getResultTypes(),
//         newOperands);
    
//     addSchedulingAttributes(newCallOp, node->parallelGroupId, node->subgraphId, node->intraSubgraphLevel, builder);


//     for (unsigned i = 0; i < mainCall->getNumResults(); ++i) {
//       mapper.map(mainCall->getResult(i), newCallOp.getResult(i));
//     }
    
//     processedOps.insert(mainCall);
    
//     // llvm::errs() << "        Processed CuLibs call: " << callOp.getCallee() << " with provided stream\n";
//   }
  
//   for (Operation* culibsOp : node->culibsOps) {
//     processedOps.insert(culibsOp);
//   }
// }

void processCuLibsNodeWithStreamEnhanced_plus(EnhancedDependencyNode* node, OpBuilder& builder,
                                             IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps,
                                             Value stream) {

  llvm::SetVector<Operation*> requiredOps;
  
  Operation* mainCall = nullptr;
  for (Operation* culibsOp : node->culibsOps) {
    if (isCuLibsCall_plus(culibsOp)) {
      mainCall = culibsOp;
      break;
    }
  }
  
  if (!mainCall) {
    llvm::errs() << "Warning: No main CuLibs call found in sequence\n";
    return;
  }
  
  // 使用增强的依赖收集，特别处理 memref 转换操作
  collectDependentOpsEnhanced_plus(mainCall, requiredOps, processedOps);
  
  // 额外收集 CuLibs 序列中其他操作的依赖
  for (Operation* culibsOp : node->culibsOps) {
    if (culibsOp != mainCall && !processedOps.count(culibsOp)) {
      collectDependentOpsEnhanced_plus(culibsOp, requiredOps, processedOps);
    }
  }
  
  // 按照依赖顺序处理所需的操作
  for (Operation* requiredOp : requiredOps) {
    if (!processedOps.count(requiredOp)) {
      Operation* newOp = builder.clone(*requiredOp, mapper);
      for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
        mapper.map(requiredOp->getResult(i), newOp->getResult(i));
      }
      processedOps.insert(requiredOp);
      
      // 调试信息：输出克隆的操作类型
      if (isMemRefTransformOp(requiredOp)) {
        // llvm::errs() << "        Cloned memref transform op: " << requiredOp->getName() << "\n";
      }
    }
  }
  
  // 处理主要的 CuLibs 调用
  if (!processedOps.count(mainCall)) {
    auto callOp = cast<func::CallOp>(mainCall);
    
    llvm::SmallVector<Value, 8> newOperands;
    for (unsigned i = 0; i < callOp.getNumOperands() - 1; ++i) {
      newOperands.push_back(mapper.lookupOrDefault(callOp.getOperand(i)));
    }
    newOperands.push_back(stream);
    
    auto newCallOp = builder.create<func::CallOp>(
        callOp.getLoc(),
        callOp.getCallee(),
        callOp.getResultTypes(),
        newOperands);
    
    addSchedulingAttributes(newCallOp, node->parallelGroupId, node->subgraphId, node->intraSubgraphLevel, builder);

    for (unsigned i = 0; i < mainCall->getNumResults(); ++i) {
      mapper.map(mainCall->getResult(i), newCallOp.getResult(i));
    }
    
    processedOps.insert(mainCall);
    
    // llvm::errs() << "        Processed CuLibs call: " << callOp.getCallee() << " with memref transform support\n";
  }
  
  // 标记所有 CuLibs 序列操作为已处理
  for (Operation* culibsOp : node->culibsOps) {
    processedOps.insert(culibsOp);
  }
}

// ===== Helper Functions Implementation =====
bool isMemRefTransformOp(Operation* op) {
  return isa<memref::ReinterpretCastOp>(op) ||
         isa<memref::CastOp>(op) ||
         isa<memref::SubViewOp>(op) ||
         isa<memref::ViewOp>(op) ||
         isa<memref::ReshapeOp>(op) ||
         isa<memref::ExpandShapeOp>(op) ||
         isa<memref::CollapseShapeOp>(op);
}

void collectMemRefTransformDependencies_plus(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
                                            const llvm::DenseSet<Operation*>& processedOps) {
  // 递归收集 memref 转换操作的依赖
  for (Value operand : op->getOperands()) {
    if (Operation* definingOp = operand.getDefiningOp()) {
      if (!processedOps.count(definingOp) && !isa<BlockArgument>(operand)) {
        // 如果是 memref 转换操作，继续递归收集
        if (isMemRefTransformOp(definingOp)) {
          collectMemRefTransformDependencies_plus(definingOp, requiredOps, processedOps);
          requiredOps.insert(definingOp);
        }
        // 如果是其他需要移动的操作
        else if (shouldMoveWithCuLibs_plus(definingOp)) {
          collectDependentOpsEnhanced_plus(definingOp, requiredOps, processedOps);
          requiredOps.insert(definingOp);
        }
      }
    }
  }
}

bool shouldMoveWithCuLibs_plus(Operation* op) {
  // 原有的判断逻辑（如果之前有实现）
  if (isa<arith::IndexCastOp>(op) ||
      isa<mlir::LLVM::IntToPtrOp>(op) ||
      isa<memref::ExtractAlignedPointerAsIndexOp>(op)) {
    return true;
  }
  
  // 新增：处理 memref 转换操作
  if (isMemRefTransformOp(op)) {
    return true;
  }
  
  // 新增：处理常量操作
  if (isa<arith::ConstantOp>(op)) {
    return true;
  }
  
  // 其他需要与 CuLibs 一起移动的操作
  return false;
}

// // p
// void collectDependentOpsEnhanced_plus(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
//                                      const llvm::DenseSet<Operation*>& processedOps) {
//   for (Value operand : op->getOperands()) {
//     if (Operation* definingOp = operand.getDefiningOp()) {

//       if (!processedOps.count(definingOp) && !isa<BlockArgument>(operand)) {

//         if (shouldMoveWithCuLibs(definingOp)) {

//           collectDependentOpsEnhanced_plus(definingOp, requiredOps, processedOps);
//           requiredOps.insert(definingOp);
//         }
//       }
//     }
//   }
// }

void collectDependentOpsEnhanced_plus(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
                                     const llvm::DenseSet<Operation*>& processedOps) {
  for (Value operand : op->getOperands()) {
    if (Operation* definingOp = operand.getDefiningOp()) {
      if (!processedOps.count(definingOp) && !isa<BlockArgument>(operand)) {
        
        // 特殊处理：如果是 memref 转换操作，使用专门的收集函数
        if (isMemRefTransformOp(definingOp)) {
          collectMemRefTransformDependencies_plus(definingOp, requiredOps, processedOps);
          requiredOps.insert(definingOp);
        }
        // 处理其他需要移动的操作
        else if (shouldMoveWithCuLibs_plus(definingOp)) {
          collectDependentOpsEnhanced_plus(definingOp, requiredOps, processedOps);
          requiredOps.insert(definingOp);
        }
      }
    }
  }
}

void analyzeGPUWaitOperations_plus(func::FuncOp funcOp, EnhancedDependencyGraph& graph,
                                  llvm::DenseMap<Operation*, bool>& waitOpShouldKeep) {
  funcOp.walk([&](gpu::WaitOp waitOp) {

    bool isAsyncWait = waitOp.getAsyncToken() != nullptr;
    bool isSyncWait = !waitOp.getAsyncDependencies().empty() && !isAsyncWait;
    
    if (isSyncWait) {
      waitOpShouldKeep[waitOp] = true;
    } else if (isAsyncWait) {
      bool usedByNonGraphOps = false;
      for (auto user : waitOp->getUsers()) {
        if (!graph.opToNodeMap.count(user)) {
          usedByNonGraphOps = true;
          break;
        }
      }
      waitOpShouldKeep[waitOp] = usedByNonGraphOps;
    } else {
      waitOpShouldKeep[waitOp] = true;
    }
  });
}

void copyPrefixOperations_plus(Block* oldBlock, Block* newBlock, OpBuilder& builder,
                              IRMapping& mapper, EnhancedDependencyGraph& graph,
                              llvm::DenseSet<Operation*>& processedOps,
                              const llvm::DenseMap<Operation*, bool>& waitOpShouldKeep,
                              llvm::SmallVector<Operation*, 16>& allocaOps) {
  
  builder.setInsertionPointToEnd(newBlock);
  
  for (auto &op : oldBlock->getOperations()) {
    if (graph.opToNodeMap.count(&op)) {
      break;
    }
    
    if (isa<memref::AllocaOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
      auto it = waitOpShouldKeep.find(waitOp);
      if (it != waitOpShouldKeep.end() && it->second) {
        Operation *newOp = op.clone(mapper);
        newBlock->push_back(newOp);
        
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          mapper.map(op.getResult(i), newOp->getResult(i));
        }
      }
      processedOps.insert(&op);
      continue;
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
    processedOps.insert(&op);
  }
}

void finalizeIRReorganization_plus(
    Block* oldBlock, Block* newBlock, OpBuilder& builder,
    IRMapping& mapper, llvm::DenseSet<Operation*>& processedOps,
    const llvm::DenseMap<Operation*, bool>& waitOpShouldKeep,
    llvm::SmallVector<Operation*, 16>& allocaOps,
    func::FuncOp funcOp, func::FuncOp descriptorReturnFunc,
    func::FuncOp workspaceReturnFunc,
    const std::map<int, std::map<int, std::vector<EnhancedDependencyNode*>>>& nodesByParallelGroupAndSubgraph) {
  
  builder.setInsertionPointToEnd(newBlock);
  
  for (auto allocaOp : allocaOps) {
    if (!processedOps.count(allocaOp)) {
      auto newAllocaOp = builder.clone(*allocaOp, mapper);
      
      for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
        mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
      }
      
      processedOps.insert(allocaOp);
    }
  }
  
  bool hasReturnOp = false;
  Operation* returnOp = nullptr;
  
  for (auto &op : oldBlock->getOperations()) {
    if (processedOps.count(&op))
      continue;
    
    if (isa<func::ReturnOp>(op)) {
      hasReturnOp = true;
      returnOp = &op;
      continue;
    }
    
    if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
      auto it = waitOpShouldKeep.find(waitOp);
      if (it != waitOpShouldKeep.end() && it->second) {
        Operation *newOp = op.clone(mapper);
        newBlock->push_back(newOp);
        
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          mapper.map(op.getResult(i), newOp->getResult(i));
        }
      }
      continue;
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
  }
  
  bool hasAnyOperations = !nodesByParallelGroupAndSubgraph.empty();
  
  if (hasAnyOperations) {
    insertDescriptorReturnCall(builder, funcOp.getLoc(), descriptorReturnFunc);
    insertWorkspaceReturnCall(builder, funcOp.getLoc(), workspaceReturnFunc);
  }
  
  if (hasReturnOp) {
    Operation *newReturnOp = returnOp->clone(mapper);
    newBlock->push_back(newReturnOp);
  }
  
  replaceFunctionBody_plus(funcOp, oldBlock, newBlock, mapper);
}

void replaceFunctionBody_plus(func::FuncOp funcOp, Block* oldBlock, Block* newBlock, IRMapping& mapper) {
  funcOp.getBody().push_back(newBlock);
  
  for (auto &op : oldBlock->getOperations()) {
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      Value oldResult = op.getResult(i);
      if (mapper.contains(oldResult)) {
        Value newResult = mapper.lookup(oldResult);
        llvm::SmallVector<mlir::OpOperand*, 4> usesToReplace;
        for (auto &use : oldResult.getUses()) {
          Operation* userOp = use.getOwner();
          if (userOp->getBlock() != oldBlock) {
            usesToReplace.push_back(&use);
          }
        }
        
        for (auto *use : usesToReplace) {
          use->set(newResult);
        }
      }
    }
  }
  
  bool hasExternalUses = false;
  for (auto &op : oldBlock->getOperations()) {
    for (auto result : op.getResults()) {
      for (auto &use : result.getUses()) {
        if (use.getOwner()->getBlock() != oldBlock) {
          if (!mapper.contains(result)) {
            llvm::errs() << "Warning: Operation still has external uses: ";
            op.print(llvm::errs());
            llvm::errs() << "\n";
            hasExternalUses = true;
          }
        }
      }
    }
  }
  
  if (hasExternalUses) {
    llvm::errs() << "Error: Cannot safely delete old block due to external references\n";
    return;
  }
  
  oldBlock->dropAllUses();
  oldBlock->erase();
}

void printEnhancedSchedulingSummary_plus(const EnhancedDependencyGraph& graph) {
//   llvm::errs() << "\n=== Enhanced Scheduling Summary ===\n";
  
  // Collect statistics by parallel group
  std::map<int, std::map<int, std::vector<EnhancedDependencyNode*>>> stats;
  int totalKernels = 0, totalCuLibs = 0, totalLoops = 0;
  
  for (const auto& nodePtr : graph.nodes) {
    EnhancedDependencyNode* node = nodePtr.get();
    stats[node->parallelGroupId][node->subgraphId].push_back(node);
    
    switch (node->type) {
      case NodeType::Kernel: totalKernels++; break;
      case NodeType::CuLibs: totalCuLibs++; break;
      case NodeType::Loop: totalLoops++; break;
    }
  }
  
  llvm::errs() << "Total Nodes: " << graph.nodes.size() 
               << " (K:" << totalKernels << ", C:" << totalCuLibs << ", L:" << totalLoops << ")\n";
  llvm::errs() << "Parallel Groups: " << stats.size() << "\n";
  llvm::errs() << "Total Subgraphs: " << graph.subgraphs.size() << "\n\n";
  
  // Print detailed breakdown
  for (const auto& [groupId, subgraphs] : stats) {
    llvm::errs() << "Parallel Group " << groupId << ": " << subgraphs.size() << " subgraphs\n";
    
    for (const auto& [subgraphId, nodes] : subgraphs) {
      int kernels = 0, culibs = 0, loops = 0;
      int maxLevel = 0;
      
      for (auto* node : nodes) {
        switch (node->type) {
          case NodeType::Kernel: kernels++; break;
          case NodeType::CuLibs: culibs++; break;
          case NodeType::Loop: loops++; break;
        }
        maxLevel = std::max(maxLevel, node->intraSubgraphLevel);
      }
      
      llvm::errs() << "  Subgraph " << subgraphId << ": " << nodes.size() << " nodes"
                   << " (K:" << kernels << ", C:" << culibs << ", L:" << loops << ")"
                   << ", depth:" << (maxLevel + 1) << "\n";
    }
  }
  
  llvm::errs() << "================================\n\n";
}

} // namespace onnx_mlir