// Enhanced Dependency Graph Implementation
#include "EnhancedDependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// ===== EnhancedDependencyGraph Class Implementation =====

EnhancedDependencyNode* EnhancedDependencyGraph::addNode(std::unique_ptr<EnhancedDependencyNode> node) {
  EnhancedDependencyNode* nodePtr = node.get();
  opToNodeMap[node->op] = nodePtr;
  
  // For culibs nodes, map all related operations
  if (node->type == NodeType::CuLibs) {
    for (Operation* culibsOp : node->culibsOps) {
      opToNodeMap[culibsOp] = nodePtr;
    }
  }
  
  nodes.push_back(std::move(node));
  return nodePtr;
}

void EnhancedDependencyGraph::addEdge(EnhancedDependencyNode* from, EnhancedDependencyNode* to) {
  outEdges[from].push_back(to);
  inEdges[to].push_back(from);
}

int EnhancedDependencyGraph::getOutDegree(EnhancedDependencyNode* node) {
  return outEdges[node].size();
}

int EnhancedDependencyGraph::getInDegree(EnhancedDependencyNode* node) {
  return inEdges[node].size();
}

Subgraph* EnhancedDependencyGraph::createSubgraph() {
  auto subgraph = std::make_unique<Subgraph>(nextSubgraphId++);
  Subgraph* subgraphPtr = subgraph.get();
  subgraphs.push_back(std::move(subgraph));
  return subgraphPtr;
}

void EnhancedDependencyGraph::addNodeToSubgraph(EnhancedDependencyNode* node, Subgraph* subgraph) {
  node->subgraphId = subgraph->id;
  subgraph->nodes.push_back(node);
  node->inCurrentSubgraph = true;
}

llvm::SmallVector<EnhancedDependencyNode*, 8> EnhancedDependencyGraph::findZeroInDegreeNodes() {
  llvm::SmallVector<EnhancedDependencyNode*, 8> result;
  for (auto& nodePtr : nodes) {
    if (getInDegree(nodePtr.get()) == 0) {
      result.push_back(nodePtr.get());
    }
  }
  return result;
}

void EnhancedDependencyGraph::dfsPartition(EnhancedDependencyNode* node, Subgraph* currentSubgraph) {
  if (node->visited || node->inCurrentSubgraph) {
    return;
  }
  
  // Add current node to subgraph
  addNodeToSubgraph(node, currentSubgraph);
  
  // If out degree is 1, continue the chain
  if (getOutDegree(node) == 1) {
    EnhancedDependencyNode* nextNode = outEdges[node][0];
    
    // If next node is not visited and doesn't belong to any subgraph
    if (!nextNode->visited && nextNode->subgraphId == -1) {
      dfsPartition(nextNode, currentSubgraph);
    }
    // If next node already belongs to a subgraph, merge current node into that subgraph
    else if (nextNode->subgraphId != -1 && !node->inCurrentSubgraph) {
      // Remove node from current subgraph
      auto& currentNodes = currentSubgraph->nodes;
      currentNodes.erase(std::remove(currentNodes.begin(), currentNodes.end(), node), currentNodes.end());
      node->inCurrentSubgraph = false;
      
      // Find the target subgraph and add node to it
      for (auto& sg : subgraphs) {
        if (sg->id == nextNode->subgraphId) {
          addNodeToSubgraph(node, sg.get());
          break;
        }
      }
    }
  }
  
  // Check predecessors with out degree 1
  if (inEdges.count(node)) {
    for (EnhancedDependencyNode* predNode : inEdges[node]) {
      if (!predNode->visited && predNode->subgraphId == -1 && getOutDegree(predNode) == 1) {
        dfsPartition(predNode, currentSubgraph);
      }
    }
  }
}

void EnhancedDependencyGraph::partitionIntoSubgraphs() {
  llvm::errs() << "Starting subgraph partitioning...\n";
  
  // Reset all node states
  for (auto& nodePtr : nodes) {
    nodePtr->visited = false;
    nodePtr->inCurrentSubgraph = false;
    nodePtr->subgraphId = -1;
  }
  
  // Start from nodes with zero in-degree
  auto zeroInDegreeNodes = findZeroInDegreeNodes();
  llvm::errs() << "Found " << zeroInDegreeNodes.size() << " zero in-degree nodes\n";
  
  // Process nodes in program order if available, otherwise use zero in-degree nodes
  std::queue<EnhancedDependencyNode*> workQueue;
  
  // Add zero in-degree nodes to work queue
  for (auto* node : zeroInDegreeNodes) {
    workQueue.push(node);
  }
  
  // If no zero in-degree nodes, add all unvisited nodes
  if (workQueue.empty()) {
    for (auto& nodePtr : nodes) {
      if (!nodePtr->visited) {
        workQueue.push(nodePtr.get());
        break; // Start with first unvisited node
      }
    }
  }
  
  while (!workQueue.empty() || std::any_of(nodes.begin(), nodes.end(), 
                                          [](const auto& n) { return n->subgraphId == -1; })) {
    
    // Get next node to process
    EnhancedDependencyNode* currentNode = nullptr;
    
    if (!workQueue.empty()) {
      currentNode = workQueue.front();
      workQueue.pop();
    } else {
      // Find first unprocessed node
      for (auto& nodePtr : nodes) {
        if (nodePtr->subgraphId == -1) {
          currentNode = nodePtr.get();
          break;
        }
      }
    }
    
    if (!currentNode || currentNode->subgraphId != -1) {
      continue;
    }
    
    // Determine how to handle this node based on its out degree
    int outDegree = getOutDegree(currentNode);
    
    if (outDegree == 1) {
      // Check if next node already has a subgraph
      EnhancedDependencyNode* nextNode = outEdges[currentNode][0];
      
      if (nextNode->subgraphId != -1) {
        // Add current node to existing subgraph
        for (auto& sg : subgraphs) {
          if (sg->id == nextNode->subgraphId) {
            addNodeToSubgraph(currentNode, sg.get());
            break;
          }
        }
      } else {
        // Create new subgraph and start DFS partitioning
        Subgraph* newSubgraph = createSubgraph();
        dfsPartition(currentNode, newSubgraph);
      }
    } else {
      // Out degree is 0 or >= 2, create individual subgraph
      Subgraph* newSubgraph = createSubgraph();
      addNodeToSubgraph(currentNode, newSubgraph);
    }
    
    currentNode->visited = true;
    
    // Add successors to work queue
    if (outEdges.count(currentNode)) {
      for (auto* successor : outEdges[currentNode]) {
        if (successor->subgraphId == -1) {
          workQueue.push(successor);
        }
      }
    }
  }
  
  llvm::errs() << "Subgraph partitioning completed. Created " << subgraphs.size() << " subgraphs\n";
}

void EnhancedDependencyGraph::buildSubgraphEdges() {
  // Build edges between subgraphs
  for (auto& nodePtr : nodes) {
    if (outEdges.count(nodePtr.get())) {
      for (auto* successor : outEdges[nodePtr.get()]) {
        if (nodePtr->subgraphId != successor->subgraphId) {
          // Find source and target subgraphs
          Subgraph* sourceSubgraph = nullptr;
          Subgraph* targetSubgraph = nullptr;
          
          for (auto& sg : subgraphs) {
            if (sg->id == nodePtr->subgraphId) {
              sourceSubgraph = sg.get();
            }
            if (sg->id == successor->subgraphId) {
              targetSubgraph = sg.get();
            }
          }
          
          if (sourceSubgraph && targetSubgraph) {
            sourceSubgraph->successors.insert(targetSubgraph);
            targetSubgraph->predecessors.insert(sourceSubgraph);
          }
        }
      }
    }
  }
}

void EnhancedDependencyGraph::topologicalSortSubgraphs() {
  llvm::errs() << "Starting subgraph topological sorting...\n";
  
  buildSubgraphEdges();
  
  // Kahn's algorithm for subgraph topological sorting
  std::queue<Subgraph*> queue;
  llvm::DenseMap<Subgraph*, int> inDegree;
  
  // Initialize in-degrees
  for (auto& sg : subgraphs) {
    inDegree[sg.get()] = sg->predecessors.size();
    if (inDegree[sg.get()] == 0) {
      queue.push(sg.get());
    }
  }
  
  int level = 0;
  while (!queue.empty()) {
    int levelSize = queue.size();
    
    // Process all subgraphs at current level
    for (int i = 0; i < levelSize; ++i) {
      Subgraph* currentSubgraph = queue.front();
      queue.pop();
      
      currentSubgraph->topologicalLevel = level;
      
      // Assign parallel group ID to all nodes in this subgraph
      for (auto* node : currentSubgraph->nodes) {
        node->parallelGroupId = level;
      }
      
      // Update successors
      for (auto* successor : currentSubgraph->successors) {
        inDegree[successor]--;
        if (inDegree[successor] == 0) {
          queue.push(successor);
        }
      }
    }
    
    level++;
  }
  
  llvm::errs() << "Subgraph topological sorting completed. Total levels: " << level << "\n";
}

void EnhancedDependencyGraph::topologicalSortWithinSubgraphs() {
  llvm::errs() << "Starting intra-subgraph topological sorting...\n";
  
  for (auto& sg : subgraphs) {
    if (sg->nodes.size() <= 1) {
      // Single node subgraph
      if (!sg->nodes.empty()) {
        sg->nodes[0]->intraSubgraphLevel = 0;
      }
      continue;
    }
    
    // Build internal edges for this subgraph
    llvm::DenseMap<EnhancedDependencyNode*, int> internalInDegree;
    std::queue<EnhancedDependencyNode*> internalQueue;
    
    // Initialize in-degrees for nodes in this subgraph
    for (auto* node : sg->nodes) {
      internalInDegree[node] = 0;
      
      if (inEdges.count(node)) {
        for (auto* pred : inEdges[node]) {
          if (pred->subgraphId == sg->id) {
            internalInDegree[node]++;
          }
        }
      }
      
      if (internalInDegree[node] == 0) {
        internalQueue.push(node);
      }
    }
    
    // Topological sort within subgraph
    int internalLevel = 0;
    while (!internalQueue.empty()) {
      int levelSize = internalQueue.size();
      
      for (int i = 0; i < levelSize; ++i) {
        EnhancedDependencyNode* currentNode = internalQueue.front();
        internalQueue.pop();
        
        currentNode->intraSubgraphLevel = internalLevel;
        
        // Update internal successors
        if (outEdges.count(currentNode)) {
          for (auto* successor : outEdges[currentNode]) {
            if (successor->subgraphId == sg->id) {
              internalInDegree[successor]--;
              if (internalInDegree[successor] == 0) {
                internalQueue.push(successor);
              }
            }
          }
        }
      }
      
      internalLevel++;
    }
  }
  
  llvm::errs() << "Intra-subgraph topological sorting completed\n";
}

void EnhancedDependencyGraph::performEnhancedScheduling() {
  llvm::errs() << "=== Starting Enhanced Dependency Graph Scheduling ===\n";
  
  partitionIntoSubgraphs();
  topologicalSortSubgraphs();
  topologicalSortWithinSubgraphs();
  
  llvm::errs() << "=== Enhanced Scheduling Completed ===\n";
}

void EnhancedDependencyGraph::dumpSubgraphInfo() {
  llvm::errs() << "\n===== Subgraph Information =====\n";
  
  for (auto& sg : subgraphs) {
    llvm::errs() << "Subgraph " << sg->id << " (Level " << sg->topologicalLevel << "):\n";
    llvm::errs() << "  Nodes (" << sg->nodes.size() << "):\n";
    
    for (auto* node : sg->nodes) {
      llvm::errs() << "    ";
      if (node->type == NodeType::Kernel) {
        llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
      } else if (node->type == NodeType::Loop) {
        llvm::errs() << "Loop at: ";
        if (node->op) node->op->getLoc().print(llvm::errs());
      } else if (node->type == NodeType::CuLibs) {
        llvm::errs() << "CuLibs: " << node->culibsFunctionName;
      }
      llvm::errs() << "\n";
    }
    
    llvm::errs() << "  Predecessors: ";
    for (auto* pred : sg->predecessors) {
      llvm::errs() << pred->id << " ";
    }
    llvm::errs() << "\n";
    
    llvm::errs() << "  Successors: ";
    for (auto* succ : sg->successors) {
      llvm::errs() << succ->id << " ";
    }
    llvm::errs() << "\n\n";
  }
}

void EnhancedDependencyGraph::dumpFinalScheduling() {
  llvm::errs() << "\n===== Final Scheduling (x,y,z) =====\n";
  
  // Group nodes by parallel group
  std::map<int, std::vector<EnhancedDependencyNode*>> parallelGroups;
  
  for (auto& nodePtr : nodes) {
    parallelGroups[nodePtr->parallelGroupId].push_back(nodePtr.get());
  }
  
  for (auto& [groupId, groupNodes] : parallelGroups) {
    llvm::errs() << "Parallel Group " << groupId << ":\n";
    
    // Group by subgraph within parallel group
    std::map<int, std::vector<EnhancedDependencyNode*>> subgraphGroups;
    for (auto* node : groupNodes) {
      subgraphGroups[node->subgraphId].push_back(node);
    }
    
    for (auto& [subgraphId, subgraphNodes] : subgraphGroups) {
      llvm::errs() << "  Subgraph " << subgraphId << ":\n";
      
      // Sort by intra-subgraph level
      std::sort(subgraphNodes.begin(), subgraphNodes.end(),
                [](EnhancedDependencyNode* a, EnhancedDependencyNode* b) {
                  return a->intraSubgraphLevel < b->intraSubgraphLevel;
                });
      
      for (auto* node : subgraphNodes) {
        llvm::errs() << "    (" << node->parallelGroupId << "," 
                     << node->subgraphId << "," 
                     << node->intraSubgraphLevel << ") ";
        
        if (node->type == NodeType::Kernel) {
          llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
        } else if (node->type == NodeType::Loop) {
          llvm::errs() << "Loop";
        } else if (node->type == NodeType::CuLibs) {
          llvm::errs() << "CuLibs: " << node->culibsFunctionName;
        }
        llvm::errs() << "\n";
      }
    }
  }
  
  llvm::errs() << "===============================\n";
}

void EnhancedDependencyGraph::dumpEnhancedGraph() {
  llvm::errs() << "\n===== Enhanced Dependency Graph =====\n";
  
  llvm::errs() << "Total nodes: " << nodes.size() << "\n";
  llvm::errs() << "Total subgraphs: " << subgraphs.size() << "\n\n";
  
  dumpSubgraphInfo();
  dumpFinalScheduling();
}

// ===== Helper Functions Implementation =====

bool isKernelLaunch_plus(Operation* op) {
  return isa<gpu::LaunchFuncOp>(op);
}

bool isLoopNest_plus(Operation* op) {
  return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
}

bool isCuLibsCall_plus(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    StringRef funcName = callOp.getCallee();
    return funcName.starts_with("mgpuCudnn") || 
           funcName.starts_with("mgpuCulibs");
  }
  return false;
}

bool isCuLibsStreamCreate_plus(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamCreate";
  }
  return false;
}

bool isCuLibsStreamSync_plus(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamSynchronize";
  }
  return false;
}

bool isCuLibsStreamDestroy_plus(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamDestroy";
  }
  return false;
}

gpu::GPUFuncOp findKernelFunc_plus(gpu::LaunchFuncOp kernelOp) {
  ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
  if (!topModule) {
    return nullptr;
  }

  StringRef kernelModuleName = kernelOp.getKernelModuleName();
  StringRef kernelName = kernelOp.getKernelName();
  
  gpu::GPUModuleOp gpuModule = nullptr;
  topModule.walk([&](gpu::GPUModuleOp module) {
    if (module.getName() == kernelModuleName) {
      gpuModule = module;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (gpuModule) {
    gpu::GPUFuncOp kernelFunc = nullptr;
    gpuModule.walk([&](gpu::GPUFuncOp func) {
      if (func.getName() == kernelName) {
        kernelFunc = func;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (kernelFunc)
      return kernelFunc;
  }
  
  gpu::GPUFuncOp result = nullptr;
  topModule.walk([&](gpu::GPUFuncOp func) {
    if (func.getName() == kernelName) {
      result = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return result;
}

// // p
// void extractKernelDependencies_plus(gpu::LaunchFuncOp kernelOp, 
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs) {
//   gpu::GPUFuncOp kernelFunc = findKernelFunc_plus(kernelOp);
  
//   if (!kernelFunc) {
//     llvm::errs() << "Warning: Could not find kernel function definition for \"" 
//                 << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
//     for (auto arg : kernelOp.getKernelOperands()) {
//       if (arg.getType().isa<MemRefType>()) {
//         inputs.insert(arg);
//         outputs.insert(arg);
//       }
//     }
//     return;
//   }

//   llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
//   unsigned memrefArgCount = 0;
//   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
//     if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
//       memrefArgCount++;
//     }
//   }
  
//   unsigned memrefOpCount = 0;
//   llvm::SmallVector<Value, 8> memrefOperands;
//   for (auto operand : kernelOp.getKernelOperands()) {
//     if (operand.getType().isa<MemRefType>()) {
//       memrefOperands.push_back(operand);
//       memrefOpCount++;
//     }
//   }
  
//   unsigned opIdx = 0;
//   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
//     BlockArgument arg = kernelFunc.getArgument(i);
//     if (arg.getType().isa<MemRefType>()) {
//       if (opIdx < memrefOperands.size()) {
//         Value operand = memrefOperands[opIdx++];
//         argOperandPairs.push_back({arg, operand});
//       }
//     }
//   }
  
//   llvm::DenseSet<BlockArgument> loadArgs;
//   llvm::DenseSet<BlockArgument> storeArgs;
  
//   unsigned loadCount = 0, storeCount = 0;
//   kernelFunc.walk([&](Operation *op) {
//     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
//       Value memref = loadOp.getMemref();
//       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
//         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
//           loadArgs.insert(blockArg);
//           loadCount++;
//         }
//       }
//     } 
//     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
//       Value memref = storeOp.getMemref();
//       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
//         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
//           storeArgs.insert(blockArg);
//           storeCount++;
//         }
//       }
//     }
//   });
  
//   for (auto &pair : argOperandPairs) {
//     BlockArgument arg = pair.first;
//     Value operand = pair.second;
    
//     bool isInput = loadArgs.count(arg) > 0;
//     bool isOutput = storeArgs.count(arg) > 0;
    
//     if (isInput) {
//       inputs.insert(operand);
//     }
    
//     if (isOutput) {
//       outputs.insert(operand);
//     }
    
//     if (!isInput && !isOutput) {
//       inputs.insert(operand);
//       llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
//     }
//   }
// }

Value traceMemRefSource_plus(Value value) {
  Value current = value;
  
  while (current) {
    Operation* definingOp = current.getDefiningOp();
    if (!definingOp) {
      break;
    }
    
    // 处理各种 memref 转换操作
    if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
      current = reinterpretCastOp.getSource();
      continue;
    }
    
    if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
      current = castOp.getSource();
      continue;
    }
    
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
      current = subViewOp.getSource();
      continue;
    }
    
    if (auto viewOp = dyn_cast<memref::ViewOp>(definingOp)) {
      current = viewOp.getSource();
      continue;
    }
    
    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(definingOp)) {
      current = reshapeOp.getSource();
      continue;
    }
    
    if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(definingOp)) {
      current = expandShapeOp.getSrc();
      continue;
    }
    
    if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(definingOp)) {
      current = collapseShapeOp.getSrc();
      continue;
    }
    
    // 如果不是 memref 转换操作，停止追踪
    break;
  }
  
  return current;
}

void extractKernelDependencies_plus(gpu::LaunchFuncOp kernelOp, 
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  gpu::GPUFuncOp kernelFunc = findKernelFunc_plus(kernelOp);
  
  if (!kernelFunc) {
    llvm::errs() << "Warning: Could not find kernel function definition for \"" 
                << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
    
    // 保守分析：处理所有 memref 操作数，包括经过转换的
    for (auto arg : kernelOp.getKernelOperands()) {
      Value sourceMemRef = traceMemRefSource_plus(arg);
      if (sourceMemRef.getType().isa<MemRefType>()) {
        inputs.insert(sourceMemRef);
        outputs.insert(sourceMemRef);
      }
    }
    return;
  }

  llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
  unsigned memrefArgCount = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
      memrefArgCount++;
    }
  }
  
  unsigned memrefOpCount = 0;
  llvm::SmallVector<Value, 8> memrefOperands;
  for (auto operand : kernelOp.getKernelOperands()) {
    // 追踪到原始的 memref，而不是经过转换的
    Value sourceMemRef = traceMemRefSource_plus(operand);
    if (sourceMemRef.getType().isa<MemRefType>()) {
      memrefOperands.push_back(sourceMemRef);
      memrefOpCount++;
    }
  }
  
  unsigned opIdx = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    BlockArgument arg = kernelFunc.getArgument(i);
    if (arg.getType().isa<MemRefType>()) {
      if (opIdx < memrefOperands.size()) {
        Value operand = memrefOperands[opIdx++];
        argOperandPairs.push_back({arg, operand});
      }
    }
  }
  
  llvm::DenseSet<BlockArgument> loadArgs;
  llvm::DenseSet<BlockArgument> storeArgs;
  
  unsigned loadCount = 0, storeCount = 0;
  kernelFunc.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      Value memref = loadOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          loadArgs.insert(blockArg);
          loadCount++;
        }
      }
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      Value memref = storeOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          storeArgs.insert(blockArg);
          storeCount++;
        }
      }
    }
  });
  
  for (auto &pair : argOperandPairs) {
    BlockArgument arg = pair.first;
    Value operand = pair.second;
    
    bool isInput = loadArgs.count(arg) > 0;
    bool isOutput = storeArgs.count(arg) > 0;
    
    if (isInput) {
      inputs.insert(operand);
    }
    
    if (isOutput) {
      outputs.insert(operand);
    }
    
    if (!isInput && !isOutput) {
      inputs.insert(operand);
      llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
    }
  }
}

void extractLoopDependencies_plus(scf::ForOp loopOp,
                           llvm::SetVector<Value> &inputs,
                           llvm::SetVector<Value> &outputs) {
  loopOp.walk([&](Operation* op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      inputs.insert(loadOp.getMemref());
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      outputs.insert(storeOp.getMemref());
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
          storeOp.getValue().getDefiningOp())) {
        inputs.insert(loadOp.getMemref());
      }
    }
  });
}

// // p
// void extractCuLibsDependencies_plus(const llvm::SmallVector<Operation*, 4> &culibsOps,
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs) {
//   Operation* mainCall = nullptr;
//   for (Operation* op : culibsOps) {
//     if (isCuLibsCall_plus(op)) {
//       mainCall = op;
//       break;
//     }
//   }
  
//   if (!mainCall) {
//     return;
//   }
  
//   auto callOp = cast<func::CallOp>(mainCall);
//   StringRef funcName = callOp.getCallee();
  
//   auto operands = callOp.getOperands();
  
//   llvm::SmallVector<Value, 8> memrefOperands;
  
//   for (Value operand : operands) {
//     if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
//       Value intToPtrInput = intToPtrOp.getArg();
//       if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
//         Value indexCastInput = indexCastOp->getOperand(0);
//         if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
//           Value memref = extractOp.getSource();
//           memrefOperands.push_back(memref);
//         }
//       }
//     }
//   }
  
//   if (funcName.contains("Conv2dForward")) {
//     if (memrefOperands.size() >= 4) {
//       inputs.insert(memrefOperands[0]);
//       inputs.insert(memrefOperands[1]);  
//       inputs.insert(memrefOperands[2]);
//       outputs.insert(memrefOperands[3]);
//     }
//   }
//   else if (funcName.contains("MaxPoolForward")) {
//     if (memrefOperands.size() >= 2) {
//       inputs.insert(memrefOperands[0]);
//       outputs.insert(memrefOperands[1]);
//     }
//   }
//   else if (funcName.contains("FullyConnectedForward")) {
//     if (memrefOperands.size() >= 4) {
//       inputs.insert(memrefOperands[0]);
//       inputs.insert(memrefOperands[1]);
//       inputs.insert(memrefOperands[2]);
//       outputs.insert(memrefOperands[3]);
//     }
//     else if (memrefOperands.size() == 3) {
//       inputs.insert(memrefOperands[0]);
//       inputs.insert(memrefOperands[1]);
//       outputs.insert(memrefOperands[2]);
//     }
//   }
//   else if (funcName.contains("MulScalar") || funcName.contains("AddScalar") || 
//            funcName.contains("SubScalar") || funcName.contains("RSubScalar")) {
//     if (memrefOperands.size() >= 3) {
//       inputs.insert(memrefOperands[0]);
//       inputs.insert(memrefOperands[1]);
//       outputs.insert(memrefOperands[2]);
//     }
//   }
//   else if (funcName.contains("Mul") || funcName.contains("Add") || funcName.contains("Sub")) {
//     if (memrefOperands.size() >= 3) {
//       inputs.insert(memrefOperands[0]);
//       inputs.insert(memrefOperands[1]);
//       outputs.insert(memrefOperands[2]);
//     }
//   }
//   else if (funcName.contains("Neg")) {
//     if (memrefOperands.size() >= 2) {
//       inputs.insert(memrefOperands[0]);
//       outputs.insert(memrefOperands[1]);
//     }
//   }
//   else {
//     for (unsigned i = 0; i < memrefOperands.size(); ++i) {
//       if (i == memrefOperands.size() - 1) {
//         outputs.insert(memrefOperands[i]);
//       } else {
//         inputs.insert(memrefOperands[i]);
//       }
//     }
//   }
// }

void extractCuLibsDependencies_plus(const llvm::SmallVector<Operation*, 4> &culibsOps,
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  Operation* mainCall = nullptr;
  for (Operation* op : culibsOps) {
    if (isCuLibsCall_plus(op)) {
      mainCall = op;
      break;
    }
  }
  
  if (!mainCall) {
    return;
  }
  
  auto callOp = cast<func::CallOp>(mainCall);
  StringRef funcName = callOp.getCallee();
  
  auto operands = callOp.getOperands();
  
  llvm::SmallVector<Value, 8> memrefOperands;
  
  for (Value operand : operands) {
    if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      Value intToPtrInput = intToPtrOp.getArg();
      if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
        Value indexCastInput = indexCastOp->getOperand(0);
        if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
          Value memref = extractOp.getSource();
          
          // 新增：追踪到原始的 memref，处理 reinterpret_cast 等转换
          Value sourceMemRef = traceMemRefSource_plus(memref);
          memrefOperands.push_back(sourceMemRef);
        }
      }
    }
  }
  
  // 保持原有的函数特定逻辑不变
  if (funcName.contains("Conv2dForward")) {
    if (memrefOperands.size() >= 4) {
      inputs.insert(memrefOperands[0]);
      inputs.insert(memrefOperands[1]);  
      inputs.insert(memrefOperands[2]);
      outputs.insert(memrefOperands[3]);
    }
  }
  else if (funcName.contains("MaxPoolForward")) {
    if (memrefOperands.size() >= 2) {
      inputs.insert(memrefOperands[0]);
      outputs.insert(memrefOperands[1]);
    }
  }
  else if (funcName.contains("FullyConnectedForward")) {
    if (memrefOperands.size() >= 4) {
      inputs.insert(memrefOperands[0]);
      inputs.insert(memrefOperands[1]);
      inputs.insert(memrefOperands[2]);
      outputs.insert(memrefOperands[3]);
    }
    else if (memrefOperands.size() == 3) {
      inputs.insert(memrefOperands[0]);
      inputs.insert(memrefOperands[1]);
      outputs.insert(memrefOperands[2]);
    }
  }
  else if (funcName.contains("MulScalar") || funcName.contains("AddScalar") || 
           funcName.contains("SubScalar") || funcName.contains("RSubScalar")) {
    if (memrefOperands.size() >= 3) {
      inputs.insert(memrefOperands[0]);
      inputs.insert(memrefOperands[1]);
      outputs.insert(memrefOperands[2]);
    }
  }
  else if (funcName.contains("Mul") || funcName.contains("Add") || funcName.contains("Sub")) {
    if (memrefOperands.size() >= 3) {
      inputs.insert(memrefOperands[0]);
      inputs.insert(memrefOperands[1]);
      outputs.insert(memrefOperands[2]);
    }
  }
  else if (funcName.contains("Neg")) {
    if (memrefOperands.size() >= 2) {
      inputs.insert(memrefOperands[0]);
      outputs.insert(memrefOperands[1]);
    }
  }
  else {
    for (unsigned i = 0; i < memrefOperands.size(); ++i) {
      if (i == memrefOperands.size() - 1) {
        outputs.insert(memrefOperands[i]);
      } else {
        inputs.insert(memrefOperands[i]);
      }
    }
  }
}

llvm::SmallVector<Operation*, 4> findCuLibsSequence_plus(Operation* streamCreateOp) {
  llvm::SmallVector<Operation*, 4> sequence;
  
  sequence.push_back(streamCreateOp);
  
  Value streamValue = streamCreateOp->getResult(0);
  
  Operation* currentOp = streamCreateOp;
  while (currentOp) {
    Operation* nextOp = currentOp->getNextNode();
    
    if (!nextOp) break;
    
    bool usesStream = false;
    for (Value operand : nextOp->getOperands()) {
      if (operand == streamValue) {
        usesStream = true;
        break;
      }
    }
    
    if (usesStream) {
      sequence.push_back(nextOp);
      
      if (isCuLibsStreamDestroy_plus(nextOp)) {
        break;
      }
    }
    
    currentOp = nextOp;
  }
  
  return sequence;
}

// ===== Main Graph Building Function =====

std::unique_ptr<EnhancedDependencyGraph> buildEnhancedDependencyGraph(func::FuncOp funcOp) {
  auto graph = std::make_unique<EnhancedDependencyGraph>();
  
  llvm::errs() << "Building enhanced dependency graph for function: " << funcOp.getName() << "\n";
  
  if (funcOp.getBody().empty()) {
    llvm::errs() << "Warning: Function " << funcOp.getName() << " has empty body, returning empty graph\n";
    return graph;
  }
  
  // Create program order mapping
  llvm::DenseMap<Operation*, unsigned> programOrder;
  unsigned orderCounter = 0;
  
  funcOp.walk([&](Operation* op) {
    programOrder[op] = orderCounter++;
  });

  // Track processed operations to avoid duplicates
  llvm::DenseSet<Operation*> processedOps;

  // First pass: create nodes for all kernels, loop nests, and culibs calls
  funcOp.walk([&](Operation* op) {
    if (processedOps.count(op)) {
      return WalkResult::advance();
    }
    
    if (isKernelLaunch_plus(op)) {
      auto kernelOp = cast<gpu::LaunchFuncOp>(op);
      auto node = std::make_unique<EnhancedDependencyNode>();
      node->type = NodeType::Kernel;
      node->op = op;
      node->kernelModuleName = kernelOp.getKernelModuleName().str();
      node->kernelName = kernelOp.getKernelName().str();
      
      extractKernelDependencies_plus(kernelOp, node->inputs, node->outputs);
      graph->addNode(std::move(node));
      processedOps.insert(op);
    } 
    else if (isLoopNest_plus(op)) {
      auto loopOp = cast<scf::ForOp>(op);
      auto node = std::make_unique<EnhancedDependencyNode>();
      node->type = NodeType::Loop;
      node->op = op;
      
      extractLoopDependencies_plus(loopOp, node->inputs, node->outputs);
      graph->addNode(std::move(node));
      processedOps.insert(op);
    }
    else if (isCuLibsStreamCreate_plus(op)) {
      auto culibsSequence = findCuLibsSequence_plus(op);
      
      Operation* mainCall = nullptr;
      for (Operation* seqOp : culibsSequence) {
        if (isCuLibsCall_plus(seqOp)) {
          mainCall = seqOp;
          break;
        }
      }
      
      if (mainCall) {
        auto node = std::make_unique<EnhancedDependencyNode>();
        node->type = NodeType::CuLibs;
        node->op = mainCall;
        node->culibsOps = culibsSequence;
        node->culibsFunctionName = cast<func::CallOp>(mainCall).getCallee().str();
        
        extractCuLibsDependencies_plus(culibsSequence, node->inputs, node->outputs);
        graph->addNode(std::move(node));
        
        for (Operation* seqOp : culibsSequence) {
          processedOps.insert(seqOp);
        }
      }
    }
    
    return WalkResult::advance();
  });
  
  llvm::errs() << "Built enhanced dependency graph with " << graph->nodes.size() << " nodes\n";
  
  // Second pass: create edges based on dependencies
  for (const auto &nodePair : graph->nodes) {
    EnhancedDependencyNode* node = nodePair.get();
    
    for (auto output : node->outputs) {
      for (const auto &otherNodePair : graph->nodes) {
        EnhancedDependencyNode* otherNode = otherNodePair.get();
        if (otherNode == node) continue;
        
        if (otherNode->inputs.count(output) && 
            programOrder[node->op] < programOrder[otherNode->op]) {
          graph->addEdge(node, otherNode);
        }
      }
    }
  }
  
  // Perform enhanced scheduling
  graph->performEnhancedScheduling();
  
  // Dump debug information
  // graph->dumpEnhancedGraph();
  
  return graph;
}

// ===== Analysis and Utility Functions =====

void analyzeSchedulingResults(const EnhancedDependencyGraph& graph) {
  // Collect statistics
  std::map<int, std::vector<EnhancedDependencyNode*>> parallelGroups;
  std::map<int, std::set<int>> subgraphsPerGroup;
  std::map<int, std::vector<EnhancedDependencyNode*>> nodesPerSubgraph;
  
  for (const auto& nodePtr : graph.nodes) {
    EnhancedDependencyNode* node = nodePtr.get();
    parallelGroups[node->parallelGroupId].push_back(node);
    subgraphsPerGroup[node->parallelGroupId].insert(node->subgraphId);
    nodesPerSubgraph[node->subgraphId].push_back(node);
  }

  // Print analysis
  llvm::errs() << "Scheduling Analysis:\n";
  llvm::errs() << "  Total nodes: " << graph.nodes.size() << "\n";
  llvm::errs() << "  Total subgraphs: " << graph.subgraphs.size() << "\n";
  llvm::errs() << "  Total parallel groups: " << parallelGroups.size() << "\n";

  // Analyze parallel efficiency
  int totalParallelizableNodes = 0;
  int maxParallelism = 0;
  
  for (const auto& [groupId, nodes] : parallelGroups) {
    int groupSize = nodes.size();
    totalParallelizableNodes += groupSize;
    maxParallelism = std::max(maxParallelism, groupSize);
    
    llvm::errs() << "  Group " << groupId << ": " << groupSize << " nodes";
    llvm::errs() << " across " << subgraphsPerGroup[groupId].size() << " subgraphs\n";
  }

  // Calculate parallelization metrics
  double avgParallelism = static_cast<double>(totalParallelizableNodes) / parallelGroups.size();
  double parallelEfficiency = avgParallelism / maxParallelism;

  llvm::errs() << "\nParallelization Metrics:\n";
  llvm::errs() << "  Average parallelism: " << avgParallelism << "\n";
  llvm::errs() << "  Maximum parallelism: " << maxParallelism << "\n";
  llvm::errs() << "  Parallel efficiency: " << (parallelEfficiency * 100) << "%\n";

  // Analyze subgraph characteristics
  llvm::errs() << "\nSubgraph Analysis:\n";
  for (const auto& [subgraphId, nodes] : nodesPerSubgraph) {
    llvm::errs() << "  Subgraph " << subgraphId << ": " << nodes.size() << " nodes";
    
    // Count node types in this subgraph
    int kernels = 0, loops = 0, culibs = 0;
    for (auto* node : nodes) {
      switch (node->type) {
        case NodeType::Kernel: kernels++; break;
        case NodeType::Loop: loops++; break;
        case NodeType::CuLibs: culibs++; break;
      }
    }
    
    llvm::errs() << " (K:" << kernels << ", L:" << loops << ", C:" << culibs << ")\n";
  }
}

void generateStreamAssignmentPlan(const EnhancedDependencyGraph& graph) {
  llvm::errs() << "Stream Assignment Plan:\n";

  // Group nodes by parallel group for stream assignment
  std::map<int, std::vector<EnhancedDependencyNode*>> parallelGroups;
  for (const auto& nodePtr : graph.nodes) {
    parallelGroups[nodePtr->parallelGroupId].push_back(nodePtr.get());
  }

  int streamCounter = 0;
  for (const auto& [groupId, nodes] : parallelGroups) {
    llvm::errs() << "\n  Parallel Group " << groupId << ":\n";
    
    // Group by subgraph within parallel group
    std::map<int, std::vector<EnhancedDependencyNode*>> subgraphGroups;
    for (auto* node : nodes) {
      subgraphGroups[node->subgraphId].push_back(node);
    }
    
    // Assign streams to subgraphs
    for (const auto& [subgraphId, subgraphNodes] : subgraphGroups) {
      int assignedStream = streamCounter++;
      
      llvm::errs() << "    Subgraph " << subgraphId << " -> Stream " << assignedStream << "\n";
      
      // Sort nodes within subgraph by intra-subgraph level
      auto sortedNodes = subgraphNodes;
      std::sort(sortedNodes.begin(), sortedNodes.end(),
                [](EnhancedDependencyNode* a, EnhancedDependencyNode* b) {
                  return a->intraSubgraphLevel < b->intraSubgraphLevel;
                });
      
      // Print execution order within stream
      for (int i = 0; i < sortedNodes.size(); ++i) {
        auto* node = sortedNodes[i];
        llvm::errs() << "      [" << (i + 1) << "] ";
        
        if (node->type == NodeType::Kernel) {
          llvm::errs() << "Kernel: " << node->kernelName;
        } else if (node->type == NodeType::Loop) {
          llvm::errs() << "Loop";
        } else if (node->type == NodeType::CuLibs) {
          llvm::errs() << "CuLibs: " << node->culibsFunctionName;
        }
        
        llvm::errs() << " (level " << node->intraSubgraphLevel << ")\n";
      }
    }
  }
  
  llvm::errs() << "\nTotal streams required: " << streamCounter << "\n";
}

void exportSchedulingInfo(const EnhancedDependencyGraph& graph, const std::string& functionName) {
  llvm::errs() << "Exporting scheduling information for function: " << functionName << "\n";
  
  // Create a structured representation that can be used by code generation
  struct ScheduleEntry {
    int parallelGroup;
    int subgraph;
    int intraLevel;
    std::string nodeType;
    std::string nodeInfo;
    Operation* op;
  };
  
  std::vector<ScheduleEntry> schedule;
  
  for (const auto& nodePtr : graph.nodes) {
    EnhancedDependencyNode* node = nodePtr.get();
    ScheduleEntry entry;
    entry.parallelGroup = node->parallelGroupId;
    entry.subgraph = node->subgraphId;
    entry.intraLevel = node->intraSubgraphLevel;
    entry.op = node->op;
    
    switch (node->type) {
      case NodeType::Kernel:
        entry.nodeType = "Kernel";
        entry.nodeInfo = node->kernelModuleName + "::" + node->kernelName;
        break;
      case NodeType::Loop:
        entry.nodeType = "Loop";
        entry.nodeInfo = "Loop nest";
        break;
      case NodeType::CuLibs:
        entry.nodeType = "CuLibs";
        entry.nodeInfo = node->culibsFunctionName;
        break;
    }
    
    schedule.push_back(entry);
  }
  
  // Sort by scheduling order
  std::sort(schedule.begin(), schedule.end(),
            [](const ScheduleEntry& a, const ScheduleEntry& b) {
              if (a.parallelGroup != b.parallelGroup) return a.parallelGroup < b.parallelGroup;
              if (a.subgraph != b.subgraph) return a.subgraph < b.subgraph;
              return a.intraLevel < b.intraLevel;
            });
  
  // Print exportable format
  llvm::errs() << "\nScheduling Table (CSV format):\n";
  llvm::errs() << "ParallelGroup,Subgraph,IntraLevel,NodeType,NodeInfo\n";
  
  for (const auto& entry : schedule) {
    llvm::errs() << entry.parallelGroup << "," 
                 << entry.subgraph << ","
                 << entry.intraLevel << ","
                 << entry.nodeType << ","
                 << entry.nodeInfo << "\n";
  }
  
  llvm::errs() << "\nScheduling information exported successfully.\n";
}

void performEnhancedSchedulingWorkflow(func::FuncOp funcOp) {
  llvm::errs() << "\n" << std::string(60, '=') << "\n";
  llvm::errs() << "ENHANCED DEPENDENCY GRAPH SCHEDULING WORKFLOW\n";
  llvm::errs() << "Function: " << funcOp.getName() << "\n";
  llvm::errs() << std::string(60, '=') << "\n";

  // Step 1: Build the enhanced dependency graph
  llvm::errs() << "\n[STEP 1] Building Enhanced Dependency Graph...\n";
  auto enhancedGraph = buildEnhancedDependencyGraph(funcOp);
  
  if (enhancedGraph->nodes.empty()) {
    llvm::errs() << "No nodes found in function, skipping scheduling.\n";
    return;
  }

  // Step 2: Analyze the scheduling results
  llvm::errs() << "\n[STEP 2] Analyzing Scheduling Results...\n";
  analyzeSchedulingResults(*enhancedGraph);

  // Step 3: Generate stream assignment plan
  llvm::errs() << "\n[STEP 3] Generating Stream Assignment Plan...\n";
  // generateStreamAssignmentPlan(*enhancedGraph);

  // Step 4: Export scheduling information
  llvm::errs() << "\n[STEP 4] Exporting Scheduling Information...\n";
  exportSchedulingInfo(*enhancedGraph, funcOp.getName().str());

  llvm::errs() << "\n" << std::string(60, '=') << "\n";
  llvm::errs() << "ENHANCED SCHEDULING WORKFLOW COMPLETED\n";
  llvm::errs() << std::string(60, '=') << "\n\n";
}

void integrateEnhancedScheduling(func::FuncOp funcOp) {
  llvm::errs() << "\nSTARTING ENHANCED DEPENDENCY ANALYSIS \n\n";
  
  try {
    performEnhancedSchedulingWorkflow(funcOp);
    llvm::errs() << "Enhanced dependency analysis completed successfully!\n\n";
  } catch (const std::exception& e) {
    llvm::errs() << "Error during enhanced dependency analysis: " << e.what() << "\n\n";
  }
}

} // namespace onnx_mlir