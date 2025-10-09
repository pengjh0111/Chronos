// // // #include "mlir/IR/Operation.h"
// // // #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// // // #include "mlir/Dialect/SCF/IR/SCF.h"
// // // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // // #include "mlir/Dialect/Func/IR/FuncOps.h"
// // // #include "mlir/Dialect/Arith/IR/Arith.h"
// // // #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// // // #include "llvm/ADT/DenseMap.h"
// // // #include "llvm/ADT/SmallVector.h"
// // // #include "llvm/ADT/SetVector.h"
// // // #include <vector>
// // // #include <set>

// // // #include "DependencyGraph.h"

// // // using namespace mlir;

// // // namespace onnx_mlir {

// // // // Implementation of DependencyGraph::addNode
// // // DependencyNode* DependencyGraph::addNode(std::unique_ptr<DependencyNode> node) {
// // //   DependencyNode* nodePtr = node.get();
// // //   opToNodeMap[node->op] = nodePtr;
  
// // //   // For culibs nodes, map all related operations
// // //   if (node->type == NodeType::CuLibs) {
// // //     for (Operation* culibsOp : node->culibsOps) {
// // //       opToNodeMap[culibsOp] = nodePtr;
// // //     }
// // //   }
  
// // //   nodes.push_back(std::move(node));
// // //   return nodePtr;
// // // }

// // // // Implementation of DependencyGraph::addEdge
// // // void DependencyGraph::addEdge(DependencyNode* from, DependencyNode* to) {
// // //   outEdges[from].push_back(to);
// // //   inEdges[to].push_back(from);
// // // }

// // // // Check if an operation is a kernel launch
// // // bool isKernelLaunch(Operation* op) {
// // //   return isa<gpu::LaunchFuncOp>(op);
// // // }

// // // // Check if an operation is a loop nest (outermost loop)
// // // bool isLoopNest(Operation* op) {
// // //   return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
// // // }

// // // // Check if an operation is a culibs function call
// // // bool isCuLibsCall(Operation* op) {
// // //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// // //     StringRef funcName = callOp.getCallee();
    
// // //     // Check for culibs function patterns based on your actual implementations
// // //     return funcName.starts_with("mgpuCudnn") || 
// // //            funcName.starts_with("mgpuCulibs");
// // //   }
// // //   return false;
// // // }

// // // // Check if an operation is culibs stream create
// // // bool isCuLibsStreamCreate(Operation* op) {
// // //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// // //     return callOp.getCallee() == "mgpuStreamCreate";
// // //   }
// // //   return false;
// // // }

// // // // Check if an operation is culibs stream sync
// // // bool isCuLibsStreamSync(Operation* op) {
// // //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// // //     return callOp.getCallee() == "mgpuStreamSynchronize";
// // //   }
// // //   return false;
// // // }

// // // // Check if an operation is culibs stream destroy
// // // bool isCuLibsStreamDestroy(Operation* op) {
// // //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// // //     return callOp.getCallee() == "mgpuStreamDestroy";
// // //   }
// // //   return false;
// // // }

// // // // Helper function: Find GPU function definition
// // // gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
// // //   // Get top-level module
// // //   ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
// // //   if (!topModule) {
// // //     return nullptr;
// // //   }

// // //   // Get kernel module and function name
// // //   StringRef kernelModuleName = kernelOp.getKernelModuleName();
// // //   StringRef kernelName = kernelOp.getKernelName();
  
// // //   // First try to find the gpu.module
// // //   gpu::GPUModuleOp gpuModule = nullptr;
// // //   topModule.walk([&](gpu::GPUModuleOp module) {
// // //     if (module.getName() == kernelModuleName) {
// // //       gpuModule = module;
// // //       return WalkResult::interrupt();
// // //     }
// // //     return WalkResult::advance();
// // //   });
  
// // //   // If GPU module is found, search for kernel function within it
// // //   if (gpuModule) {
// // //     gpu::GPUFuncOp kernelFunc = nullptr;
// // //     gpuModule.walk([&](gpu::GPUFuncOp func) {
// // //       if (func.getName() == kernelName) {
// // //         kernelFunc = func;
// // //         return WalkResult::interrupt();
// // //       }
// // //       return WalkResult::advance();
// // //     });
    
// // //     if (kernelFunc)
// // //       return kernelFunc;
// // //   }
  
// // //   // Fallback: Search throughout the entire top-level module
// // //   gpu::GPUFuncOp result = nullptr;
// // //   topModule.walk([&](gpu::GPUFuncOp func) {
// // //     if (func.getName() == kernelName) {
// // //       result = func;
// // //       return WalkResult::interrupt();
// // //     }
// // //     return WalkResult::advance();
// // //   });
  
// // //   return result;
// // // }

// // // // Extract memref inputs and outputs from a kernel launch, analyze kernel function definition
// // // void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
// // //                               llvm::SetVector<Value> &inputs,
// // //                               llvm::SetVector<Value> &outputs) {
// // //   // Find kernel function definition
// // //   gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
// // //   if (!kernelFunc) {
// // //     // If function not found, fall back to conservative analysis
// // //     llvm::errs() << "Warning: Could not find kernel function definition for \"" 
// // //                 << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
// // //     for (auto arg : kernelOp.getKernelOperands()) {
// // //       if (arg.getType().isa<MemRefType>()) {
// // //         inputs.insert(arg);
// // //         outputs.insert(arg);
// // //       }
// // //     }
// // //     return;
// // //   }

// // //   llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
// // //   // Count the number of MemRef type parameters and operands
// // //   unsigned memrefArgCount = 0;
// // //   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
// // //     if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
// // //       memrefArgCount++;
// // //     }
// // //   }
  
// // //   unsigned memrefOpCount = 0;
// // //   llvm::SmallVector<Value, 8> memrefOperands;
// // //   for (auto operand : kernelOp.getKernelOperands()) {
// // //     if (operand.getType().isa<MemRefType>()) {
// // //       memrefOperands.push_back(operand);
// // //       memrefOpCount++;
// // //     }
// // //   }
  
// // //   // Traverse all MemRef type function parameters
// // //   unsigned opIdx = 0;
// // //   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
// // //     BlockArgument arg = kernelFunc.getArgument(i);
// // //     if (arg.getType().isa<MemRefType>()) {
// // //       // Ensure operand index is within valid range
// // //       if (opIdx < memrefOperands.size()) {
// // //         Value operand = memrefOperands[opIdx++];
// // //         argOperandPairs.push_back({arg, operand});
// // //       }
// // //     }
// // //   }
  
// // //   // Track which parameters are used for load and store
// // //   llvm::DenseSet<BlockArgument> loadArgs;
// // //   llvm::DenseSet<BlockArgument> storeArgs;
  
// // //   // Analyze memory operations in kernel function body
// // //   unsigned loadCount = 0, storeCount = 0;
// // //   kernelFunc.walk([&](Operation *op) {
// // //     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
// // //       // Check if the loaded memref is a function parameter
// // //       Value memref = loadOp.getMemref();
// // //       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
// // //         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
// // //           loadArgs.insert(blockArg);
// // //           loadCount++;
// // //         }
// // //       }
// // //     } 
// // //     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
// // //       // Check if the stored memref is a function parameter
// // //       Value memref = storeOp.getMemref();
// // //       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
// // //         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
// // //           storeArgs.insert(blockArg);
// // //           storeCount++;
// // //         }
// // //       }
// // //     }
// // //   });
  
// // //   // Map function parameter analysis to kernel operands
// // //   for (auto &pair : argOperandPairs) {
// // //     BlockArgument arg = pair.first;
// // //     Value operand = pair.second;
    
// // //     bool isInput = loadArgs.count(arg) > 0;
// // //     bool isOutput = storeArgs.count(arg) > 0;
    
// // //     if (isInput) {
// // //       inputs.insert(operand);
// // //     }
    
// // //     if (isOutput) {
// // //       outputs.insert(operand);
// // //     }
    
// // //     // If the parameter is neither loaded nor stored, treat it as input to be conservative
// // //     if (!isInput && !isOutput) {
// // //       inputs.insert(operand);
// // //       llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
// // //     }
// // //   }
// // // }

// // // // Extract memref inputs and outputs from a loop nest
// // // void extractLoopDependencies(scf::ForOp loopOp,
// // //                            llvm::SetVector<Value> &inputs,
// // //                            llvm::SetVector<Value> &outputs) {
// // //   // Walk through the loop body to find all memref accesses
// // //   loopOp.walk([&](Operation* op) {
// // //     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
// // //       inputs.insert(loadOp.getMemref());
// // //     } 
// // //     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
// // //       outputs.insert(storeOp.getMemref());
// // //       // The stored value might also be a load from another memref
// // //       if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
// // //           storeOp.getValue().getDefiningOp())) {
// // //         inputs.insert(loadOp.getMemref());
// // //       }
// // //     }
// // //   });
// // // }

// // // // Extract memref inputs and outputs from culibs function calls
// // // void extractCuLibsDependencies(const llvm::SmallVector<Operation*, 4> &culibsOps,
// // //                               llvm::SetVector<Value> &inputs,
// // //                               llvm::SetVector<Value> &outputs) {
// // //   // Find the main culibs function call (not stream management calls)
// // //   Operation* mainCall = nullptr;
// // //   for (Operation* op : culibsOps) {
// // //     if (isCuLibsCall(op)) {
// // //       mainCall = op;
// // //       break;
// // //     }
// // //   }
  
// // //   if (!mainCall) {
// // //     return;
// // //   }
  
// // //   auto callOp = cast<func::CallOp>(mainCall);
// // //   StringRef funcName = callOp.getCallee();
  
// // //   // Analyze operands - the last few operands are typically memref pointers
// // //   auto operands = callOp.getOperands();
  
// // //   // For most culibs functions, the pattern is:
// // //   // - First several operands are scalar parameters (dimensions, strides, etc.)
// // //   // - Last few operands are memref pointers (input, weight, bias, output, stream)
// // //   // - Stream is always the last operand
  
// // //   llvm::SmallVector<Value, 8> memrefOperands;
  
// // //   // Extract memref operands by looking for LLVM pointer types that come from memref.extract_aligned_pointer_as_index
// // //   for (Value operand : operands) {
// // //     // Check if this operand comes from a memref pointer extraction
// // //     if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
// // //       Value intToPtrInput = intToPtrOp.getArg();
// // //       if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
// // //         // Use generic Operation API for IndexCastOp input
// // //         Value indexCastInput = indexCastOp->getOperand(0);
// // //         if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
// // //           Value memref = extractOp.getSource();
// // //           memrefOperands.push_back(memref);
// // //         }
// // //       }
// // //     }
// // //   }
  
// // //   // Function-specific dependency analysis based on your actual function implementations
// // //   if (funcName.contains("Conv2dForward")) {
// // //     // mgpuCudnnConv2dForward: input, weight, bias, output
// // //     if (memrefOperands.size() >= 4) {
// // //       inputs.insert(memrefOperands[0]);  // input (x_data)
// // //       inputs.insert(memrefOperands[1]);  // weight (w_data)  
// // //       inputs.insert(memrefOperands[2]);  // bias (bias_data)
// // //       outputs.insert(memrefOperands[3]); // output (y_data)
// // //     }
// // //   }
// // //   else if (funcName.contains("MaxPoolForward")) {
// // //     // mgpuCudnnMaxPoolForward: input, output
// // //     if (memrefOperands.size() >= 2) {
// // //       inputs.insert(memrefOperands[0]);  // input_data
// // //       outputs.insert(memrefOperands[1]); // output_data
// // //     }
// // //   }
// // //   else if (funcName.contains("FullyConnectedForward")) {
// // //     // mgpuCulibsFullyConnectedForward or mgpuCulibsFlattenFullyConnectedForward
// // //     // input, weight, bias, output
// // //     if (memrefOperands.size() >= 4) {
// // //       inputs.insert(memrefOperands[0]);  // input_data
// // //       inputs.insert(memrefOperands[1]);  // weight_data
// // //       inputs.insert(memrefOperands[2]);  // bias_data
// // //       outputs.insert(memrefOperands[3]); // output_data
// // //     }
// // //     else if (memrefOperands.size() == 3) {
// // //       // 无偏置的情况：input, weight, output, stream（bias_data是空指针，不产生memref操作数）
// // //       inputs.insert(memrefOperands[0]);  // input_data
// // //       inputs.insert(memrefOperands[1]);  // weight_data
// // //       outputs.insert(memrefOperands[2]); // output_data
// // //       // memrefOperands[3] 是 stream，不作为数据依赖处理
// // //     }
// // //   }
// // //   else if (funcName.contains("MulScalar") || funcName.contains("AddScalar") || 
// // //            funcName.contains("SubScalar") || funcName.contains("RSubScalar")) {
// // //     // Scalar operations: input, scalar, output
// // //     if (memrefOperands.size() >= 3) {
// // //       inputs.insert(memrefOperands[0]);  // input
// // //       inputs.insert(memrefOperands[1]);  // scalar
// // //       outputs.insert(memrefOperands[2]); // output
// // //     }
// // //   }
// // //   else if (funcName.contains("Mul") || funcName.contains("Add") || funcName.contains("Sub")) {
// // //     // Element-wise binary operations: inputA, inputB, output
// // //     if (memrefOperands.size() >= 3) {
// // //       inputs.insert(memrefOperands[0]);  // inputA
// // //       inputs.insert(memrefOperands[1]);  // inputB
// // //       outputs.insert(memrefOperands[2]); // output
// // //     }
// // //   }
// // //   else if (funcName.contains("Neg")) {
// // //     // Unary operations: input, output
// // //     if (memrefOperands.size() >= 2) {
// // //       inputs.insert(memrefOperands[0]);  // input
// // //       outputs.insert(memrefOperands[1]); // output
// // //     }
// // //   }
// // //   else {
// // //     // Conservative analysis for unknown functions
// // //     // Treat all but the last memref as inputs, and the last as output
// // //     for (unsigned i = 0; i < memrefOperands.size(); ++i) {
// // //       if (i == memrefOperands.size() - 1) {
// // //         outputs.insert(memrefOperands[i]); // Assume last is output
// // //       } else {
// // //         inputs.insert(memrefOperands[i]);  // Others are inputs
// // //       }
// // //     }
// // //   }
// // // }

// // // // Find culibs operation sequence starting from stream create
// // // llvm::SmallVector<Operation*, 4> findCuLibsSequence(Operation* streamCreateOp) {
// // //   llvm::SmallVector<Operation*, 4> sequence;
  
// // //   // Add stream create
// // //   sequence.push_back(streamCreateOp);
  
// // //   // Get the stream value
// // //   Value streamValue = streamCreateOp->getResult(0);
  
// // //   // Find operations that use this stream
// // //   Operation* currentOp = streamCreateOp;
// // //   while (currentOp) {
// // //     Operation* nextOp = currentOp->getNextNode();
    
// // //     if (!nextOp) break;
    
// // //     // Check if this operation uses the stream
// // //     bool usesStream = false;
// // //     for (Value operand : nextOp->getOperands()) {
// // //       if (operand == streamValue) {
// // //         usesStream = true;
// // //         break;
// // //       }
// // //     }
    
// // //     if (usesStream) {
// // //       sequence.push_back(nextOp);
      
// // //       // If this is stream destroy, we're done
// // //       if (isCuLibsStreamDestroy(nextOp)) {
// // //         break;
// // //       }
// // //     }
    
// // //     currentOp = nextOp;
// // //   }
  
// // //   return sequence;
// // // }

// // // // Print text representation of the dependency graph
// // // void dumpDependencyGraph(DependencyGraph &graph) {
// // //   llvm::errs() << "===== Dependency Graph =====\n";
  
// // //   // Print all nodes
// // //   llvm::errs() << "Nodes (" << graph.nodes.size() << " total):\n";
// // //   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
// // //     DependencyNode* node = graph.nodes[i].get();
    
// // //     llvm::errs() << "  [" << i << "] ";
// // //     if (node->type == NodeType::Kernel) {
// // //       llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
// // //     } else if (node->type == NodeType::Loop) {
// // //       llvm::errs() << "Loop at: ";
// // //       if (node->op) node->op->getLoc().print(llvm::errs());
// // //       else llvm::errs() << "<unknown location>";
// // //     } else if (node->type == NodeType::CuLibs) {
// // //       llvm::errs() << "CuLibs: " << node->culibsFunctionName 
// // //                    << " (ops: " << node->culibsOps.size() << ")";
// // //     }
// // //     llvm::errs() << "\n";
    
// // //     // Print input dependencies
// // //     llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
// // //     for (Value input : node->inputs) {
// // //       llvm::errs() << "      ";
// // //       input.print(llvm::errs());
// // //       llvm::errs() << "\n";
// // //     }
    
// // //     // Print output dependencies
// // //     llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
// // //     for (Value output : node->outputs) {
// // //       llvm::errs() << "      ";
// // //       output.print(llvm::errs());
// // //       llvm::errs() << "\n";
// // //     }
    
// // //     // Print topological sort level (if calculated)
// // //     if (node->topologicalLevel > 0) {
// // //       llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
// // //     }
    
// // //     llvm::errs() << "\n";
// // //   }
  
// // //   // Print all edges
// // //   llvm::errs() << "Edges:\n";
// // //   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
// // //     DependencyNode* node = graph.nodes[i].get();
    
// // //     // Get outgoing edges for this node
// // //     if (graph.outEdges.count(node)) {
// // //       const auto &edges = graph.outEdges[node];
// // //       if (!edges.empty()) {
// // //         llvm::errs() << "  From [" << i << "] to:\n";
        
// // //         for (DependencyNode* target : edges) {
// // //           // Find target node index
// // //           for (unsigned j = 0; j < graph.nodes.size(); ++j) {
// // //             if (graph.nodes[j].get() == target) {
// // //               llvm::errs() << "    [" << j << "]";
              
// // //               // Output shared memory references causing this dependency
// // //               bool foundSharedMem = false;
// // //               for (Value out : node->outputs) {
// // //                 for (Value in : target->inputs) {
// // //                   if (out == in) {
// // //                     if (!foundSharedMem) {
// // //                       llvm::errs() << " via: ";
// // //                       foundSharedMem = true;
// // //                     } else {
// // //                       llvm::errs() << ", ";
// // //                     }
// // //                     out.print(llvm::errs());
// // //                   }
// // //                 }
// // //               }
              
// // //               llvm::errs() << "\n";
// // //               break;
// // //             }
// // //           }
// // //         }
// // //       }
// // //     }
// // //   }
  
// // //   llvm::errs() << "===========================\n";
// // // }

// // // // Build the dependency graph from a function
// // // std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp) {
// // //   auto graph = std::make_unique<DependencyGraph>();
  
// // //   // 安全检查：确保函数体不为空
// // //   if (funcOp.getBody().empty()) {
// // //     llvm::errs() << "Warning: Function " << funcOp.getName() << " has empty body, returning empty graph\n";
// // //     return graph;
// // //   }
  
// // //   // Create program order mapping
// // //   llvm::DenseMap<Operation*, unsigned> programOrder;
// // //   unsigned orderCounter = 0;
  
// // //   // Traverse function body to collect operation order
// // //   funcOp.walk([&](Operation* op) {
// // //     programOrder[op] = orderCounter++;
// // //   });

// // //   // Track processed operations to avoid duplicates
// // //   llvm::DenseSet<Operation*> processedOps;

// // //   // First pass: create nodes for all kernels, loop nests, and culibs calls
// // //   funcOp.walk([&](Operation* op) {
// // //     if (processedOps.count(op)) {
// // //       return WalkResult::advance();
// // //     }
    
// // //     if (isKernelLaunch(op)) {
// // //       auto kernelOp = cast<gpu::LaunchFuncOp>(op);
// // //       auto node = std::make_unique<DependencyNode>();
// // //       node->type = NodeType::Kernel;
// // //       node->op = op;
// // //       node->kernelModuleName = kernelOp.getKernelModuleName();
// // //       node->kernelName = kernelOp.getKernelName();
      
// // //       extractKernelDependencies(kernelOp, node->inputs, node->outputs);
// // //       graph->addNode(std::move(node));
// // //       processedOps.insert(op);
// // //     } 
// // //     else if (isLoopNest(op)) {
// // //       auto loopOp = cast<scf::ForOp>(op);
// // //       auto node = std::make_unique<DependencyNode>();
// // //       node->type = NodeType::Loop;
// // //       node->op = op;
      
// // //       extractLoopDependencies(loopOp, node->inputs, node->outputs);
// // //       graph->addNode(std::move(node));
// // //       processedOps.insert(op);
// // //     }
// // //     else if (isCuLibsStreamCreate(op)) {
// // //       // Find the complete culibs sequence
// // //       auto culibsSequence = findCuLibsSequence(op);
      
// // //       // Find the main culibs function call
// // //       Operation* mainCall = nullptr;
// // //       for (Operation* seqOp : culibsSequence) {
// // //         if (isCuLibsCall(seqOp)) {
// // //           mainCall = seqOp;
// // //           break;
// // //         }
// // //       }
      
// // //       if (mainCall) {
// // //         auto node = std::make_unique<DependencyNode>();
// // //         node->type = NodeType::CuLibs;
// // //         node->op = mainCall;  // Use main call as representative operation
// // //         node->culibsOps = culibsSequence;
// // //         node->culibsFunctionName = cast<func::CallOp>(mainCall).getCallee();
        
// // //         extractCuLibsDependencies(culibsSequence, node->inputs, node->outputs);
// // //         graph->addNode(std::move(node));
        
// // //         // Mark all operations in sequence as processed
// // //         for (Operation* seqOp : culibsSequence) {
// // //           processedOps.insert(seqOp);
// // //         }
// // //       }
// // //     }
    
// // //     return WalkResult::advance();
// // //   });
  
// // //   // Debug: Print how many nodes we found
// // //   llvm::errs() << "Built dependency graph with " << graph->nodes.size() << " nodes\n";
  
// // //   // Second pass: create edges based on dependencies
// // //   for (const auto &nodePair : graph->nodes) {
// // //     DependencyNode* node = nodePair.get();
    
// // //     // For each output of this node
// // //     for (auto output : node->outputs) {
// // //       // Check if it's an input to any other node
// // //       for (const auto &otherNodePair : graph->nodes) {
// // //         DependencyNode* otherNode = otherNodePair.get();
// // //         if (otherNode == node) continue;
        
// // //         // If this node's output is another node's input, add an edge
// // //         if (otherNode->inputs.count(output) && 
// // //             programOrder[node->op] < programOrder[otherNode->op]) {
// // //           graph->addEdge(node, otherNode);
// // //         }
// // //       }
// // //     }
// // //   }
  
// // //   return graph;
// // // }

// // // } // namespace onnx_mlir

// // #include "mlir/IR/Operation.h"
// // #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// // #include "mlir/Dialect/SCF/IR/SCF.h"
// // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // #include "mlir/Dialect/Func/IR/FuncOps.h"
// // #include "mlir/Dialect/Arith/IR/Arith.h"
// // #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// // #include "llvm/ADT/DenseMap.h"
// // #include "llvm/ADT/SmallVector.h"
// // #include "llvm/ADT/SetVector.h"
// // #include <vector>
// // #include <set>

// // #include "DependencyGraph.h"

// // using namespace mlir;

// // namespace onnx_mlir {

// // // Implementation of DependencyGraph::addNode
// // DependencyNode* DependencyGraph::addNode(std::unique_ptr<DependencyNode> node) {
// //   DependencyNode* nodePtr = node.get();
// //   opToNodeMap[node->op] = nodePtr;
  
// //   // For culibs nodes, map all related operations
// //   if (node->type == NodeType::CuLibs) {
// //     for (Operation* culibsOp : node->culibsOps) {
// //       opToNodeMap[culibsOp] = nodePtr;
// //     }
// //   }
  
// //   nodes.push_back(std::move(node));
// //   return nodePtr;
// // }

// // // Implementation of DependencyGraph::addEdge
// // void DependencyGraph::addEdge(DependencyNode* from, DependencyNode* to) {
// //   outEdges[from].push_back(to);
// //   inEdges[to].push_back(from);
// // }

// // // Check if an operation is a kernel launch
// // bool isKernelLaunch(Operation* op) {
// //   return isa<gpu::LaunchFuncOp>(op);
// // }

// // // Check if an operation is a loop nest (outermost loop)
// // bool isLoopNest(Operation* op) {
// //   return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
// // }

// // // Check if an operation is a culibs function call
// // bool isCuLibsCall(Operation* op) {
// //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// //     StringRef funcName = callOp.getCallee();
    
// //     // Check for culibs function patterns based on your actual implementations
// //     return funcName.starts_with("mgpuCudnn") || 
// //            funcName.starts_with("mgpuCulibs");
// //   }
// //   return false;
// // }

// // // Check if an operation is culibs stream create
// // bool isCuLibsStreamCreate(Operation* op) {
// //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// //     return callOp.getCallee() == "mgpuStreamCreate";
// //   }
// //   return false;
// // }

// // // Check if an operation is culibs stream sync
// // bool isCuLibsStreamSync(Operation* op) {
// //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// //     return callOp.getCallee() == "mgpuStreamSynchronize";
// //   }
// //   return false;
// // }

// // // Check if an operation is culibs stream destroy
// // bool isCuLibsStreamDestroy(Operation* op) {
// //   if (auto callOp = dyn_cast<func::CallOp>(op)) {
// //     return callOp.getCallee() == "mgpuStreamDestroy";
// //   }
// //   return false;
// // }

// // // NEW: Check if an operation is a gpu.memcpy
// // bool isGpuMemcpy(Operation* op) {
// //   return isa<gpu::MemcpyOp>(op);
// // }

// // // NEW: Check if an operation is a memref.reinterpret_cast
// // bool isMemrefReinterpretCast(Operation* op) {
// //   return isa<memref::ReinterpretCastOp>(op);
// // }

// // // NEW: Check if an operation should be included in CuLibs sequence
// // bool shouldIncludeInCuLibsSequence(Operation* op) {
// //   return isGpuMemcpy(op) || 
// //          isa<gpu::WaitOp>(op) ||
// //          isa<memref::ReinterpretCastOp>(op) ||
// //          isa<memref::ExtractAlignedPointerAsIndexOp>(op) ||
// //          isa<arith::IndexCastOp>(op) ||
// //          isa<mlir::LLVM::IntToPtrOp>(op) ||
// //          isa<memref::AllocOp>(op) ||
// //          isa<arith::ConstantOp>(op);
// // }

// // // Helper function: Find GPU function definition
// // gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
// //   // Get top-level module
// //   ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
// //   if (!topModule) {
// //     return nullptr;
// //   }

// //   // Get kernel module and function name
// //   StringRef kernelModuleName = kernelOp.getKernelModuleName();
// //   StringRef kernelName = kernelOp.getKernelName();
  
// //   // First try to find the gpu.module
// //   gpu::GPUModuleOp gpuModule = nullptr;
// //   topModule.walk([&](gpu::GPUModuleOp module) {
// //     if (module.getName() == kernelModuleName) {
// //       gpuModule = module;
// //       return WalkResult::interrupt();
// //     }
// //     return WalkResult::advance();
// //   });
  
// //   // If GPU module is found, search for kernel function within it
// //   if (gpuModule) {
// //     gpu::GPUFuncOp kernelFunc = nullptr;
// //     gpuModule.walk([&](gpu::GPUFuncOp func) {
// //       if (func.getName() == kernelName) {
// //         kernelFunc = func;
// //         return WalkResult::interrupt();
// //       }
// //       return WalkResult::advance();
// //     });
    
// //     if (kernelFunc)
// //       return kernelFunc;
// //   }
  
// //   // Fallback: Search throughout the entire top-level module
// //   gpu::GPUFuncOp result = nullptr;
// //   topModule.walk([&](gpu::GPUFuncOp func) {
// //     if (func.getName() == kernelName) {
// //       result = func;
// //       return WalkResult::interrupt();
// //     }
// //     return WalkResult::advance();
// //   });
  
// //   return result;
// // }

// // // Extract memref inputs and outputs from a kernel launch, analyze kernel function definition
// // void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
// //                               llvm::SetVector<Value> &inputs,
// //                               llvm::SetVector<Value> &outputs) {
// //   // Find kernel function definition
// //   gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
// //   if (!kernelFunc) {
// //     // If function not found, fall back to conservative analysis
// //     llvm::errs() << "Warning: Could not find kernel function definition for \"" 
// //                 << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
// //     for (auto arg : kernelOp.getKernelOperands()) {
// //       if (arg.getType().isa<MemRefType>()) {
// //         inputs.insert(arg);
// //         outputs.insert(arg);
// //       }
// //     }
// //     return;
// //   }

// //   llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
// //   // Count the number of MemRef type parameters and operands
// //   unsigned memrefArgCount = 0;
// //   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
// //     if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
// //       memrefArgCount++;
// //     }
// //   }
  
// //   unsigned memrefOpCount = 0;
// //   llvm::SmallVector<Value, 8> memrefOperands;
// //   for (auto operand : kernelOp.getKernelOperands()) {
// //     if (operand.getType().isa<MemRefType>()) {
// //       memrefOperands.push_back(operand);
// //       memrefOpCount++;
// //     }
// //   }
  
// //   // Traverse all MemRef type function parameters
// //   unsigned opIdx = 0;
// //   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
// //     BlockArgument arg = kernelFunc.getArgument(i);
// //     if (arg.getType().isa<MemRefType>()) {
// //       // Ensure operand index is within valid range
// //       if (opIdx < memrefOperands.size()) {
// //         Value operand = memrefOperands[opIdx++];
// //         argOperandPairs.push_back({arg, operand});
// //       }
// //     }
// //   }
  
// //   // Track which parameters are used for load and store
// //   llvm::DenseSet<BlockArgument> loadArgs;
// //   llvm::DenseSet<BlockArgument> storeArgs;
  
// //   // Analyze memory operations in kernel function body
// //   unsigned loadCount = 0, storeCount = 0;
// //   kernelFunc.walk([&](Operation *op) {
// //     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
// //       // Check if the loaded memref is a function parameter
// //       Value memref = loadOp.getMemref();
// //       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
// //         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
// //           loadArgs.insert(blockArg);
// //           loadCount++;
// //         }
// //       }
// //     } 
// //     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
// //       // Check if the stored memref is a function parameter
// //       Value memref = storeOp.getMemref();
// //       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
// //         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
// //           storeArgs.insert(blockArg);
// //           storeCount++;
// //         }
// //       }
// //     }
// //   });
  
// //   // Map function parameter analysis to kernel operands
// //   for (auto &pair : argOperandPairs) {
// //     BlockArgument arg = pair.first;
// //     Value operand = pair.second;
    
// //     bool isInput = loadArgs.count(arg) > 0;
// //     bool isOutput = storeArgs.count(arg) > 0;
    
// //     if (isInput) {
// //       inputs.insert(operand);
// //     }
    
// //     if (isOutput) {
// //       outputs.insert(operand);
// //     }
    
// //     // If the parameter is neither loaded nor stored, treat it as input to be conservative
// //     if (!isInput && !isOutput) {
// //       inputs.insert(operand);
// //       llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
// //     }
// //   }
// // }

// // // Extract memref inputs and outputs from a loop nest
// // void extractLoopDependencies(scf::ForOp loopOp,
// //                            llvm::SetVector<Value> &inputs,
// //                            llvm::SetVector<Value> &outputs) {
// //   // Walk through the loop body to find all memref accesses
// //   loopOp.walk([&](Operation* op) {
// //     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
// //       inputs.insert(loadOp.getMemref());
// //     } 
// //     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
// //       outputs.insert(storeOp.getMemref());
// //       // The stored value might also be a load from another memref
// //       if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
// //           storeOp.getValue().getDefiningOp())) {
// //         inputs.insert(loadOp.getMemref());
// //       }
// //     }
// //   });
// // }

// // // Extract memref inputs and outputs from culibs function calls
// // void extractCuLibsDependencies(const llvm::SmallVector<Operation*, 4> &culibsOps,
// //                               llvm::SetVector<Value> &inputs,
// //                               llvm::SetVector<Value> &outputs) {
// //   // Find the main culibs function call (not stream management calls)
// //   Operation* mainCall = nullptr;
// //   for (Operation* op : culibsOps) {
// //     if (isCuLibsCall(op)) {
// //       mainCall = op;
// //       break;
// //     }
// //   }
  
// //   if (!mainCall) {
// //     return;
// //   }
  
// //   auto callOp = cast<func::CallOp>(mainCall);
// //   StringRef funcName = callOp.getCallee();
  
// //   // Analyze operands - the last few operands are typically memref pointers
// //   auto operands = callOp.getOperands();
  
// //   // For most culibs functions, the pattern is:
// //   // - First several operands are scalar parameters (dimensions, strides, etc.)
// //   // - Last few operands are memref pointers (input, weight, bias, output, stream)
// //   // - Stream is always the last operand
  
// //   llvm::SmallVector<Value, 8> memrefOperands;
  
// //   // Extract memref operands by looking for LLVM pointer types that come from memref.extract_aligned_pointer_as_index
// //   for (Value operand : operands) {
// //     // Check if this operand comes from a memref pointer extraction
// //     if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
// //       Value intToPtrInput = intToPtrOp.getArg();
// //       if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
// //         // Use generic Operation API for IndexCastOp input
// //         Value indexCastInput = indexCastOp->getOperand(0);
// //         if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
// //           Value memref = extractOp.getSource();
// //           memrefOperands.push_back(memref);
// //         }
// //       }
// //     }
// //   }
  
// //   // NEW: Also extract memrefs from related gpu.memcpy operations
// //   for (Operation* op : culibsOps) {
// //     if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(op)) {
// //       Value src = memcpyOp.getSrc();
// //       Value dst = memcpyOp.getDst();
      
// //       // Determine if this is input or output memcpy based on position relative to main call
// //       if (op->isBeforeInBlock(mainCall)) {
// //         // Input memcpy: src is input, dst might be used in culibs call
// //         inputs.insert(src);
// //         // The dst might be used as input to the culibs call
// //         inputs.insert(dst);
// //       } else {
// //         // Output memcpy: src might be output from culibs call, dst is final output
// //         outputs.insert(dst);
// //         // The src might be output from the culibs call
// //         outputs.insert(src);
// //       }
// //     }
// //   }
  
// //   // Function-specific dependency analysis based on your actual function implementations
// //   if (funcName.contains("Conv2dForward")) {
// //     // mgpuCudnnConv2dForward: input, weight, bias, output
// //     if (memrefOperands.size() >= 4) {
// //       inputs.insert(memrefOperands[0]);  // input (x_data)
// //       inputs.insert(memrefOperands[1]);  // weight (w_data)  
// //       inputs.insert(memrefOperands[2]);  // bias (bias_data)
// //       outputs.insert(memrefOperands[3]); // output (y_data)
// //     }
// //   }
// //   else if (funcName.contains("MaxPoolForward")) {
// //     // mgpuCudnnMaxPoolForward: input, output
// //     if (memrefOperands.size() >= 2) {
// //       inputs.insert(memrefOperands[0]);  // input_data
// //       outputs.insert(memrefOperands[1]); // output_data
// //     }
// //   }
// //   else if (funcName.contains("FullyConnectedForward")) {
// //     // mgpuCulibsFullyConnectedForward or mgpuCulibsFlattenFullyConnectedForward
// //     // input, weight, bias, output
// //     if (memrefOperands.size() >= 4) {
// //       inputs.insert(memrefOperands[0]);  // input_data
// //       inputs.insert(memrefOperands[1]);  // weight_data
// //       inputs.insert(memrefOperands[2]);  // bias_data
// //       outputs.insert(memrefOperands[3]); // output_data
// //     }
// //     else if (memrefOperands.size() == 3) {
// //       // 无偏置的情况：input, weight, output, stream（bias_data是空指针，不产生memref操作数）
// //       inputs.insert(memrefOperands[0]);  // input_data
// //       inputs.insert(memrefOperands[1]);  // weight_data
// //       outputs.insert(memrefOperands[2]); // output_data
// //       // memrefOperands[3] 是 stream，不作为数据依赖处理
// //     }
// //   }
// //   else if (funcName.contains("MulScalar") || funcName.contains("AddScalar") || 
// //            funcName.contains("SubScalar") || funcName.contains("RSubScalar")) {
// //     // Scalar operations: input, scalar, output
// //     if (memrefOperands.size() >= 3) {
// //       inputs.insert(memrefOperands[0]);  // input
// //       inputs.insert(memrefOperands[1]);  // scalar
// //       outputs.insert(memrefOperands[2]); // output
// //     }
// //   }
// //   else if (funcName.contains("Mul") || funcName.contains("Add") || funcName.contains("Sub")) {
// //     // Element-wise binary operations: inputA, inputB, output
// //     if (memrefOperands.size() >= 3) {
// //       inputs.insert(memrefOperands[0]);  // inputA
// //       inputs.insert(memrefOperands[1]);  // inputB
// //       outputs.insert(memrefOperands[2]); // output
// //     }
// //   }
// //   else if (funcName.contains("Neg")) {
// //     // Unary operations: input, output
// //     if (memrefOperands.size() >= 2) {
// //       inputs.insert(memrefOperands[0]);  // input
// //       outputs.insert(memrefOperands[1]); // output
// //     }
// //   }
// //   else {
// //     // Conservative analysis for unknown functions
// //     // Treat all but the last memref as inputs, and the last as output
// //     for (unsigned i = 0; i < memrefOperands.size(); ++i) {
// //       if (i == memrefOperands.size() - 1) {
// //         outputs.insert(memrefOperands[i]); // Assume last is output
// //       } else {
// //         inputs.insert(memrefOperands[i]);  // Others are inputs
// //       }
// //     }
// //   }
// // }

// // // MODIFIED: Find extended culibs operation sequence including memcpy and reinterpret_cast
// // llvm::SmallVector<Operation*, 4> findExtendedCuLibsSequence(Operation* streamCreateOp) {
// //   llvm::SmallVector<Operation*, 4> sequence;
  
// //   // First, collect operations before stream create that should be included
// //   Operation* currentOp = streamCreateOp->getPrevNode();
// //   llvm::SmallVector<Operation*, 8> beforeOps;
  
// //   // Look backwards for related operations
// //   while (currentOp) {
// //     if (shouldIncludeInCuLibsSequence(currentOp)) {
// //       beforeOps.insert(beforeOps.begin(), currentOp);
// //       currentOp = currentOp->getPrevNode();
// //     } else {
// //       // Stop when we hit an unrelated operation
// //       break;
// //     }
// //   }
  
// //   // Add the before operations to sequence
// //   sequence.append(beforeOps.begin(), beforeOps.end());
  
// //   // Add stream create
// //   sequence.push_back(streamCreateOp);
  
// //   // Get the stream value
// //   Value streamValue = streamCreateOp->getResult(0);
  
// //   // Find operations that use this stream (original logic)
// //   currentOp = streamCreateOp;
// //   while (currentOp) {
// //     Operation* nextOp = currentOp->getNextNode();
    
// //     if (!nextOp) break;
    
// //     // Check if this operation uses the stream
// //     bool usesStream = false;
// //     for (Value operand : nextOp->getOperands()) {
// //       if (operand == streamValue) {
// //         usesStream = true;
// //         break;
// //       }
// //     }
    
// //     if (usesStream) {
// //       sequence.push_back(nextOp);
      
// //       // If this is stream destroy, continue looking for post operations
// //       if (isCuLibsStreamDestroy(nextOp)) {
// //         // Look for operations after stream destroy (like output memcpy)
// //         Operation* postOp = nextOp->getNextNode();
// //         while (postOp) {
// //           if (isGpuMemcpy(postOp) || isa<gpu::WaitOp>(postOp)) {
// //             sequence.push_back(postOp);
// //             postOp = postOp->getNextNode();
// //           } else {
// //             // Stop when we hit an unrelated operation
// //             break;
// //           }
// //         }
// //         break;
// //       }
// //     }
    
// //     currentOp = nextOp;
// //   }
  
// //   return sequence;
// // }

// // // Keep the original function for compatibility, but use the extended version
// // llvm::SmallVector<Operation*, 4> findCuLibsSequence(Operation* streamCreateOp) {
// //   return findExtendedCuLibsSequence(streamCreateOp);
// // }

// // // Print text representation of the dependency graph
// // void dumpDependencyGraph(DependencyGraph &graph) {
// //   llvm::errs() << "===== Dependency Graph =====\n";
  
// //   // Print all nodes
// //   llvm::errs() << "Nodes (" << graph.nodes.size() << " total):\n";
// //   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
// //     DependencyNode* node = graph.nodes[i].get();
    
// //     llvm::errs() << "  [" << i << "] ";
// //     if (node->type == NodeType::Kernel) {
// //       llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
// //     } else if (node->type == NodeType::Loop) {
// //       llvm::errs() << "Loop at: ";
// //       if (node->op) node->op->getLoc().print(llvm::errs());
// //       else llvm::errs() << "<unknown location>";
// //     } else if (node->type == NodeType::CuLibs) {
// //       llvm::errs() << "CuLibs: " << node->culibsFunctionName 
// //                    << " (ops: " << node->culibsOps.size() << ")";
// //     }
// //     llvm::errs() << "\n";
    
// //     // Print input dependencies
// //     llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
// //     for (Value input : node->inputs) {
// //       llvm::errs() << "      ";
// //       input.print(llvm::errs());
// //       llvm::errs() << "\n";
// //     }
    
// //     // Print output dependencies
// //     llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
// //     for (Value output : node->outputs) {
// //       llvm::errs() << "      ";
// //       output.print(llvm::errs());
// //       llvm::errs() << "\n";
// //     }
    
// //     // Print topological sort level (if calculated)
// //     if (node->topologicalLevel > 0) {
// //       llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
// //     }
    
// //     llvm::errs() << "\n";
// //   }
  
// //   // Print all edges
// //   llvm::errs() << "Edges:\n";
// //   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
// //     DependencyNode* node = graph.nodes[i].get();
    
// //     // Get outgoing edges for this node
// //     if (graph.outEdges.count(node)) {
// //       const auto &edges = graph.outEdges[node];
// //       if (!edges.empty()) {
// //         llvm::errs() << "  From [" << i << "] to:\n";
        
// //         for (DependencyNode* target : edges) {
// //           // Find target node index
// //           for (unsigned j = 0; j < graph.nodes.size(); ++j) {
// //             if (graph.nodes[j].get() == target) {
// //               llvm::errs() << "    [" << j << "]";
              
// //               // Output shared memory references causing this dependency
// //               bool foundSharedMem = false;
// //               for (Value out : node->outputs) {
// //                 for (Value in : target->inputs) {
// //                   if (out == in) {
// //                     if (!foundSharedMem) {
// //                       llvm::errs() << " via: ";
// //                       foundSharedMem = true;
// //                     } else {
// //                       llvm::errs() << ", ";
// //                     }
// //                     out.print(llvm::errs());
// //                   }
// //                 }
// //               }
              
// //               llvm::errs() << "\n";
// //               break;
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
  
// //   llvm::errs() << "===========================\n";
// // }

// // // Build the dependency graph from a function
// // std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp) {
// //   auto graph = std::make_unique<DependencyGraph>();
  
// //   // 安全检查：确保函数体不为空
// //   if (funcOp.getBody().empty()) {
// //     llvm::errs() << "Warning: Function " << funcOp.getName() << " has empty body, returning empty graph\n";
// //     return graph;
// //   }
  
// //   // Create program order mapping
// //   llvm::DenseMap<Operation*, unsigned> programOrder;
// //   unsigned orderCounter = 0;
  
// //   // Traverse function body to collect operation order
// //   funcOp.walk([&](Operation* op) {
// //     programOrder[op] = orderCounter++;
// //   });

// //   // Track processed operations to avoid duplicates
// //   llvm::DenseSet<Operation*> processedOps;

// //   // First pass: create nodes for all kernels, loop nests, and culibs calls
// //   funcOp.walk([&](Operation* op) {
// //     if (processedOps.count(op)) {
// //       return WalkResult::advance();
// //     }
    
// //     if (isKernelLaunch(op)) {
// //       auto kernelOp = cast<gpu::LaunchFuncOp>(op);
// //       auto node = std::make_unique<DependencyNode>();
// //       node->type = NodeType::Kernel;
// //       node->op = op;
// //       node->kernelModuleName = kernelOp.getKernelModuleName();
// //       node->kernelName = kernelOp.getKernelName();
      
// //       extractKernelDependencies(kernelOp, node->inputs, node->outputs);
// //       graph->addNode(std::move(node));
// //       processedOps.insert(op);
// //     } 
// //     else if (isLoopNest(op)) {
// //       auto loopOp = cast<scf::ForOp>(op);
// //       auto node = std::make_unique<DependencyNode>();
// //       node->type = NodeType::Loop;
// //       node->op = op;
      
// //       extractLoopDependencies(loopOp, node->inputs, node->outputs);
// //       graph->addNode(std::move(node));
// //       processedOps.insert(op);
// //     }
// //     else if (isCuLibsStreamCreate(op)) {
// //       // Find the complete culibs sequence (now includes memcpy and reinterpret_cast)
// //       auto culibsSequence = findExtendedCuLibsSequence(op);
      
// //       // Find the main culibs function call
// //       Operation* mainCall = nullptr;
// //       for (Operation* seqOp : culibsSequence) {
// //         if (isCuLibsCall(seqOp)) {
// //           mainCall = seqOp;
// //           break;
// //         }
// //       }
      
// //       if (mainCall) {
// //         auto node = std::make_unique<DependencyNode>();
// //         node->type = NodeType::CuLibs;
// //         node->op = mainCall;  // Use main call as representative operation
// //         node->culibsOps = culibsSequence;
// //         node->culibsFunctionName = cast<func::CallOp>(mainCall).getCallee();
        
// //         extractCuLibsDependencies(culibsSequence, node->inputs, node->outputs);
// //         graph->addNode(std::move(node));
        
// //         // Mark all operations in sequence as processed
// //         for (Operation* seqOp : culibsSequence) {
// //           processedOps.insert(seqOp);
// //         }
// //       }
// //     }
    
// //     return WalkResult::advance();
// //   });
  
// //   // Debug: Print how many nodes we found
// //   llvm::errs() << "Built dependency graph with " << graph->nodes.size() << " nodes\n";
  
// //   // Second pass: create edges based on dependencies
// //   for (const auto &nodePair : graph->nodes) {
// //     DependencyNode* node = nodePair.get();
    
// //     // For each output of this node
// //     for (auto output : node->outputs) {
// //       // Check if it's an input to any other node
// //       for (const auto &otherNodePair : graph->nodes) {
// //         DependencyNode* otherNode = otherNodePair.get();
// //         if (otherNode == node) continue;
        
// //         // If this node's output is another node's input, add an edge
// //         if (otherNode->inputs.count(output) && 
// //             programOrder[node->op] < programOrder[otherNode->op]) {
// //           graph->addEdge(node, otherNode);
// //         }
// //       }
// //     }
// //   }
  
// //   return graph;
// // }

// // } // namespace onnx_mlir

// #include "mlir/IR/Operation.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/ADT/SetVector.h"
// #include <vector>
// #include <set>

// #include "DependencyGraph.h"

// using namespace mlir;

// namespace onnx_mlir {

// // Implementation of DependencyGraph::addNode
// DependencyNode* DependencyGraph::addNode(std::unique_ptr<DependencyNode> node) {
//   DependencyNode* nodePtr = node.get();
//   opToNodeMap[node->op] = nodePtr;
  
//   // For culibs nodes, map all related operations
//   if (node->type == NodeType::CuLibs) {
//     for (Operation* culibsOp : node->culibsOps) {
//       opToNodeMap[culibsOp] = nodePtr;
//     }
//   }
  
//   nodes.push_back(std::move(node));
//   return nodePtr;
// }

// // Implementation of DependencyGraph::addEdge
// void DependencyGraph::addEdge(DependencyNode* from, DependencyNode* to) {
//   outEdges[from].push_back(to);
//   inEdges[to].push_back(from);
// }

// // Check if an operation is a kernel launch
// bool isKernelLaunch(Operation* op) {
//   return isa<gpu::LaunchFuncOp>(op);
// }

// // Check if an operation is a loop nest (outermost loop)
// bool isLoopNest(Operation* op) {
//   return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
// }

// // Check if an operation is a culibs function call
// bool isCuLibsCall(Operation* op) {
//   if (auto callOp = dyn_cast<func::CallOp>(op)) {
//     StringRef funcName = callOp.getCallee();
    
//     // Check for culibs function patterns based on your actual implementations
//     return funcName.starts_with("mgpuCudnn") || 
//            funcName.starts_with("mgpuCulibs");
//   }
//   return false;
// }

// // Check if an operation is culibs stream create
// bool isCuLibsStreamCreate(Operation* op) {
//   if (auto callOp = dyn_cast<func::CallOp>(op)) {
//     return callOp.getCallee() == "mgpuStreamCreate";
//   }
//   return false;
// }

// // Check if an operation is culibs stream sync
// bool isCuLibsStreamSync(Operation* op) {
//   if (auto callOp = dyn_cast<func::CallOp>(op)) {
//     return callOp.getCallee() == "mgpuStreamSynchronize";
//   }
//   return false;
// }

// // Check if an operation is culibs stream destroy
// bool isCuLibsStreamDestroy(Operation* op) {
//   if (auto callOp = dyn_cast<func::CallOp>(op)) {
//     return callOp.getCallee() == "mgpuStreamDestroy";
//   }
//   return false;
// }

// // NEW: Check if an operation is a gpu.memcpy
// bool isGpuMemcpy(Operation* op) {
//   return isa<gpu::MemcpyOp>(op);
// }

// // NEW: Check if an operation is a memref.reinterpret_cast
// bool isMemrefReinterpretCast(Operation* op) {
//   return isa<memref::ReinterpretCastOp>(op);
// }

// // NEW: Check if an operation should be included in CuLibs sequence
// bool shouldIncludeInCuLibsSequence(Operation* op) {
//   return isGpuMemcpy(op) || 
//          isa<gpu::WaitOp>(op) ||
//          isa<memref::ReinterpretCastOp>(op) ||
//          isa<memref::ExtractAlignedPointerAsIndexOp>(op) ||
//          isa<arith::IndexCastOp>(op) ||
//          isa<mlir::LLVM::IntToPtrOp>(op) ||
//          isa<memref::AllocOp>(op) ||
//          isa<arith::ConstantOp>(op);
// }

// // Helper function: Find GPU function definition
// gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
//   // Get top-level module
//   ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
//   if (!topModule) {
//     return nullptr;
//   }

//   // Get kernel module and function name
//   StringRef kernelModuleName = kernelOp.getKernelModuleName();
//   StringRef kernelName = kernelOp.getKernelName();
  
//   // First try to find the gpu.module
//   gpu::GPUModuleOp gpuModule = nullptr;
//   topModule.walk([&](gpu::GPUModuleOp module) {
//     if (module.getName() == kernelModuleName) {
//       gpuModule = module;
//       return WalkResult::interrupt();
//     }
//     return WalkResult::advance();
//   });
  
//   // If GPU module is found, search for kernel function within it
//   if (gpuModule) {
//     gpu::GPUFuncOp kernelFunc = nullptr;
//     gpuModule.walk([&](gpu::GPUFuncOp func) {
//       if (func.getName() == kernelName) {
//         kernelFunc = func;
//         return WalkResult::interrupt();
//       }
//       return WalkResult::advance();
//     });
    
//     if (kernelFunc)
//       return kernelFunc;
//   }
  
//   // Fallback: Search throughout the entire top-level module
//   gpu::GPUFuncOp result = nullptr;
//   topModule.walk([&](gpu::GPUFuncOp func) {
//     if (func.getName() == kernelName) {
//       result = func;
//       return WalkResult::interrupt();
//     }
//     return WalkResult::advance();
//   });
  
//   return result;
// }

// // Extract memref inputs and outputs from a kernel launch, analyze kernel function definition
// void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs) {
//   // Find kernel function definition
//   gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
//   if (!kernelFunc) {
//     // If function not found, fall back to conservative analysis
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
  
//   // Count the number of MemRef type parameters and operands
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
  
//   // Traverse all MemRef type function parameters
//   unsigned opIdx = 0;
//   for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
//     BlockArgument arg = kernelFunc.getArgument(i);
//     if (arg.getType().isa<MemRefType>()) {
//       // Ensure operand index is within valid range
//       if (opIdx < memrefOperands.size()) {
//         Value operand = memrefOperands[opIdx++];
//         argOperandPairs.push_back({arg, operand});
//       }
//     }
//   }
  
//   // Track which parameters are used for load and store
//   llvm::DenseSet<BlockArgument> loadArgs;
//   llvm::DenseSet<BlockArgument> storeArgs;
  
//   // Analyze memory operations in kernel function body
//   unsigned loadCount = 0, storeCount = 0;
//   kernelFunc.walk([&](Operation *op) {
//     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
//       // Check if the loaded memref is a function parameter
//       Value memref = loadOp.getMemref();
//       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
//         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
//           loadArgs.insert(blockArg);
//           loadCount++;
//         }
//       }
//     } 
//     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
//       // Check if the stored memref is a function parameter
//       Value memref = storeOp.getMemref();
//       if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
//         if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
//           storeArgs.insert(blockArg);
//           storeCount++;
//         }
//       }
//     }
//   });
  
//   // Map function parameter analysis to kernel operands
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
    
//     // If the parameter is neither loaded nor stored, treat it as input to be conservative
//     if (!isInput && !isOutput) {
//       inputs.insert(operand);
//       llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
//     }
//   }
// }

// // Extract memref inputs and outputs from a loop nest
// void extractLoopDependencies(scf::ForOp loopOp,
//                            llvm::SetVector<Value> &inputs,
//                            llvm::SetVector<Value> &outputs) {
//   // Walk through the loop body to find all memref accesses
//   loopOp.walk([&](Operation* op) {
//     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
//       inputs.insert(loadOp.getMemref());
//     } 
//     else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
//       outputs.insert(storeOp.getMemref());
//       // The stored value might also be a load from another memref
//       if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
//           storeOp.getValue().getDefiningOp())) {
//         inputs.insert(loadOp.getMemref());
//       }
//     }
//   });
// }

// // Extract memref inputs and outputs from culibs function calls
// void extractCuLibsDependencies(const llvm::SmallVector<Operation*, 4> &culibsOps,
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs) {
//   // Find the main culibs function call (not stream management calls)
//   Operation* mainCall = nullptr;
//   for (Operation* op : culibsOps) {
//     if (isCuLibsCall(op)) {
//       mainCall = op;
//       break;
//     }
//   }
  
//   if (!mainCall) {
//     return;
//   }
  
//   auto callOp = cast<func::CallOp>(mainCall);
//   StringRef funcName = callOp.getCallee();
  
//   // Analyze operands - the last few operands are typically memref pointers
//   auto operands = callOp.getOperands();
  
//   // For most culibs functions, the pattern is:
//   // - First several operands are scalar parameters (dimensions, strides, etc.)
//   // - Last few operands are memref pointers (input, weight, bias, output, stream)
//   // - Stream is always the last operand
  
//   llvm::SmallVector<Value, 8> memrefOperands;
  
//   // Extract memref operands by looking for LLVM pointer types that come from memref.extract_aligned_pointer_as_index
//   for (Value operand : operands) {
//     // Check if this operand comes from a memref pointer extraction
//     if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
//       Value intToPtrInput = intToPtrOp.getArg();
//       if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
//         // Use generic Operation API for IndexCastOp input
//         Value indexCastInput = indexCastOp->getOperand(0);
//         if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
//           Value memref = extractOp.getSource();
//           memrefOperands.push_back(memref);
//         }
//       }
//     }
//   }
  
//   // NEW: Also extract memrefs from related gpu.memcpy operations
//   for (Operation* op : culibsOps) {
//     if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(op)) {
//       Value src = memcpyOp.getSrc();
//       Value dst = memcpyOp.getDst();
      
//       // Determine if this is input or output memcpy based on position relative to main call
//       if (op->isBeforeInBlock(mainCall)) {
//         // Input memcpy: src is input, dst might be used in culibs call
//         inputs.insert(src);
//         // The dst might be used as input to the culibs call
//         inputs.insert(dst);
//       } else {
//         // Output memcpy: src might be output from culibs call, dst is final output
//         outputs.insert(dst);
//         // The src might be output from the culibs call
//         outputs.insert(src);
//       }
//     }
//   }
  
//   // Function-specific dependency analysis based on your actual function implementations
//   if (funcName.contains("Conv2dForward")) {
//     // mgpuCudnnConv2dForward: input, weight, bias, output
//     if (memrefOperands.size() >= 4) {
//       inputs.insert(memrefOperands[0]);  // input (x_data)
//       inputs.insert(memrefOperands[1]);  // weight (w_data)  
//       inputs.insert(memrefOperands[2]);  // bias (bias_data)
//       outputs.insert(memrefOperands[3]); // output (y_data)
//     }
//   }
//   else if (funcName.contains("MaxPoolForward")) {
//     // mgpuCudnnMaxPoolForward: input, output
//     if (memrefOperands.size() >= 2) {
//       inputs.insert(memrefOperands[0]);  // input_data
//       outputs.insert(memrefOperands[1]); // output_data
//     }
//   }
//   else if (funcName.contains("FullyConnectedForward")) {
//     // mgpuCulibsFullyConnectedForward or mgpuCulibsFlattenFullyConnectedForward
//     // input, weight, bias, output
//     if (memrefOperands.size() >= 4) {
//       inputs.insert(memrefOperands[0]);  // input_data
//       inputs.insert(memrefOperands[1]);  // weight_data
//       inputs.insert(memrefOperands[2]);  // bias_data
//       outputs.insert(memrefOperands[3]); // output_data
//     }
//     else if (memrefOperands.size() == 3) {
//       // 无偏置的情况：input, weight, output, stream（bias_data是空指针，不产生memref操作数）
//       inputs.insert(memrefOperands[0]);  // input_data
//       inputs.insert(memrefOperands[1]);  // weight_data
//       outputs.insert(memrefOperands[2]); // output_data
//       // memrefOperands[3] 是 stream，不作为数据依赖处理
//     }
//   }
//   else if (funcName.contains("MulScalar") || funcName.contains("AddScalar") || 
//            funcName.contains("SubScalar") || funcName.contains("RSubScalar")) {
//     // Scalar operations: input, scalar, output
//     if (memrefOperands.size() >= 3) {
//       inputs.insert(memrefOperands[0]);  // input
//       inputs.insert(memrefOperands[1]);  // scalar
//       outputs.insert(memrefOperands[2]); // output
//     }
//   }
//   else if (funcName.contains("Mul") || funcName.contains("Add") || funcName.contains("Sub")) {
//     // Element-wise binary operations: inputA, inputB, output
//     if (memrefOperands.size() >= 3) {
//       inputs.insert(memrefOperands[0]);  // inputA
//       inputs.insert(memrefOperands[1]);  // inputB
//       outputs.insert(memrefOperands[2]); // output
//     }
//   }
//   else if (funcName.contains("Neg")) {
//     // Unary operations: input, output
//     if (memrefOperands.size() >= 2) {
//       inputs.insert(memrefOperands[0]);  // input
//       outputs.insert(memrefOperands[1]); // output
//     }
//   }
//   else {
//     // Conservative analysis for unknown functions
//     // Treat all but the last memref as inputs, and the last as output
//     for (unsigned i = 0; i < memrefOperands.size(); ++i) {
//       if (i == memrefOperands.size() - 1) {
//         outputs.insert(memrefOperands[i]); // Assume last is output
//       } else {
//         inputs.insert(memrefOperands[i]);  // Others are inputs
//       }
//     }
//   }
// }

// // MODIFIED: Find extended culibs operation sequence including memcpy and reinterpret_cast
// llvm::SmallVector<Operation*, 4> findExtendedCuLibsSequence(Operation* streamCreateOp) {
//   llvm::SmallVector<Operation*, 4> sequence;
  
//   // First, collect operations before stream create that should be included
//   Operation* currentOp = streamCreateOp->getPrevNode();
//   llvm::SmallVector<Operation*, 8> beforeOps;
  
//   // Look backwards for related operations
//   while (currentOp) {
//     if (shouldIncludeInCuLibsSequence(currentOp)) {
//       beforeOps.insert(beforeOps.begin(), currentOp);
//       currentOp = currentOp->getPrevNode();
//     } else {
//       // Stop when we hit an unrelated operation
//       break;
//     }
//   }
  
//   // Add the before operations to sequence
//   sequence.append(beforeOps.begin(), beforeOps.end());
  
//   // Add stream create
//   sequence.push_back(streamCreateOp);
  
//   // Get the stream value
//   Value streamValue = streamCreateOp->getResult(0);
  
//   // Find operations that use this stream (original logic)
//   currentOp = streamCreateOp;
//   while (currentOp) {
//     Operation* nextOp = currentOp->getNextNode();
    
//     if (!nextOp) break;
    
//     // Check if this operation uses the stream
//     bool usesStream = false;
//     for (Value operand : nextOp->getOperands()) {
//       if (operand == streamValue) {
//         usesStream = true;
//         break;
//       }
//     }
    
//     if (usesStream) {
//       sequence.push_back(nextOp);
      
//       // If this is stream destroy, continue looking for post operations
//       if (isCuLibsStreamDestroy(nextOp)) {
//         // Look for operations after stream destroy (like output memcpy)
//         Operation* postOp = nextOp->getNextNode();
//         while (postOp) {
//           if (isGpuMemcpy(postOp) || isa<gpu::WaitOp>(postOp)) {
//             sequence.push_back(postOp);
//             postOp = postOp->getNextNode();
//           } else {
//             // Stop when we hit an unrelated operation
//             break;
//           }
//         }
//         break;
//       }
//     }
    
//     currentOp = nextOp;
//   }
  
//   return sequence;
// }

// // Keep the original function for compatibility, but use the extended version
// llvm::SmallVector<Operation*, 4> findCuLibsSequence(Operation* streamCreateOp) {
//   return findExtendedCuLibsSequence(streamCreateOp);
// }

// // Print text representation of the dependency graph
// void dumpDependencyGraph(DependencyGraph &graph) {
//   llvm::errs() << "===== Dependency Graph =====\n";
  
//   // Print all nodes
//   llvm::errs() << "Nodes (" << graph.nodes.size() << " total):\n";
//   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
//     DependencyNode* node = graph.nodes[i].get();
    
//     llvm::errs() << "  [" << i << "] ";
//     if (node->type == NodeType::Kernel) {
//       llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
//     } else if (node->type == NodeType::Loop) {
//       llvm::errs() << "Loop at: ";
//       if (node->op) node->op->getLoc().print(llvm::errs());
//       else llvm::errs() << "<unknown location>";
//     } else if (node->type == NodeType::CuLibs) {
//       llvm::errs() << "CuLibs: " << node->culibsFunctionName 
//                    << " (ops: " << node->culibsOps.size() << ")";
//     }
//     llvm::errs() << "\n";
    
//     // Print input dependencies
//     llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
//     for (Value input : node->inputs) {
//       llvm::errs() << "      ";
//       input.print(llvm::errs());
//       llvm::errs() << "\n";
//     }
    
//     // Print output dependencies
//     llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
//     for (Value output : node->outputs) {
//       llvm::errs() << "      ";
//       output.print(llvm::errs());
//       llvm::errs() << "\n";
//     }
    
//     // Print topological sort level (if calculated)
//     if (node->topologicalLevel > 0) {
//       llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
//     }
    
//     llvm::errs() << "\n";
//   }
  
//   // Print all edges
//   llvm::errs() << "Edges:\n";
//   for (unsigned i = 0; i < graph.nodes.size(); ++i) {
//     DependencyNode* node = graph.nodes[i].get();
    
//     // Get outgoing edges for this node
//     if (graph.outEdges.count(node)) {
//       const auto &edges = graph.outEdges[node];
//       if (!edges.empty()) {
//         llvm::errs() << "  From [" << i << "] to:\n";
        
//         for (DependencyNode* target : edges) {
//           // Find target node index
//           for (unsigned j = 0; j < graph.nodes.size(); ++j) {
//             if (graph.nodes[j].get() == target) {
//               llvm::errs() << "    [" << j << "]";
              
//               // Output shared memory references causing this dependency
//               bool foundSharedMem = false;
//               for (Value out : node->outputs) {
//                 for (Value in : target->inputs) {
//                   if (out == in) {
//                     if (!foundSharedMem) {
//                       llvm::errs() << " via: ";
//                       foundSharedMem = true;
//                     } else {
//                       llvm::errs() << ", ";
//                     }
//                     out.print(llvm::errs());
//                   }
//                 }
//               }
              
//               llvm::errs() << "\n";
//               break;
//             }
//           }
//         }
//       }
//     }
//   }
  
//   llvm::errs() << "===========================\n";
// }

// // Build the dependency graph from a function
// std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp) {
//   auto graph = std::make_unique<DependencyGraph>();
  
//   // 安全检查：确保函数体不为空
//   if (funcOp.getBody().empty()) {
//     llvm::errs() << "Warning: Function " << funcOp.getName() << " has empty body, returning empty graph\n";
//     return graph;
//   }
  
//   // Create program order mapping
//   llvm::DenseMap<Operation*, unsigned> programOrder;
//   unsigned orderCounter = 0;
  
//   // Traverse function body to collect operation order
//   funcOp.walk([&](Operation* op) {
//     programOrder[op] = orderCounter++;
//   });

//   // Track processed operations to avoid duplicates
//   llvm::DenseSet<Operation*> processedOps;

//   // First pass: create nodes for all kernels, loop nests, and culibs calls
//   funcOp.walk([&](Operation* op) {
//     if (processedOps.count(op)) {
//       return WalkResult::advance();
//     }
    
//     if (isKernelLaunch(op)) {
//       auto kernelOp = cast<gpu::LaunchFuncOp>(op);
//       auto node = std::make_unique<DependencyNode>();
//       node->type = NodeType::Kernel;
//       node->op = op;
//       node->kernelModuleName = kernelOp.getKernelModuleName();
//       node->kernelName = kernelOp.getKernelName();
      
//       extractKernelDependencies(kernelOp, node->inputs, node->outputs);
//       graph->addNode(std::move(node));
//       processedOps.insert(op);
//     } 
//     else if (isLoopNest(op)) {
//       auto loopOp = cast<scf::ForOp>(op);
//       auto node = std::make_unique<DependencyNode>();
//       node->type = NodeType::Loop;
//       node->op = op;
      
//       extractLoopDependencies(loopOp, node->inputs, node->outputs);
//       graph->addNode(std::move(node));
//       processedOps.insert(op);
//     }
//     else if (isCuLibsStreamCreate(op)) {
//       // Find the complete culibs sequence (now includes memcpy and reinterpret_cast)
//       auto culibsSequence = findExtendedCuLibsSequence(op);
      
//       // Find the main culibs function call
//       Operation* mainCall = nullptr;
//       for (Operation* seqOp : culibsSequence) {
//         if (isCuLibsCall(seqOp)) {
//           mainCall = seqOp;
//           break;
//         }
//       }
      
//       if (mainCall) {
//         auto node = std::make_unique<DependencyNode>();
//         node->type = NodeType::CuLibs;
//         node->op = mainCall;  // Use main call as representative operation
//         node->culibsOps = culibsSequence;
//         node->culibsFunctionName = cast<func::CallOp>(mainCall).getCallee();
        
//         extractCuLibsDependencies(culibsSequence, node->inputs, node->outputs);
//         graph->addNode(std::move(node));
        
//         // Mark all operations in sequence as processed
//         for (Operation* seqOp : culibsSequence) {
//           processedOps.insert(seqOp);
//         }
//       }
//     }
    
//     return WalkResult::advance();
//   });
  
//   // Debug: Print how many nodes we found
//   llvm::errs() << "Built dependency graph with " << graph->nodes.size() << " nodes\n";
  
//   // Second pass: create edges based on dependencies
//   for (const auto &nodePair : graph->nodes) {
//     DependencyNode* node = nodePair.get();
    
//     // For each output of this node
//     for (auto output : node->outputs) {
//       // Check if it's an input to any other node
//       for (const auto &otherNodePair : graph->nodes) {
//         DependencyNode* otherNode = otherNodePair.get();
//         if (otherNode == node) continue;
        
//         // If this node's output is another node's input, add an edge
//         if (otherNode->inputs.count(output) && 
//             programOrder[node->op] < programOrder[otherNode->op]) {
//           graph->addEdge(node, otherNode);
//         }
//       }
//     }
//   }
  
//   return graph;
// }

// } // namespace onnx_mlir

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

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Implementation of DependencyGraph::addNode
DependencyNode* DependencyGraph::addNode(std::unique_ptr<DependencyNode> node) {
  DependencyNode* nodePtr = node.get();
  opToNodeMap[node->op] = nodePtr;
  
  // For kernel nodes, map all related operations
  if (node->type == NodeType::Kernel) {
    for (Operation* kernelOp : node->kernelOps) {
      opToNodeMap[kernelOp] = nodePtr;
    }
  }
  
  // For culibs nodes, map all related operations
  if (node->type == NodeType::CuLibs) {
    for (Operation* culibsOp : node->culibsOps) {
      opToNodeMap[culibsOp] = nodePtr;
    }
  }
  
  nodes.push_back(std::move(node));
  return nodePtr;
}

// Implementation of DependencyGraph::addEdge
void DependencyGraph::addEdge(DependencyNode* from, DependencyNode* to) {
  outEdges[from].push_back(to);
  inEdges[to].push_back(from);
}

// Check if an operation is a kernel launch
bool isKernelLaunch(Operation* op) {
  return isa<gpu::LaunchFuncOp>(op);
}

// Check if an operation is a loop nest (outermost loop)
bool isLoopNest(Operation* op) {
  return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
}

// Check if an operation is a culibs function call
bool isCuLibsCall(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    StringRef funcName = callOp.getCallee();
    
    // Check for culibs function patterns based on your actual implementations
    return funcName.starts_with("mgpuCudnn") || 
           funcName.starts_with("mgpuCulibs");
  }
  return false;
}

// Check if an operation is culibs stream create
bool isCuLibsStreamCreate(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamCreate";
  }
  return false;
}

// Check if an operation is culibs stream sync
bool isCuLibsStreamSync(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamSynchronize";
  }
  return false;
}

// Check if an operation is culibs stream destroy
bool isCuLibsStreamDestroy(Operation* op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callOp.getCallee() == "mgpuStreamDestroy";
  }
  return false;
}

// Check if an operation is a gpu.memcpy
bool isGpuMemcpy(Operation* op) {
  return isa<gpu::MemcpyOp>(op);
}

// Check if an operation is a memref.reinterpret_cast
bool isMemrefReinterpretCast(Operation* op) {
  return isa<memref::ReinterpretCastOp>(op);
}

// CONSERVATIVE: Only include reinterpret_cast that directly feeds into kernel arguments
bool shouldIncludeInKernelSequence(Operation* op) {
  return isMemrefReinterpretCast(op);
}

// CONSERVATIVE: Find only the reinterpret_cast operations that are direct kernel arguments
llvm::SmallVector<Operation*, 4> findExtendedKernelSequence(Operation* kernelLaunchOp) {
  llvm::SmallVector<Operation*, 4> sequence;
  auto kernelOp = cast<gpu::LaunchFuncOp>(kernelLaunchOp);
  
  // Collect reinterpret_cast operations that are direct operands of the kernel
  llvm::SmallVector<Operation*, 8> directReinterpretCasts;
  
  // Check each kernel operand to see if it comes directly from a reinterpret_cast
  for (Value operand : kernelOp.getKernelOperands()) {
    if (auto definingOp = operand.getDefiningOp()) {
      if (isa<memref::ReinterpretCastOp>(definingOp)) {
        directReinterpretCasts.push_back(definingOp);
      }
    }
  }
  
  // Sort the reinterpret_cast operations by their position in the block
  llvm::sort(directReinterpretCasts, [](Operation* a, Operation* b) {
    return a->isBeforeInBlock(b);
  });
  
  // Add the reinterpret_cast operations to sequence
  sequence.append(directReinterpretCasts.begin(), directReinterpretCasts.end());
  
  // Add the kernel launch itself
  sequence.push_back(kernelLaunchOp);
  
  return sequence;
}

// Check if an operation should be included in CuLibs sequence
bool shouldIncludeInCuLibsSequence(Operation* op) {
  return isGpuMemcpy(op) || 
         isa<gpu::WaitOp>(op) ||
         isa<memref::ReinterpretCastOp>(op) ||
         isa<memref::ExtractAlignedPointerAsIndexOp>(op) ||
         isa<arith::IndexCastOp>(op) ||
         isa<mlir::LLVM::IntToPtrOp>(op) ||
         isa<memref::AllocOp>(op) ||
         isa<arith::ConstantOp>(op);
}

// Helper function: Find GPU function definition
gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
  // Get top-level module
  ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
  if (!topModule) {
    return nullptr;
  }

  // Get kernel module and function name
  StringRef kernelModuleName = kernelOp.getKernelModuleName();
  StringRef kernelName = kernelOp.getKernelName();
  
  // First try to find the gpu.module
  gpu::GPUModuleOp gpuModule = nullptr;
  topModule.walk([&](gpu::GPUModuleOp module) {
    if (module.getName() == kernelModuleName) {
      gpuModule = module;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  // If GPU module is found, search for kernel function within it
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
  
  // Fallback: Search throughout the entire top-level module
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

// MODIFIED: Extract memref inputs and outputs from a kernel launch and its extended sequence
void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
                              const llvm::SmallVector<Operation*, 4> &kernelSequence,
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  // Find kernel function definition
  gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
  if (!kernelFunc) {
    // If function not found, fall back to conservative analysis
    llvm::errs() << "Warning: Could not find kernel function definition for \"" 
                << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
    
    // Analyze kernel operands directly, tracing through reinterpret_cast if present
    for (auto arg : kernelOp.getKernelOperands()) {
      if (arg.getType().isa<MemRefType>()) {
        // Trace back through reinterpret_cast to find original memref
        Value originalMemref = arg;
        while (auto reinterpretOp = originalMemref.getDefiningOp<memref::ReinterpretCastOp>()) {
          originalMemref = reinterpretOp.getSource();
        }
        
        inputs.insert(originalMemref);
        outputs.insert(originalMemref);
      }
    }
    return;
  }

  // Analyze kernel function to understand which parameters are read/written
  llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
  // Count the number of MemRef type parameters and operands
  unsigned memrefArgCount = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
      memrefArgCount++;
    }
  }
  
  unsigned memrefOpCount = 0;
  llvm::SmallVector<Value, 8> memrefOperands;
  for (auto operand : kernelOp.getKernelOperands()) {
    if (operand.getType().isa<MemRefType>()) {
      memrefOperands.push_back(operand);
      memrefOpCount++;
    }
  }
  
  // Map kernel function arguments to launch operands
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
  
  // Track which parameters are used for load and store
  llvm::DenseSet<BlockArgument> loadArgs;
  llvm::DenseSet<BlockArgument> storeArgs;
  
  // Analyze memory operations in kernel function body
  kernelFunc.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      Value memref = loadOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          loadArgs.insert(blockArg);
        }
      }
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      Value memref = storeOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          storeArgs.insert(blockArg);
        }
      }
    }
  });
  
  // Map function parameter analysis to kernel operands, tracing through reinterpret_cast
  for (auto &pair : argOperandPairs) {
    BlockArgument arg = pair.first;
    Value operand = pair.second;
    
    bool isInput = loadArgs.count(arg) > 0;
    bool isOutput = storeArgs.count(arg) > 0;
    
    // Trace the operand back through any reinterpret_cast operations to find the original memref
    Value originalMemref = operand;
    while (auto reinterpretOp = originalMemref.getDefiningOp<memref::ReinterpretCastOp>()) {
      originalMemref = reinterpretOp.getSource();
    }
    
    if (isInput) {
      inputs.insert(originalMemref);
    }
    
    if (isOutput) {
      outputs.insert(originalMemref);
    }
    
    // If the parameter is neither loaded nor stored, treat it as input to be conservative
    if (!isInput && !isOutput) {
      inputs.insert(originalMemref);
      llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
    }
  }
}

// Extract memref inputs and outputs from a loop nest
void extractLoopDependencies(scf::ForOp loopOp,
                           llvm::SetVector<Value> &inputs,
                           llvm::SetVector<Value> &outputs) {
  // Walk through the loop body to find all memref accesses
  loopOp.walk([&](Operation* op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      inputs.insert(loadOp.getMemref());
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      outputs.insert(storeOp.getMemref());
      // The stored value might also be a load from another memref
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
          storeOp.getValue().getDefiningOp())) {
        inputs.insert(loadOp.getMemref());
      }
    }
  });
}

// Extract memref inputs and outputs from culibs function calls
void extractCuLibsDependencies(const llvm::SmallVector<Operation*, 4> &culibsOps,
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  // Find the main culibs function call (not stream management calls)
  Operation* mainCall = nullptr;
  for (Operation* op : culibsOps) {
    if (isCuLibsCall(op)) {
      mainCall = op;
      break;
    }
  }
  
  if (!mainCall) {
    return;
  }
  
  auto callOp = cast<func::CallOp>(mainCall);
  StringRef funcName = callOp.getCallee();
  
  // Analyze operands - the last few operands are typically memref pointers
  auto operands = callOp.getOperands();
  
  // Extract memref operands by looking for LLVM pointer types that come from memref.extract_aligned_pointer_as_index
  llvm::SmallVector<Value, 8> memrefOperands;
  
  for (Value operand : operands) {
    // Check if this operand comes from a memref pointer extraction
    if (auto intToPtrOp = operand.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      Value intToPtrInput = intToPtrOp.getArg();
      if (auto indexCastOp = intToPtrInput.getDefiningOp<mlir::arith::IndexCastOp>()) {
        Value indexCastInput = indexCastOp->getOperand(0);
        if (auto extractOp = indexCastInput.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
          Value memref = extractOp.getSource();
          memrefOperands.push_back(memref);
        }
      }
    }
  }
  
  // Extract memrefs from related gpu.memcpy operations
  for (Operation* op : culibsOps) {
    if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(op)) {
      Value src = memcpyOp.getSrc();
      Value dst = memcpyOp.getDst();
      
      // Determine if this is input or output memcpy based on position relative to main call
      if (op->isBeforeInBlock(mainCall)) {
        inputs.insert(src);
        inputs.insert(dst);
      } else {
        outputs.insert(dst);
        outputs.insert(src);
      }
    }
  }
  
  // Function-specific dependency analysis
  if (funcName.contains("Conv2dForward")) {
    if (memrefOperands.size() >= 4) {
      inputs.insert(memrefOperands[0]);  // input
      inputs.insert(memrefOperands[1]);  // weight
      inputs.insert(memrefOperands[2]);  // bias
      outputs.insert(memrefOperands[3]); // output
    }
  }
  else if (funcName.contains("MaxPoolForward")) {
    if (memrefOperands.size() >= 2) {
      inputs.insert(memrefOperands[0]);  // input
      outputs.insert(memrefOperands[1]); // output
    }
  }
  else if (funcName.contains("FullyConnectedForward")) {
    if (memrefOperands.size() >= 4) {
      inputs.insert(memrefOperands[0]);  // input
      inputs.insert(memrefOperands[1]);  // weight
      inputs.insert(memrefOperands[2]);  // bias
      outputs.insert(memrefOperands[3]); // output
    }
    else if (memrefOperands.size() == 3) {
      inputs.insert(memrefOperands[0]);  // input
      inputs.insert(memrefOperands[1]);  // weight
      outputs.insert(memrefOperands[2]); // output
    }
  }
  else if (funcName.contains("ReduceSum") || funcName.contains("ReduceMean")) {
    // mgpuCudnnReduceSum/ReduceMean: input_data -> output_data
    if (memrefOperands.size() >= 2) {
      inputs.insert(memrefOperands[0]);  // input_data
      outputs.insert(memrefOperands[1]); // output_data
    }
  }
  else if (funcName.contains("Transpose_0213") || funcName.contains("Transpose_0231")) {
    // mgpuCulibsTranspose_*: input_data -> output_data
    if (memrefOperands.size() >= 2) {
      inputs.insert(memrefOperands[0]);  // input_data
      outputs.insert(memrefOperands[1]); // output_data
    }
  }
  else if (funcName.contains("BatchedMatMulForward")) {
    // mgpuCulibsBatchedMatMulForward: input_a, input_b -> output_c
    if (memrefOperands.size() >= 3) {
      inputs.insert(memrefOperands[0]);  // input_a
      inputs.insert(memrefOperands[1]);  // input_b
      outputs.insert(memrefOperands[2]); // output_c
    }
  }
  else {
    // Conservative analysis for unknown functions
    for (unsigned i = 0; i < memrefOperands.size(); ++i) {
      if (i == memrefOperands.size() - 1) {
        outputs.insert(memrefOperands[i]);
      } else {
        inputs.insert(memrefOperands[i]);
      }
    }
  }
}

// Find extended culibs operation sequence including memcpy and reinterpret_cast
// llvm::SmallVector<Operation*, 4> findExtendedCuLibsSequence(Operation* streamCreateOp) {
llvm::SmallVector<Operation*, 4> findBasicCuLibsSequence(Operation* streamCreateOp) {
  llvm::SmallVector<Operation*, 4> sequence;
  
  // First, collect operations before stream create that should be included
  Operation* currentOp = streamCreateOp->getPrevNode();
  llvm::SmallVector<Operation*, 8> beforeOps;
  
  // Look backwards for related operations
  while (currentOp) {
    if (shouldIncludeInCuLibsSequence(currentOp)) {
      beforeOps.insert(beforeOps.begin(), currentOp);
      currentOp = currentOp->getPrevNode();
    } else {
      break;
    }
  }
  
  // Add the before operations to sequence
  sequence.append(beforeOps.begin(), beforeOps.end());
  
  // Add stream create
  sequence.push_back(streamCreateOp);
  
  // Get the stream value
  Value streamValue = streamCreateOp->getResult(0);
  
  // Find operations that use this stream
  currentOp = streamCreateOp;
  while (currentOp) {
    Operation* nextOp = currentOp->getNextNode();
    
    if (!nextOp) break;
    
    // Check if this operation uses the stream
    bool usesStream = false;
    for (Value operand : nextOp->getOperands()) {
      if (operand == streamValue) {
        usesStream = true;
        break;
      }
    }
    
    if (usesStream) {
      sequence.push_back(nextOp);
      
      // If this is stream destroy, continue looking for post operations
      if (isCuLibsStreamDestroy(nextOp)) {
        // Look for operations after stream destroy (like output memcpy)
        Operation* postOp = nextOp->getNextNode();
        while (postOp) {
          if (isGpuMemcpy(postOp) || isa<gpu::WaitOp>(postOp)) {
            sequence.push_back(postOp);
            postOp = postOp->getNextNode();
          } else {
            break;
          }
        }
        break;
      }
    }
    
    currentOp = nextOp;
  }
  
  return sequence;
}

// Enhanced version that collects reinterpret_cast dependencies
llvm::SmallVector<Operation*, 4> findExtendedCuLibsSequence(Operation* streamCreateOp) {
  // First, get the basic sequence using the original logic (重命名后的函数)
  auto sequence = findBasicCuLibsSequence(streamCreateOp);
  
  // Get already processed operations (if available from context)
  llvm::DenseSet<Operation*> emptyProcessed;
  
  // Now collect all reinterpret_cast dependencies
  llvm::SetVector<Operation*> reinterpretCasts;
  collectReinterpretCastDependencies(sequence, reinterpretCasts, emptyProcessed);
  
  // Create the final sequence with reinterpret_cast operations
  llvm::SmallVector<Operation*, 4> finalSequence;
  
  // Sort reinterpret_cast operations by their position in the block to maintain order
  llvm::SmallVector<Operation*, 8> sortedReinterpretCasts(reinterpretCasts.begin(), reinterpretCasts.end());
  llvm::sort(sortedReinterpretCasts, [](Operation* a, Operation* b) {
    return a->isBeforeInBlock(b);
  });
  
  // Add reinterpret_cast operations first (they are dependencies)
  finalSequence.append(sortedReinterpretCasts.begin(), sortedReinterpretCasts.end());
  
  // Then add the original sequence
  finalSequence.append(sequence.begin(), sequence.end());
  
  return finalSequence;
}


// Keep the original function for compatibility
llvm::SmallVector<Operation*, 4> findCuLibsSequence(Operation* streamCreateOp) {
  return findExtendedCuLibsSequence(streamCreateOp);
}

// Print text representation of the dependency graph
void dumpDependencyGraph(DependencyGraph &graph) {
  llvm::errs() << "===== Dependency Graph =====\n";
  
  // Print all nodes
  llvm::errs() << "Nodes (" << graph.nodes.size() << " total):\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    DependencyNode* node = graph.nodes[i].get();
    
    llvm::errs() << "  [" << i << "] ";
    if (node->type == NodeType::Kernel) {
      llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
      if (!node->kernelOps.empty() && node->kernelOps.size() > 1) {
        llvm::errs() << " (extended sequence: " << node->kernelOps.size() << " ops)";
      }
    } else if (node->type == NodeType::Loop) {
      llvm::errs() << "Loop at: ";
      if (node->op) node->op->getLoc().print(llvm::errs());
      else llvm::errs() << "<unknown location>";
    } else if (node->type == NodeType::CuLibs) {
      llvm::errs() << "CuLibs: " << node->culibsFunctionName 
                   << " (ops: " << node->culibsOps.size() << ")";
    }
    llvm::errs() << "\n";
    
    // Print input dependencies
    llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
    for (Value input : node->inputs) {
      llvm::errs() << "      ";
      input.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // Print output dependencies
    llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
    for (Value output : node->outputs) {
      llvm::errs() << "      ";
      output.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // Print topological sort level (if calculated)
    if (node->topologicalLevel > 0) {
      llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
    }
    
    llvm::errs() << "\n";
  }
  
  // Print all edges
  llvm::errs() << "Edges:\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    DependencyNode* node = graph.nodes[i].get();
    
    if (graph.outEdges.count(node)) {
      const auto &edges = graph.outEdges[node];
      if (!edges.empty()) {
        llvm::errs() << "  From [" << i << "] to:\n";
        
        for (DependencyNode* target : edges) {
          for (unsigned j = 0; j < graph.nodes.size(); ++j) {
            if (graph.nodes[j].get() == target) {
              llvm::errs() << "    [" << j << "]";
              
              bool foundSharedMem = false;
              for (Value out : node->outputs) {
                for (Value in : target->inputs) {
                  if (out == in) {
                    if (!foundSharedMem) {
                      llvm::errs() << " via: ";
                      foundSharedMem = true;
                    } else {
                      llvm::errs() << ", ";
                    }
                    out.print(llvm::errs());
                  }
                }
              }
              
              llvm::errs() << "\n";
              break;
            }
          }
        }
      }
    }
  }
  
  llvm::errs() << "===========================\n";
}


// Helper function to recursively collect all reinterpret_cast dependencies
void collectReinterpretCastDependencies(const llvm::SmallVector<Operation*, 4> &sequence, 
                                       llvm::SetVector<Operation*> &reinterpretCasts,
                                       const llvm::DenseSet<Operation*> &alreadyProcessed = {}) {
  llvm::DenseSet<Operation*> visited;
  
  // Helper lambda to recursively trace reinterpret_cast dependencies
  std::function<void(Value)> traceReinterpretCastDeps = [&](Value val) {
    if (auto definingOp = val.getDefiningOp()) {
      // If it's already processed by another node, skip it
      if (alreadyProcessed.count(definingOp)) {
        return;
      }
      
      // Avoid infinite recursion
      if (visited.count(definingOp)) {
        return;
      }
      visited.insert(definingOp);
      
      // If this is a reinterpret_cast operation
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
        reinterpretCasts.insert(reinterpretOp);
        
        // Recursively check the source of reinterpret_cast
        traceReinterpretCastDeps(reinterpretOp.getSource());
      }
    }
  };
  
  // Check all operands of all operations in the sequence
  for (Operation* op : sequence) {
    for (Value operand : op->getOperands()) {
      traceReinterpretCastDeps(operand);
    }
  }
}


// CONSERVATIVE: Build the dependency graph with minimal changes
std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp) {
  auto graph = std::make_unique<DependencyGraph>();
  
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
    
    if (isKernelLaunch(op)) {
      auto kernelOp = cast<gpu::LaunchFuncOp>(op);
      
      // CONSERVATIVE: Find only direct reinterpret_cast operations for this kernel
      auto kernelSequence = findExtendedKernelSequence(op);
      
      auto node = std::make_unique<DependencyNode>();
      node->type = NodeType::Kernel;
      node->op = op;
      node->kernelOps = kernelSequence;  // Store the limited sequence
      node->kernelModuleName = kernelOp.getKernelModuleName();
      node->kernelName = kernelOp.getKernelName();
      
      // Extract dependencies considering only the direct sequence
      extractKernelDependencies(kernelOp, kernelSequence, node->inputs, node->outputs);
      graph->addNode(std::move(node));
      
      // Mark ONLY the operations in the limited sequence as processed
      for (Operation* seqOp : kernelSequence) {
        processedOps.insert(seqOp);
      }
    } 
    else if (isLoopNest(op)) {
      auto loopOp = cast<scf::ForOp>(op);
      auto node = std::make_unique<DependencyNode>();
      node->type = NodeType::Loop;
      node->op = op;
      
      extractLoopDependencies(loopOp, node->inputs, node->outputs);
      graph->addNode(std::move(node));
      processedOps.insert(op);
    }
    else if (isCuLibsStreamCreate(op)) {
      // Find the complete culibs sequence
      auto culibsSequence = findExtendedCuLibsSequence(op);
      
      // Find the main culibs function call
      Operation* mainCall = nullptr;
      for (Operation* seqOp : culibsSequence) {
        if (isCuLibsCall(seqOp)) {
          mainCall = seqOp;
          break;
        }
      }
      
      if (mainCall) {
        auto node = std::make_unique<DependencyNode>();
        node->type = NodeType::CuLibs;
        node->op = mainCall;
        node->culibsOps = culibsSequence;
        node->culibsFunctionName = cast<func::CallOp>(mainCall).getCallee();
        
        extractCuLibsDependencies(culibsSequence, node->inputs, node->outputs);
        graph->addNode(std::move(node));
        
        // Mark all operations in sequence as processed
        for (Operation* seqOp : culibsSequence) {
          processedOps.insert(seqOp);
        }
      }
    }
    
    return WalkResult::advance();
  });
  
  llvm::errs() << "Built dependency graph with " << graph->nodes.size() << " nodes\n";
  
  // Second pass: create edges based on dependencies
  for (const auto &nodePair : graph->nodes) {
    DependencyNode* node = nodePair.get();
    
    for (auto output : node->outputs) {
      for (const auto &otherNodePair : graph->nodes) {
        DependencyNode* otherNode = otherNodePair.get();
        if (otherNode == node) continue;
        
        if (otherNode->inputs.count(output) && 
            programOrder[node->op] < programOrder[otherNode->op]) {
          graph->addEdge(node, otherNode);
        }
      }
    }
  }
  
  return graph;
}

} // namespace onnx_mlir