// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "llvm/Support/Debug.h"
// #include <map>
// #include <vector>
// #include <string>
// #include <set>

// using namespace mlir;

// #define DEBUG_TYPE "stream-unification"

// namespace {

// struct SchedulingInfo {
//   int parallelGroupId;
//   int subgraphId;
//   int intraLevel;
//   std::string nodeType;
//   std::string functionName;
//   Operation* operation;
//   Value streamValue;
// };

// struct SerialExecutionChain {
//   int parallelGroupId;
//   int subgraphId;
//   int startLevel;
//   int endLevel;
//   std::vector<SchedulingInfo*> nodes;
//   bool hasMixedTypes = false;
//   Value unifiedStream;
// };

// class UnifyMixedSerialChainPattern : public RewritePattern {
// private:
//   std::vector<SerialExecutionChain> &serialChains;
//   mutable std::set<Operation*> processedOps; // 跟踪已处理的操作
  
// public:
//   UnifyMixedSerialChainPattern(MLIRContext *context, 
//                               std::vector<SerialExecutionChain> &chains)
//       : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
//         serialChains(chains) {}
    
//   LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
//     // 检查是否已经处理过这个操作
//     if (processedOps.count(op)) {
//       return failure();
//     }
    
//     // 只处理GPU LaunchFuncOp
//     auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op);
//     if (!launchOp) {
//       return failure();
//     }
    
//     // 查找这个操作是否属于需要统一的串行链中的GPU kernel
//     for (const auto &chain : serialChains) {
//       if (!chain.hasMixedTypes || !chain.unifiedStream) {
//         continue;
//       }
      
//       // 查找链中的GPU kernel操作
//       for (const auto *nodeInfo : chain.nodes) {
//         if (nodeInfo->operation == op && nodeInfo->nodeType == "KERNEL") {
//           // 标记为已处理
//           processedOps.insert(op);
          
//           return unifyKernelStreamLLVM(launchOp, chain.unifiedStream, rewriter);
//         }
//       }
//     }

//     return failure();
//   }

// private:
//   LogicalResult unifyKernelStreamLLVM(gpu::LaunchFuncOp launchOp, Value culibsStream, 
//                                      PatternRewriter &rewriter) const {
//     // 检查是否需要修改
//     Value currentStream = getCurrentStream(launchOp);
//     if (currentStream == culibsStream) {
//       return failure(); // 已经是目标stream，无需修改
//     }
    
//     // 方法1：如果LaunchFuncOp有async token，直接修改
//     if (launchOp.getAsyncToken()) {
//       rewriter.modifyOpInPlace(launchOp, [&]() {
//         launchOp.getAsyncObjectMutable().assign(culibsStream);
//       });
//       return success();
//     }
    
//     // 方法2：检查和修改操作数中的stream
//     auto operands = launchOp->getOpOperands();
//     for (unsigned i = 0; i < operands.size(); ++i) {
//       Value operand = operands[i].get();
//       if (operand.getType().isa<LLVM::LLVMPointerType>()) {
//         rewriter.modifyOpInPlace(launchOp, [&]() {
//           operands[i].set(culibsStream);
//         });
//         return success();
//       }
//     }
    
//     // 方法3：重构LaunchFuncOp
//     return reconstructLaunchFuncOpWithStream(launchOp, culibsStream, rewriter);
//   }
  
//   Value getCurrentStream(gpu::LaunchFuncOp launchOp) const {
//     if (launchOp.getAsyncToken()) {
//       return launchOp.getAsyncToken();
//     }
    
//     auto asyncDeps = launchOp.getAsyncDependencies();
//     if (!asyncDeps.empty()) {
//       return asyncDeps[0];
//     }
    
//     for (Value operand : launchOp->getOperands()) {
//       if (operand.getType().isa<LLVM::LLVMPointerType>()) {
//         return operand;
//       }
//     }
    
//     return {};
//   }
  
//   LogicalResult reconstructLaunchFuncOpWithStream(gpu::LaunchFuncOp launchOp, Value culibsStream,
//                                                  PatternRewriter &rewriter) const {
//     Location loc = launchOp.getLoc();
//     auto kernelAttr = launchOp.getKernelAttr();
    
//     auto gridSizeValues = launchOp.getGridSizeOperandValues();
//     auto blockSizeValues = launchOp.getBlockSizeOperandValues();
    
//     gpu::KernelDim3 gridSize = {
//         gridSizeValues.x, gridSizeValues.y, gridSizeValues.z
//     };
    
//     gpu::KernelDim3 blockSize = {
//         blockSizeValues.x, blockSizeValues.y, blockSizeValues.z
//     };
    
//     Value dynamicSharedMemorySize = launchOp.getDynamicSharedMemorySize();
//     ValueRange kernelOperands = launchOp.getKernelOperands();
    
//     std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
//     if (launchOp.hasClusterSize()) {
//       auto clusterSizeValues = launchOp.getClusterSizeOperandValues();
//       clusterSize = gpu::KernelDim3{
//           clusterSizeValues.x, clusterSizeValues.y, clusterSizeValues.z
//       };
//     }
    
//     auto newLaunchOp = rewriter.create<gpu::LaunchFuncOp>(
//         loc,
//         kernelAttr,
//         gridSize,
//         blockSize,
//         dynamicSharedMemorySize,
//         kernelOperands,
//         culibsStream,
//         clusterSize
//     );
    
//     if (launchOp->getNumResults() > 0) {
//       rewriter.replaceOp(launchOp, {culibsStream});
//     } else {
//       rewriter.eraseOp(launchOp);
//     }
    
//     return success();
//   }
// };

// struct StreamUnificationPass
//     : public PassWrapper<StreamUnificationPass, OperationPass<ModuleOp>> {
  
//   StringRef getArgument() const final { return "stream-unification"; }
//   StringRef getDescription() const final {
//     return "Unify streams for mixed CuLibs-Kernel serial execution chains";
//   }
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<LLVM::LLVMDialect>();
//   }
  
//   void runOnOperation() override {
//     ModuleOp moduleOp = getOperation();
//     MLIRContext *context = &getContext();
    
//     LLVM::LLVMFuncOp mainGraphFunc = nullptr;
    
//     for (auto &op : moduleOp.getBody()->getOperations()) {
//       if (auto llvmFuncOp = dyn_cast<LLVM::LLVMFuncOp>(&op)) {
//         if (llvmFuncOp.getName() == "main_graph") {
//           mainGraphFunc = llvmFuncOp;
//           break;
//         }
//       }
//     }
    
//     if (!mainGraphFunc || mainGraphFunc.isExternal()) {
//       return;
//     }
    
//     if (failed(processMainGraphFunction(mainGraphFunc, context))) {
//       signalPassFailure();
//     }
//   }

// private:
//   std::optional<SchedulingInfo> parseSchedulingInfoFromKernelName(const std::string& kernelName, Operation* op) {
//     size_t schedPos = kernelName.find("sched_");
//     if (schedPos == std::string::npos) {
//       return std::nullopt;
//     }
    
//     std::string schedulePart = kernelName.substr(schedPos + 6);
    
//     size_t pPos = schedulePart.find('P');
//     size_t sPos = schedulePart.find("_S");
//     size_t lPos = schedulePart.find("_L");
    
//     if (pPos == std::string::npos || sPos == std::string::npos || lPos == std::string::npos) {
//       return std::nullopt;
//     }
    
//     try {
//       SchedulingInfo info;
//       info.parallelGroupId = std::stoi(schedulePart.substr(pPos + 1, sPos - pPos - 1));
//       info.subgraphId = std::stoi(schedulePart.substr(sPos + 2, lPos - sPos - 2));
      
//       size_t nextUnderscore = schedulePart.find('_', lPos + 2);
//       if (nextUnderscore != std::string::npos) {
//         info.intraLevel = std::stoi(schedulePart.substr(lPos + 2, nextUnderscore - lPos - 2));
//       } else {
//         info.intraLevel = std::stoi(schedulePart.substr(lPos + 2));
//       }
      
//       info.operation = op;
//       info.nodeType = "KERNEL";
//       info.functionName = kernelName;
      
//       return info;
//     } catch (const std::exception&) {
//       return std::nullopt;
//     }
//   }

//   LogicalResult processMainGraphFunction(LLVM::LLVMFuncOp mainGraphFunc, MLIRContext *context) {
//     auto schedulingInfos = recoverSchedulingInfoFromMainGraph(mainGraphFunc);
//     if (schedulingInfos.empty()) {
//       return success();
//     }
    
//     auto serialChains = identifySerialChains(schedulingInfos);
//     if (serialChains.empty()) {
//       return success();
//     }
    
//     if (failed(prepareUnifiedStreams(serialChains))) {
//       return failure();
//     }
    
//     RewritePatternSet patterns(context);
//     patterns.add<UnifyMixedSerialChainPattern>(context, serialChains);
    
//     // 使用更保守的配置
//     GreedyRewriteConfig config;
//     config.maxIterations = 10; // 限制最大迭代次数
    
//     if (failed(applyPatternsAndFoldGreedily(mainGraphFunc, std::move(patterns), config))) {
//       return failure();
//     }
    
//     return success();
//   }
  
//   std::vector<SchedulingInfo> recoverSchedulingInfoFromMainGraph(LLVM::LLVMFuncOp mainGraphFunc) {
//     std::vector<SchedulingInfo> infos;
    
//     mainGraphFunc.walk([&](Operation *op) {
//       auto scheduleInfoAttr = op->getAttrOfType<StringAttr>("schedule.info");
//       if (scheduleInfoAttr) {
//         std::string scheduleStr = scheduleInfoAttr.getValue().str();
//         auto schedulingInfo = parseSchedulingInfo(scheduleStr, op);
        
//         if (schedulingInfo.has_value()) {
//           schedulingInfo->streamValue = extractStreamFromOperationLLVM(op);
//           infos.push_back(*schedulingInfo);
//         }
//         return;
//       }
      
//       if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//         auto kernelName = launchOp.getKernelName().str();
//         auto schedulingInfo = parseSchedulingInfoFromKernelName(kernelName, op);
        
//         if (schedulingInfo.has_value()) {
//           schedulingInfo->streamValue = extractStreamFromOperationLLVM(op);
//           infos.push_back(*schedulingInfo);
//         }
//       }
//     });
    
//     std::sort(infos.begin(), infos.end(), 
//         [](const SchedulingInfo& a, const SchedulingInfo& b) {
//             if (a.parallelGroupId != b.parallelGroupId) return a.parallelGroupId < b.parallelGroupId;
//             if (a.subgraphId != b.subgraphId) return a.subgraphId < b.subgraphId;
//             return a.intraLevel < b.intraLevel;
//         });
    
//     return infos;
//   }

//   Value extractStreamFromOperationLLVM(Operation* op) {
//     if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//       if (launchOp.getAsyncToken()) {
//         return launchOp.getAsyncToken();
//       }
      
//       auto asyncDeps = launchOp.getAsyncDependencies();
//       if (!asyncDeps.empty()) {
//         return asyncDeps[0];
//       }
      
//       for (Value operand : launchOp->getOperands()) {
//         if (operand.getType().isa<LLVM::LLVMPointerType>()) {
//           return operand;
//         }
//       }
      
//       return {};
//     }
    
//     if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
//       auto operands = callOp.getOperands();
      
//       if (isCuLibsCallLLVM(op)) {
//         if (!operands.empty()) {
//           return operands.back();
//         }
//       }
      
//       return {};
//     }
    
//     return {};
//   }

//   bool isCuLibsCallLLVM(Operation* op) {
//     if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
//       if (auto calleeAttr = callOp.getCalleeAttr()) {
//         StringRef funcName = calleeAttr.getValue();
//         return funcName.starts_with("mgpuCudnn") || 
//                funcName.starts_with("mgpuCulibs");
//       }
//     }
//     return false;
//   }
  
//   bool isGpuKernelLaunchLLVM(Operation* op) {
//     if (isa<gpu::LaunchFuncOp>(op)) {
//       return true;
//     }
    
//     if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
//       if (auto calleeAttr = callOp.getCalleeAttr()) {
//         StringRef funcName = calleeAttr.getValue();
//         return funcName.contains("enhanced_scheduled_kernels") ||
//                funcName.contains("_kernel") ||
//                funcName.contains("sched_P") ||
//                op->hasAttr("gpu.kernel");
//       }
//     }
    
//     return false;
//   }
  
//   std::string extractKernelNameLLVM(Operation* op) {
//     if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//       return launchOp.getKernelName().str();
//     }
    
//     if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
//       if (auto calleeAttr = callOp.getCalleeAttr()) {
//         return calleeAttr.getValue().str();
//       }
//     }
//     return "unknown_kernel";
//   }
  
//   std::string extractCuLibsNameLLVM(Operation* op) {
//     if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
//       if (auto calleeAttr = callOp.getCalleeAttr()) {
//         return calleeAttr.getValue().str();
//       }
//     }
//     return "unknown_culibs";
//   }
  
//   std::optional<SchedulingInfo> parseSchedulingInfo(const std::string& scheduleStr, Operation* op) {
//     size_t pPos = scheduleStr.find('P');
//     size_t sPos = scheduleStr.find("_S");
//     size_t lPos = scheduleStr.find("_L");
    
//     if (pPos == std::string::npos || sPos == std::string::npos || lPos == std::string::npos) {
//       return std::nullopt;
//     }
    
//     try {
//       SchedulingInfo info;
//       info.parallelGroupId = std::stoi(scheduleStr.substr(pPos + 1, sPos - pPos - 1));
//       info.subgraphId = std::stoi(scheduleStr.substr(sPos + 2, lPos - sPos - 2));
      
//       size_t lastUnderscore = scheduleStr.rfind('_');
//       if (lastUnderscore > lPos) {
//         info.intraLevel = std::stoi(scheduleStr.substr(lPos + 2, lastUnderscore - lPos - 2));
//       } else {
//         info.intraLevel = std::stoi(scheduleStr.substr(lPos + 2));
//       }
      
//       info.operation = op;
      
//       if (isGpuKernelLaunchLLVM(op)) {
//         info.nodeType = "KERNEL";
//         info.functionName = extractKernelNameLLVM(op);
//       } else if (isCuLibsCallLLVM(op)) {
//         info.nodeType = "CULIBS";
//         info.functionName = extractCuLibsNameLLVM(op);
//       } else {
//         return std::nullopt;
//       }
      
//       return info;
//     } catch (const std::exception&) {
//       return std::nullopt;
//     }
//   }

//   std::vector<SerialExecutionChain> identifySerialChains(const std::vector<SchedulingInfo>& infos) {
//     std::vector<SerialExecutionChain> chains;
    
//     std::map<std::pair<int,int>, std::vector<SchedulingInfo*>> nodesBySubgraph;
//     for (const auto& info : infos) {
//       nodesBySubgraph[{info.parallelGroupId, info.subgraphId}].push_back(
//           const_cast<SchedulingInfo*>(&info));
//     }
    
//     for (const auto& [subgraphKey, subgraphNodes] : nodesBySubgraph) {
//       analyzeSubgraphForSerialChains(subgraphKey.first, subgraphKey.second, 
//                                    subgraphNodes, chains);
//     }
    
//     return chains;
//   }
  
//   void analyzeSubgraphForSerialChains(int parallelGroupId, int subgraphId,
//                                     const std::vector<SchedulingInfo*>& nodes,
//                                     std::vector<SerialExecutionChain>& chains) {
//     if (nodes.size() <= 1) {
//       return;
//     }
    
//     std::map<int, std::vector<SchedulingInfo*>> nodesByLevel;
//     for (auto* node : nodes) {
//       nodesByLevel[node->intraLevel].push_back(node);
//     }
    
//     SerialExecutionChain currentChain;
//     currentChain.parallelGroupId = parallelGroupId;
//     currentChain.subgraphId = subgraphId;
    
//     for (const auto& [level, levelNodes] : nodesByLevel) {
//       if (levelNodes.size() == 1) {
//         auto* node = levelNodes[0];
        
//         if (currentChain.nodes.empty()) {
//           currentChain.startLevel = level;
//           currentChain.endLevel = level;
//           currentChain.nodes.push_back(node);
//         } else if (level == currentChain.endLevel + 1) {
//           currentChain.endLevel = level;
//           currentChain.nodes.push_back(node);
//         } else {
//           finalizePotentialChain(currentChain, chains);
          
//           currentChain.nodes.clear();
//           currentChain.startLevel = level;
//           currentChain.endLevel = level;
//           currentChain.nodes.push_back(node);
//           currentChain.hasMixedTypes = false;
//         }
//       } else {
//         if (!currentChain.nodes.empty()) {
//           finalizePotentialChain(currentChain, chains);
//           currentChain.nodes.clear();
//           currentChain.hasMixedTypes = false;
//         }
//       }
//     }
    
//     if (!currentChain.nodes.empty()) {
//       finalizePotentialChain(currentChain, chains);
//     }
//   }

//   void finalizePotentialChain(SerialExecutionChain& chain, 
//                             std::vector<SerialExecutionChain>& chains) {
//     if (chain.nodes.size() < 2) {
//       return;
//     }
    
//     bool hasKernel = false;
//     bool hasCuLibs = false;
    
//     for (const auto* node : chain.nodes) {
//       if (node->nodeType == "KERNEL") hasKernel = true;
//       if (node->nodeType == "CULIBS") hasCuLibs = true;
//     }
    
//     chain.hasMixedTypes = hasKernel && hasCuLibs;
    
//     if (chain.hasMixedTypes) {
//       chains.push_back(chain);
//     }
//   }

//   LogicalResult prepareUnifiedStreams(std::vector<SerialExecutionChain>& chains) {
//     for (auto& chain : chains) {
//       for (const auto* node : chain.nodes) {
//         if (node->nodeType == "CULIBS" && node->streamValue) {
//           chain.unifiedStream = node->streamValue;
//           break;
//         }
//       }
      
//       if (!chain.unifiedStream) {
//         return failure();
//       }
//     }
    
//     return success();
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {

// void addSchedulingAttributes(Operation* op, int parallelGroupId, int subgraphId, int intraLevel, OpBuilder& builder) {
//   auto context = builder.getContext();
  
//   op->setAttr("schedule.parallel_group", 
//               IntegerAttr::get(IntegerType::get(context, 32), parallelGroupId));
//   op->setAttr("schedule.subgraph_id", 
//               IntegerAttr::get(IntegerType::get(context, 32), subgraphId));
//   op->setAttr("schedule.intra_level", 
//               IntegerAttr::get(IntegerType::get(context, 32), intraLevel));
  
//   std::string scheduleInfo = "P" + std::to_string(parallelGroupId) + 
//                             "_S" + std::to_string(subgraphId) + 
//                             "_L" + std::to_string(intraLevel);
//   op->setAttr("schedule.info", StringAttr::get(context, scheduleInfo));
  
//   if (isa<gpu::LaunchFuncOp>(op)) {
//     op->setAttr("schedule.node_type", StringAttr::get(context, "KERNEL"));
//   } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
//     StringRef funcName = callOp.getCallee();
//     if (funcName.starts_with("mgpuCudnn") || funcName.starts_with("mgpuCulibs")) {
//       op->setAttr("schedule.node_type", StringAttr::get(context, "CULIBS"));
//     }
//   }
// }

// std::unique_ptr<Pass> createStreamUnificationPass() {
//   return std::make_unique<StreamUnificationPass>();
// }

// } // namespace onnx_mlir

// static mlir::PassRegistration<StreamUnificationPass> pass;


#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <vector>
#include <string>
#include <set>

using namespace mlir;

#define DEBUG_TYPE "stream-unification"

namespace {

struct SchedulingInfo {
  int parallelGroupId;
  int subgraphId;
  int intraLevel;
  std::string nodeType;
  std::string functionName;
  Operation* operation;
  Value streamValue;
};

struct SerialExecutionChain {
  int parallelGroupId;
  int subgraphId;
  int startLevel;
  int endLevel;
  std::vector<SchedulingInfo*> nodes;
  bool hasMixedTypes = false;
  Value unifiedStream;
};

class UnifyMixedSerialChainPattern : public RewritePattern {
private:
  std::vector<SerialExecutionChain> &serialChains;
  mutable std::set<Operation*> processedOps; // 跟踪已处理的操作
  
public:
  UnifyMixedSerialChainPattern(MLIRContext *context, 
                              std::vector<SerialExecutionChain> &chains)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        serialChains(chains) {}
    
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // 检查是否已经处理过这个操作
    if (processedOps.count(op)) {
      return failure();
    }
    
    // 只处理GPU LaunchFuncOp
    auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op);
    if (!launchOp) {
      return failure();
    }
    
    // 查找这个操作是否属于需要统一的串行链中的GPU kernel
    for (const auto &chain : serialChains) {
      if (!chain.hasMixedTypes || !chain.unifiedStream) {
        continue;
      }
      
      // 查找链中的GPU kernel操作
      for (const auto *nodeInfo : chain.nodes) {
        if (nodeInfo->operation == op && nodeInfo->nodeType == "KERNEL") {
          // 标记为已处理
          processedOps.insert(op);
          
          return unifyKernelStreamLLVM(launchOp, chain.unifiedStream, rewriter);
        }
      }
    }

    return failure();
  }

private:
  LogicalResult unifyKernelStreamLLVM(gpu::LaunchFuncOp launchOp, Value culibsStream, 
                                     PatternRewriter &rewriter) const {
    // 检查是否需要修改
    Value currentStream = getCurrentStream(launchOp);
    if (currentStream == culibsStream) {
      return failure(); // 已经是目标stream，无需修改
    }
    
    // 在LLVM dialect阶段，始终使用重构方法来确保正确的stream替换
    // 这避免了就地修改可能导致的错误参数替换问题
    return reconstructLaunchFuncOpWithStream(launchOp, culibsStream, rewriter);
  }
  
  Value getCurrentStream(gpu::LaunchFuncOp launchOp) const {
    if (launchOp.getAsyncToken()) {
      return launchOp.getAsyncToken();
    }
    
    auto asyncDeps = launchOp.getAsyncDependencies();
    if (!asyncDeps.empty()) {
      return asyncDeps[0];
    }
    
    for (Value operand : launchOp->getOperands()) {
      if (operand.getType().isa<LLVM::LLVMPointerType>()) {
        return operand;
      }
    }
    
    return {};
  }
  
  LogicalResult reconstructLaunchFuncOpWithStream(gpu::LaunchFuncOp launchOp, Value culibsStream,
                                                 PatternRewriter &rewriter) const {
    Location loc = launchOp.getLoc();
    auto kernelAttr = launchOp.getKernelAttr();
    
    auto gridSizeValues = launchOp.getGridSizeOperandValues();
    auto blockSizeValues = launchOp.getBlockSizeOperandValues();
    
    gpu::KernelDim3 gridSize = {
        gridSizeValues.x, gridSizeValues.y, gridSizeValues.z
    };
    
    gpu::KernelDim3 blockSize = {
        blockSizeValues.x, blockSizeValues.y, blockSizeValues.z
    };
    
    Value dynamicSharedMemorySize = launchOp.getDynamicSharedMemorySize();
    ValueRange kernelOperands = launchOp.getKernelOperands();
    
    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (launchOp.hasClusterSize()) {
      auto clusterSizeValues = launchOp.getClusterSizeOperandValues();
      clusterSize = gpu::KernelDim3{
          clusterSizeValues.x, clusterSizeValues.y, clusterSizeValues.z
      };
    }
    
    auto newLaunchOp = rewriter.create<gpu::LaunchFuncOp>(
        loc,
        kernelAttr,
        gridSize,
        blockSize,
        dynamicSharedMemorySize,
        kernelOperands,
        culibsStream,  // 新的统一stream作为async object
        clusterSize
    );
    
    // 正确处理替换：如果原操作有返回值，用新操作的返回值替换
    if (launchOp->getNumResults() > 0) {
      rewriter.replaceOp(launchOp, newLaunchOp->getResults());
    } else {
      rewriter.eraseOp(launchOp);
    }
    
    return success();
  }
};

struct StreamUnificationPass
    : public PassWrapper<StreamUnificationPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "stream-unification"; }
  StringRef getDescription() const final {
    return "Unify streams for mixed CuLibs-Kernel serial execution chains";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM::LLVMFuncOp mainGraphFunc = nullptr;
    
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto llvmFuncOp = dyn_cast<LLVM::LLVMFuncOp>(&op)) {
        if (llvmFuncOp.getName() == "main_graph") {
          mainGraphFunc = llvmFuncOp;
          break;
        }
      }
    }
    
    if (!mainGraphFunc || mainGraphFunc.isExternal()) {
      return;
    }
    
    if (failed(processMainGraphFunction(mainGraphFunc, context))) {
      signalPassFailure();
    }
  }

private:
  std::optional<SchedulingInfo> parseSchedulingInfoFromKernelName(const std::string& kernelName, Operation* op) {
    size_t schedPos = kernelName.find("sched_");
    if (schedPos == std::string::npos) {
      return std::nullopt;
    }
    
    std::string schedulePart = kernelName.substr(schedPos + 6);
    
    size_t pPos = schedulePart.find('P');
    size_t sPos = schedulePart.find("_S");
    size_t lPos = schedulePart.find("_L");
    
    if (pPos == std::string::npos || sPos == std::string::npos || lPos == std::string::npos) {
      return std::nullopt;
    }
    
    try {
      SchedulingInfo info;
      info.parallelGroupId = std::stoi(schedulePart.substr(pPos + 1, sPos - pPos - 1));
      info.subgraphId = std::stoi(schedulePart.substr(sPos + 2, lPos - sPos - 2));
      
      size_t nextUnderscore = schedulePart.find('_', lPos + 2);
      if (nextUnderscore != std::string::npos) {
        info.intraLevel = std::stoi(schedulePart.substr(lPos + 2, nextUnderscore - lPos - 2));
      } else {
        info.intraLevel = std::stoi(schedulePart.substr(lPos + 2));
      }
      
      info.operation = op;
      info.nodeType = "KERNEL";
      info.functionName = kernelName;
      
      return info;
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }

  LogicalResult processMainGraphFunction(LLVM::LLVMFuncOp mainGraphFunc, MLIRContext *context) {
    auto schedulingInfos = recoverSchedulingInfoFromMainGraph(mainGraphFunc);
    if (schedulingInfos.empty()) {
      return success();
    }
    
    auto serialChains = identifySerialChains(schedulingInfos);
    if (serialChains.empty()) {
      return success();
    }
    
    if (failed(prepareUnifiedStreams(serialChains))) {
      return failure();
    }
    
    RewritePatternSet patterns(context);
    patterns.add<UnifyMixedSerialChainPattern>(context, serialChains);
    
    // 使用更保守的配置
    GreedyRewriteConfig config;
    config.maxIterations = 10; // 限制最大迭代次数
    
    if (failed(applyPatternsAndFoldGreedily(mainGraphFunc, std::move(patterns), config))) {
      return failure();
    }
    
    return success();
  }
  
  std::vector<SchedulingInfo> recoverSchedulingInfoFromMainGraph(LLVM::LLVMFuncOp mainGraphFunc) {
    std::vector<SchedulingInfo> infos;
    
    mainGraphFunc.walk([&](Operation *op) {
      auto scheduleInfoAttr = op->getAttrOfType<StringAttr>("schedule.info");
      if (scheduleInfoAttr) {
        std::string scheduleStr = scheduleInfoAttr.getValue().str();
        auto schedulingInfo = parseSchedulingInfo(scheduleStr, op);
        
        if (schedulingInfo.has_value()) {
          schedulingInfo->streamValue = extractStreamFromOperationLLVM(op);
          infos.push_back(*schedulingInfo);
        }
        return;
      }
      
      if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        auto kernelName = launchOp.getKernelName().str();
        auto schedulingInfo = parseSchedulingInfoFromKernelName(kernelName, op);
        
        if (schedulingInfo.has_value()) {
          schedulingInfo->streamValue = extractStreamFromOperationLLVM(op);
          infos.push_back(*schedulingInfo);
        }
      }
    });
    
    std::sort(infos.begin(), infos.end(), 
        [](const SchedulingInfo& a, const SchedulingInfo& b) {
            if (a.parallelGroupId != b.parallelGroupId) return a.parallelGroupId < b.parallelGroupId;
            if (a.subgraphId != b.subgraphId) return a.subgraphId < b.subgraphId;
            return a.intraLevel < b.intraLevel;
        });
    
    return infos;
  }

  Value extractStreamFromOperationLLVM(Operation* op) {
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      if (launchOp.getAsyncToken()) {
        return launchOp.getAsyncToken();
      }
      
      auto asyncDeps = launchOp.getAsyncDependencies();
      if (!asyncDeps.empty()) {
        return asyncDeps[0];
      }
      
      for (Value operand : launchOp->getOperands()) {
        if (operand.getType().isa<LLVM::LLVMPointerType>()) {
          return operand;
        }
      }
      
      return {};
    }
    
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      auto operands = callOp.getOperands();
      
      if (isCuLibsCallLLVM(op)) {
        if (!operands.empty()) {
          return operands.back();
        }
      }
      
      return {};
    }
    
    return {};
  }

  bool isCuLibsCallLLVM(Operation* op) {
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        StringRef funcName = calleeAttr.getValue();
        return funcName.starts_with("mgpuCudnn") || 
               funcName.starts_with("mgpuCulibs");
      }
    }
    return false;
  }
  
  bool isGpuKernelLaunchLLVM(Operation* op) {
    if (isa<gpu::LaunchFuncOp>(op)) {
      return true;
    }
    
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        StringRef funcName = calleeAttr.getValue();
        return funcName.contains("enhanced_scheduled_kernels") ||
               funcName.contains("_kernel") ||
               funcName.contains("sched_P") ||
               op->hasAttr("gpu.kernel");
      }
    }
    
    return false;
  }
  
  std::string extractKernelNameLLVM(Operation* op) {
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      return launchOp.getKernelName().str();
    }
    
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        return calleeAttr.getValue().str();
      }
    }
    return "unknown_kernel";
  }
  
  std::string extractCuLibsNameLLVM(Operation* op) {
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        return calleeAttr.getValue().str();
      }
    }
    return "unknown_culibs";
  }
  
  std::optional<SchedulingInfo> parseSchedulingInfo(const std::string& scheduleStr, Operation* op) {
    size_t pPos = scheduleStr.find('P');
    size_t sPos = scheduleStr.find("_S");
    size_t lPos = scheduleStr.find("_L");
    
    if (pPos == std::string::npos || sPos == std::string::npos || lPos == std::string::npos) {
      return std::nullopt;
    }
    
    try {
      SchedulingInfo info;
      info.parallelGroupId = std::stoi(scheduleStr.substr(pPos + 1, sPos - pPos - 1));
      info.subgraphId = std::stoi(scheduleStr.substr(sPos + 2, lPos - sPos - 2));
      
      size_t lastUnderscore = scheduleStr.rfind('_');
      if (lastUnderscore > lPos) {
        info.intraLevel = std::stoi(scheduleStr.substr(lPos + 2, lastUnderscore - lPos - 2));
      } else {
        info.intraLevel = std::stoi(scheduleStr.substr(lPos + 2));
      }
      
      info.operation = op;
      
      if (isGpuKernelLaunchLLVM(op)) {
        info.nodeType = "KERNEL";
        info.functionName = extractKernelNameLLVM(op);
      } else if (isCuLibsCallLLVM(op)) {
        info.nodeType = "CULIBS";
        info.functionName = extractCuLibsNameLLVM(op);
      } else {
        return std::nullopt;
      }
      
      return info;
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }

  std::vector<SerialExecutionChain> identifySerialChains(const std::vector<SchedulingInfo>& infos) {
    std::vector<SerialExecutionChain> chains;
    
    std::map<std::pair<int,int>, std::vector<SchedulingInfo*>> nodesBySubgraph;
    for (const auto& info : infos) {
      nodesBySubgraph[{info.parallelGroupId, info.subgraphId}].push_back(
          const_cast<SchedulingInfo*>(&info));
    }
    
    for (const auto& [subgraphKey, subgraphNodes] : nodesBySubgraph) {
      analyzeSubgraphForSerialChains(subgraphKey.first, subgraphKey.second, 
                                   subgraphNodes, chains);
    }
    
    return chains;
  }
  
  void analyzeSubgraphForSerialChains(int parallelGroupId, int subgraphId,
                                    const std::vector<SchedulingInfo*>& nodes,
                                    std::vector<SerialExecutionChain>& chains) {
    if (nodes.size() <= 1) {
      return;
    }
    
    std::map<int, std::vector<SchedulingInfo*>> nodesByLevel;
    for (auto* node : nodes) {
      nodesByLevel[node->intraLevel].push_back(node);
    }
    
    SerialExecutionChain currentChain;
    currentChain.parallelGroupId = parallelGroupId;
    currentChain.subgraphId = subgraphId;
    
    for (const auto& [level, levelNodes] : nodesByLevel) {
      if (levelNodes.size() == 1) {
        auto* node = levelNodes[0];
        
        if (currentChain.nodes.empty()) {
          currentChain.startLevel = level;
          currentChain.endLevel = level;
          currentChain.nodes.push_back(node);
        } else if (level == currentChain.endLevel + 1) {
          currentChain.endLevel = level;
          currentChain.nodes.push_back(node);
        } else {
          finalizePotentialChain(currentChain, chains);
          
          currentChain.nodes.clear();
          currentChain.startLevel = level;
          currentChain.endLevel = level;
          currentChain.nodes.push_back(node);
          currentChain.hasMixedTypes = false;
        }
      } else {
        if (!currentChain.nodes.empty()) {
          finalizePotentialChain(currentChain, chains);
          currentChain.nodes.clear();
          currentChain.hasMixedTypes = false;
        }
      }
    }
    
    if (!currentChain.nodes.empty()) {
      finalizePotentialChain(currentChain, chains);
    }
  }

  void finalizePotentialChain(SerialExecutionChain& chain, 
                            std::vector<SerialExecutionChain>& chains) {
    if (chain.nodes.size() < 2) {
      return;
    }
    
    bool hasKernel = false;
    bool hasCuLibs = false;
    
    for (const auto* node : chain.nodes) {
      if (node->nodeType == "KERNEL") hasKernel = true;
      if (node->nodeType == "CULIBS") hasCuLibs = true;
    }
    
    chain.hasMixedTypes = hasKernel && hasCuLibs;
    
    if (chain.hasMixedTypes) {
      chains.push_back(chain);
    }
  }

  LogicalResult prepareUnifiedStreams(std::vector<SerialExecutionChain>& chains) {
    for (auto& chain : chains) {
      for (const auto* node : chain.nodes) {
        if (node->nodeType == "CULIBS" && node->streamValue) {
          chain.unifiedStream = node->streamValue;
          break;
        }
      }
      
      if (!chain.unifiedStream) {
        return failure();
      }
    }
    
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

void addSchedulingAttributes(Operation* op, int parallelGroupId, int subgraphId, int intraLevel, OpBuilder& builder) {
  auto context = builder.getContext();
  
  op->setAttr("schedule.parallel_group", 
              IntegerAttr::get(IntegerType::get(context, 32), parallelGroupId));
  op->setAttr("schedule.subgraph_id", 
              IntegerAttr::get(IntegerType::get(context, 32), subgraphId));
  op->setAttr("schedule.intra_level", 
              IntegerAttr::get(IntegerType::get(context, 32), intraLevel));
  
  std::string scheduleInfo = "P" + std::to_string(parallelGroupId) + 
                            "_S" + std::to_string(subgraphId) + 
                            "_L" + std::to_string(intraLevel);
  op->setAttr("schedule.info", StringAttr::get(context, scheduleInfo));
  
  if (isa<gpu::LaunchFuncOp>(op)) {
    op->setAttr("schedule.node_type", StringAttr::get(context, "KERNEL"));
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    StringRef funcName = callOp.getCallee();
    if (funcName.starts_with("mgpuCudnn") || funcName.starts_with("mgpuCulibs")) {
      op->setAttr("schedule.node_type", StringAttr::get(context, "CULIBS"));
    }
  }
}

std::unique_ptr<Pass> createStreamUnificationPass() {
  return std::make_unique<StreamUnificationPass>();
}

} // namespace onnx_mlir

static mlir::PassRegistration<StreamUnificationPass> pass;