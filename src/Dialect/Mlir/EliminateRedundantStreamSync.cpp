// #include "mlir/Pass/Pass.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/DenseSet.h"

// using namespace mlir;
// using namespace mlir::LLVM;

// #define DEBUG_TYPE "redundant-stream-sync-elimination"

// namespace {

// class RedundantStreamSyncEliminationPass
//     : public PassWrapper<RedundantStreamSyncEliminationPass, OperationPass<ModuleOp>> {
  
//   StringRef getArgument() const final { return "eliminate-redundant-stream-sync"; }
//   StringRef getDescription() const final {
//     return "Eliminate redundant GPU stream synchronization calls";
//   }
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<LLVMDialect>();
//     registry.insert<gpu::GPUDialect>();
//   }
  
//   void runOnOperation() override {
//     ModuleOp moduleOp = getOperation();
    
//     LLVM_DEBUG(llvm::dbgs() << "Running RedundantStreamSyncEliminationPass\n");
    
//     moduleOp.walk([&](LLVMFuncOp funcOp) {
//       eliminateRedundantSyncsInFunction(funcOp);
//     });
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed RedundantStreamSyncEliminationPass\n");
//   }

// private:
//   void eliminateRedundantSyncsInFunction(LLVMFuncOp funcOp) {
//     llvm::DenseMap<Value, CallOp> lastSyncOp;
//     llvm::DenseSet<Value> usedSinceLastSync;  // 真正的使用，不包括release
//     llvm::DenseMap<Value, CallOp> lastReleaseOp;  // 跟踪每个stream的最后一次release
//     llvm::SmallVector<CallOp> redundantSyncs;
//     llvm::SmallVector<CallOp> redundantReleases;
    
//     funcOp.walk([&](Operation *op) {
//       if (auto callOp = dyn_cast<CallOp>(op)) {
//         handleCallOperation(callOp, lastSyncOp, usedSinceLastSync, 
//                           lastReleaseOp, redundantSyncs, redundantReleases);
//       } else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//         handleGpuLaunchOperation(launchOp, usedSinceLastSync);
//       }
//     });
    
//     // 删除冗余的同步操作
//     for (auto redundantSync : redundantSyncs) {
//       LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream sync: " << redundantSync << "\n");
//       redundantSync.erase();
//     }
    
//     // 删除冗余的释放操作
//     for (auto redundantRelease : redundantReleases) {
//       LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream release: " << redundantRelease << "\n");
//       redundantRelease.erase();
//     }
//   }
  
//   void handleCallOperation(CallOp callOp,
//                           llvm::DenseMap<Value, CallOp> &lastSyncOp,
//                           llvm::DenseSet<Value> &usedSinceLastSync,
//                           llvm::DenseMap<Value, CallOp> &lastReleaseOp,
//                           llvm::SmallVector<CallOp> &redundantSyncs,
//                           llvm::SmallVector<CallOp> &redundantReleases) {
    
//     auto callee = callOp.getCallee();
//     if (!callee)
//       return;
    
//     StringRef functionName = *callee;
    
//     if (functionName == "mgpuStreamSynchronize") {
//       handleStreamSyncOperation(callOp, lastSyncOp, usedSinceLastSync, redundantSyncs);
//     } else if (functionName == "mgpuReleasePooledStream") {
//       handleStreamReleaseOperation(callOp, lastReleaseOp, redundantReleases);
//     } else if (isStreamUsingOperation(functionName)) {
//       handleStreamUsingOperation(callOp, usedSinceLastSync);
//     }
//   }
  
//   void handleStreamSyncOperation(CallOp syncOp,
//                                 llvm::DenseMap<Value, CallOp> &lastSyncOp,
//                                 llvm::DenseSet<Value> &usedSinceLastSync,
//                                 llvm::SmallVector<CallOp> &redundantSyncs) {
    
//     if (syncOp.getOperands().empty())
//       return;
      
//     Value streamValue = syncOp.getOperands()[0];
    
//     // 检查是否存在前一次同步且自那以后没有真正使用过
//     auto it = lastSyncOp.find(streamValue);
//     if (it != lastSyncOp.end()) {
//       // 如果在最后一次同步之后没有真正的使用（release不算使用），则当前同步是冗余的
//       if (usedSinceLastSync.find(streamValue) == usedSinceLastSync.end()) {
//         LLVM_DEBUG(llvm::dbgs() << "Found redundant sync for stream (no real usage since last sync)\n");
//         redundantSyncs.push_back(syncOp);
//         return;
//       }
//     }
    
//     // 更新最后一次同步操作
//     lastSyncOp[streamValue] = syncOp;
//     // 清除使用标记，因为我们刚刚同步了
//     usedSinceLastSync.erase(streamValue);
//   }
  
//   void handleStreamReleaseOperation(CallOp releaseOp,
//                                    llvm::DenseMap<Value, CallOp> &lastReleaseOp,
//                                    llvm::SmallVector<CallOp> &redundantReleases) {
    
//     if (releaseOp.getOperands().empty())
//       return;
      
//     Value streamValue = releaseOp.getOperands()[0];
    
//     // 如果已经有一个release操作，则前一个是冗余的
//     auto it = lastReleaseOp.find(streamValue);
//     if (it != lastReleaseOp.end()) {
//       LLVM_DEBUG(llvm::dbgs() << "Found redundant release for stream (consecutive releases)\n");
//       redundantReleases.push_back(it->second);
//     }
    
//     // 更新最后一次release操作
//     lastReleaseOp[streamValue] = releaseOp;
    
//     // 注意：我们不在这里修改usedSinceLastSync，因为release不算真正的使用
//   }
  
//   void handleGpuLaunchOperation(gpu::LaunchFuncOp launchOp,
//                                llvm::DenseSet<Value> &usedSinceLastSync) {
    
//     // gpu.launch_func 的stream通过AsyncToken获取
//     if (launchOp.getAsyncToken()) {
//       Value streamValue = launchOp.getAsyncToken();
//       if (isStreamValue(streamValue)) {
//         usedSinceLastSync.insert(streamValue);
//       }
//     }
//   }
  
//   void handleStreamUsingOperation(CallOp callOp,
//                                  llvm::DenseSet<Value> &usedSinceLastSync) {
    
//     if (!callOp.getOperands().empty()) {
//       Value lastOperand = callOp.getOperands().back();
//       if (isStreamValue(lastOperand)) {
//         usedSinceLastSync.insert(lastOperand);
//       }
//     }
//   }
  
//   bool isStreamUsingOperation(StringRef functionName) const {
//     return functionName.starts_with("mgpuCudnn") ||
//            functionName.starts_with("mgpuCulibs") ||
//            functionName.starts_with("mgpuCublas");
//   }
  
//   bool isStreamValue(Value value) const {
//     if (auto ptrType = value.getType().dyn_cast<LLVMPointerType>()) {
//       return true;
//     }
//     return false;
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
//     std::unique_ptr<Pass> createRedundantStreamSyncEliminationPass() {
//       return std::make_unique<RedundantStreamSyncEliminationPass>();
//     }
// } // namespace onnx_mlir

// static mlir::PassRegistration<RedundantStreamSyncEliminationPass> pass;

// #include "mlir/Pass/Pass.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/DenseSet.h"

// using namespace mlir;
// using namespace mlir::LLVM;

// #define DEBUG_TYPE "redundant-stream-sync-elimination"

// namespace {

// class RedundantStreamSyncEliminationPass
//     : public PassWrapper<RedundantStreamSyncEliminationPass, OperationPass<ModuleOp>> {
  
//   StringRef getArgument() const final { return "eliminate-redundant-stream-sync"; }
//   StringRef getDescription() const final {
//     return "Eliminate redundant GPU stream synchronization calls";
//   }
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<LLVMDialect>();
//     registry.insert<gpu::GPUDialect>();
//   }
  
//   void runOnOperation() override {
//     ModuleOp moduleOp = getOperation();
    
//     LLVM_DEBUG(llvm::dbgs() << "Running RedundantStreamSyncEliminationPass\n");
    
//     moduleOp.walk([&](LLVMFuncOp funcOp) {
//       eliminateRedundantSyncsInFunction(funcOp);
//     });
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed RedundantStreamSyncEliminationPass\n");
//   }

// private:
//   void eliminateRedundantSyncsInFunction(LLVMFuncOp funcOp) {
//     // 收集所有stream相关的操作，按执行顺序
//     llvm::SmallVector<Operation*> allOps;
//     llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamAcquires;
//     llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamUses;
//     llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamSyncs;
//     llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamReleases;
    
//     // 按顺序收集操作
//     funcOp.walk([&](Operation *op) {
//       allOps.push_back(op);
      
//       if (auto callOp = dyn_cast<CallOp>(op)) {
//         handleCallOperation(callOp, streamAcquires, streamUses, streamSyncs, streamReleases);
//       } else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//         handleGpuLaunchOperation(launchOp, streamUses);
//       }
//     });
    
//     // 获取所有相关的stream
//     llvm::DenseSet<Value> allStreams;
//     for (auto& [stream, _] : streamAcquires) allStreams.insert(stream);
//     for (auto& [stream, _] : streamUses) allStreams.insert(stream);
//     for (auto& [stream, _] : streamSyncs) allStreams.insert(stream);
//     for (auto& [stream, _] : streamReleases) allStreams.insert(stream);
    
//     llvm::SmallVector<Operation*> toErase;
    
//     for (Value stream : allStreams) {
//       analyzeStreamAndMarkRedundant(stream, streamAcquires, streamUses, streamSyncs, streamReleases, toErase);
//     }
    
//     // 删除标记的冗余操作 - 按正确的顺序删除
//     if (toErase.empty()) {
//       return; // 没有要删除的操作
//     }
    
//     // 先删除使用者(release, sync)，再删除定义者(acquire)
//     // 按操作在程序中的位置进行排序，然后反向删除
//     llvm::DenseMap<Operation*, size_t> opOrder;
//     size_t order = 0;
    
//     // 建立操作顺序映射
//     funcOp.walk([&](Operation* op) {
//       opOrder[op] = order++;
//     });
    
//     // 按程序顺序排序，然后反向删除
//     llvm::sort(toErase, [&](Operation* a, Operation* b) {
//       return opOrder[a] < opOrder[b];
//     });
    
//     // 反向删除，确保先删除使用者再删除定义者
//     for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
//       LLVM_DEBUG(llvm::dbgs() << "Removing redundant operation: " << **it << "\n");
//       (*it)->erase();
//     }
//   }
  
//   void handleCallOperation(CallOp callOp,
//                           llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamAcquires,
//                           llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses,
//                           llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamSyncs,
//                           llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamReleases) {
//     auto callee = callOp.getCallee();
//     if (!callee) return;
    
//     StringRef functionName = *callee;
    
//     if (functionName == "mgpuAcquirePooledStream") {
//       if (!callOp.getResults().empty()) {
//         Value stream = callOp.getResults()[0];
//         streamAcquires[stream].push_back(callOp);
//       }
//     } else if (functionName == "mgpuStreamSynchronize") {
//       if (!callOp.getOperands().empty()) {
//         Value stream = callOp.getOperands()[0];
//         streamSyncs[stream].push_back(callOp);
//       }
//     } else if (functionName == "mgpuReleasePooledStream") {
//       if (!callOp.getOperands().empty()) {
//         Value stream = callOp.getOperands()[0];
//         streamReleases[stream].push_back(callOp);
//       }
//     } else if (isStreamUsingOperation(functionName)) {
//       // 检查所有操作数，找到stream参数
//       for (Value operand : callOp.getOperands()) {
//         if (isStreamValue(operand)) {
//           streamUses[operand].push_back(callOp);
//           break; // 通常一个操作只使用一个stream，找到后就退出
//         }
//       }
//     }
//   }
  
//   void handleGpuLaunchOperation(gpu::LaunchFuncOp launchOp,
//                                llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses) {
//     if (launchOp.getAsyncToken() && isStreamValue(launchOp.getAsyncToken())) {
//       Value stream = launchOp.getAsyncToken();
//       streamUses[stream].push_back(launchOp);
//     }
//   }
  
//   void analyzeStreamAndMarkRedundant(Value stream,
//                                     const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamAcquires,
//                                     const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses,
//                                     const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamSyncs,
//                                     const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamReleases,
//                                     llvm::SmallVector<Operation*>& toErase) {
    
//     // 规则1: 如果stream从未被实际使用过（只有acquire、sync、release），则整个stream都是冗余的
//     auto usesIt = streamUses.find(stream);
//     bool hasActualUse = (usesIt != streamUses.end() && !usesIt->second.empty());
    
//     if (!hasActualUse) {
//       LLVM_DEBUG(llvm::dbgs() << "Stream never actually used, marking all operations as redundant\n");
      
//       // 标记所有相关操作为冗余
//       auto acquireIt = streamAcquires.find(stream);
//       if (acquireIt != streamAcquires.end()) {
//         for (auto op : acquireIt->second) {
//           toErase.push_back(op);
//         }
//       }
      
//       auto syncIt = streamSyncs.find(stream);
//       if (syncIt != streamSyncs.end()) {
//         for (auto op : syncIt->second) {
//           toErase.push_back(op);
//         }
//       }
      
//       auto releaseIt = streamReleases.find(stream);
//       if (releaseIt != streamReleases.end()) {
//         for (auto op : releaseIt->second) {
//           toErase.push_back(op);
//         }
//       }
//       return;
//     }
    
//     // 规则2: 对于同一个stream，只保留最后一次释放
//     auto releaseIt = streamReleases.find(stream);
//     if (releaseIt != streamReleases.end() && releaseIt->second.size() > 1) {
//       LLVM_DEBUG(llvm::dbgs() << "Multiple releases found, keeping only the last one\n");
      
//       // 删除除了最后一个之外的所有释放操作
//       for (size_t i = 0; i < releaseIt->second.size() - 1; i++) {
//         toErase.push_back(releaseIt->second[i]);
//       }
//     }
    
//     // 规则3: 删除冗余的同步操作
//     auto syncIt = streamSyncs.find(stream);
//     if (syncIt != streamSyncs.end() && syncIt->second.size() > 1) {
//       markRedundantSyncs(stream, syncIt->second, usesIt->second, toErase);
//     }
//   }
  
//   void markRedundantSyncs(Value stream,
//                          const llvm::SmallVector<Operation*>& syncOps,
//                          const llvm::SmallVector<Operation*>& useOps,
//                          llvm::SmallVector<Operation*>& toErase) {
    
//     // 创建所有操作的时间顺序映射
//     llvm::DenseMap<Operation*, size_t> opOrder;
//     size_t order = 0;
    
//     // 遍历包含函数，建立操作顺序
//     auto parentFunc = syncOps[0]->getParentOfType<LLVMFuncOp>();
//     parentFunc.walk([&](Operation* op) {
//       opOrder[op] = order++;
//     });
    
//     // 按时间顺序排序同步操作
//     llvm::SmallVector<Operation*> sortedSyncs = syncOps;
//     llvm::sort(sortedSyncs, [&](Operation* a, Operation* b) {
//       return opOrder[a] < opOrder[b];
//     });
    
//     // 检查每对相邻的同步操作
//     for (size_t i = 0; i < sortedSyncs.size() - 1; i++) {
//       Operation* firstSync = sortedSyncs[i];
//       Operation* secondSync = sortedSyncs[i + 1];
      
//       size_t firstSyncOrder = opOrder[firstSync];
//       size_t secondSyncOrder = opOrder[secondSync];
      
//       // 检查两个同步之间是否有实际使用
//       bool hasUseBetween = false;
//       for (Operation* useOp : useOps) {
//         size_t useOrder = opOrder[useOp];
//         if (useOrder > firstSyncOrder && useOrder < secondSyncOrder) {
//           hasUseBetween = true;
//           break;
//         }
//       }
      
//       // 如果两个同步之间没有实际使用，第二个同步是冗余的
//       if (!hasUseBetween) {
//         LLVM_DEBUG(llvm::dbgs() << "Found redundant sync (no use between syncs)\n");
//         toErase.push_back(secondSync);
//       }
//     }
//   }
  
//   bool isStreamUsingOperation(StringRef functionName) const {
//     return functionName.starts_with("mgpuCudnn") ||
//            functionName.starts_with("mgpuCulibs") ||
//            functionName.starts_with("mgpuCublas") ||
//            functionName == "mgpuAcquirePooledHandles" ||
//            functionName == "mgpuMemAlloc" ||
//            functionName == "mgpuMemcpy" ||
//            functionName == "mgpuMemcpyAsync" ||
//            functionName == "mgpuMemset" ||
//            functionName == "mgpuMemsetAsync";
//   }
  
//   bool isStreamValue(Value value) const {
//     return value.getType().isa<LLVMPointerType>();
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
//     std::unique_ptr<Pass> createRedundantStreamSyncEliminationPass() {
//       return std::make_unique<RedundantStreamSyncEliminationPass>();
//     }
// } // namespace onnx_mlir

// static mlir::PassRegistration<RedundantStreamSyncEliminationPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::LLVM;

#define DEBUG_TYPE "redundant-stream-sync-elimination"

namespace {

class RedundantStreamSyncEliminationPass
    : public PassWrapper<RedundantStreamSyncEliminationPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "eliminate-redundant-stream-sync"; }
  StringRef getDescription() const final {
    return "Eliminate redundant GPU stream synchronization calls";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVMDialect>();
    registry.insert<gpu::GPUDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "Running RedundantStreamSyncEliminationPass\n");
    
    moduleOp.walk([&](LLVMFuncOp funcOp) {
      eliminateRedundantSyncsInFunction(funcOp);
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Completed RedundantStreamSyncEliminationPass\n");
  }

private:
  void eliminateRedundantSyncsInFunction(LLVMFuncOp funcOp) {
    // 收集所有stream相关的操作，按执行顺序
    llvm::SmallVector<Operation*> allOps;
    llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamAcquires;
    llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamUses;
    llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamSyncs;
    llvm::DenseMap<Value, llvm::SmallVector<Operation*>> streamReleases;
    
    // 按顺序收集操作
    funcOp.walk([&](Operation *op) {
      allOps.push_back(op);
      
      if (auto callOp = dyn_cast<CallOp>(op)) {
        handleCallOperation(callOp, streamAcquires, streamUses, streamSyncs, streamReleases);
      } else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        handleGpuLaunchOperation(launchOp, streamUses);
      }
    });
    
    // 获取所有相关的stream
    llvm::DenseSet<Value> allStreams;
    for (auto& [stream, _] : streamAcquires) allStreams.insert(stream);
    for (auto& [stream, _] : streamUses) allStreams.insert(stream);
    for (auto& [stream, _] : streamSyncs) allStreams.insert(stream);
    for (auto& [stream, _] : streamReleases) allStreams.insert(stream);
    
    llvm::SmallVector<Operation*> toErase;
    
    for (Value stream : allStreams) {
      analyzeStreamAndMarkRedundant(stream, streamAcquires, streamUses, streamSyncs, streamReleases, toErase);
    }
    
    // 删除标记的冗余操作 - 按正确的顺序删除
    if (toErase.empty()) {
      return; // 没有要删除的操作
    }
    
    // 先删除使用者(release, sync)，再删除定义者(acquire)
    // 按操作在程序中的位置进行排序，然后反向删除
    llvm::DenseMap<Operation*, size_t> opOrder;
    size_t order = 0;
    
    // 建立操作顺序映射
    funcOp.walk([&](Operation* op) {
      opOrder[op] = order++;
    });
    
    // 按程序顺序排序，然后反向删除
    llvm::sort(toErase, [&](Operation* a, Operation* b) {
      return opOrder[a] < opOrder[b];
    });
    
    // 反向删除，确保先删除使用者再删除定义者
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant operation: " << **it << "\n");
      (*it)->erase();
    }
  }
  
  void handleCallOperation(CallOp callOp,
                          llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamAcquires,
                          llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses,
                          llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamSyncs,
                          llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamReleases) {
    auto callee = callOp.getCallee();
    if (!callee) return;
    
    StringRef functionName = *callee;
    
    if (functionName == "mgpuAcquirePooledStream") {
      if (!callOp.getResults().empty()) {
        Value stream = callOp.getResults()[0];
        streamAcquires[stream].push_back(callOp);
      }
    } else if (functionName == "mgpuStreamSynchronize") {
      if (!callOp.getOperands().empty()) {
        Value stream = callOp.getOperands()[0];
        streamSyncs[stream].push_back(callOp);
      }
    } else if (functionName == "mgpuReleasePooledStream") {
      if (!callOp.getOperands().empty()) {
        Value stream = callOp.getOperands()[0];
        streamReleases[stream].push_back(callOp);
      }
    } else if (isStreamUsingOperation(functionName)) {
      // 检查所有操作数，找到stream参数
      for (Value operand : callOp.getOperands()) {
        if (isStreamValue(operand)) {
          streamUses[operand].push_back(callOp);
          break; // 通常一个操作只使用一个stream，找到后就退出
        }
      }
    }
  }
  
  void handleGpuLaunchOperation(gpu::LaunchFuncOp launchOp,
                               llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses) {
    // 检查 async dependencies (<%stream : !llvm.ptr> 语法)
    for (Value asyncDep : launchOp.getAsyncDependencies()) {
      if (isStreamValue(asyncDep)) {
        streamUses[asyncDep].push_back(launchOp);
        LLVM_DEBUG(llvm::dbgs() << "Found stream usage in gpu.launch_func async dependency: " << asyncDep << "\n");
      }
    }
    
    // 检查 async token (如果存在)
    if (launchOp.getAsyncToken() && isStreamValue(launchOp.getAsyncToken())) {
      Value stream = launchOp.getAsyncToken();
      streamUses[stream].push_back(launchOp);
      LLVM_DEBUG(llvm::dbgs() << "Found stream usage in gpu.launch_func async token: " << stream << "\n");
    }
    
    // 作为备选方案，也检查所有操作数中的stream类型
    for (Value operand : launchOp.getOperands()) {
      if (isStreamValue(operand)) {
        streamUses[operand].push_back(launchOp);
        LLVM_DEBUG(llvm::dbgs() << "Found stream usage in gpu.launch_func operand: " << operand << "\n");
        break; // 通常只有一个stream，找到后就退出
      }
    }
  }
  
  void analyzeStreamAndMarkRedundant(Value stream,
                                    const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamAcquires,
                                    const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamUses,
                                    const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamSyncs,
                                    const llvm::DenseMap<Value, llvm::SmallVector<Operation*>>& streamReleases,
                                    llvm::SmallVector<Operation*>& toErase) {
    
    // 规则1: 如果stream从未被实际使用过（只有acquire、sync、release），则整个stream都是冗余的
    // 暂时注释掉这个逻辑，因为存在识别问题
    /*
    auto usesIt = streamUses.find(stream);
    bool hasActualUse = (usesIt != streamUses.end() && !usesIt->second.empty());
    
    if (!hasActualUse) {
      LLVM_DEBUG(llvm::dbgs() << "Stream never actually used, marking all operations as redundant\n");
      
      // 标记所有相关操作为冗余
      auto acquireIt = streamAcquires.find(stream);
      if (acquireIt != streamAcquires.end()) {
        for (auto op : acquireIt->second) {
          toErase.push_back(op);
        }
      }
      
      auto syncIt = streamSyncs.find(stream);
      if (syncIt != streamSyncs.end()) {
        for (auto op : syncIt->second) {
          toErase.push_back(op);
        }
      }
      
      auto releaseIt = streamReleases.find(stream);
      if (releaseIt != streamReleases.end()) {
        for (auto op : releaseIt->second) {
          toErase.push_back(op);
        }
      }
      return;
    }
    */
    
    LLVM_DEBUG(llvm::dbgs() << "Analyzing stream for partial redundancy\n");
    
    // 规则2: 对于同一个stream，只保留最后一次释放
    auto releaseIt = streamReleases.find(stream);
    if (releaseIt != streamReleases.end() && releaseIt->second.size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "Multiple releases found, keeping only the last one\n");
      
      // 创建操作顺序映射来确定哪个是最后一个释放
      llvm::DenseMap<Operation*, size_t> opOrder;
      size_t order = 0;
      
      // 遍历包含函数，建立操作顺序
      auto parentFunc = releaseIt->second[0]->getParentOfType<LLVMFuncOp>();
      parentFunc.walk([&](Operation* op) {
        opOrder[op] = order++;
      });
      
      // 按时间顺序排序释放操作
      llvm::SmallVector<Operation*> sortedReleases = releaseIt->second;
      llvm::sort(sortedReleases, [&](Operation* a, Operation* b) {
        return opOrder[a] < opOrder[b];
      });
      
      // 删除除了最后一个之外的所有释放操作
      for (size_t i = 0; i < sortedReleases.size() - 1; i++) {
        toErase.push_back(sortedReleases[i]);
      }
    }
    
    // 规则3: 删除冗余的同步操作
    auto usesIt = streamUses.find(stream);
    auto syncIt = streamSyncs.find(stream);
    if (syncIt != streamSyncs.end() && syncIt->second.size() > 1) {
      if (usesIt != streamUses.end()) {
        markRedundantSyncs(stream, syncIt->second, usesIt->second, toErase);
      }
    }
  }
  
  void markRedundantSyncs(Value stream,
                         const llvm::SmallVector<Operation*>& syncOps,
                         const llvm::SmallVector<Operation*>& useOps,
                         llvm::SmallVector<Operation*>& toErase) {
    
    // 创建所有操作的时间顺序映射
    llvm::DenseMap<Operation*, size_t> opOrder;
    size_t order = 0;
    
    // 遍历包含函数，建立操作顺序
    auto parentFunc = syncOps[0]->getParentOfType<LLVMFuncOp>();
    parentFunc.walk([&](Operation* op) {
      opOrder[op] = order++;
    });
    
    // 按时间顺序排序同步操作
    llvm::SmallVector<Operation*> sortedSyncs = syncOps;
    llvm::sort(sortedSyncs, [&](Operation* a, Operation* b) {
      return opOrder[a] < opOrder[b];
    });
    
    // 检查每对相邻的同步操作
    for (size_t i = 0; i < sortedSyncs.size() - 1; i++) {
      Operation* firstSync = sortedSyncs[i];
      Operation* secondSync = sortedSyncs[i + 1];
      
      size_t firstSyncOrder = opOrder[firstSync];
      size_t secondSyncOrder = opOrder[secondSync];
      
      // 检查两个同步之间是否有实际使用
      bool hasUseBetween = false;
      for (Operation* useOp : useOps) {
        size_t useOrder = opOrder[useOp];
        if (useOrder > firstSyncOrder && useOrder < secondSyncOrder) {
          hasUseBetween = true;
          break;
        }
      }
      
      // 如果两个同步之间没有实际使用，第二个同步是冗余的
      if (!hasUseBetween) {
        LLVM_DEBUG(llvm::dbgs() << "Found redundant sync (no use between syncs)\n");
        toErase.push_back(secondSync);
      }
    }
  }
  
  bool isStreamUsingOperation(StringRef functionName) const {
    return functionName.starts_with("mgpuCudnn") ||
           functionName.starts_with("mgpuCulibs") ||
           functionName.starts_with("mgpuCublas") ||
           functionName == "mgpuAcquirePooledHandles" ||
           functionName == "mgpuMemAlloc" ||
           functionName == "mgpuMemcpy" ||
           functionName == "mgpuMemcpyAsync" ||
           functionName == "mgpuMemset" ||
           functionName == "mgpuMemsetAsync";
  }
  
  bool isStreamValue(Value value) const {
    return value.getType().isa<LLVMPointerType>();
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createRedundantStreamSyncEliminationPass() {
      return std::make_unique<RedundantStreamSyncEliminationPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<RedundantStreamSyncEliminationPass> pass;