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
//     llvm::DenseSet<Value> usedSinceLastSync;
//     llvm::SmallVector<CallOp> redundantSyncs;
    
//     funcOp.walk([&](Operation *op) {
//       if (auto callOp = dyn_cast<CallOp>(op)) {
//         handleCallOperation(callOp, lastSyncOp, usedSinceLastSync, redundantSyncs);
//       } else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
//         handleGpuLaunchOperation(launchOp, usedSinceLastSync);
//       }
//     });
    
//     for (auto redundantSync : redundantSyncs) {
//       LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream sync: " << redundantSync << "\n");
//       redundantSync.erase();
//     }
//   }
  
//   void handleCallOperation(CallOp callOp,
//                           llvm::DenseMap<Value, CallOp> &lastSyncOp,
//                           llvm::DenseSet<Value> &usedSinceLastSync,
//                           llvm::SmallVector<CallOp> &redundantSyncs) {
    
//     auto callee = callOp.getCallee();
//     if (!callee)
//       return;
    
//     StringRef functionName = *callee;
    
//     if (functionName == "mgpuStreamSynchronize") {
//       handleStreamSyncOperation(callOp, lastSyncOp, usedSinceLastSync, redundantSyncs);
//     } else if (isStreamUsingOperation(functionName)) {
//       handleStreamUsingOperation(callOp, usedSinceLastSync);
//     } else if (functionName == "mgpuReleasePooledStream") {
//       handleStreamReleaseOperation(callOp, lastSyncOp, usedSinceLastSync);
//     }
//   }
  
//   void handleStreamSyncOperation(CallOp syncOp,
//                                 llvm::DenseMap<Value, CallOp> &lastSyncOp,
//                                 llvm::DenseSet<Value> &usedSinceLastSync,
//                                 llvm::SmallVector<CallOp> &redundantSyncs) {
    
//     if (syncOp.getOperands().empty())
//       return;
      
//     Value streamValue = syncOp.getOperands()[0];
    
//     auto it = lastSyncOp.find(streamValue);
//     if (it != lastSyncOp.end()) {
//       if (usedSinceLastSync.find(streamValue) == usedSinceLastSync.end()) {
//         redundantSyncs.push_back(syncOp);
//         return;
//       }
//     }
    
//     lastSyncOp[streamValue] = syncOp;
//     usedSinceLastSync.erase(streamValue);
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
  
//   void handleStreamReleaseOperation(CallOp releaseOp,
//                                    llvm::DenseMap<Value, CallOp> &lastSyncOp,
//                                    llvm::DenseSet<Value> &usedSinceLastSync) {
    
//     if (releaseOp.getOperands().empty())
//       return;
      
//     Value streamValue = releaseOp.getOperands()[0];
    
//     lastSyncOp.erase(streamValue);
//     usedSinceLastSync.erase(streamValue);
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
    llvm::DenseMap<Value, CallOp> lastSyncOp;
    llvm::DenseSet<Value> usedSinceLastSync;  // 真正的使用，不包括release
    llvm::DenseMap<Value, CallOp> lastReleaseOp;  // 跟踪每个stream的最后一次release
    llvm::SmallVector<CallOp> redundantSyncs;
    llvm::SmallVector<CallOp> redundantReleases;
    
    funcOp.walk([&](Operation *op) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        handleCallOperation(callOp, lastSyncOp, usedSinceLastSync, 
                          lastReleaseOp, redundantSyncs, redundantReleases);
      } else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        handleGpuLaunchOperation(launchOp, usedSinceLastSync);
      }
    });
    
    // 删除冗余的同步操作
    for (auto redundantSync : redundantSyncs) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream sync: " << redundantSync << "\n");
      redundantSync.erase();
    }
    
    // 删除冗余的释放操作
    for (auto redundantRelease : redundantReleases) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream release: " << redundantRelease << "\n");
      redundantRelease.erase();
    }
  }
  
  void handleCallOperation(CallOp callOp,
                          llvm::DenseMap<Value, CallOp> &lastSyncOp,
                          llvm::DenseSet<Value> &usedSinceLastSync,
                          llvm::DenseMap<Value, CallOp> &lastReleaseOp,
                          llvm::SmallVector<CallOp> &redundantSyncs,
                          llvm::SmallVector<CallOp> &redundantReleases) {
    
    auto callee = callOp.getCallee();
    if (!callee)
      return;
    
    StringRef functionName = *callee;
    
    if (functionName == "mgpuStreamSynchronize") {
      handleStreamSyncOperation(callOp, lastSyncOp, usedSinceLastSync, redundantSyncs);
    } else if (functionName == "mgpuReleasePooledStream") {
      handleStreamReleaseOperation(callOp, lastReleaseOp, redundantReleases);
    } else if (isStreamUsingOperation(functionName)) {
      handleStreamUsingOperation(callOp, usedSinceLastSync);
    }
  }
  
  void handleStreamSyncOperation(CallOp syncOp,
                                llvm::DenseMap<Value, CallOp> &lastSyncOp,
                                llvm::DenseSet<Value> &usedSinceLastSync,
                                llvm::SmallVector<CallOp> &redundantSyncs) {
    
    if (syncOp.getOperands().empty())
      return;
      
    Value streamValue = syncOp.getOperands()[0];
    
    // 检查是否存在前一次同步且自那以后没有真正使用过
    auto it = lastSyncOp.find(streamValue);
    if (it != lastSyncOp.end()) {
      // 如果在最后一次同步之后没有真正的使用（release不算使用），则当前同步是冗余的
      if (usedSinceLastSync.find(streamValue) == usedSinceLastSync.end()) {
        LLVM_DEBUG(llvm::dbgs() << "Found redundant sync for stream (no real usage since last sync)\n");
        redundantSyncs.push_back(syncOp);
        return;
      }
    }
    
    // 更新最后一次同步操作
    lastSyncOp[streamValue] = syncOp;
    // 清除使用标记，因为我们刚刚同步了
    usedSinceLastSync.erase(streamValue);
  }
  
  void handleStreamReleaseOperation(CallOp releaseOp,
                                   llvm::DenseMap<Value, CallOp> &lastReleaseOp,
                                   llvm::SmallVector<CallOp> &redundantReleases) {
    
    if (releaseOp.getOperands().empty())
      return;
      
    Value streamValue = releaseOp.getOperands()[0];
    
    // 如果已经有一个release操作，则前一个是冗余的
    auto it = lastReleaseOp.find(streamValue);
    if (it != lastReleaseOp.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Found redundant release for stream (consecutive releases)\n");
      redundantReleases.push_back(it->second);
    }
    
    // 更新最后一次release操作
    lastReleaseOp[streamValue] = releaseOp;
    
    // 注意：我们不在这里修改usedSinceLastSync，因为release不算真正的使用
  }
  
  void handleGpuLaunchOperation(gpu::LaunchFuncOp launchOp,
                               llvm::DenseSet<Value> &usedSinceLastSync) {
    
    // gpu.launch_func 的stream通过AsyncToken获取
    if (launchOp.getAsyncToken()) {
      Value streamValue = launchOp.getAsyncToken();
      if (isStreamValue(streamValue)) {
        usedSinceLastSync.insert(streamValue);
      }
    }
  }
  
  void handleStreamUsingOperation(CallOp callOp,
                                 llvm::DenseSet<Value> &usedSinceLastSync) {
    
    if (!callOp.getOperands().empty()) {
      Value lastOperand = callOp.getOperands().back();
      if (isStreamValue(lastOperand)) {
        usedSinceLastSync.insert(lastOperand);
      }
    }
  }
  
  bool isStreamUsingOperation(StringRef functionName) const {
    return functionName.starts_with("mgpuCudnn") ||
           functionName.starts_with("mgpuCulibs") ||
           functionName.starts_with("mgpuCublas");
  }
  
  bool isStreamValue(Value value) const {
    if (auto ptrType = value.getType().dyn_cast<LLVMPointerType>()) {
      return true;
    }
    return false;
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createRedundantStreamSyncEliminationPass() {
      return std::make_unique<RedundantStreamSyncEliminationPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<RedundantStreamSyncEliminationPass> pass;