#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace mlir;
using namespace mlir::gpu;

#define DEBUG_TYPE "convert-gpu-memcpy-to-memref-copy"

namespace {

// 存储 gpu.memcpy 操作信息
struct GpuMemcpyOpInfo {
  Operation* memcpyOp;
  Value dst;
  Value src;
  Type dstType;
  Type srcType;
  SmallVector<Value> asyncDependencies;
  Value asyncToken;  // 返回的异步token（如果有）
  Location loc;
  
  GpuMemcpyOpInfo(Operation* op) : memcpyOp(op), loc(op->getLoc()) {}
};

class ConvertGpuMemcpyToMemrefCopyPass
    : public PassWrapper<ConvertGpuMemcpyToMemrefCopyPass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { 
    return "convert-gpu-memcpy-to-memref-copy"; 
  }
  
  StringRef getDescription() const final {
    return "Convert gpu.memcpy operations to memref.copy operations";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Convert GPU Memcpy to Memref Copy Pass ===\n");
    
    // 收集所有 gpu.memcpy 操作
    std::vector<GpuMemcpyOpInfo> memcpyOps;
    
    if (failed(collectGpuMemcpyOps(funcOp, memcpyOps))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << memcpyOps.size() 
               << " gpu.memcpy ops\n");
    
    // 转换 gpu.memcpy 操作
    for (auto& info : memcpyOps) {
      if (failed(convertGpuMemcpyToMemrefCopy(info))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert a gpu.memcpy op\n");
        signalPassFailure();
        return;
      }
    }
    
    // 清理 gpu.wait 操作
    if (failed(cleanupGpuWaitOps(funcOp))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to cleanup gpu.wait operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted all operations\n");
  }

private:
  // 收集所有 gpu.memcpy 操作
  LogicalResult collectGpuMemcpyOps(
      func::FuncOp funcOp,
      std::vector<GpuMemcpyOpInfo>& memcpyOps) {
    
    funcOp.walk([&](Operation* op) {
      if (op->getName().getStringRef() == "gpu.memcpy") {
        GpuMemcpyOpInfo info(op);
        if (succeeded(extractGpuMemcpyInfo(op, info))) {
          memcpyOps.push_back(info);
        }
      }
    });
    
    return success();
  }

  // 提取 gpu.memcpy 操作信息
  LogicalResult extractGpuMemcpyInfo(Operation* op, GpuMemcpyOpInfo& info) {
    // gpu.memcpy 的格式：
    // %result = gpu.memcpy async [%token1, %token2, ...] %dst, %src : dst_type, src_type
    
    if (op->getNumOperands() < 2) {
      LLVM_DEBUG(llvm::dbgs() << "gpu.memcpy: insufficient operands\n");
      return failure();
    }
    
    // 获取操作数
    // 需要区分异步依赖token和实际的dst/src操作数
    size_t numOperands = op->getNumOperands();
    
    // 通常最后两个操作数是 dst 和 src
    info.dst = op->getOperand(numOperands - 2);
    info.src = op->getOperand(numOperands - 1);
    
    // 前面的操作数是异步依赖token（如果有的话）
    for (size_t i = 0; i < numOperands - 2; ++i) {
      info.asyncDependencies.push_back(op->getOperand(i));
    }
    
    info.dstType = info.dst.getType();
    info.srcType = info.src.getType();
    
    // 检查是否有返回值（异步token）
    if (op->getNumResults() > 0) {
      info.asyncToken = op->getResult(0);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "gpu.memcpy: dst type = " << info.dstType 
               << ", src type = " << info.srcType 
               << ", async deps = " << info.asyncDependencies.size()
               << ", has token result = " << (bool)info.asyncToken << "\n");
    
    return success();
  }

  // 将 gpu.memcpy 转换为 memref.copy
LogicalResult convertGpuMemcpyToMemrefCopy(GpuMemcpyOpInfo& info) {
  LLVM_DEBUG(llvm::dbgs() << "\nConverting gpu.memcpy operation\n");
  
  OpBuilder builder(info.memcpyOp);
  builder.setInsertionPoint(info.memcpyOp);
  
  // 创建 memref.copy 操作
  builder.create<memref::CopyOp>(info.loc, info.src, info.dst);
  
  LLVM_DEBUG(llvm::dbgs() << "  Created memref.copy from " 
             << info.srcType << " to " << info.dstType << "\n");
  
  // 如果 gpu.memcpy 返回了异步token，需要先处理所有使用者
  if (info.asyncToken) {
    // 收集所有使用该token的操作
    SmallVector<Operation*> usersToErase;
    for (auto& use : info.asyncToken.getUses()) {
      Operation* user = use.getOwner();
      // 如果是 gpu.wait 操作，标记删除
      if (user->getName().getStringRef() == "gpu.wait") {
        usersToErase.push_back(user);
      }
    }
    
    // 先删除所有使用者
    for (Operation* user : usersToErase) {
      LLVM_DEBUG(llvm::dbgs() << "  Erasing user: gpu.wait\n");
      user->erase();
    }
    
    // 如果还有其他使用者，用一个dummy值替换
    if (!info.asyncToken.use_empty()) {
      LLVM_DEBUG(llvm::dbgs() << "  Warning: token still has " 
                 << std::distance(info.asyncToken.use_begin(), 
                                  info.asyncToken.use_end())
                 << " remaining uses\n");
      // 可以选择报错或者强制替换为null
      return failure();
    }
  }
  
  // 现在可以安全删除 gpu.memcpy 操作
  info.memcpyOp->erase();
  
  LLVM_DEBUG(llvm::dbgs() << "  Successfully converted gpu.memcpy\n");
  return success();
}

  // 清理 gpu.wait 操作
    LogicalResult cleanupGpuWaitOps(func::FuncOp funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "\nCleaning up remaining gpu.wait operations\n");
    
    std::vector<Operation*> waitOpsToRemove;
    
    funcOp.walk([&](Operation* op) {
        if (op->getName().getStringRef() == "gpu.wait") {
        // 清理剩余的孤立 gpu.wait 操作（没有操作数或操作数无效）
        if (op->getNumOperands() == 0) {
            waitOpsToRemove.push_back(op);
            LLVM_DEBUG(llvm::dbgs() << "  Marking gpu.wait for removal\n");
        }
        }
    });
    
    for (auto* op : waitOpsToRemove) {
        op->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Removed " << waitOpsToRemove.size() 
                << " gpu.wait operations\n");
    
    return success();
    }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createConvertGpuMemcpyToMemrefCopyPass() {
  return std::make_unique<ConvertGpuMemcpyToMemrefCopyPass>();
}

} // namespace onnx_mlir

// Pass 注册
static mlir::PassRegistration<ConvertGpuMemcpyToMemrefCopyPass> pass;