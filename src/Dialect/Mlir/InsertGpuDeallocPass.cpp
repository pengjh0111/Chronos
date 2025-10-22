#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "insert-gpu-dealloc"

namespace {

// 存储 GPU Alloc 操作信息
struct GpuAllocInfo {
  gpu::AllocOp allocOp;
  Value memref;
  Value asyncToken;
  Location loc;
  
  GpuAllocInfo(gpu::AllocOp op) : allocOp(op), loc(op.getLoc()) {
    memref = op.getMemref();
    asyncToken = op.getAsyncToken();
  }
};

class InsertGpuDeallocPass
    : public PassWrapper<InsertGpuDeallocPass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { 
    return "insert-gpu-dealloc"; 
  }
  
  StringRef getDescription() const final {
    return "Insert gpu.dealloc operations before func.return for all gpu.alloc operations";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Insert GPU Dealloc Pass ===\n");
    
    // 收集所有 gpu.alloc 操作
    std::vector<GpuAllocInfo> allocOps;
    if (failed(collectGpuAllocOps(funcOp, allocOps))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect gpu.alloc operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << allocOps.size() << " gpu.alloc ops\n");
    
    // 如果没有找到 gpu.alloc，直接返回
    if (allocOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No gpu.alloc operations found, skipping\n");
      return;
    }
    
    // 找到所有的 func.return 操作并插入 gpu.dealloc
    if (failed(insertDeallocsBeforeReturn(funcOp, allocOps))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to insert gpu.dealloc operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully inserted all dealloc operations\n");
  }

private:
  // 收集所有 gpu.alloc 操作
  LogicalResult collectGpuAllocOps(
      func::FuncOp funcOp,
      std::vector<GpuAllocInfo>& allocOps) {
    
    funcOp.walk([&](gpu::AllocOp allocOp) {
      GpuAllocInfo info(allocOp);
      allocOps.push_back(info);
      LLVM_DEBUG(llvm::dbgs() << "  Found gpu.alloc: " << allocOp << "\n");
    });
    
    return success();
  }

  // 在 func.return 之前插入 gpu.dealloc 操作
  LogicalResult insertDeallocsBeforeReturn(
      func::FuncOp funcOp,
      const std::vector<GpuAllocInfo>& allocOps) {
    
    // 找到所有的 func.return 操作
    SmallVector<func::ReturnOp, 4> returnOps;
    funcOp.walk([&](func::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
    });
    
    if (returnOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No func.return operations found\n");
      return failure();
    }
    
    // 为每个 return 操作插入 dealloc
    for (auto returnOp : returnOps) {
      OpBuilder builder(returnOp);
      
      LLVM_DEBUG(llvm::dbgs() << "Inserting deallocations before return\n");
      
      // 在 return 之前插入 gpu.wait async (创建同步点)
      auto waitBeforeFree = builder.create<gpu::WaitOp>(
          returnOp.getLoc(),
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange{});  // 空的依赖列表，会等待所有之前的 GPU 操作
      
      Value lastToken = waitBeforeFree.getAsyncToken();
      
      // 为每个 alloc 插入对应的 dealloc
      for (const auto& allocInfo : allocOps) {
        // 创建 gpu.dealloc async，依赖前一个 token
        auto deallocOp = builder.create<gpu::DeallocOp>(
            allocInfo.loc,
            gpu::AsyncTokenType::get(builder.getContext()),
            ValueRange{lastToken},  // 依赖前一个 token
            allocInfo.memref);
        
        lastToken = deallocOp.getAsyncToken();
        
        LLVM_DEBUG(llvm::dbgs() << "  Inserted gpu.dealloc for memref\n");
      }
      
      // 在所有 dealloc 之后插入 gpu.wait 来同步
      // 这确保在 return 之前所有内存都已经释放
      builder.create<gpu::WaitOp>(
          returnOp.getLoc(),
          Type{},  // 没有返回 async token
          ValueRange{lastToken});  // 等待最后一个 dealloc 完成
      
      LLVM_DEBUG(llvm::dbgs() << "  Inserted final gpu.wait\n");
    }
    
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createInsertGpuDeallocPass() {
  return std::make_unique<InsertGpuDeallocPass>();
}

} // namespace mlir

// Pass 注册
static mlir::PassRegistration<InsertGpuDeallocPass> pass;