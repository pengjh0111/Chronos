#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

using namespace mlir;
using namespace mlir::gpu;

#define DEBUG_TYPE "optimize-gpu-stream"

namespace {

// 存储stream相关的操作信息
struct StreamInfo {
  LLVM::CallOp createOp;           // stream创建操作
  LLVM::CallOp handleOp;           // handle创建操作（可选）
  std::vector<LLVM::CallOp> wrapperCalls;  // 包装函数调用
  std::vector<gpu::LaunchFuncOp> kernelLaunches;  // kernel启动
  LLVM::CallOp syncOp;             // stream同步操作
  LLVM::CallOp destroyOp;          // stream销毁操作
  Value streamValue;               // stream值
  
  StreamInfo() : createOp(nullptr), handleOp(nullptr), 
                 syncOp(nullptr), destroyOp(nullptr) {}
};

class OptimizeGPUStreamPass
    : public PassWrapper<OptimizeGPUStreamPass, OperationPass<LLVM::LLVMFuncOp>> {

public:
  StringRef getArgument() const final { 
    return "optimize-gpu-stream"; 
  }
  
  StringRef getDescription() const final {
    return "Optimize GPU stream usage by merging multiple streams into one";
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Optimize GPU Stream Pass ===\n");
    
    // 收集所有stream信息
    std::vector<StreamInfo> streamInfos;
    if (failed(collectStreamInfo(funcOp, streamInfos))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect stream info\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << streamInfos.size() 
               << " stream sequences\n");
    
    // 如果只有0或1个stream，无需优化
    if (streamInfos.size() <= 1) {
      LLVM_DEBUG(llvm::dbgs() << "Not enough streams to optimize\n");
      return;
    }
    
    // 执行优化
    if (failed(optimizeStreams(funcOp, streamInfos))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to optimize streams\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully optimized stream usage\n\n");
  }

private:
  // 检查是否是包装函数
  bool isWrapperFunction(StringRef funcName) {
    return funcName == "mgpuCulibsFullyConnectedForward" ||
           funcName == "mgpuCudnnAveragePoolForward" ||
           funcName == "mgpuCudnnMaxPoolForward" ||
           funcName == "mgpuCudnnConv2dForward";
  }
  
  // 收集stream信息 - 改进版本：按顺序遍历，更准确地关联操作
  LogicalResult collectStreamInfo(LLVM::LLVMFuncOp funcOp,
                                  std::vector<StreamInfo>& streamInfos) {
    
    DenseMap<Value, StreamInfo*> streamMap;
    StreamInfo* currentStream = nullptr;
    
    // 按顺序遍历所有操作
    funcOp.walk([&](Operation* op) {
      // 处理LLVM CallOp
      if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
        auto callee = callOp.getCallee();
        if (!callee) return;
        
        StringRef funcName = *callee;
        
        // Stream创建 - 开始新的stream序列
        if (funcName == "mgpuStreamCreate") {
          if (callOp.getNumResults() > 0) {
            StreamInfo info;
            info.createOp = callOp;
            info.streamValue = callOp.getResult();
            streamInfos.push_back(info);
            currentStream = &streamInfos.back();
            streamMap[info.streamValue] = currentStream;
            
            LLVM_DEBUG(llvm::dbgs() << "  Found stream create #" 
                       << streamInfos.size() << "\n");
          }
        }
        // Handle创建
        else if (funcName == "mgpuCreateHandlesForStream") {
          if (currentStream && callOp.getNumOperands() > 0) {
            Value stream = callOp.getOperand(0);
            if (streamMap.count(stream) && streamMap[stream] == currentStream) {
              currentStream->handleOp = callOp;
              LLVM_DEBUG(llvm::dbgs() << "  Associated handle create\n");
            }
          }
        }
        // Stream同步
        else if (funcName == "mgpuStreamSynchronize") {
          if (currentStream && callOp.getNumOperands() > 0) {
            Value stream = callOp.getOperand(0);
            if (streamMap.count(stream) && streamMap[stream] == currentStream) {
              currentStream->syncOp = callOp;
              LLVM_DEBUG(llvm::dbgs() << "  Associated stream sync\n");
            }
          }
        }
        // Stream销毁
        else if (funcName == "mgpuStreamDestroy") {
          if (currentStream && callOp.getNumOperands() > 0) {
            Value stream = callOp.getOperand(0);
            if (streamMap.count(stream) && streamMap[stream] == currentStream) {
              currentStream->destroyOp = callOp;
              LLVM_DEBUG(llvm::dbgs() << "  Associated stream destroy\n");
              // stream序列结束
              currentStream = nullptr;
            }
          }
        }
        // 包装函数
        else if (isWrapperFunction(funcName)) {
          if (currentStream && callOp.getNumOperands() > 0) {
            // 最后一个参数是stream
            Value stream = callOp.getOperand(callOp.getNumOperands() - 1);
            if (streamMap.count(stream) && streamMap[stream] == currentStream) {
              currentStream->wrapperCalls.push_back(callOp);
              LLVM_DEBUG(llvm::dbgs() << "  Associated wrapper: " 
                         << funcName << "\n");
            }
          }
        }
      }
      // 处理gpu.launch_func
      else if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
        // 方法1: 尝试通过 asyncToken 关联
        auto asyncToken = launchOp.getAsyncToken();
        bool associated = false;
        
        if (asyncToken && currentStream) {
          if (streamMap.count(asyncToken) && streamMap[asyncToken] == currentStream) {
            currentStream->kernelLaunches.push_back(launchOp);
            associated = true;
            LLVM_DEBUG(llvm::dbgs() << "  Associated gpu.launch_func via asyncToken\n");
          }
        }
        
        // 方法2: 如果方法1失败，尝试通过操作数查找
        if (!associated && currentStream) {
          // gpu.launch_func 可能通过某个 operand 引用 stream
          for (auto operand : launchOp->getOperands()) {
            if (streamMap.count(operand) && streamMap[operand] == currentStream) {
              currentStream->kernelLaunches.push_back(launchOp);
              associated = true;
              LLVM_DEBUG(llvm::dbgs() << "  Associated gpu.launch_func via operand\n");
              break;
            }
          }
        }
        
        // 方法3: 如果还是失败，检查这个 launch 是否在当前 stream 的作用域内
        if (!associated && currentStream) {
          // 如果这个 launch 在当前 stream create 之后，且在 destroy 之前，
          // 就认为它属于当前 stream
          currentStream->kernelLaunches.push_back(launchOp);
          LLVM_DEBUG(llvm::dbgs() << "  Associated gpu.launch_func by scope\n");
        }
      }
    });
    
    return success();
  }

  // 优化streams
  LogicalResult optimizeStreams(LLVM::LLVMFuncOp funcOp,
                               std::vector<StreamInfo>& streamInfos) {
    
    if (streamInfos.empty()) return success();
    
    // 使用第一个stream作为统一stream
    StreamInfo& firstInfo = streamInfos[0];
    Value unifiedStream = firstInfo.streamValue;
    
    LLVM_DEBUG(llvm::dbgs() << "Using first stream as unified stream\n");
    
    OpBuilder builder(firstInfo.createOp);
    Location loc = firstInfo.createOp.getLoc();
    
    // 检查是否需要handle
    bool needsHandle = false;
    for (const auto& info : streamInfos) {
      if (info.handleOp || !info.wrapperCalls.empty()) {
        needsHandle = true;
        break;
      }
    }
    
    // 如果需要且第一个没有，创建handle
    if (needsHandle && !firstInfo.handleOp) {
      builder.setInsertionPointAfter(firstInfo.createOp);
      builder.create<LLVM::CallOp>(
          loc, TypeRange{}, "mgpuCreateHandlesForStream", 
          ValueRange{unifiedStream});
      LLVM_DEBUG(llvm::dbgs() << "Created unified handle\n");
    }
    
    // 第一步：收集并更新所有包装函数调用
    SmallVector<LLVM::CallOp> allWrapperCalls;
    for (const auto& info : streamInfos) {
      allWrapperCalls.append(info.wrapperCalls.begin(), info.wrapperCalls.end());
    }
    
    // 按照在代码中的位置排序
    std::sort(allWrapperCalls.begin(), allWrapperCalls.end(),
              [](LLVM::CallOp a, LLVM::CallOp b) {
                return a->isBeforeInBlock(b.getOperation());
              });
    
    LLVM_DEBUG(llvm::dbgs() << "Updating " << allWrapperCalls.size() 
               << " wrapper calls\n");
    
    // 更新所有包装函数使用统一的stream
    for (auto callOp : allWrapperCalls) {
      builder.setInsertionPoint(callOp);
      
      auto callee = callOp.getCallee();
      if (!callee) continue;
      
      // 构建新的操作数列表（替换最后一个stream参数）
      SmallVector<Value> newOperands;
      for (size_t i = 0; i < callOp.getNumOperands() - 1; ++i) {
        newOperands.push_back(callOp.getOperand(i));
      }
      newOperands.push_back(unifiedStream);
      
      // 创建新的调用
      auto newCallOp = builder.create<LLVM::CallOp>(
          callOp.getLoc(),
          callOp.getResultTypes(),
          *callee,
          newOperands);
      
      // 替换使用
      callOp.replaceAllUsesWith(newCallOp);
      callOp->erase();
      
      LLVM_DEBUG(llvm::dbgs() << "  Updated wrapper: " << *callee << "\n");
    }
    
    // 第二步：收集并更新所有kernel launch
    SmallVector<gpu::LaunchFuncOp> allKernelLaunches;
    for (const auto& info : streamInfos) {
      allKernelLaunches.append(info.kernelLaunches.begin(), 
                               info.kernelLaunches.end());
    }
    
    // 按照在代码中的位置排序
    std::sort(allKernelLaunches.begin(), allKernelLaunches.end(),
              [](gpu::LaunchFuncOp a, gpu::LaunchFuncOp b) {
                return a->isBeforeInBlock(b.getOperation());
              });
    
    LLVM_DEBUG(llvm::dbgs() << "Updating " << allKernelLaunches.size() 
               << " kernel launches\n");
    
    // 更新所有kernel launch的asyncToken
    for (auto launchOp : allKernelLaunches) {
      bool updated = false;
      
      // 方法1: 尝试通过 asyncToken 更新
      auto asyncToken = launchOp.getAsyncToken();
      if (asyncToken && asyncToken != unifiedStream) {
        // 找到asyncToken在operands中的位置并替换
        for (unsigned i = 0; i < launchOp->getNumOperands(); ++i) {
          if (launchOp->getOperand(i) == asyncToken) {
            launchOp->setOperand(i, unifiedStream);
            updated = true;
            LLVM_DEBUG(llvm::dbgs() << "  Updated gpu.launch_func asyncToken via method 1\n");
            break;
          }
        }
      }
      
      // 方法2: 如果方法1失败，遍历所有operands，替换任何匹配的stream值
      if (!updated) {
        for (unsigned i = 0; i < launchOp->getNumOperands(); ++i) {
          Value operand = launchOp->getOperand(i);
          // 检查这个operand是否是某个stream值
          for (size_t j = 1; j < streamInfos.size(); ++j) {
            if (operand == streamInfos[j].streamValue) {
              launchOp->setOperand(i, unifiedStream);
              updated = true;
              LLVM_DEBUG(llvm::dbgs() << "  Updated gpu.launch_func operand #" << i 
                         << " via method 2\n");
              break;
            }
          }
          if (updated) break;
        }
      }
      
      if (!updated) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not update gpu.launch_func, "
                   << "may already use unified stream or has no stream reference\n");
      }
    }
    
    // 删除其他stream的create/handle/sync/destroy操作
    // 关键：必须先替换所有使用，再删除操作
    // 并且要先删除使用这些值的操作，最后删除创建这些值的操作
    for (size_t i = 1; i < streamInfos.size(); ++i) {
      auto& info = streamInfos[i];
      
      LLVM_DEBUG({
        if (info.streamValue && !info.streamValue.use_empty()) {
          llvm::dbgs() << "  Stream #" << i << " still has " 
                       << std::distance(info.streamValue.use_begin(), 
                                       info.streamValue.use_end()) 
                       << " uses before cleanup\n";
        }
      });
      
      // 步骤1: 先删除使用stream值的操作（handle, sync, destroy）
      if (info.handleOp) {
        info.handleOp->erase();
        LLVM_DEBUG(llvm::dbgs() << "  Removed handle create #" << i << "\n");
      }
      
      if (info.syncOp) {
        info.syncOp->erase();
        LLVM_DEBUG(llvm::dbgs() << "  Removed stream sync #" << i << "\n");
      }
      
      if (info.destroyOp) {
        info.destroyOp->erase();
        LLVM_DEBUG(llvm::dbgs() << "  Removed stream destroy #" << i << "\n");
      }
      
      // 步骤2: 如果stream值还有使用，用统一stream替换
      if (info.streamValue && !info.streamValue.use_empty()) {
        LLVM_DEBUG({
          llvm::dbgs() << "  Stream #" << i << " still has uses after deleting handle/sync/destroy, "
                       << "replacing with unified stream\n";
        });
        
        // 确保类型匹配
        if (info.streamValue.getType() == unifiedStream.getType()) {
          info.streamValue.replaceAllUsesWith(unifiedStream);
          LLVM_DEBUG(llvm::dbgs() << "  Successfully replaced all remaining uses\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  ERROR: Type mismatch, cannot replace uses\n");
          // 类型不匹配，不能继续
          signalPassFailure();
          return failure();
        }
      }
      
      // 步骤3: 最后删除创建stream值的操作
      if (info.createOp) {
        info.createOp->erase();
        LLVM_DEBUG(llvm::dbgs() << "  Removed stream create #" << i << "\n");
      }
    }
    
    // 移动第一个stream的sync和destroy到最后
    Operation* lastOp = nullptr;
    if (!allKernelLaunches.empty()) {
      lastOp = allKernelLaunches.back().getOperation();
    } else if (!allWrapperCalls.empty() && allWrapperCalls.back().getOperation()->getBlock()) {
      // 需要检查操作是否还在block中（因为可能被删除了）
      for (auto it = allWrapperCalls.rbegin(); it != allWrapperCalls.rend(); ++it) {
        if ((*it).getOperation()->getBlock()) {
          lastOp = (*it).getOperation();
          break;
        }
      }
    }
    
    if (lastOp) {
      if (firstInfo.syncOp && firstInfo.syncOp->getBlock()) {
        firstInfo.syncOp->moveAfter(lastOp);
        LLVM_DEBUG(llvm::dbgs() << "Moved final sync to end\n");
        lastOp = firstInfo.syncOp;
      }
      
      if (firstInfo.destroyOp && lastOp && firstInfo.destroyOp->getBlock()) {
        firstInfo.destroyOp->moveAfter(lastOp);
        LLVM_DEBUG(llvm::dbgs() << "Moved final destroy to end\n");
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Stream optimization completed\n");
    
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createOptimizeGPUStreamPass() {
  return std::make_unique<OptimizeGPUStreamPass>();
}

} // namespace onnx_mlir

// Pass 注册
static mlir::PassRegistration<OptimizeGPUStreamPass> pass;