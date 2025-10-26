#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringMap.h"
#include <vector>
#include <string>
#include <map>

using namespace mlir;
using namespace mlir::gpu;

#define DEBUG_TYPE "merge-gpu-modules"

namespace {

// 存储kernel函数信息
struct KernelInfo {
  gpu::GPUFuncOp kernelFunc;        // kernel函数（会被更新为克隆后的）
  gpu::GPUModuleOp originalModule;  // 原始所属module
  std::string originalModuleName;   // 原始module名称
  std::string originalKernelName;   // 原始kernel名称（克隆前）
  std::string kernelName;           // 新kernel名称（克隆后，可能重命名）
  
  KernelInfo(gpu::GPUFuncOp func, gpu::GPUModuleOp module)
      : kernelFunc(func), originalModule(module) {
    originalModuleName = module.getName().str();
    originalKernelName = func.getName().str();
    kernelName = func.getName().str();
  }
};

class MergeGPUModulesPass
    : public PassWrapper<MergeGPUModulesPass, OperationPass<ModuleOp>> {

public:
  StringRef getArgument() const final { 
    return "merge-gpu-modules"; 
  }
  
  StringRef getDescription() const final {
    return "Merge all GPU modules into a single unified module";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Merge GPU Modules Pass ===\n");
    
    // 第一步：收集所有GPU模块和kernel信息
    SmallVector<gpu::GPUModuleOp> gpuModules;
    SmallVector<KernelInfo> allKernels;
    
    if (failed(collectGPUModules(moduleOp, gpuModules, allKernels))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect GPU modules\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << gpuModules.size() 
               << " GPU modules with " << allKernels.size() 
               << " total kernels\n");
    
    // 如果只有0或1个GPU module，无需合并
    if (gpuModules.size() <= 1) {
      LLVM_DEBUG(llvm::dbgs() << "Not enough GPU modules to merge\n");
      return;
    }
    
    // 第二步：创建统一的GPU module
    gpu::GPUModuleOp unifiedModule = createUnifiedModule(moduleOp);
    if (!unifiedModule) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to create unified module\n");
      signalPassFailure();
      return;
    }
    
    // 第三步：将所有kernel移动到统一module中
    if (failed(moveKernelsToUnifiedModule(unifiedModule, allKernels))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to move kernels\n");
      signalPassFailure();
      return;
    }
    
    // 第四步：更新所有gpu.launch_func的引用
    if (failed(updateLaunchFuncReferences(moduleOp, unifiedModule, allKernels))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to update launch_func references\n");
      signalPassFailure();
      return;
    }
    
    // 第五步：删除原有的空GPU modules
    if (failed(removeEmptyModules(gpuModules))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to remove empty modules\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully merged all GPU modules\n\n");
  }

private:
  // 收集所有GPU模块和kernel信息
  LogicalResult collectGPUModules(ModuleOp moduleOp,
                                  SmallVector<gpu::GPUModuleOp>& gpuModules,
                                  SmallVector<KernelInfo>& allKernels) {
    
    moduleOp.walk([&](gpu::GPUModuleOp gpuModule) {
      gpuModules.push_back(gpuModule);
      
      LLVM_DEBUG(llvm::dbgs() << "  Found GPU module: " 
                 << gpuModule.getName() << "\n");
      
      // 收集该模块中的所有kernel函数
      gpuModule.walk([&](gpu::GPUFuncOp kernelFunc) {
        // 只处理kernel函数（有gpu.kernel属性）
        if (kernelFunc->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
          allKernels.emplace_back(kernelFunc, gpuModule);
          LLVM_DEBUG(llvm::dbgs() << "    Found kernel: " 
                     << kernelFunc.getName() << "\n");
        }
      });
    });
    
    return success();
  }
  
  // 创建统一的GPU module
  gpu::GPUModuleOp createUnifiedModule(ModuleOp moduleOp) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // 创建统一module，命名为 "unified_gpu_module"
    auto unifiedModule = builder.create<gpu::GPUModuleOp>(
        moduleOp.getLoc(), "unified_gpu_module");
    
    LLVM_DEBUG(llvm::dbgs() << "Created unified GPU module: " 
               << unifiedModule.getName() << "\n");
    
    return unifiedModule;
  }
  
  // 将所有kernel移动到统一module中
  LogicalResult moveKernelsToUnifiedModule(gpu::GPUModuleOp unifiedModule,
                                           SmallVector<KernelInfo>& allKernels) {
    
    OpBuilder builder(unifiedModule.getContext());
    Block* unifiedBlock = &unifiedModule.getBodyRegion().front();
    
    // 用于检测名称冲突
    llvm::StringSet<> usedNames;
    
    for (auto& kernelInfo : allKernels) {
      gpu::GPUFuncOp originalKernel = kernelInfo.kernelFunc;
      std::string originalName = kernelInfo.kernelName;
      std::string newName = originalName;
      
      // 处理名称冲突
      if (usedNames.count(newName)) {
        int suffix = 1;
        do {
          newName = originalName + "_" + std::to_string(suffix);
          suffix++;
        } while (usedNames.count(newName));
        
        LLVM_DEBUG(llvm::dbgs() << "  Renaming kernel " << originalName 
                   << " to " << newName << " to avoid conflict\n");
      }
      
      usedNames.insert(newName);
      
      // 克隆kernel函数到统一module中
      builder.setInsertionPointToEnd(unifiedBlock);
      
      auto newKernel = originalKernel.clone();
      newKernel.setName(newName);
      
      builder.insert(newKernel);
      
      // 更新kernelInfo以便后续引用更新
      kernelInfo.kernelFunc = cast<gpu::GPUFuncOp>(newKernel);
      kernelInfo.kernelName = newName;
      
      LLVM_DEBUG(llvm::dbgs() << "  Moved kernel: " << originalName 
                 << " -> " << newName << "\n");
    }
    
    return success();
  }
  
  // 更新所有gpu.launch_func的引用
  LogicalResult updateLaunchFuncReferences(ModuleOp moduleOp,
                                           gpu::GPUModuleOp unifiedModule,
                                           SmallVector<KernelInfo>& allKernels) {
    
    // 构建从原始module::kernel到新kernel的映射
    // 使用 "module_name::kernel_name" 作为key
    llvm::StringMap<std::string> refMap;
    for (auto& kernelInfo : allKernels) {
      // 使用原始的模块名和kernel名构建key
      std::string key = kernelInfo.originalModuleName + "::" + 
                        kernelInfo.originalKernelName;
      refMap[key] = kernelInfo.kernelName;
      LLVM_DEBUG(llvm::dbgs() << "  Mapping: " << key 
                 << " -> " << kernelInfo.kernelName << "\n");
    }
    
    std::string unifiedModuleName = unifiedModule.getName().str();
    
    // 遍历所有launch_func操作并更新引用
    int updateCount = 0;
    WalkResult result = moduleOp.walk([&](gpu::LaunchFuncOp launchOp) {
      // 直接使用字符串"kernel"获取属性
      auto kernelAttr = launchOp->getAttrOfType<SymbolRefAttr>("kernel");
      if (!kernelAttr) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: launch_func without kernel attribute\n");
        return WalkResult::advance();
      }
      
      // 获取module和kernel名称
      StringRef oldModuleName = kernelAttr.getRootReference().getValue();
      StringRef oldKernelName = kernelAttr.getLeafReference().getValue();
      
      std::string key = oldModuleName.str() + "::" + oldKernelName.str();
      auto it = refMap.find(key);
      
      if (it != refMap.end()) {
        std::string newKernelName = it->second;
        
        // 创建新的SymbolRef属性并设置
        auto newKernelAttr = SymbolRefAttr::get(
            launchOp.getContext(), 
            unifiedModuleName,
            {SymbolRefAttr::get(launchOp.getContext(), newKernelName)});
        
        launchOp->setAttr("kernel", newKernelAttr);
        updateCount++;
        
        LLVM_DEBUG(llvm::dbgs() << "  Updated launch_func: " 
                   << oldModuleName << "::" << oldKernelName 
                   << " -> " << unifiedModuleName << "::" << newKernelName << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not find mapping for " 
                   << oldModuleName << "::" << oldKernelName << "\n");
      }
      
      return WalkResult::advance();
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Updated " << updateCount << " launch_func operations\n");
    
    if (result.wasInterrupted()) {
      return failure();
    }
    
    return success();
  }
  
  // 删除原有的空GPU modules
  LogicalResult removeEmptyModules(SmallVector<gpu::GPUModuleOp>& gpuModules) {
    for (auto gpuModule : gpuModules) {
      // 检查module是否为空（只考虑kernel函数）
      bool isEmpty = true;
      gpuModule.walk([&](gpu::GPUFuncOp kernelFunc) {
        if (kernelFunc->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
          isEmpty = false;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      
      if (isEmpty || gpuModule.getName() != "unified_gpu_module") {
        LLVM_DEBUG(llvm::dbgs() << "  Removing module: " 
                   << gpuModule.getName() << "\n");
        gpuModule->erase();
      }
    }
    
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createMergeGPUModulesPass() {
  return std::make_unique<MergeGPUModulesPass>();
}

} // namespace onnx_mlir

// Pass 注册
static mlir::PassRegistration<MergeGPUModulesPass> pass;