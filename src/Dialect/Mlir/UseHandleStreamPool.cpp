#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "cuda-pool-conversion"

namespace {

// Pattern for mgpuStreamCreate -> mgpuAcquirePooledStream
class StreamCreateToPooledPattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, 
                               PatternRewriter &rewriter) const override {
    if (!callOp.getCallee() || callOp.getCallee().value() != "mgpuStreamCreate")
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Converting mgpuStreamCreate to mgpuAcquirePooledStream\n");
    
    // Create new call to mgpuAcquirePooledStream
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(),
        callOp.getResultTypes(),
        "mgpuAcquirePooledStream",
        ValueRange{} // No arguments
    );
    
    rewriter.replaceOp(callOp, newCallOp.getResults());
    return success();
  }
};

// Pattern for mgpuStreamDestroy -> mgpuReleasePooledStream
class StreamDestroyToPooledPattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, 
                               PatternRewriter &rewriter) const override {
    if (!callOp.getCallee() || callOp.getCallee().value() != "mgpuStreamDestroy")
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Converting mgpuStreamDestroy to mgpuReleasePooledStream\n");
    
    // Create new call to mgpuReleasePooledStream
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(),
        callOp.getResultTypes(),
        "mgpuReleasePooledStream",
        callOp.getOperands() // Same arguments (stream pointer)
    );
    
    rewriter.replaceOp(callOp, newCallOp.getResults());
    return success();
  }
};

// Pattern for mgpuCreateHandlesForStream -> mgpuAcquirePooledHandles
class CreateHandlesToPooledPattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, 
                               PatternRewriter &rewriter) const override {
    if (!callOp.getCallee() || callOp.getCallee().value() != "mgpuCreateHandlesForStream")
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Converting mgpuCreateHandlesForStream to mgpuAcquirePooledHandles\n");
    
    // Create new call to mgpuAcquirePooledHandles
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(),
        callOp.getResultTypes(),
        "mgpuAcquirePooledHandles",
        callOp.getOperands() // Same arguments (stream pointer)
    );
    
    rewriter.replaceOp(callOp, newCallOp.getResults());
    return success();
  }
};

// Pattern to remove mgpuDestroyHandlePool calls (we'll add them at the end)
class RemoveDestroyHandlePoolPattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, 
                               PatternRewriter &rewriter) const override {
    if (!callOp.getCallee() || callOp.getCallee().value() != "mgpuDestroyHandlePool")
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Removing existing mgpuDestroyHandlePool call\n");
    
    // Simply remove the call - we'll add proper cleanup later
    rewriter.eraseOp(callOp);
    return success();
  }
};

struct CudaPoolConversionPass
    : public PassWrapper<CudaPoolConversionPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "use-cuda-pools"; }
  StringRef getDescription() const final {
    return "Convert CUDA stream/handle functions to use pools";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM_DEBUG(llvm::dbgs() << "Running CudaPoolConversionPass\n");
    
    // First, add function declarations if they don't exist
    if (failed(addFunctionDeclarations(moduleOp))) {
      signalPassFailure();
      return;
    }
    
    // Then, apply the patterns to convert function calls
    RewritePatternSet patterns(context);
    patterns.add<StreamCreateToPooledPattern>(context);
    patterns.add<StreamDestroyToPooledPattern>(context);
    patterns.add<CreateHandlesToPooledPattern>(context);
    // patterns.add<RemoveDestroyHandlePoolPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    
    // Finally, add initialization and cleanup calls
    if (failed(addPoolManagement(moduleOp))) {
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed CudaPoolConversionPass\n");
  }

private:
  LogicalResult addFunctionDeclarations(ModuleOp moduleOp) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // Get common types
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto i32Type = IntegerType::get(builder.getContext(), 32);
    
    // List of function declarations to add
    struct FuncDecl {
      StringRef name;
      Type resultType;
      SmallVector<Type> argTypes;
    };
    
    SmallVector<FuncDecl> functionDecls = {
      // Stream pool functions
      {"mgpuInitStreamPool", voidType, {i32Type}},
      {"mgpuDestroyStreamPool", voidType, {}},
      {"mgpuAcquirePooledStream", ptrType, {}},
      {"mgpuReleasePooledStream", voidType, {ptrType}},
      
      // Handle pool functions  
      {"mgpuInitHandlePool", voidType, {i32Type}},
      {"mgpuDestroyHandlePool", voidType, {}},
      {"mgpuAcquirePooledHandles", voidType, {ptrType}},

      // Descriptor pool functions
      {"mgpuInitDescriptorPool", voidType, {i32Type, i32Type, i32Type, i32Type, i32Type}},
      {"mgpuDestroyDescriptorPool", voidType, {}},

      {"mgpuInitWorkspacePool", voidType, {i32Type, i32Type}},
      {"mgpuDestroyWorkspacePool", voidType, {}} 
    };
    
    LLVM_DEBUG(llvm::dbgs() << "Adding function declarations\n");
    
    for (const auto& decl : functionDecls) {
      // Check if function already exists
      if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(decl.name)) {
        LLVM_DEBUG(llvm::dbgs() << "Function " << decl.name << " already exists, skipping\n");
        continue;
      }
      
      // Create function type
      auto funcType = LLVM::LLVMFunctionType::get(decl.resultType, decl.argTypes);
      
      // Create function declaration
      auto funcOp = builder.create<LLVM::LLVMFuncOp>(
          moduleOp.getLoc(),
          decl.name,
          funcType);
      
      // Set private visibility
      funcOp.setSymVisibilityAttr(builder.getStringAttr("private"));
      
      LLVM_DEBUG(llvm::dbgs() << "Added function declaration: " << decl.name << "\n");
    }
    
    return success();
  }
  
  // LogicalResult addPoolManagement(ModuleOp moduleOp) {
  //   // Find all functions in the module
  //   auto functions = moduleOp.getOps<LLVM::LLVMFuncOp>();
  //   if (functions.empty()) {
  //     LLVM_DEBUG(llvm::dbgs() << "No functions found in module\n");
  //     return success();
  //   }
    
  //   // Find the main function or a function that contains CUDA operations
  //   LLVM::LLVMFuncOp targetFunc = nullptr;
  //   for (auto func : functions) {
  //     // Skip the function declarations we just added
  //     if (func.isExternal()) {
  //       continue;
  //     }
      
  //     // Look for a function that contains CUDA calls
  //     bool hasCudaCalls = false;
  //     func.walk([&](LLVM::CallOp callOp) {
  //       if (callOp.getCallee() && 
  //           (callOp.getCallee().value().contains("mgpu") || 
  //            callOp.getCallee().value().contains("cuda"))) {
  //         hasCudaCalls = true;
  //       }
  //     });
      
  //     if (hasCudaCalls) {
  //       targetFunc = func;
  //       break;
  //     }
  //   }
    
  //   if (!targetFunc) {
  //     LLVM_DEBUG(llvm::dbgs() << "No function with CUDA calls found\n");
  //     return success();
  //   }
    
  //   LLVM_DEBUG(llvm::dbgs() << "Adding pool management to function: " 
  //              << targetFunc.getName() << "\n");
    
  //   // Add initialization at the beginning
  //   addPoolInitialization(targetFunc);
    
  //   // Add cleanup before all return statements
  //   addPoolCleanup(targetFunc);
    
  //   return success();
  // }
  
LogicalResult addPoolManagement(ModuleOp moduleOp) {
  // 直接查找名为"main"的函数
  LLVM::LLVMFuncOp targetFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("main");
  
  if (!targetFunc) {
    LLVM_DEBUG(llvm::dbgs() << "Main function not found in module\n");
    // 可以选择以下策略之一：
    
    // 策略1: 直接返回失败
    // return failure();
    
    // 策略2: 发出警告并使用原来的逻辑作为fallback
    LLVM_DEBUG(llvm::dbgs() << "Falling back to finding function with CUDA calls\n");
    
    auto functions = moduleOp.getOps<LLVM::LLVMFuncOp>();
    if (functions.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No functions found in module\n");
      return success();
    }
    
    // 原来的fallback逻辑：查找包含CUDA调用的函数
    for (auto func : functions) {
      // Skip the function declarations we just added
      if (func.isExternal()) {
        continue;
      }
      
      // Look for a function that contains CUDA calls
      bool hasCudaCalls = false;
      func.walk([&](LLVM::CallOp callOp) {
        if (callOp.getCallee() && 
            (callOp.getCallee().value().contains("mgpu") || 
             callOp.getCallee().value().contains("cuda"))) {
          hasCudaCalls = true;
        }
      });
      
      if (hasCudaCalls) {
        targetFunc = func;
        break;
      }
    }
    
    if (!targetFunc) {
      LLVM_DEBUG(llvm::dbgs() << "No function with CUDA calls found\n");
      return success();
    }
  }
  
  // 检查目标函数是否为函数声明（external）
  if (targetFunc.isExternal()) {
    LLVM_DEBUG(llvm::dbgs() << "Target function is external, cannot modify\n");
    return failure();
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Adding pool management to function: " 
             << targetFunc.getName() << "\n");
  
  // Add initialization at the beginning
  addPoolInitialization(targetFunc);
  
  // Add cleanup before all return statements
  addPoolCleanup(targetFunc);
  
  return success();
}

  void addPoolInitialization(LLVM::LLVMFuncOp func) {
    OpBuilder builder(func.getContext());
    Block &entryBlock = func.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Create constant for pool size (35)
    auto i32Type = IntegerType::get(builder.getContext(), 32);
    auto poolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(110));
    
    // Descriptor pool sizes (75, 20, 20, 20, 20)
    auto tensorPoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(100));
    auto filterPoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(50));
    auto convPoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(50));
    auto poolingPoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(50));
    auto opTensorPoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(50));

    auto workspacePoolSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(50));
    auto workspaceSize = builder.create<LLVM::ConstantOp>(
        func.getLoc(), i32Type, builder.getI32IntegerAttr(128));
    LLVM_DEBUG(llvm::dbgs() << "Adding pool initialization calls\n");
    
    // Add mgpuInitStreamPool(35)
    builder.create<LLVM::CallOp>(
        func.getLoc(),
        TypeRange{},
        "mgpuInitStreamPool",
        ValueRange{poolSize});
    
    // Add mgpuInitHandlePool(35)
    builder.create<LLVM::CallOp>(
        func.getLoc(),
        TypeRange{},
        "mgpuInitHandlePool",
        ValueRange{poolSize});

    // Add mgpuInitDescriptorPool(75, 20, 20, 20, 20)
    builder.create<LLVM::CallOp>(
        func.getLoc(),
        TypeRange{},
        "mgpuInitDescriptorPool",
        ValueRange{tensorPoolSize, filterPoolSize, convPoolSize, 
                   poolingPoolSize, opTensorPoolSize});

    // Add mgpuInitDescriptorPool
    builder.create<LLVM::CallOp>(
        func.getLoc(),
        TypeRange{},
        "mgpuInitWorkspacePool",
        ValueRange{workspacePoolSize, workspaceSize});
  }
  
  void addPoolCleanup(LLVM::LLVMFuncOp func) {
    OpBuilder builder(func.getContext());
    
    // Find all return operations and add cleanup before them
    SmallVector<LLVM::ReturnOp> returnOps;
    func.walk([&](LLVM::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Adding cleanup calls before " 
               << returnOps.size() << " return statements\n");
    
    for (auto returnOp : returnOps) {
      builder.setInsertionPoint(returnOp);
      
      // Add mgpuDestroyStreamPool()
      builder.create<LLVM::CallOp>(
          returnOp.getLoc(),
          TypeRange{},
          "mgpuDestroyStreamPool",
          ValueRange{});
      
      // Add mgpuDestroyHandlePool()
      builder.create<LLVM::CallOp>(
          returnOp.getLoc(),
          TypeRange{},
          "mgpuDestroyHandlePool",
          ValueRange{});

      // Add mgpuDestroyDescriptorPool()
      builder.create<LLVM::CallOp>(
          returnOp.getLoc(),
          TypeRange{},
          "mgpuDestroyDescriptorPool",
          ValueRange{});

      // Add mgpuDestroyDescriptorPool()
      builder.create<LLVM::CallOp>(
          returnOp.getLoc(),
          TypeRange{},
          "mgpuDestroyWorkspacePool",
          ValueRange{});
    }
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createCudaPoolConversionPass() {
      return std::make_unique<CudaPoolConversionPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<CudaPoolConversionPass> pass;