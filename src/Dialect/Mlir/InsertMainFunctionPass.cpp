// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "llvm/Support/Debug.h"

// using namespace mlir;

// #define DEBUG_TYPE "insert-main-function"

// namespace {

// struct InsertMainFunctionPass
//     : public PassWrapper<InsertMainFunctionPass, OperationPass<ModuleOp>> {
  
//   StringRef getArgument() const final { return "insert-main-function"; }
//   StringRef getDescription() const final {
//     return "Insert a main function for testing that calls main_graph with GPU allocated inputs";
//   }
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<func::FuncDialect>();
//     registry.insert<memref::MemRefDialect>();
//     registry.insert<gpu::GPUDialect>();
//     registry.insert<arith::ArithDialect>();
//   }
  
//   void runOnOperation() override {
//     ModuleOp moduleOp = getOperation();
//     MLIRContext *context = &getContext();
    
//     LLVM_DEBUG(llvm::dbgs() << "Running InsertMainFunctionPass\n");
    
//     // Find the main_graph function
//     func::FuncOp mainGraphFunc = moduleOp.lookupSymbol<func::FuncOp>("main_graph");
//     if (!mainGraphFunc) {
//       LLVM_DEBUG(llvm::dbgs() << "No main_graph function found, skipping\n");
//       return;
//     }
    
//     // Check if main function already exists
//     if (moduleOp.lookupSymbol<func::FuncOp>("main")) {
//       LLVM_DEBUG(llvm::dbgs() << "main function already exists, skipping\n");
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Found main_graph function, creating main function\n");
    
//     // Create global constants and main function
//     if (failed(createMainFunction(moduleOp, mainGraphFunc))) {
//       signalPassFailure();
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed InsertMainFunctionPass\n");
//   }

// private:
//   LogicalResult createMainFunction(ModuleOp moduleOp, func::FuncOp mainGraphFunc) {
//     OpBuilder builder(moduleOp.getContext());
//     Location loc = moduleOp.getLoc();
    
//     // Get the input types of main_graph
//     auto funcType = mainGraphFunc.getFunctionType();
//     auto inputTypes = funcType.getInputs();
    
//     LLVM_DEBUG(llvm::dbgs() << "main_graph has " << inputTypes.size() << " inputs\n");
    
//     // Create global constants for each input
//     // 修复: 使用 std::string 向量来确保字符串生命周期
//     SmallVector<std::string> globalNameStrings;
//     SmallVector<StringRef> globalNames;
    
//     for (auto [idx, inputType] : llvm::enumerate(inputTypes)) {
//       if (auto memrefType = inputType.dyn_cast<MemRefType>()) {
//         std::string globalName = "__constant_input_" + std::to_string(idx);
        
//         // 先存储到 vector 中确保生命周期
//         globalNameStrings.push_back(globalName);
//         globalNames.push_back(StringRef(globalNameStrings.back()));
        
//         // Create the global constant
//         if (failed(createGlobalConstant(builder, moduleOp, globalNameStrings.back(), memrefType, loc))) {
//           return failure();
//         }
        
//         LLVM_DEBUG(llvm::dbgs() << "Created global constant: " << globalName << "\n");
//       } else {
//         // Skip non-memref inputs for now
//         LLVM_DEBUG(llvm::dbgs() << "Skipping non-memref input type\n");
//         globalNameStrings.push_back("");
//         globalNames.push_back("");
//       }
//     }
    
//     // Create the main function
//     return createMainFunctionBody(builder, moduleOp, mainGraphFunc, globalNames, loc);
//   }
  
//   LogicalResult createGlobalConstant(OpBuilder &builder, ModuleOp moduleOp, 
//                                    const std::string &name, MemRefType memrefType, Location loc) {
//     builder.setInsertionPointToStart(moduleOp.getBody());
    
//     // Create dense attribute filled with 2.0
//     Type elementType = memrefType.getElementType();
    
//     Attribute fillValue;
//     if (elementType.isa<FloatType>()) {
//       fillValue = builder.getFloatAttr(elementType, 2.0);
//     } else if (elementType.isa<IntegerType>()) {
//       fillValue = builder.getIntegerAttr(elementType, 2);
//     } else {
//       return failure();
//     }
    
//     auto denseAttr = DenseElementsAttr::get(
//         RankedTensorType::get(memrefType.getShape(), elementType), fillValue);
    
//     // Create the global operation
//     auto globalOp = builder.create<memref::GlobalOp>(
//         loc,
//         builder.getStringAttr(name),
//         builder.getStringAttr("private"),
//         memrefType,
//         denseAttr,
//         /*constant=*/true,
//         /*alignment=*/nullptr);
    
//     return success();
//   }
  
//   LogicalResult createMainFunctionBody(OpBuilder &builder, ModuleOp moduleOp, 
//                                      func::FuncOp mainGraphFunc, 
//                                      ArrayRef<StringRef> globalNames, Location loc) {
//     // Create main function type (no inputs, no outputs)
//     auto mainFuncType = builder.getFunctionType({}, {});
    
//     // Create main function
//     auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainFuncType);
    
//     // Create function body
//     Block *entryBlock = mainFunc.addEntryBlock();
//     builder.setInsertionPointToStart(entryBlock);
    
//     // Get input types for main_graph
//     auto inputTypes = mainGraphFunc.getFunctionType().getInputs();
//     SmallVector<Value> gpuMemrefs;
    
//     // Process each input
//     for (auto [idx, inputType] : llvm::enumerate(inputTypes)) {
//       if (auto memrefType = inputType.dyn_cast<MemRefType>()) {
//         StringRef globalName = globalNames[idx];
//         if (globalName.empty()) continue;
        
//         // Get global memref
//         auto globalGetOp = builder.create<memref::GetGlobalOp>(
//             loc, memrefType, globalName);
        
//         // Create async token
//         auto tokenType = gpu::AsyncTokenType::get(builder.getContext());
//         auto waitOp = builder.create<gpu::WaitOp>(loc, tokenType, ValueRange{});
        
//         // Allocate GPU memory
//         auto allocOp = builder.create<gpu::AllocOp>(
//             loc, 
//             memrefType,
//             tokenType,
//             /*asyncDependencies=*/ValueRange{waitOp.getAsyncToken()},
//             /*dynamicSizes=*/ValueRange{},
//             /*symbolOperands=*/ValueRange{});
        
//         // Copy to GPU memory
//         auto memcpyOp = builder.create<gpu::MemcpyOp>(
//             loc,
//             tokenType,
//             /*asyncDependencies=*/ValueRange{allocOp.getAsyncToken()},
//             /*dst=*/allocOp.getMemref(),
//             /*src=*/globalGetOp.getResult());
        
//         // Wait for copy to complete
//         builder.create<gpu::WaitOp>(loc, TypeRange{}, 
//                                    ValueRange{memcpyOp.getAsyncToken()});
        
//         gpuMemrefs.push_back(allocOp.getMemref());
        
//         LLVM_DEBUG(llvm::dbgs() << "Added GPU allocation and copy for input " << idx << "\n");
//       }
//     }
    
//     // Call main_graph with GPU allocated memrefs
//     auto callOp = builder.create<func::CallOp>(
//         loc,
//         mainGraphFunc.getFunctionType().getResults(),
//         "main_graph",
//         gpuMemrefs);
    
//     LLVM_DEBUG(llvm::dbgs() << "Added call to main_graph\n");
    
//     // Add return
//     builder.create<func::ReturnOp>(loc);
    
//     // Move main function to the end of the module
//     mainFunc->moveBefore(&moduleOp.getBody()->back());
    
//     return success();
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
//     std::unique_ptr<Pass> createInsertMainFunctionPass() {
//       return std::make_unique<InsertMainFunctionPass>();
//     }
// } // namespace onnx_mlir

// static mlir::PassRegistration<InsertMainFunctionPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "insert-main-function"

namespace {

struct InsertMainFunctionPass
    : public PassWrapper<InsertMainFunctionPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "insert-main-function"; }
  StringRef getDescription() const final {
    return "Insert a main function for testing that calls main_graph with GPU allocated inputs";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM_DEBUG(llvm::dbgs() << "Running InsertMainFunctionPass\n");
    
    // Find the main_graph function
    func::FuncOp mainGraphFunc = moduleOp.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainGraphFunc) {
      LLVM_DEBUG(llvm::dbgs() << "No main_graph function found, skipping\n");
      return;
    }
    
    // Check if main function already exists
    if (moduleOp.lookupSymbol<func::FuncOp>("main")) {
      LLVM_DEBUG(llvm::dbgs() << "main function already exists, skipping\n");
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found main_graph function, creating main function\n");
    
    // Create global constants and main function
    if (failed(createMainFunction(moduleOp, mainGraphFunc))) {
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed InsertMainFunctionPass\n");
  }

private:
  LogicalResult createMainFunction(ModuleOp moduleOp, func::FuncOp mainGraphFunc) {
    OpBuilder builder(moduleOp.getContext());
    Location loc = moduleOp.getLoc();
    
    // Get the input types of main_graph
    auto funcType = mainGraphFunc.getFunctionType();
    auto inputTypes = funcType.getInputs();
    
    LLVM_DEBUG(llvm::dbgs() << "main_graph has " << inputTypes.size() << " inputs\n");
    
    // 判断是否有输入参数
    if (inputTypes.empty()) {
      // 无输入参数，直接创建main函数
      return createMainFunctionBody(builder, moduleOp, mainGraphFunc, {}, loc);
    }
    
    // 有输入参数，创建全局常量
    SmallVector<std::string> globalNameStrings;
    SmallVector<StringRef> globalNames;
    
    for (auto [idx, inputType] : llvm::enumerate(inputTypes)) {
      if (auto memrefType = inputType.dyn_cast<MemRefType>()) {
        std::string globalName = "__constant_input_" + std::to_string(idx);
        
        // 先存储到 vector 中确保生命周期
        globalNameStrings.push_back(globalName);
        globalNames.push_back(StringRef(globalNameStrings.back()));
        
        // Create the global constant
        if (failed(createGlobalConstant(builder, moduleOp, globalNameStrings.back(), memrefType, loc))) {
          return failure();
        }
        
        LLVM_DEBUG(llvm::dbgs() << "Created global constant: " << globalName << "\n");
      } else {
        // Skip non-memref inputs for now
        LLVM_DEBUG(llvm::dbgs() << "Skipping non-memref input type\n");
        globalNameStrings.push_back("");
        globalNames.push_back("");
      }
    }
    
    // Create the main function
    return createMainFunctionBody(builder, moduleOp, mainGraphFunc, globalNames, loc);
  }
  
  LogicalResult createGlobalConstant(OpBuilder &builder, ModuleOp moduleOp, 
                                   const std::string &name, MemRefType memrefType, Location loc) {
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // Create dense attribute filled with 2.0
    Type elementType = memrefType.getElementType();
    
    Attribute fillValue;
    if (elementType.isa<FloatType>()) {
      fillValue = builder.getFloatAttr(elementType, 2.0);
    } else if (elementType.isa<IntegerType>()) {
      fillValue = builder.getIntegerAttr(elementType, 2);
    } else {
      return failure();
    }
    
    auto denseAttr = DenseElementsAttr::get(
        RankedTensorType::get(memrefType.getShape(), elementType), fillValue);
    
    // Create the global operation
    auto globalOp = builder.create<memref::GlobalOp>(
        loc,
        builder.getStringAttr(name),
        builder.getStringAttr("private"),
        memrefType,
        denseAttr,
        /*constant=*/true,
        /*alignment=*/nullptr);
    
    return success();
  }
  
  LogicalResult createMainFunctionBody(OpBuilder &builder, ModuleOp moduleOp, 
                                     func::FuncOp mainGraphFunc, 
                                     ArrayRef<StringRef> globalNames, Location loc) {
    // Create main function type (no inputs, no outputs)
    auto mainFuncType = builder.getFunctionType({}, {});
    
    // Create main function
    auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainFuncType);
    
    // Create function body
    Block *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Get input types for main_graph
    auto inputTypes = mainGraphFunc.getFunctionType().getInputs();
    SmallVector<Value> gpuMemrefs;
    
    // 如果有输入参数，处理GPU内存分配
    if (!inputTypes.empty()) {
      // Process each input
      for (auto [idx, inputType] : llvm::enumerate(inputTypes)) {
        if (auto memrefType = inputType.dyn_cast<MemRefType>()) {
          StringRef globalName = globalNames[idx];
          if (globalName.empty()) continue;
          
          // Get global memref
          auto globalGetOp = builder.create<memref::GetGlobalOp>(
              loc, memrefType, globalName);
          
          // Create async token
          auto tokenType = gpu::AsyncTokenType::get(builder.getContext());
          auto waitOp = builder.create<gpu::WaitOp>(loc, tokenType, ValueRange{});
          
          // Allocate GPU memory
          auto allocOp = builder.create<gpu::AllocOp>(
              loc, 
              memrefType,
              tokenType,
              /*asyncDependencies=*/ValueRange{waitOp.getAsyncToken()},
              /*dynamicSizes=*/ValueRange{},
              /*symbolOperands=*/ValueRange{});
          
          // Copy to GPU memory
          auto memcpyOp = builder.create<gpu::MemcpyOp>(
              loc,
              tokenType,
              /*asyncDependencies=*/ValueRange{allocOp.getAsyncToken()},
              /*dst=*/allocOp.getMemref(),
              /*src=*/globalGetOp.getResult());
          
          // Wait for copy to complete
          builder.create<gpu::WaitOp>(loc, TypeRange{}, 
                                     ValueRange{memcpyOp.getAsyncToken()});
          
          gpuMemrefs.push_back(allocOp.getMemref());
          
          LLVM_DEBUG(llvm::dbgs() << "Added GPU allocation and copy for input " << idx << "\n");
        }
      }
    }
    
    // 创建scf.for循环的常量
    auto indexType = builder.getIndexType();
    auto c0 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    auto c2 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(2));
    auto c1 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
    
    // 创建scf.for循环
    auto forOp = builder.create<scf::ForOp>(loc, c0, c2, c1);
    
    // 在循环体内调用main_graph
    builder.setInsertionPointToStart(forOp.getBody());
    
    // Call main_graph with GPU allocated memrefs (or no args if empty)
    auto callOp = builder.create<func::CallOp>(
        loc,
        mainGraphFunc.getFunctionType().getResults(),
        "main_graph",
        gpuMemrefs);
    
    LLVM_DEBUG(llvm::dbgs() << "Added scf.for loop with call to main_graph\n");
    
    // 回到main函数的结尾添加return
    builder.setInsertionPointAfter(forOp);
    builder.create<func::ReturnOp>(loc);
    
    // Move main function to the end of the module
    mainFunc->moveBefore(&moduleOp.getBody()->back());
    
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createInsertMainFunctionPass() {
      return std::make_unique<InsertMainFunctionPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<InsertMainFunctionPass> pass;