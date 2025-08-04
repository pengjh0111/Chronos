// // #include "mlir/IR/Builders.h"
// // #include "mlir/IR/Diagnostics.h"
// // #include "mlir/IR/Operation.h"
// // #include "mlir/IR/PatternMatch.h"
// // #include "mlir/IR/OpDefinition.h"
// // #include "mlir/Dialect/Func/IR/FuncOps.h"
// // #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// // #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // #include "mlir/Pass/Pass.h"
// // #include "mlir/Dialect/Arith/IR/Arith.h"

// // #include "llvm/Support/ErrorHandling.h"

// // using namespace mlir;

// // namespace {
  
// // struct InsertGPUAllocPass 
// //     : public PassWrapper<InsertGPUAllocPass, OperationPass<ModuleOp>> {
// // public:
// //     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocPass)
// //   void runOnOperation() override;
// //   StringRef getArgument() const final { return "insert-gpu-alloc"; }
// //   StringRef getDescription() const final { 
// //     return "Convert host-side memref.alloc, memref.reinterpret_cast, krnl.global, memref.get_global, and gpu.alloc host_shared "
// //            "to async gpu.alloc, and insert necessary async gpu.memcpy operations. "
// //            "Adds explicit synchronization to decouple memory operations from compute operations. "
// //            "Now includes scalar parameters in GPU memory conversion and scalar constant optimization."; 
// //   }
// //   private:
// //   // Member function declaration
// //   Value processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers);
// //   Value processScalarConstant(OpBuilder &builder, Location loc, Value operand);
  
// //   // 新增：追踪通过reinterpret_cast链找到最终源操作
// //   std::pair<Value, Operation*> traceSourceThroughReinterpretCast(Value memref) {
// //     Value currentMemref = memref;
// //     Operation* currentDefOp = currentMemref.getDefiningOp();
    
// //     // 追踪reinterpret_cast链直到找到非reinterpret_cast的源操作
// //     while (currentDefOp && isa<memref::ReinterpretCastOp>(currentDefOp)) {
// //       auto reinterpretOp = cast<memref::ReinterpretCastOp>(currentDefOp);
// //       currentMemref = reinterpretOp.getSource();
// //       currentDefOp = currentMemref.getDefiningOp();
// //     }
    
// //     return {currentMemref, currentDefOp};
// //   }

// //   // Global replacement map to avoid duplicate GPU allocations
// //   DenseMap<Value, Value> replacementMap;
// // };

// // static Value createGpuAllocAndCopyAsync(OpBuilder &builder, Location loc, Value origVal,
// //                                       ArrayRef<Value> dims, bool doCopy) {
// //   // This pass does not support dynamic cases; if dims is not empty, report an error
// //   if (!dims.empty()) {
// //     llvm::report_fatal_error("InsertGPUAllocPass: Dynamic dimensions not supported");
// //   }

// //   auto origType = origVal.getType().cast<MemRefType>();
  
// //   if (!origType.getLayout().isIdentity()) {
// //     llvm::report_fatal_error("InsertGPUAllocPass: Non-identity (non-contiguous) layouts are not supported");
// //   }

// //   auto allocType = MemRefType::get(
// //       origType.getShape(), 
// //       origType.getElementType(), 
// //       /*use the original layout*/ origType.getLayout(),
// //       origType.getMemorySpace());
  
// //   // Create an initial token
// //   Value initialToken = builder.create<gpu::WaitOp>(
// //       loc, 
// //       builder.getType<gpu::AsyncTokenType>(),
// //       ValueRange{}).getAsyncToken();
  
// //   // Insert the async gpu.alloc operation without host_shared flag
// //   auto gpuAlloc = builder.create<gpu::AllocOp>(
// //       loc, allocType, 
// //       builder.getType<gpu::AsyncTokenType>(), // Return an async token
// //       ValueRange{initialToken}, // Use the initial token
// //       dims, /*symbolOperands=*/std::nullopt,
// //       /*hostShared=*/false); // No host_shared flag
  
// //   Value allocResult = gpuAlloc.getResult(0);
// //   Value allocToken = gpuAlloc.getAsyncToken();

// //   // If a copy is needed, insert async gpu.memcpy operation
// //   if (doCopy) {
// //     auto memcpyOp = builder.create<gpu::MemcpyOp>(
// //         loc, 
// //         builder.getType<gpu::AsyncTokenType>(),
// //         ValueRange{allocToken}, // Chain with allocation token
// //         allocResult, origVal);
    
// //     // Update token to the memcpy token
// //     allocToken = memcpyOp.getAsyncToken();
// //   }
  
// //   // Add explicit synchronization point
// //   builder.create<gpu::WaitOp>(loc, TypeRange{}, ValueRange{allocToken});
  
// //   // If the allocated type does not match the original type, insert a cast operation.
// //   if (allocType != origType) {
// //     allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
// //   }
  
// //   return allocResult;
// // }


// // // 辅助函数：检查是否为标量操作函数
// // static bool isScalarOperation(StringRef funcName) {
// //   return funcName.contains("AddScalar") || 
// //          funcName.contains("SubScalar") || 
// //          funcName.contains("MulScalar") || 
// //          funcName.contains("RSubScalar");
// // }

// // // 辅助函数：检查参数是否为标量参数（保留用于调试和日志记录）
// // static bool isScalarParameter(func::CallOp callOp, Value paramValue) {
// //   if (!isScalarOperation(callOp.getCallee())) {
// //     return false;
// //   }
  
// //   // 对于标量操作函数，第二个参数（index 1）是标量参数
// //   // 函数签名：mgpuCudnnXXXScalar(input_ptr, scalar_ptr, output_ptr, n, c, h, w, stream)
// //   auto operands = callOp.getOperands();
  
// //   // 需要追踪 extract_aligned_pointer -> index_cast -> inttoptr -> call operand 的链
// //   // 找到该参数在call操作中的位置
// //   for (unsigned i = 0; i < operands.size(); ++i) {
// //     Value operand = operands[i];
    
// //     // 追踪指针操作链，看是否最终来源于我们关心的paramValue
// //     Value currentVal = operand;
    
// //     // 反向追踪：inttoptr -> index_cast -> extract_aligned_pointer
// //     if (auto intToPtrOp = currentVal.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
// //       currentVal = intToPtrOp.getArg();
      
// //       if (auto indexCastOp = currentVal.getDefiningOp<mlir::arith::IndexCastOp>()) {
// //         currentVal = indexCastOp->getOperand(0);
        
// //         if (auto extractOp = currentVal.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
// //           Value sourceMemref = extractOp.getSource();
          
// //           // 检查源memref是否与我们的paramValue匹配
// //           if (sourceMemref == paramValue) {
// //             // 找到了参数位置，检查是否为标量参数位置
// //             // 标量参数通常在第二个位置（index 1）
// //             return (i == 1);
// //           }
// //         }
// //       }
// //     }
// //   }
  
// //   return false;
// // }

// // Value InsertGPUAllocPass::processScalarConstant(OpBuilder &builder, Location loc, Value operand) {
// //   // 检查操作数是否为f32或f16标量类型
// //   if (auto floatType = operand.getType().dyn_cast<FloatType>()) {
// //     if (floatType.isF32() || floatType.isF16()) {
// //       // 检查是否来自memref.load
// //       if (auto loadOp = operand.getDefiningOp<memref::LoadOp>()) {
// //         Value memref = loadOp.getMemref();
        
// //         // 检查memref是否来自krnl.global
// //         if (auto defOp = memref.getDefiningOp()) {
// //           if (defOp->getName().getStringRef() == "krnl.global") {
// //             // 尝试提取krnl.global的常量值
// //             if (auto valueAttr = defOp->getAttrOfType<mlir::Attribute>("value")) {
// //               if (auto denseAttr = valueAttr.dyn_cast<DenseElementsAttr>()) {
// //                 // 检查是否为标量常量（shape为空）
// //                 if (denseAttr.getType().getRank() == 0) {
// //                   // 直接使用DenseElementsAttr中的APFloat，它已经具有正确的精度
// //                   APFloat originalValue = denseAttr.getSplatValue<APFloat>();
                  
// //                   // 创建arith.constant替换memref.load
// //                   Value constantOp = builder.create<arith::ConstantFloatOp>(
// //                       loc, originalValue, floatType);
                  
// //                   // 替换所有使用该load操作的地方
// //                   loadOp.getResult().replaceAllUsesWith(constantOp);
// //                   loadOp.erase();
                  
// //                   return constantOp;
// //                 }
// //               }
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
  
// //   // 如果不符合条件，返回原始操作数
// //   return operand;
// // }

// // /// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, "krnl.global",
// // /// memref.get_global, or gpu.alloc host_shared, call createGpuAllocAndCopyAsync as appropriate to obtain a GPU memory variable.
// // Value InsertGPUAllocPass::processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers) {
// //   Operation *defOp = operand.getDefiningOp();

// //   // Check if we already have a replacement for this value
// //   if (replacementMap.count(operand)) {
// //     needUpdateUsers = false;
// //     return replacementMap[operand];
// //   }

// //   // Handle memref.reinterpret_cast specially
// //   if (defOp && isa<memref::ReinterpretCastOp>(defOp)) {
// //     auto reinterpretOp = cast<memref::ReinterpretCastOp>(defOp);
// //     Value sourceOperand = reinterpretOp.getSource();
    
// //     // Recursively process the source operand
// //     bool sourceNeedUpdateUsers = false;
// //     Value newSource = processOperand(builder, loc, sourceOperand, sourceNeedUpdateUsers);
    
// //     // If the source was converted to GPU memory, update the reinterpret_cast operand
// //     if (newSource != sourceOperand) {
// //       // Directly modify the existing reinterpret_cast operation's operand
// //       reinterpretOp.getOperation()->setOperand(0, newSource);
      
// //       // Store in replacement map
// //       replacementMap[operand] = operand; // The result is still the same Value
// //       needUpdateUsers = false;
// //       return operand; // Return the original result, but now it operates on GPU memory
// //     } else {
// //       // Source wasn't converted, return original
// //       needUpdateUsers = false;
// //       return operand;
// //     }
// //   }

// //   llvm::SmallVector<Value, 4> dims;
// //   if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
// //     for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
// //       if (memType.isDynamicDim(i))
// //         dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
// //     }
// //     if (!dims.empty())
// //       llvm::report_fatal_error("InsertGPUAllocsPass: Dynamic dimensions not supported");
// //   }

// //   if (defOp && (isa<memref::AllocOp>(defOp) ||
// //                 isa<memref::GetGlobalOp>(defOp) ||
// //                 defOp->getName().getStringRef() == "krnl.global" ||
// //                 (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
// //     bool doCopy = false;
    
// //     // For krnl.global or get_global, need to copy data
// //     if (defOp->getName().getStringRef() == "krnl.global" || 
// //         isa<memref::GetGlobalOp>(defOp)) {
// //       doCopy = true;
// //     }
    
// //     // For memref.alloc and gpu.alloc host_shared, update insertion point to function start
// //     if (isa<memref::AllocOp>(defOp) || (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
// //       auto funcOp = defOp->getParentOfType<func::FuncOp>();
// //       if (!funcOp || funcOp.isExternal() || funcOp.getBody().empty()) {
// //         // Skip external functions or functions without a body
// //         needUpdateUsers = false;
// //         return operand;
// //       }
// //       builder.setInsertionPointToStart(&funcOp.getBody().front());
// //       needUpdateUsers = true;
// //     } else {
// //       // For krnl.global or get_global, set insertion point after the op
// //       if (defOp->getName().getStringRef() == "krnl.global" || isa<memref::GetGlobalOp>(defOp)) {
// //         builder.setInsertionPointAfter(defOp);
// //       }
// //       needUpdateUsers = false;
// //     }
    
// //     // Create async GPU allocation and copy if needed with explicit synchronization
// //     Value newVal = createGpuAllocAndCopyAsync(builder, loc, operand, dims, doCopy);
    
// //     // Store the replacement in the map
// //     replacementMap[operand] = newVal;
    
// //     return newVal;
// //   }

// //   // In all other cases, return the original value directly.
// //   needUpdateUsers = false;
// //   return operand;
// // }

// // void InsertGPUAllocPass::runOnOperation() {
// //   ModuleOp module = getOperation();
// //   OpBuilder builder(module.getContext());

// //   // Clear the replacement map
// //   replacementMap.clear();

// //   // First pass: Process all functions to handle memory allocations at function start
// //   module.walk([&](func::FuncOp funcOp) {
// //     // Skip external functions (functions without a body)
// //     if (funcOp.isExternal() || funcOp.getBody().empty())
// //       return;
    
// //     builder.setInsertionPointToStart(&funcOp.getBody().front());
    
// //     // Process gpu.launch_func operations
// //     llvm::SmallVector<gpu::LaunchFuncOp, 8> launchOps;
// //     funcOp.walk([&](gpu::LaunchFuncOp op) {
// //       launchOps.push_back(op);
// //     });
    
// //     for (auto launchOp : launchOps) {
// //       auto operands = launchOp.getOperands();
// //       llvm::SmallVector<Value, 4> newOperands;
// //       bool launchChanged = false;

// //       // Process each operand
// //       for (Value operand : operands) {
// //         bool updateUsers = false;

// //         Value processedOperand = processScalarConstant(builder, launchOp.getLoc(), operand);
// //         if (processedOperand != operand) {
// //           // 如果进行了标量常量优化，使用新的常量值
// //           newOperands.push_back(processedOperand);
// //           launchChanged = true;
// //           continue;
// //         }

// //         Operation *defOp = operand.getDefiningOp();
// //         if (defOp && (isa<memref::AllocOp>(defOp) ||
// //                     isa<memref::ReinterpretCastOp>(defOp) ||
// //                     isa<memref::GetGlobalOp>(defOp) ||
// //                     defOp->getName().getStringRef() == "krnl.global" ||
// //                     (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
          
// //           // Process the operand, getting new GPU memory value
// //           Value newVal = processOperand(builder, launchOp.getLoc(), operand, updateUsers);
          
// //           newOperands.push_back(newVal);
// //           launchChanged = true;
          
// //           // If updating users is needed (for memref.alloc or gpu.alloc host_shared)
// //           if (updateUsers) {
// //             // Replace all users of the original allocation
// //             SmallVector<OpOperand*, 8> operandUses;
// //             for (OpOperand &use : operand.getUses())
// //               operandUses.push_back(&use);
// //             for (OpOperand *use : operandUses) {
// //               use->set(newVal);
// //             }
// //             // Erase the original allocation op
// //             defOp->erase();
// //           }
// //         } else {
// //           newOperands.push_back(operand);
// //         }
// //       }
      
// //       // Only update the operands if they changed
// //       if (launchChanged) {
// //         launchOp.getOperation()->setOperands(newOperands);
// //       }
// //     }
// //   });

// //   // Second pass: Handle memref.extract_aligned_pointer_as_index operations for CUDA library calls
// //   module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
// //     Value memref = extractOp.getSource();
    
// //     // 使用新的追踪函数找到最终源操作
// //     auto [ultimateMemref, ultimateDefOp] = traceSourceThroughReinterpretCast(memref);
    
// //     // Check if the ultimate source comes from krnl.global, memref.alloc, memref.get_global or gpu.alloc with host_shared
// //     bool isTargetOp = false;
    
// //     if (ultimateDefOp) {
// //       if (ultimateDefOp->getName().getStringRef() == "krnl.global" || 
// //           isa<memref::AllocOp>(ultimateDefOp) ||
// //           isa<memref::GetGlobalOp>(ultimateDefOp) ||
// //           (isa<gpu::AllocOp>(ultimateDefOp) && cast<gpu::AllocOp>(ultimateDefOp).getHostShared())) {
// //         isTargetOp = true;
// //       }
// //     }
    
// //     if (isTargetOp) {
// //       // Check if we've already created a replacement for the ultimate source
// //       if (replacementMap.count(ultimateMemref)) {
// //         // 如果原始memref经过了reinterpret_cast，需要使用processOperand来正确处理
// //         if (memref != ultimateMemref) {
// //           bool needUpdateUsers = false;
// //           Value processedMemref = processOperand(builder, extractOp.getLoc(), memref, needUpdateUsers);
// //           extractOp.setOperand(processedMemref);
// //         } else {
// //           extractOp.setOperand(replacementMap[ultimateMemref]);
// //         }
// //         return;
// //       }
      
// //       // Check if the result flows into a CUDA library call
// //       bool flowsToGpuLibCall = false;
// //       bool isScalarParam = false;  // 保留用于调试和日志记录
      
// //       for (Operation *user : extractOp->getResult(0).getUsers()) {
// //         // Trace through inttoptr and index_cast operations
// //         Operation* currentUser = user;
// //         while (currentUser && (isa<arith::IndexCastOp>(currentUser) || currentUser->getName().getStringRef() == "llvm.inttoptr")) {
// //           if (currentUser->getNumResults() > 0 && !currentUser->getResult(0).getUses().empty()) {
// //             currentUser = *currentUser->getResult(0).getUsers().begin();
// //           } else {
// //             break;
// //           }
// //         }
        
// //         // Check if we reached a call operation to a CUDA library function
// //         if (currentUser && isa<func::CallOp>(currentUser)) {
// //           auto callOp = cast<func::CallOp>(currentUser);
// //           if (callOp.getCallee().starts_with("mgpuCudnn") || 
// //               callOp.getCallee().starts_with("mgpuCulibs") ||
// //               callOp.getCallee().starts_with("mgpu")) {
// //             flowsToGpuLibCall = true;
            
// //             // 检查是否为标量参数（用于调试和日志记录）
// //             isScalarParam = isScalarParameter(callOp, ultimateMemref);
            
// //             break;
// //           }
// //         }
// //       }
      
// //       // 现在所有流向GPU库调用的参数都进行GPU内存转换
// //       if (flowsToGpuLibCall) {
// //         // For gpu.alloc host_shared, replace directly
// //         if (auto gpuAllocOp = dyn_cast_or_null<gpu::AllocOp>(ultimateDefOp)) {
// //           if (gpuAllocOp.getHostShared()) {
// //             auto funcOp = gpuAllocOp->getParentOfType<func::FuncOp>();
// //             if (!funcOp) {
// //               llvm::report_fatal_error("gpu.alloc op is not within a function");
// //             }
            
// //             // Move replacement to the start of the function
// //             builder.setInsertionPointToStart(&funcOp.getBody().front());
            
// //             // Create new async GPU allocation without host_shared
// //             Value initialToken = builder.create<gpu::WaitOp>(
// //                 gpuAllocOp.getLoc(),
// //                 builder.getType<gpu::AsyncTokenType>(),
// //                 ValueRange{}).getAsyncToken();
                
// //             auto newGpuAlloc = builder.create<gpu::AllocOp>(
// //                 gpuAllocOp.getLoc(),
// //                 gpuAllocOp.getMemref().getType(),
// //                 builder.getType<gpu::AsyncTokenType>(),
// //                 ValueRange{initialToken},
// //                 gpuAllocOp.getDynamicSizes(),
// //                 gpuAllocOp.getSymbolOperands(),
// //                 /*hostShared=*/false);
            
// //             // Add explicit synchronization 
// //             builder.create<gpu::WaitOp>(
// //                 gpuAllocOp.getLoc(), 
// //                 TypeRange{}, 
// //                 ValueRange{newGpuAlloc.getAsyncToken()});
            
// //             // Replace and store in map
// //             Value newMem = newGpuAlloc.getMemref();
// //             replacementMap[ultimateMemref] = newMem;
            
// //             // Replace all uses
// //             gpuAllocOp.getMemref().replaceAllUsesWith(newMem);
// //             gpuAllocOp.erase();
            
// //             // 如果原始memref经过了reinterpret_cast，使用processOperand处理
// //             if (memref != ultimateMemref) {
// //               bool needUpdateUsers = false;
// //               Value processedMemref = processOperand(builder, extractOp.getLoc(), memref, needUpdateUsers);
// //               extractOp.setOperand(processedMemref);
// //             } else {
// //               extractOp.setOperand(newMem);
// //             }
// //           }
// //         } else {
// //           // For memref.alloc, memref.get_global or krnl.global, process at function start
// //           auto funcOp = extractOp->getParentOfType<func::FuncOp>();
// //           if (!funcOp) {
// //             llvm::report_fatal_error("extract operation is not within a function");
// //           }
          
// //           builder.setInsertionPointToStart(&funcOp.getBody().front());
          
// //           // 如果原始memref经过了reinterpret_cast，使用processOperand处理整个链
// //           Value processedMemref;
// //           bool updateUsers = false;
          
// //           if (memref != ultimateMemref) {
// //             // 处理包含reinterpret_cast的情况
// //             processedMemref = processOperand(builder, extractOp.getLoc(), memref, updateUsers);
// //           } else {
// //             // 处理直接的情况
// //             processedMemref = processOperand(builder, extractOp.getLoc(), ultimateMemref, updateUsers);
// //           }
          
// //           // Replace the operand of the extract operation
// //           extractOp.setOperand(processedMemref);
          
// //           // Handle memref.alloc replacement if needed
// //           if (updateUsers && isa<memref::AllocOp>(ultimateDefOp)) {
// //             ultimateDefOp->getResult(0).replaceAllUsesWith(processedMemref);
// //             ultimateDefOp->erase();
// //           }
// //         }
// //       }
// //     }
// //   });
// // }

// // } // end anonymous namespace

// // namespace onnx_mlir {
// //   namespace krnl {
  
// //   std::unique_ptr<mlir::Pass> createInsertGPUAllocPass() {
// //       return std::make_unique<InsertGPUAllocPass>();
// //   }
  
// //   } // namespace krnl
// // } // namespace onnx_mlir
  
// // static mlir::PassRegistration<InsertGPUAllocPass> pass;

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/Diagnostics.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/IR/OpDefinition.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"

// #include "llvm/Support/ErrorHandling.h"

// using namespace mlir;

// namespace {
  
// struct InsertGPUAllocPass 
//     : public PassWrapper<InsertGPUAllocPass, OperationPass<ModuleOp>> {
// public:
//     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocPass)
//   void runOnOperation() override;
//   StringRef getArgument() const final { return "insert-gpu-alloc"; }
//   StringRef getDescription() const final { 
//     return "Convert host-side memref.alloc, memref.reinterpret_cast, krnl.global, memref.get_global, and gpu.alloc host_shared "
//            "to async gpu.alloc, and insert necessary async gpu.memcpy operations. "
//            "Adds explicit synchronization to decouple memory operations from compute operations. "
//            "Now includes scalar parameters in GPU memory conversion and scalar constant optimization."; 
//   }
//   private:
//   // Member function declarations
//   Value processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers);
//   Value processScalarConstant(OpBuilder &builder, Location loc, Value operand);
//   void collectAndMoveAllocationsToFunctionStart(func::FuncOp funcOp);
  
//   // 新增：追踪通过reinterpret_cast链找到最终源操作
//   std::pair<Value, Operation*> traceSourceThroughReinterpretCast(Value memref) {
//     Value currentMemref = memref;
//     Operation* currentDefOp = currentMemref.getDefiningOp();
    
//     // 追踪reinterpret_cast链直到找到非reinterpret_cast的源操作
//     while (currentDefOp && isa<memref::ReinterpretCastOp>(currentDefOp)) {
//       auto reinterpretOp = cast<memref::ReinterpretCastOp>(currentDefOp);
//       currentMemref = reinterpretOp.getSource();
//       currentDefOp = currentMemref.getDefiningOp();
//     }
    
//     return {currentMemref, currentDefOp};
//   }

//   // Global replacement map to avoid duplicate GPU allocations
//   DenseMap<Value, Value> replacementMap;
  
//   // 新增：存储需要移动的操作
//   DenseMap<func::FuncOp, SmallVector<Operation*, 8>> allocsToMove;
//   DenseMap<func::FuncOp, SmallVector<Operation*, 8>> subviewsToMove;
// };

// static Value createGpuAllocAndCopyAsync(OpBuilder &builder, Location loc, Value origVal,
//                                       ArrayRef<Value> dims, bool doCopy) {
//   // This pass does not support dynamic cases; if dims is not empty, report an error
//   if (!dims.empty()) {
//     llvm::report_fatal_error("InsertGPUAllocPass: Dynamic dimensions not supported");
//   }

//   auto origType = origVal.getType().cast<MemRefType>();
  
//   if (!origType.getLayout().isIdentity()) {
//     llvm::report_fatal_error("InsertGPUAllocPass: Non-identity (non-contiguous) layouts are not supported");
//   }

//   auto allocType = MemRefType::get(
//       origType.getShape(), 
//       origType.getElementType(), 
//       /*use the original layout*/ origType.getLayout(),
//       origType.getMemorySpace());
  
//   // Create an initial token
//   Value initialToken = builder.create<gpu::WaitOp>(
//       loc, 
//       builder.getType<gpu::AsyncTokenType>(),
//       ValueRange{}).getAsyncToken();
  
//   // Insert the async gpu.alloc operation without host_shared flag
//   auto gpuAlloc = builder.create<gpu::AllocOp>(
//       loc, allocType, 
//       builder.getType<gpu::AsyncTokenType>(), // Return an async token
//       ValueRange{initialToken}, // Use the initial token
//       dims, /*symbolOperands=*/std::nullopt,
//       /*hostShared=*/false); // No host_shared flag
  
//   Value allocResult = gpuAlloc.getResult(0);
//   Value allocToken = gpuAlloc.getAsyncToken();

//   // If a copy is needed, insert async gpu.memcpy operation
//   if (doCopy) {
//     auto memcpyOp = builder.create<gpu::MemcpyOp>(
//         loc, 
//         builder.getType<gpu::AsyncTokenType>(),
//         ValueRange{allocToken}, // Chain with allocation token
//         allocResult, origVal);
    
//     // Update token to the memcpy token
//     allocToken = memcpyOp.getAsyncToken();
//   }
  
//   // Add explicit synchronization point
//   builder.create<gpu::WaitOp>(loc, TypeRange{}, ValueRange{allocToken});
  
//   // If the allocated type does not match the original type, insert a cast operation.
//   if (allocType != origType) {
//     allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
//   }
  
//   return allocResult;
// }

// // 辅助函数：检查是否为标量操作函数
// static bool isScalarOperation(StringRef funcName) {
//   return funcName.contains("AddScalar") || 
//          funcName.contains("SubScalar") || 
//          funcName.contains("MulScalar") || 
//          funcName.contains("RSubScalar");
// }

// // 辅助函数：检查参数是否为标量参数（保留用于调试和日志记录）
// static bool isScalarParameter(func::CallOp callOp, Value paramValue) {
//   if (!isScalarOperation(callOp.getCallee())) {
//     return false;
//   }
  
//   // 对于标量操作函数，第二个参数（index 1）是标量参数
//   // 函数签名：mgpuCudnnXXXScalar(input_ptr, scalar_ptr, output_ptr, n, c, h, w, stream)
//   auto operands = callOp.getOperands();
  
//   // 需要追踪 extract_aligned_pointer -> index_cast -> inttoptr -> call operand 的链
//   // 找到该参数在call操作中的位置
//   for (unsigned i = 0; i < operands.size(); ++i) {
//     Value operand = operands[i];
    
//     // 追踪指针操作链，看是否最终来源于我们关心的paramValue
//     Value currentVal = operand;
    
//     // 反向追踪：inttoptr -> index_cast -> extract_aligned_pointer
//     if (auto intToPtrOp = currentVal.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
//       currentVal = intToPtrOp.getArg();
      
//       if (auto indexCastOp = currentVal.getDefiningOp<mlir::arith::IndexCastOp>()) {
//         currentVal = indexCastOp->getOperand(0);
        
//         if (auto extractOp = currentVal.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
//           Value sourceMemref = extractOp.getSource();
          
//           // 检查源memref是否与我们的paramValue匹配
//           if (sourceMemref == paramValue) {
//             // 找到了参数位置，检查是否为标量参数位置
//             // 标量参数通常在第二个位置（index 1）
//             return (i == 1);
//           }
//         }
//       }
//     }
//   }
  
//   return false;
// }

// Value InsertGPUAllocPass::processScalarConstant(OpBuilder &builder, Location loc, Value operand) {
//   // 检查操作数是否为f32或f16标量类型
//   if (auto floatType = operand.getType().dyn_cast<FloatType>()) {
//     if (floatType.isF32() || floatType.isF16()) {
//       // 检查是否来自memref.load
//       if (auto loadOp = operand.getDefiningOp<memref::LoadOp>()) {
//         Value memref = loadOp.getMemref();
        
//         // 检查memref是否来自krnl.global
//         if (auto defOp = memref.getDefiningOp()) {
//           if (defOp->getName().getStringRef() == "krnl.global") {
//             // 尝试提取krnl.global的常量值
//             if (auto valueAttr = defOp->getAttrOfType<mlir::Attribute>("value")) {
//               if (auto denseAttr = valueAttr.dyn_cast<DenseElementsAttr>()) {
//                 // 检查是否为标量常量（shape为空）
//                 if (denseAttr.getType().getRank() == 0) {
//                   // 直接使用DenseElementsAttr中的APFloat，它已经具有正确的精度
//                   APFloat originalValue = denseAttr.getSplatValue<APFloat>();
                  
//                   // 创建arith.constant替换memref.load
//                   Value constantOp = builder.create<arith::ConstantFloatOp>(
//                       loc, originalValue, floatType);
                  
//                   // 替换所有使用该load操作的地方
//                   loadOp.getResult().replaceAllUsesWith(constantOp);
//                   loadOp.erase();
                  
//                   return constantOp;
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
  
//   // 如果不符合条件，返回原始操作数
//   return operand;
// }

// /// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, "krnl.global",
// /// memref.get_global, or gpu.alloc host_shared, call createGpuAllocAndCopyAsync as appropriate to obtain a GPU memory variable.
// Value InsertGPUAllocPass::processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers) {
//   Operation *defOp = operand.getDefiningOp();

//   // Check if we already have a replacement for this value
//   if (replacementMap.count(operand)) {
//     needUpdateUsers = false;
//     return replacementMap[operand];
//   }

//   // Handle memref.reinterpret_cast specially
//   if (defOp && isa<memref::ReinterpretCastOp>(defOp)) {
//     auto reinterpretOp = cast<memref::ReinterpretCastOp>(defOp);
//     Value sourceOperand = reinterpretOp.getSource();
    
//     // Recursively process the source operand
//     bool sourceNeedUpdateUsers = false;
//     Value newSource = processOperand(builder, loc, sourceOperand, sourceNeedUpdateUsers);
    
//     // If the source was converted to GPU memory, update the reinterpret_cast operand
//     if (newSource != sourceOperand) {
//       // Directly modify the existing reinterpret_cast operation's operand
//       reinterpretOp.getOperation()->setOperand(0, newSource);
      
//       // Store in replacement map
//       replacementMap[operand] = operand; // The result is still the same Value
//       needUpdateUsers = false;
//       return operand; // Return the original result, but now it operates on GPU memory
//     } else {
//       // Source wasn't converted, return original
//       needUpdateUsers = false;
//       return operand;
//     }
//   }

//   llvm::SmallVector<Value, 4> dims;
//   if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
//     for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
//       if (memType.isDynamicDim(i))
//         dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
//     }
//     if (!dims.empty())
//       llvm::report_fatal_error("InsertGPUAllocsPass: Dynamic dimensions not supported");
//   }

//   if (defOp && (isa<memref::AllocOp>(defOp) ||
//                 isa<memref::GetGlobalOp>(defOp) ||
//                 defOp->getName().getStringRef() == "krnl.global" ||
//                 (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
//     bool doCopy = false;
    
//     // For krnl.global or get_global, need to copy data
//     if (defOp->getName().getStringRef() == "krnl.global" || 
//         isa<memref::GetGlobalOp>(defOp)) {
//       doCopy = true;
//     }
    
//     // For memref.alloc and gpu.alloc host_shared, set insertion point to function start
//     if (isa<memref::AllocOp>(defOp) || (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
//       needUpdateUsers = true;
//     } else {
//       // For krnl.global or get_global, set insertion point after the op
//       if (defOp->getName().getStringRef() == "krnl.global" || isa<memref::GetGlobalOp>(defOp)) {
//         builder.setInsertionPointAfter(defOp);
//       }
//       needUpdateUsers = false;
//     }
    
//     // Create async GPU allocation and copy if needed with explicit synchronization
//     Value newVal = createGpuAllocAndCopyAsync(builder, loc, operand, dims, doCopy);
    
//     // Store the replacement in the map
//     replacementMap[operand] = newVal;
    
//     return newVal;
//   }

//   // In all other cases, return the original value directly.
//   needUpdateUsers = false;
//   return operand;
// }

// // 新增函数：收集并移动分配操作到函数开头
// void InsertGPUAllocPass::collectAndMoveAllocationsToFunctionStart(func::FuncOp funcOp) {
//   if (funcOp.isExternal() || funcOp.getBody().empty()) {
//     return;
//   }

//   OpBuilder builder(&funcOp.getBody().front(), funcOp.getBody().front().begin());
  
//   // 收集所有需要转换的 memref.alloc 操作
//   SmallVector<memref::AllocOp, 8> allocOpsToProcess;
//   SmallVector<gpu::AllocOp, 8> hostSharedAllocOps;
  
//   funcOp.walk([&](memref::AllocOp allocOp) {
//     allocOpsToProcess.push_back(allocOp);
//   });
  
//   funcOp.walk([&](gpu::AllocOp allocOp) {
//     if (allocOp.getHostShared()) {
//       hostSharedAllocOps.push_back(allocOp);
//     }
//   });
  
//   // 处理 memref.alloc 操作
//   for (auto allocOp : allocOpsToProcess) {
//     // 创建新的 GPU 分配
//     auto origType = allocOp.getType().cast<MemRefType>();
//     auto allocType = MemRefType::get(
//         origType.getShape(), 
//         origType.getElementType(), 
//         origType.getLayout(),
//         origType.getMemorySpace());
    
//     // Create initial token
//     Value initialToken = builder.create<gpu::WaitOp>(
//         allocOp.getLoc(), 
//         builder.getType<gpu::AsyncTokenType>(),
//         ValueRange{}).getAsyncToken();
    
//     // Create GPU allocation
//     auto gpuAlloc = builder.create<gpu::AllocOp>(
//         allocOp.getLoc(), allocType, 
//         builder.getType<gpu::AsyncTokenType>(),
//         ValueRange{initialToken},
//         SmallVector<Value>{}, // no dynamic dims for now
//         std::nullopt,
//         false); // not host_shared
    
//     // Add synchronization
//     builder.create<gpu::WaitOp>(
//         allocOp.getLoc(), 
//         TypeRange{}, 
//         ValueRange{gpuAlloc.getAsyncToken()});
    
//     Value gpuMemref = gpuAlloc.getMemref();
    
//     // Store replacement
//     replacementMap[allocOp.getResult()] = gpuMemref;
//   }
  
//   // 处理 gpu.alloc host_shared 操作
//   for (auto hostSharedOp : hostSharedAllocOps) {
//     auto origType = hostSharedOp.getType().cast<MemRefType>();
    
//     // Create initial token
//     Value initialToken = builder.create<gpu::WaitOp>(
//         hostSharedOp.getLoc(), 
//         builder.getType<gpu::AsyncTokenType>(),
//         ValueRange{}).getAsyncToken();
    
//     // Create new GPU allocation without host_shared
//     auto newGpuAlloc = builder.create<gpu::AllocOp>(
//         hostSharedOp.getLoc(),
//         origType,
//         builder.getType<gpu::AsyncTokenType>(),
//         ValueRange{initialToken},
//         hostSharedOp.getDynamicSizes(),
//         hostSharedOp.getSymbolOperands(),
//         false); // not host_shared
    
//     // Add synchronization
//     builder.create<gpu::WaitOp>(
//         hostSharedOp.getLoc(), 
//         TypeRange{}, 
//         ValueRange{newGpuAlloc.getAsyncToken()});
    
//     Value gpuMemref = newGpuAlloc.getMemref();
    
//     // Store replacement
//     replacementMap[hostSharedOp.getMemref()] = gpuMemref;
//   }
  
//   // 收集并移动相关的 memref.subview 操作
//   SmallVector<memref::SubViewOp, 8> subviewOpsToMove;
  
//   for (auto allocOp : allocOpsToProcess) {
//     for (auto user : allocOp.getResult().getUsers()) {
//       if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
//         subviewOpsToMove.push_back(subviewOp);
//       }
//     }
//   }
  
//   for (auto hostSharedOp : hostSharedAllocOps) {
//     for (auto user : hostSharedOp.getMemref().getUsers()) {
//       if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
//         subviewOpsToMove.push_back(subviewOp);
//       }
//     }
//   }
  
//   // 移动 subview 操作到分配之后
//   for (auto subviewOp : subviewOpsToMove) {
//     // 更新 subview 的源操作数
//     Value sourceMemref = subviewOp.getSource();
//     if (replacementMap.count(sourceMemref)) {
//       subviewOp.getOperation()->setOperand(0, replacementMap[sourceMemref]);
//     }
    
//     // 将操作移动到函数开头
//     subviewOp->moveBefore(&funcOp.getBody().front(), builder.getInsertionPoint());
//   }
  
//   // 替换所有使用并删除原始操作
//   for (auto allocOp : allocOpsToProcess) {
//     Value originalResult = allocOp.getResult();
//     Value replacement = replacementMap[originalResult];
    
//     // 替换所有使用
//     originalResult.replaceAllUsesWith(replacement);
    
//     // 删除原始操作
//     allocOp.erase();
//   }
  
//   for (auto hostSharedOp : hostSharedAllocOps) {
//     Value originalResult = hostSharedOp.getMemref();
//     Value replacement = replacementMap[originalResult];
    
//     // 替换所有使用
//     originalResult.replaceAllUsesWith(replacement);
    
//     // 删除原始操作
//     hostSharedOp.erase();
//   }
// }

// void InsertGPUAllocPass::runOnOperation() {
//   ModuleOp module = getOperation();
//   OpBuilder builder(module.getContext());

//   // Clear the replacement map
//   replacementMap.clear();
//   allocsToMove.clear();
//   subviewsToMove.clear();

//   // 新的处理逻辑：首先处理每个函数中的分配操作
//   module.walk([&](func::FuncOp funcOp) {
//     collectAndMoveAllocationsToFunctionStart(funcOp);
//   });

//   // First pass: Process all functions to handle memory allocations at function start
//   module.walk([&](func::FuncOp funcOp) {
//     // Skip external functions (functions without a body)
//     if (funcOp.isExternal() || funcOp.getBody().empty())
//       return;
    
//     builder.setInsertionPointToStart(&funcOp.getBody().front());
    
//     // Process gpu.launch_func operations
//     llvm::SmallVector<gpu::LaunchFuncOp, 8> launchOps;
//     funcOp.walk([&](gpu::LaunchFuncOp op) {
//       launchOps.push_back(op);
//     });
    
//     for (auto launchOp : launchOps) {
//       auto operands = launchOp.getOperands();
//       llvm::SmallVector<Value, 4> newOperands;
//       bool launchChanged = false;

//       // Process each operand
//       for (Value operand : operands) {
//         bool updateUsers = false;

//         Value processedOperand = processScalarConstant(builder, launchOp.getLoc(), operand);
//         if (processedOperand != operand) {
//           // 如果进行了标量常量优化，使用新的常量值
//           newOperands.push_back(processedOperand);
//           launchChanged = true;
//           continue;
//         }

//         // Check if we already have a replacement
//         if (replacementMap.count(operand)) {
//           newOperands.push_back(replacementMap[operand]);
//           launchChanged = true;
//         } else {
//           newOperands.push_back(operand);
//         }
//       }
      
//       // Only update the operands if they changed
//       if (launchChanged) {
//         launchOp.getOperation()->setOperands(newOperands);
//       }
//     }
//   });

//   // Second pass: Handle memref.extract_aligned_pointer_as_index operations for CUDA library calls
//   module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
//     Value memref = extractOp.getSource();
    
//     // 使用新的追踪函数找到最终源操作
//     auto [ultimateMemref, ultimateDefOp] = traceSourceThroughReinterpretCast(memref);
    
//     // Check if we have a replacement for the ultimate source
//     if (replacementMap.count(ultimateMemref)) {
//       // 如果原始memref经过了reinterpret_cast，需要相应地处理
//       if (memref != ultimateMemref) {
//         // 处理reinterpret_cast链
//         Operation* currentOp = memref.getDefiningOp();
//         while (currentOp && isa<memref::ReinterpretCastOp>(currentOp)) {
//           auto reinterpretOp = cast<memref::ReinterpretCastOp>(currentOp);
//           Value source = reinterpretOp.getSource();
//           if (replacementMap.count(source)) {
//             reinterpretOp.getOperation()->setOperand(0, source);
//             break;
//           }
//           currentOp = source.getDefiningOp();
//         }
//       } else {
//         extractOp.getOperation()->setOperand(0, replacementMap[ultimateMemref]);
//       }
//     }
//   });
// }

// } // end anonymous namespace

// namespace onnx_mlir {
//   namespace krnl {
  
//   std::unique_ptr<mlir::Pass> createInsertGPUAllocPass() {
//       return std::make_unique<InsertGPUAllocPass>();
//   }
  
//   } // namespace krnl
// } // namespace onnx_mlir
  
// static mlir::PassRegistration<InsertGPUAllocPass> pass;


#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {
  
struct InsertGPUAllocPass 
    : public PassWrapper<InsertGPUAllocPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocPass)
  void runOnOperation() override;
  StringRef getArgument() const final { return "insert-gpu-alloc"; }
  StringRef getDescription() const final { 
    return "Convert host-side memref.alloc, memref.reinterpret_cast, krnl.global, memref.get_global, and gpu.alloc host_shared "
           "to async gpu.alloc, and insert necessary async gpu.memcpy operations. "
           "Adds explicit synchronization to decouple memory operations from compute operations. "
           "Now includes scalar parameters in GPU memory conversion and scalar constant optimization."; 
  }
  private:
  // Member function declarations
  Value processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers);
  Value processScalarConstant(OpBuilder &builder, Location loc, Value operand);
  void collectAndMoveAllocationsToFunctionStart(func::FuncOp funcOp);
  
  // 新增：追踪通过reinterpret_cast链找到最终源操作
  std::pair<Value, Operation*> traceSourceThroughReinterpretCast(Value memref) {
    Value currentMemref = memref;
    Operation* currentDefOp = currentMemref.getDefiningOp();
    
    // 追踪reinterpret_cast链直到找到非reinterpret_cast的源操作
    while (currentDefOp && isa<memref::ReinterpretCastOp>(currentDefOp)) {
      auto reinterpretOp = cast<memref::ReinterpretCastOp>(currentDefOp);
      currentMemref = reinterpretOp.getSource();
      currentDefOp = currentMemref.getDefiningOp();
    }
    
    return {currentMemref, currentDefOp};
  }

  // Global replacement map to avoid duplicate GPU allocations
  DenseMap<Value, Value> replacementMap;
  
  // 新增：存储需要移动的操作
  DenseMap<func::FuncOp, SmallVector<Operation*, 8>> allocsToMove;
  DenseMap<func::FuncOp, SmallVector<Operation*, 8>> subviewsToMove;
};

static Value createGpuAllocAndCopyAsync(OpBuilder &builder, Location loc, Value origVal,
                                      ArrayRef<Value> dims, bool doCopy) {
  // This pass does not support dynamic cases; if dims is not empty, report an error
  if (!dims.empty()) {
    llvm::report_fatal_error("InsertGPUAllocPass: Dynamic dimensions not supported");
  }

  auto origType = origVal.getType().cast<MemRefType>();
  
  if (!origType.getLayout().isIdentity()) {
    llvm::report_fatal_error("InsertGPUAllocPass: Non-identity (non-contiguous) layouts are not supported");
  }

  auto allocType = MemRefType::get(
      origType.getShape(), 
      origType.getElementType(), 
      /*use the original layout*/ origType.getLayout(),
      origType.getMemorySpace());
  
  // Create an initial token
  Value initialToken = builder.create<gpu::WaitOp>(
      loc, 
      builder.getType<gpu::AsyncTokenType>(),
      ValueRange{}).getAsyncToken();
  
  // Insert the async gpu.alloc operation without host_shared flag
  auto gpuAlloc = builder.create<gpu::AllocOp>(
      loc, allocType, 
      builder.getType<gpu::AsyncTokenType>(), // Return an async token
      ValueRange{initialToken}, // Use the initial token
      dims, /*symbolOperands=*/std::nullopt,
      /*hostShared=*/false); // No host_shared flag
  
  Value allocResult = gpuAlloc.getResult(0);
  Value allocToken = gpuAlloc.getAsyncToken();

  // If a copy is needed, insert async gpu.memcpy operation
  if (doCopy) {
    auto memcpyOp = builder.create<gpu::MemcpyOp>(
        loc, 
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{allocToken}, // Chain with allocation token
        allocResult, origVal);
    
    // Update token to the memcpy token
    allocToken = memcpyOp.getAsyncToken();
  }
  
  // Add explicit synchronization point
  builder.create<gpu::WaitOp>(loc, TypeRange{}, ValueRange{allocToken});
  
  // If the allocated type does not match the original type, insert a cast operation.
  if (allocType != origType) {
    allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
  }
  
  return allocResult;
}

// 辅助函数：检查是否为标量操作函数
static bool isScalarOperation(StringRef funcName) {
  return funcName.contains("AddScalar") || 
         funcName.contains("SubScalar") || 
         funcName.contains("MulScalar") || 
         funcName.contains("RSubScalar");
}

// 辅助函数：检查参数是否为标量参数（保留用于调试和日志记录）
static bool isScalarParameter(func::CallOp callOp, Value paramValue) {
  if (!isScalarOperation(callOp.getCallee())) {
    return false;
  }
  
  // 对于标量操作函数，第二个参数（index 1）是标量参数
  // 函数签名：mgpuCudnnXXXScalar(input_ptr, scalar_ptr, output_ptr, n, c, h, w, stream)
  auto operands = callOp.getOperands();
  
  // 需要追踪 extract_aligned_pointer -> index_cast -> inttoptr -> call operand 的链
  // 找到该参数在call操作中的位置
  for (unsigned i = 0; i < operands.size(); ++i) {
    Value operand = operands[i];
    
    // 追踪指针操作链，看是否最终来源于我们关心的paramValue
    Value currentVal = operand;
    
    // 反向追踪：inttoptr -> index_cast -> extract_aligned_pointer
    if (auto intToPtrOp = currentVal.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      currentVal = intToPtrOp.getArg();
      
      if (auto indexCastOp = currentVal.getDefiningOp<mlir::arith::IndexCastOp>()) {
        currentVal = indexCastOp->getOperand(0);
        
        if (auto extractOp = currentVal.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
          Value sourceMemref = extractOp.getSource();
          
          // 检查源memref是否与我们的paramValue匹配
          if (sourceMemref == paramValue) {
            // 找到了参数位置，检查是否为标量参数位置
            // 标量参数通常在第二个位置（index 1）
            return (i == 1);
          }
        }
      }
    }
  }
  
  return false;
}

Value InsertGPUAllocPass::processScalarConstant(OpBuilder &builder, Location loc, Value operand) {
  // 检查操作数是否为f32或f16标量类型
  if (auto floatType = operand.getType().dyn_cast<FloatType>()) {
    if (floatType.isF32() || floatType.isF16()) {
      // 检查是否来自memref.load
      if (auto loadOp = operand.getDefiningOp<memref::LoadOp>()) {
        Value memref = loadOp.getMemref();
        
        // 检查memref是否来自krnl.global
        if (auto defOp = memref.getDefiningOp()) {
          if (defOp->getName().getStringRef() == "krnl.global") {
            // 尝试提取krnl.global的常量值
            if (auto valueAttr = defOp->getAttrOfType<mlir::Attribute>("value")) {
              if (auto denseAttr = valueAttr.dyn_cast<DenseElementsAttr>()) {
                // 检查是否为标量常量（shape为空）
                if (denseAttr.getType().getRank() == 0) {
                  // 直接使用DenseElementsAttr中的APFloat，它已经具有正确的精度
                  APFloat originalValue = denseAttr.getSplatValue<APFloat>();
                  
                  // 创建arith.constant替换memref.load
                  Value constantOp = builder.create<arith::ConstantFloatOp>(
                      loc, originalValue, floatType);
                  
                  // 替换所有使用该load操作的地方
                  loadOp.getResult().replaceAllUsesWith(constantOp);
                  loadOp.erase();
                  
                  return constantOp;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // 如果不符合条件，返回原始操作数
  return operand;
}

/// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, "krnl.global",
/// memref.get_global, or gpu.alloc host_shared, call createGpuAllocAndCopyAsync as appropriate to obtain a GPU memory variable.
Value InsertGPUAllocPass::processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers) {
  Operation *defOp = operand.getDefiningOp();

  // Check if we already have a replacement for this value
  if (replacementMap.count(operand)) {
    needUpdateUsers = false;
    return replacementMap[operand];
  }

  // Handle memref.reinterpret_cast specially
  if (defOp && isa<memref::ReinterpretCastOp>(defOp)) {
    auto reinterpretOp = cast<memref::ReinterpretCastOp>(defOp);
    Value sourceOperand = reinterpretOp.getSource();
    
    // Recursively process the source operand
    bool sourceNeedUpdateUsers = false;
    Value newSource = processOperand(builder, loc, sourceOperand, sourceNeedUpdateUsers);
    
    // If the source was converted to GPU memory, update the reinterpret_cast operand
    if (newSource != sourceOperand) {
      // Directly modify the existing reinterpret_cast operation's operand
      reinterpretOp.getOperation()->setOperand(0, newSource);
      
      // Store in replacement map
      replacementMap[operand] = operand; // The result is still the same Value
      needUpdateUsers = false;
      return operand; // Return the original result, but now it operates on GPU memory
    } else {
      // Source wasn't converted, return original
      needUpdateUsers = false;
      return operand;
    }
  }

  llvm::SmallVector<Value, 4> dims;
  if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
    for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
      if (memType.isDynamicDim(i))
        dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
    }
    if (!dims.empty())
      llvm::report_fatal_error("InsertGPUAllocsPass: Dynamic dimensions not supported");
  }

  if (defOp && (isa<memref::AllocOp>(defOp) ||
                isa<memref::GetGlobalOp>(defOp) ||
                defOp->getName().getStringRef() == "krnl.global" ||
                (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
    bool doCopy = false;
    
    // For krnl.global or get_global, need to copy data
    if (defOp->getName().getStringRef() == "krnl.global" || 
        isa<memref::GetGlobalOp>(defOp)) {
      doCopy = true;
    }
    
    // For memref.alloc and gpu.alloc host_shared, set insertion point to function start
    if (isa<memref::AllocOp>(defOp) || (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
      needUpdateUsers = true;
    } else {
      // For krnl.global or get_global, set insertion point after the op
      if (defOp->getName().getStringRef() == "krnl.global" || isa<memref::GetGlobalOp>(defOp)) {
        builder.setInsertionPointAfter(defOp);
      }
      needUpdateUsers = false;
    }
    
    // Create async GPU allocation and copy if needed with explicit synchronization
    Value newVal = createGpuAllocAndCopyAsync(builder, loc, operand, dims, doCopy);
    
    // Store the replacement in the map
    replacementMap[operand] = newVal;
    
    return newVal;
  }

  // In all other cases, return the original value directly.
  needUpdateUsers = false;
  return operand;
}

// 修复：收集并移动分配操作到函数开头
void InsertGPUAllocPass::collectAndMoveAllocationsToFunctionStart(func::FuncOp funcOp) {
  if (funcOp.isExternal() || funcOp.getBody().empty()) {
    return;
  }

  OpBuilder builder(&funcOp.getBody().front(), funcOp.getBody().front().begin());
  
  // 收集所有需要转换的 memref.alloc 操作
  SmallVector<memref::AllocOp, 8> allocOpsToProcess;
  SmallVector<gpu::AllocOp, 8> hostSharedAllocOps;
  
  funcOp.walk([&](memref::AllocOp allocOp) {
    allocOpsToProcess.push_back(allocOp);
  });
  
  funcOp.walk([&](gpu::AllocOp allocOp) {
    if (allocOp.getHostShared()) {
      hostSharedAllocOps.push_back(allocOp);
    }
  });
  
  // 处理 memref.alloc 操作
  for (auto allocOp : allocOpsToProcess) {
    // 创建新的 GPU 分配
    auto origType = allocOp.getType().cast<MemRefType>();
    auto allocType = MemRefType::get(
        origType.getShape(), 
        origType.getElementType(), 
        origType.getLayout(),
        origType.getMemorySpace());
    
    // Create initial token
    Value initialToken = builder.create<gpu::WaitOp>(
        allocOp.getLoc(), 
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{}).getAsyncToken();
    
    // Create GPU allocation
    auto gpuAlloc = builder.create<gpu::AllocOp>(
        allocOp.getLoc(), allocType, 
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{initialToken},
        SmallVector<Value>{}, // no dynamic dims for now
        std::nullopt,
        false); // not host_shared
    
    // Add synchronization
    builder.create<gpu::WaitOp>(
        allocOp.getLoc(), 
        TypeRange{}, 
        ValueRange{gpuAlloc.getAsyncToken()});
    
    Value gpuMemref = gpuAlloc.getMemref();
    
    // Store replacement
    replacementMap[allocOp.getResult()] = gpuMemref;
  }
  
  // 处理 gpu.alloc host_shared 操作
  for (auto hostSharedOp : hostSharedAllocOps) {
    auto origType = hostSharedOp.getType().cast<MemRefType>();
    
    // Create initial token
    Value initialToken = builder.create<gpu::WaitOp>(
        hostSharedOp.getLoc(), 
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{}).getAsyncToken();
    
    // Create new GPU allocation without host_shared
    auto newGpuAlloc = builder.create<gpu::AllocOp>(
        hostSharedOp.getLoc(),
        origType,
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{initialToken},
        hostSharedOp.getDynamicSizes(),
        hostSharedOp.getSymbolOperands(),
        false); // not host_shared
    
    // Add synchronization
    builder.create<gpu::WaitOp>(
        hostSharedOp.getLoc(), 
        TypeRange{}, 
        ValueRange{newGpuAlloc.getAsyncToken()});
    
    Value gpuMemref = newGpuAlloc.getMemref();
    
    // Store replacement
    replacementMap[hostSharedOp.getMemref()] = gpuMemref;
  }
  
  // 收集并移动相关的 memref.subview 操作
  SmallVector<memref::SubViewOp, 8> subviewOpsToMove;
  
  for (auto allocOp : allocOpsToProcess) {
    for (auto user : allocOp.getResult().getUsers()) {
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        subviewOpsToMove.push_back(subviewOp);
      }
    }
  }
  
  for (auto hostSharedOp : hostSharedAllocOps) {
    for (auto user : hostSharedOp.getMemref().getUsers()) {
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        subviewOpsToMove.push_back(subviewOp);
      }
    }
  }
  
  // 移动 subview 操作到分配之后
  for (auto subviewOp : subviewOpsToMove) {
    // 更新 subview 的源操作数
    Value sourceMemref = subviewOp.getSource();
    if (replacementMap.count(sourceMemref)) {
      subviewOp.getOperation()->setOperand(0, replacementMap[sourceMemref]);
    }
    
    // 将操作移动到函数开头
    subviewOp->moveBefore(&funcOp.getBody().front(), builder.getInsertionPoint());
  }
  
  // 替换所有使用并删除原始操作
  for (auto allocOp : allocOpsToProcess) {
    Value originalResult = allocOp.getResult();
    Value replacement = replacementMap[originalResult];
    
    // 替换所有使用
    originalResult.replaceAllUsesWith(replacement);
    
    // 删除原始操作
    allocOp.erase();
  }
  
  for (auto hostSharedOp : hostSharedAllocOps) {
    Value originalResult = hostSharedOp.getMemref();
    Value replacement = replacementMap[originalResult];
    
    // 替换所有使用
    originalResult.replaceAllUsesWith(replacement);
    
    // 删除原始操作
    hostSharedOp.erase();
  }
}

void InsertGPUAllocPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Clear the replacement map
  replacementMap.clear();
  allocsToMove.clear();
  subviewsToMove.clear();

  // 新的处理逻辑：首先处理每个函数中的分配操作
  module.walk([&](func::FuncOp funcOp) {
    collectAndMoveAllocationsToFunctionStart(funcOp);
  });

  // First pass: Process all functions to handle memory allocations at function start
  module.walk([&](func::FuncOp funcOp) {
    // Skip external functions (functions without a body)
    if (funcOp.isExternal() || funcOp.getBody().empty())
      return;
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Process gpu.launch_func operations
    llvm::SmallVector<gpu::LaunchFuncOp, 8> launchOps;
    funcOp.walk([&](gpu::LaunchFuncOp op) {
      launchOps.push_back(op);
    });
    
    for (auto launchOp : launchOps) {
      auto operands = launchOp.getOperands();
      llvm::SmallVector<Value, 4> newOperands;
      bool launchChanged = false;

      // Process each operand
      for (Value operand : operands) {
        bool updateUsers = false;

        Value processedOperand = processScalarConstant(builder, launchOp.getLoc(), operand);
        if (processedOperand != operand) {
          // 如果进行了标量常量优化，使用新的常量值
          newOperands.push_back(processedOperand);
          launchChanged = true;
          continue;
        }

        // Check if we already have a replacement
        if (replacementMap.count(operand)) {
          newOperands.push_back(replacementMap[operand]);
          launchChanged = true;
        } else {
          newOperands.push_back(operand);
        }
      }
      
      // Only update the operands if they changed
      if (launchChanged) {
        launchOp.getOperation()->setOperands(newOperands);
      }
    }
  });

  // Second pass: Handle memref.extract_aligned_pointer_as_index operations for CUDA library calls
  // 这里需要特别处理 krnl.global 和 memref.get_global 的情况
  module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
    Value memref = extractOp.getSource();
    
    // 使用新的追踪函数找到最终源操作
    auto [ultimateMemref, ultimateDefOp] = traceSourceThroughReinterpretCast(memref);
    
    // Check if the ultimate source comes from krnl.global, memref.alloc, memref.get_global or gpu.alloc with host_shared
    bool isTargetOp = false;
    
    if (ultimateDefOp) {
      if (ultimateDefOp->getName().getStringRef() == "krnl.global" || 
          isa<memref::AllocOp>(ultimateDefOp) ||
          isa<memref::GetGlobalOp>(ultimateDefOp) ||
          (isa<gpu::AllocOp>(ultimateDefOp) && cast<gpu::AllocOp>(ultimateDefOp).getHostShared())) {
        isTargetOp = true;
      }
    }
    
    if (isTargetOp) {
      // Check if we've already created a replacement for the ultimate source
      if (replacementMap.count(ultimateMemref)) {
        // 如果原始memref经过了reinterpret_cast，需要使用processOperand来正确处理
        if (memref != ultimateMemref) {
          bool needUpdateUsers = false;
          Value processedMemref = processOperand(builder, extractOp.getLoc(), memref, needUpdateUsers);
          extractOp.setOperand(processedMemref);
        } else {
          extractOp.setOperand(replacementMap[ultimateMemref]);
        }
        return;
      }
      
      // Check if the result flows into a CUDA library call
      bool flowsToGpuLibCall = false;
      bool isScalarParam = false;  // 保留用于调试和日志记录
      
      for (Operation *user : extractOp->getResult(0).getUsers()) {
        // Trace through inttoptr and index_cast operations
        Operation* currentUser = user;
        while (currentUser && (isa<arith::IndexCastOp>(currentUser) || currentUser->getName().getStringRef() == "llvm.inttoptr")) {
          if (currentUser->getNumResults() > 0 && !currentUser->getResult(0).getUses().empty()) {
            currentUser = *currentUser->getResult(0).getUsers().begin();
          } else {
            break;
          }
        }
        
        // Check if we reached a call operation to a CUDA library function
        if (currentUser && isa<func::CallOp>(currentUser)) {
          auto callOp = cast<func::CallOp>(currentUser);
          if (callOp.getCallee().starts_with("mgpuCudnn") || 
              callOp.getCallee().starts_with("mgpuCulibs") ||
              callOp.getCallee().starts_with("mgpu")) {
            flowsToGpuLibCall = true;
            
            // 检查是否为标量参数（用于调试和日志记录）
            isScalarParam = isScalarParameter(callOp, ultimateMemref);
            
            break;
          }
        }
      }
      
      // 现在所有流向GPU库调用的参数都进行GPU内存转换
      if (flowsToGpuLibCall) {
        // 对于 krnl.global 和 memref.get_global，需要特殊处理以确保数据复制
        if (ultimateDefOp->getName().getStringRef() == "krnl.global" || 
            isa<memref::GetGlobalOp>(ultimateDefOp)) {
          
          // 设置插入点在操作之后
          builder.setInsertionPointAfter(ultimateDefOp);
          
          // 如果原始memref经过了reinterpret_cast，使用processOperand处理整个链
          Value processedMemref;
          bool updateUsers = false;
          
          if (memref != ultimateMemref) {
            // 处理包含reinterpret_cast的情况
            processedMemref = processOperand(builder, extractOp.getLoc(), memref, updateUsers);
          } else {
            // 处理直接的情况
            processedMemref = processOperand(builder, extractOp.getLoc(), ultimateMemref, updateUsers);
          }
          
          // Replace the operand of the extract operation
          extractOp.setOperand(processedMemref);
          
        } else {
          // For gpu.alloc host_shared, replace directly
          if (auto gpuAllocOp = dyn_cast_or_null<gpu::AllocOp>(ultimateDefOp)) {
            if (gpuAllocOp.getHostShared()) {
              auto funcOp = gpuAllocOp->getParentOfType<func::FuncOp>();
              if (!funcOp) {
                llvm::report_fatal_error("gpu.alloc op is not within a function");
              }
              
              // Move replacement to the start of the function
              builder.setInsertionPointToStart(&funcOp.getBody().front());
              
              // Create new async GPU allocation without host_shared
              Value initialToken = builder.create<gpu::WaitOp>(
                  gpuAllocOp.getLoc(),
                  builder.getType<gpu::AsyncTokenType>(),
                  ValueRange{}).getAsyncToken();
                  
              auto newGpuAlloc = builder.create<gpu::AllocOp>(
                  gpuAllocOp.getLoc(),
                  gpuAllocOp.getMemref().getType(),
                  builder.getType<gpu::AsyncTokenType>(),
                  ValueRange{initialToken},
                  gpuAllocOp.getDynamicSizes(),
                  gpuAllocOp.getSymbolOperands(),
                  /*hostShared=*/false);
              
              // Add explicit synchronization 
              builder.create<gpu::WaitOp>(
                  gpuAllocOp.getLoc(), 
                  TypeRange{}, 
                  ValueRange{newGpuAlloc.getAsyncToken()});
              
              // Replace and store in map
              Value newMem = newGpuAlloc.getMemref();
              replacementMap[ultimateMemref] = newMem;
              
              // Replace all uses
              gpuAllocOp.getMemref().replaceAllUsesWith(newMem);
              gpuAllocOp.erase();
              
              // 如果原始memref经过了reinterpret_cast，使用processOperand处理
              if (memref != ultimateMemref) {
                bool needUpdateUsers = false;
                Value processedMemref = processOperand(builder, extractOp.getLoc(), memref, needUpdateUsers);
                extractOp.setOperand(processedMemref);
              } else {
                extractOp.setOperand(newMem);
              }
            }
          } else {
            // For memref.alloc, check if we have a replacement already
            if (replacementMap.count(ultimateMemref)) {
              if (memref != ultimateMemref) {
                // 处理reinterpret_cast链
                Operation* currentOp = memref.getDefiningOp();
                while (currentOp && isa<memref::ReinterpretCastOp>(currentOp)) {
                  auto reinterpretOp = cast<memref::ReinterpretCastOp>(currentOp);
                  Value source = reinterpretOp.getSource();
                  if (replacementMap.count(source)) {
                    reinterpretOp.getOperation()->setOperand(0, replacementMap[source]);
                    break;
                  }
                  currentOp = source.getDefiningOp();
                }
              } else {
                extractOp.getOperation()->setOperand(0, replacementMap[ultimateMemref]);
              }
            }
          }
        }
      }
    }
  });
}

} // end anonymous namespace

namespace onnx_mlir {
  namespace krnl {
  
  std::unique_ptr<mlir::Pass> createInsertGPUAllocPass() {
      return std::make_unique<InsertGPUAllocPass>();
  }
  
  } // namespace krnl
} // namespace onnx_mlir
  
static mlir::PassRegistration<InsertGPUAllocPass> pass;