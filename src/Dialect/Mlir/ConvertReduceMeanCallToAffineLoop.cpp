// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "llvm/Support/Debug.h"
// #include <vector>

// using namespace mlir;
// using namespace mlir::affine;

// #define DEBUG_TYPE "convert-reduce-mean-to-affine"

// namespace {

//   // 存储一个 ReduceMean 调用序列的完整信息
// struct ReduceMeanCallSequence {
//   // 输入相关操作
//   Value inputMemref;
//   memref::ExtractAlignedPointerAsIndexOp inputExtract;
//   arith::IndexCastOp inputIndexCast;
//   LLVM::IntToPtrOp inputIntToPtr;
  
//   // 输出相关操作
//   memref::AllocOp outputAlloc;
//   memref::ExtractAlignedPointerAsIndexOp outputExtract;
//   arith::IndexCastOp outputIndexCast;
//   LLVM::IntToPtrOp outputIntToPtr;
  
//   // Stream 相关操作
//   LLVM::CallOp streamCreate;
//   LLVM::CallOp createHandles;
//   LLVM::CallOp reduceMeanCall;
//   LLVM::CallOp streamSync;
//   LLVM::CallOp streamDestroy;
  
//   // 从 memref 类型中提取的维度信息
//   std::vector<int64_t> inputShape;
//   std::vector<int64_t> outputShape;
//   int reduceAxis;  // 被reduce的维度索引
  
//   Location loc;
//   ReduceMeanCallSequence(Location l) : loc(l) {}
// };

// class ConvertReduceMeanToAffinePass
//     : public PassWrapper<ConvertReduceMeanToAffinePass, OperationPass<func::FuncOp>> {

// public:
//   StringRef getArgument() const final { 
//     return "convert-reduce-mean-to-affine"; 
//   }
  
//   StringRef getDescription() const final {
//     return "Convert mgpuCudnnReduceMean calls to affine loop implementations";
//   }

//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
    
//     LLVM_DEBUG(llvm::dbgs() << "=== Convert ReduceMean to Affine Pass ===\n");
    
//     // 收集所有需要转换的 reduce mean 调用序列
//     std::vector<ReduceMeanCallSequence> sequences;
//     if (failed(collectReduceMeanSequences(funcOp, sequences))) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to collect reduce mean sequences\n");
//       signalPassFailure();
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Found " << sequences.size() 
//                << " reduce mean call sequences\n");
    
//     // 转换每个序列
//     for (auto& seq : sequences) {
//       if (failed(convertSequenceToAffineLoop(seq))) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to convert a sequence\n");
//         signalPassFailure();
//         return;
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully converted all sequences\n");
//   }

// private:
//   // 收集所有 ReduceMean 调用序列
//   LogicalResult collectReduceMeanSequences(
//       func::FuncOp funcOp, 
//       std::vector<ReduceMeanCallSequence>& sequences) {
    
//     funcOp.walk([&](LLVM::CallOp callOp) {
//       // 查找 mgpuCudnnReduceMean 调用
//       if (callOp.getCallee() && 
//           callOp.getCallee()->str() == "mgpuCudnnReduceMean") {
        
//         ReduceMeanCallSequence seq(callOp.getLoc());
//         if (succeeded(extractSequenceInfo(callOp, seq))) {
//           sequences.push_back(seq);
//         }
//       }
//     });
    
//     return success();
//   }

//   // 从 ReduceMean 调用中提取完整的操作序列信息
//   LogicalResult extractSequenceInfo(
//       LLVM::CallOp reduceMeanCall, 
//       ReduceMeanCallSequence& seq) {
    
//     seq.reduceMeanCall = reduceMeanCall;
//     seq.loc = reduceMeanCall.getLoc();
    
//     auto operands = reduceMeanCall.getOperands();
//     if (operands.size() < 15) {
//       LLVM_DEBUG(llvm::dbgs() << "Invalid number of operands\n");
//       return failure();
//     }
    
//     // 获取输入和输出指针以及 stream
//     Value inputPtr = operands[12];  // 第13个参数是输入指针
//     Value outputPtr = operands[13]; // 第14个参数是输出指针
//     Value streamPtr = operands[14]; // 第15个参数是stream
    
//     // 回溯操作链
//     if (failed(traceInputChain(inputPtr, seq)) ||
//         failed(traceOutputChain(outputPtr, seq)) ||
//         failed(traceStreamOps(streamPtr, reduceMeanCall, seq))) {
//       return failure();
//     }
    
//     // 从 memref 类型中提取维度信息
//     if (failed(extractDimensionsFromMemref(seq))) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to extract dimensions from memref\n");
//       return failure();
//     }
    
//     return success();
//   }

//   // 从 memref 类型中提取维度信息
//   LogicalResult extractDimensionsFromMemref(ReduceMeanCallSequence& seq) {
//     // 获取输入 memref 的类型
//     auto inputMemrefType = seq.inputMemref.getType().dyn_cast<MemRefType>();
//     if (!inputMemrefType) {
//       LLVM_DEBUG(llvm::dbgs() << "Input is not a memref type\n");
//       return failure();
//     }
    
//     // 获取输出 memref 的类型
//     auto outputMemrefType = seq.outputAlloc.getType().dyn_cast<MemRefType>();
//     if (!outputMemrefType) {
//       LLVM_DEBUG(llvm::dbgs() << "Output is not a memref type\n");
//       return failure();
//     }
    
//     // 检查维度必须是静态的
//     if (!inputMemrefType.hasStaticShape() || !outputMemrefType.hasStaticShape()) {
//       LLVM_DEBUG(llvm::dbgs() << "Dynamic shapes not supported\n");
//       return failure();
//     }
    
//     // 提取输入维度
//     ArrayRef<int64_t> inputShapeRef = inputMemrefType.getShape();
//     seq.inputShape.assign(inputShapeRef.begin(), inputShapeRef.end());
    
//     // 提取输出维度
//     ArrayRef<int64_t> outputShapeRef = outputMemrefType.getShape();
//     seq.outputShape.assign(outputShapeRef.begin(), outputShapeRef.end());
    
//     // 检查维度数量是否匹配
//     if (seq.inputShape.size() != seq.outputShape.size()) {
//       LLVM_DEBUG(llvm::dbgs() << "Input and output rank mismatch\n");
//       return failure();
//     }
    
//     // 找出哪个维度被 reduce 了（输出维度为1）
//     seq.reduceAxis = -1;
//     for (size_t i = 0; i < seq.inputShape.size(); ++i) {
//       if (seq.outputShape[i] == 1 && seq.inputShape[i] != 1) {
//         if (seq.reduceAxis != -1) {
//           LLVM_DEBUG(llvm::dbgs() << "Multiple reduce axes not supported\n");
//           return failure();
//         }
//         seq.reduceAxis = i;
//       } else if (seq.inputShape[i] != seq.outputShape[i]) {
//         LLVM_DEBUG(llvm::dbgs() << "Unsupported dimension change\n");
//         return failure();
//       }
//     }
    
//     if (seq.reduceAxis == -1) {
//       LLVM_DEBUG(llvm::dbgs() << "Could not identify reduce axis\n");
//       return failure();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Extracted dimensions:\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Input shape: [");
//     for (size_t i = 0; i < seq.inputShape.size(); ++i) {
//       LLVM_DEBUG(llvm::dbgs() << seq.inputShape[i]);
//       if (i < seq.inputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//     }
//     LLVM_DEBUG(llvm::dbgs() << "]\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Output shape: [");
//     for (size_t i = 0; i < seq.outputShape.size(); ++i) {
//       LLVM_DEBUG(llvm::dbgs() << seq.outputShape[i]);
//       if (i < seq.outputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//     }
//     LLVM_DEBUG(llvm::dbgs() << "]\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Reduce axis: " << seq.reduceAxis << "\n");
    
//     return success();
//   }

//   // 从常量操作中提取整数值
//   LogicalResult extractIntFromConstant(Value val, int& result) {
//     if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
//       if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
//         result = intAttr.getInt();
//         return success();
//       }
//     }
//     return failure();
//   }

//   // 回溯输入操作链: memref -> extract -> index_cast -> inttoptr
//   LogicalResult traceInputChain(Value inputPtr, ReduceMeanCallSequence& seq) {
//     // inttoptr
//     auto intToPtrOp = inputPtr.getDefiningOp<LLVM::IntToPtrOp>();
//     if (!intToPtrOp) return failure();
//     seq.inputIntToPtr = intToPtrOp;
    
//     // index_cast
//     auto indexCastOp = intToPtrOp.getOperand().getDefiningOp<arith::IndexCastOp>();
//     if (!indexCastOp) return failure();
//     seq.inputIndexCast = indexCastOp;
    
//     // extract_aligned_pointer
//     auto extractOp = indexCastOp.getOperand()
//         .getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
//     if (!extractOp) return failure();
//     seq.inputExtract = extractOp;
    
//     // memref (可能来自 alloc 或函数参数)
//     seq.inputMemref = extractOp.getOperand();
    
//     return success();
//   }

//   // 回溯输出操作链: alloc -> extract -> index_cast -> inttoptr
//   LogicalResult traceOutputChain(Value outputPtr, ReduceMeanCallSequence& seq) {
//     // inttoptr
//     auto intToPtrOp = outputPtr.getDefiningOp<LLVM::IntToPtrOp>();
//     if (!intToPtrOp) return failure();
//     seq.outputIntToPtr = intToPtrOp;
    
//     // index_cast
//     auto indexCastOp = intToPtrOp.getOperand().getDefiningOp<arith::IndexCastOp>();
//     if (!indexCastOp) return failure();
//     seq.outputIndexCast = indexCastOp;
    
//     // extract_aligned_pointer
//     auto extractOp = indexCastOp.getOperand()
//         .getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
//     if (!extractOp) return failure();
//     seq.outputExtract = extractOp;
    
//     // alloc
//     auto allocOp = extractOp.getOperand().getDefiningOp<memref::AllocOp>();
//     if (!allocOp) return failure();
//     seq.outputAlloc = allocOp;
    
//     return success();
//   }

//   // 回溯 stream 相关操作
//   LogicalResult traceStreamOps(
//       Value streamPtr, 
//       LLVM::CallOp reduceMeanCall,
//       ReduceMeanCallSequence& seq) {
    
//     // stream 来自 mgpuStreamCreate
//     auto streamCreateOp = streamPtr.getDefiningOp<LLVM::CallOp>();
//     if (!streamCreateOp || !streamCreateOp.getCallee() ||
//         streamCreateOp.getCallee()->str() != "mgpuStreamCreate") {
//       return failure();
//     }
//     seq.streamCreate = streamCreateOp;
    
//     // 向后查找 createHandles, streamSync, streamDestroy
//     Operation* nextOp = streamCreateOp.getOperation()->getNextNode();
    
//     // mgpuCreateHandlesForStream
//     if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(nextOp)) {
//       if (callOp.getCallee() && 
//           callOp.getCallee()->str() == "mgpuCreateHandlesForStream") {
//         seq.createHandles = callOp;
//         nextOp = nextOp->getNextNode();
//       }
//     }
    
//     // 跳过 reduceMeanCall (已经有了)
//     if (nextOp == reduceMeanCall.getOperation()) {
//       nextOp = nextOp->getNextNode();
//     }
    
//     // mgpuStreamSynchronize
//     if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(nextOp)) {
//       if (callOp.getCallee() && 
//           callOp.getCallee()->str() == "mgpuStreamSynchronize") {
//         seq.streamSync = callOp;
//         nextOp = nextOp->getNextNode();
//       }
//     }
    
//     // mgpuStreamDestroy
//     if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(nextOp)) {
//       if (callOp.getCallee() && 
//           callOp.getCallee()->str() == "mgpuStreamDestroy") {
//         seq.streamDestroy = callOp;
//       }
//     }
    
//     return success();
//   }

//   // 将序列转换为 affine loop
//   LogicalResult convertSequenceToAffineLoop(ReduceMeanCallSequence& seq) {
//     LLVM_DEBUG(llvm::dbgs() << "\nConverting sequence:\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Input shape: [");
//     for (size_t i = 0; i < seq.inputShape.size(); ++i) {
//       LLVM_DEBUG(llvm::dbgs() << seq.inputShape[i]);
//       if (i < seq.inputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//     }
//     LLVM_DEBUG(llvm::dbgs() << "]\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Output shape: [");
//     for (size_t i = 0; i < seq.outputShape.size(); ++i) {
//       LLVM_DEBUG(llvm::dbgs() << seq.outputShape[i]);
//       if (i < seq.outputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//     }
//     LLVM_DEBUG(llvm::dbgs() << "]\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Reduce axis: " << seq.reduceAxis << "\n");
    
//     // 在 outputAlloc 之后插入 affine loop
//     OpBuilder builder(seq.outputAlloc.getOperation()->getNextNode());
    
//     // 创建常量
//     Value zero_f32 = builder.create<arith::ConstantOp>(
//         seq.loc, builder.getF32Type(), builder.getF32FloatAttr(0.0));
    
//     // reduce 维度的大小（用于计算平均值）
//     int64_t reduceSize = seq.inputShape[seq.reduceAxis];
//     Value reduceSize_f32 = builder.create<arith::ConstantOp>(
//         seq.loc, builder.getF32Type(), 
//         builder.getF32FloatAttr(static_cast<float>(reduceSize)));
    
//     Value inputMemref = seq.inputMemref;
//     Value outputMemref = seq.outputAlloc.getResult();
    
//     // 创建外层循环（遍历所有非reduce维度）
//     std::vector<AffineForOp> outerLoops;
//     std::vector<Value> outerIVs;
    
//     for (size_t dim = 0; dim < seq.outputShape.size(); ++dim) {
//       if (dim != seq.reduceAxis) {
//         auto forOp = builder.create<AffineForOp>(seq.loc, 0, seq.outputShape[dim]);
//         builder.setInsertionPointToStart(forOp.getBody());
//         outerLoops.push_back(forOp);
//         outerIVs.push_back(forOp.getInductionVar());
//       }
//     }
    
//     // 构建输出的索引（对于reduce维度使用常量0）
//     std::vector<Value> outputIndices;
//     Value c0 = builder.create<arith::ConstantIndexOp>(seq.loc, 0);
    
//     size_t outerIVIndex = 0;
//     for (size_t dim = 0; dim < seq.outputShape.size(); ++dim) {
//       if (dim == seq.reduceAxis) {
//         outputIndices.push_back(c0);
//       } else {
//         outputIndices.push_back(outerIVs[outerIVIndex++]);
//       }
//     }
    
//     // 初始化累加器为 0
//     builder.create<AffineStoreOp>(
//         seq.loc, zero_f32, outputMemref, outputIndices);
    
//     // 内层循环：遍历需要 reduce 的维度
//     auto reduceLoop = builder.create<AffineForOp>(seq.loc, 0, reduceSize);
//     builder.setInsertionPointToStart(reduceLoop.getBody());
//     Value reduceIV = reduceLoop.getInductionVar();
    
//     // 构建输入的索引
//     std::vector<Value> inputIndices;
//     outerIVIndex = 0;
//     for (size_t dim = 0; dim < seq.inputShape.size(); ++dim) {
//       if (dim == seq.reduceAxis) {
//         inputIndices.push_back(reduceIV);
//       } else {
//         inputIndices.push_back(outerIVs[outerIVIndex++]);
//       }
//     }
    
//     // 加载输入值
//     Value inputVal = builder.create<AffineLoadOp>(
//         seq.loc, inputMemref, inputIndices);
    
//     // 加载当前累加值
//     Value accumVal = builder.create<AffineLoadOp>(
//         seq.loc, outputMemref, outputIndices);
    
//     // 累加
//     Value sum = builder.create<arith::AddFOp>(seq.loc, accumVal, inputVal);
    
//     // 存储累加结果
//     builder.create<AffineStoreOp>(
//         seq.loc, sum, outputMemref, outputIndices);
    
//     // 在 reduce 循环之后，计算 mean (sum / count)
//     builder.setInsertionPointAfter(reduceLoop);
    
//     Value finalSum = builder.create<AffineLoadOp>(
//         seq.loc, outputMemref, outputIndices);
//     Value mean = builder.create<arith::DivFOp>(seq.loc, finalSum, reduceSize_f32);
//     builder.create<AffineStoreOp>(
//         seq.loc, mean, outputMemref, outputIndices);
    
//     // 删除原始的操作序列
//     eraseSequence(seq);
    
//     LLVM_DEBUG(llvm::dbgs() << "  Successfully converted to affine loop\n");
    
//     return success();
//   }

//   // 删除原始的操作序列
//   void eraseSequence(ReduceMeanCallSequence& seq) {
//     // 按照相反的顺序删除操作,避免use-after-free
    
//     // 删除 stream 相关操作
//     if (seq.streamDestroy) seq.streamDestroy.erase();
//     if (seq.streamSync) seq.streamSync.erase();
//     seq.reduceMeanCall.erase();
//     if (seq.createHandles) seq.createHandles.erase();
//     seq.streamCreate.erase();
    
//     // 删除输出指针转换链
//     seq.outputIntToPtr.erase();
//     seq.outputIndexCast.erase();
//     seq.outputExtract.erase();
    
//     // 删除输入指针转换链
//     seq.inputIntToPtr.erase();
//     seq.inputIndexCast.erase();
//     seq.inputExtract.erase();
    
//     // 注意: 不删除 inputMemref 和 outputAlloc,
//     // 因为它们在 affine loop 中还会使用
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {

// std::unique_ptr<Pass> createConvertReduceMeanToAffinePass() {
//   return std::make_unique<ConvertReduceMeanToAffinePass>();
// }

// } // 

// // Pass 注册
// static mlir::PassRegistration<ConvertReduceMeanToAffinePass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "convert-reduce-mean-to-affine"

namespace {

  // 存储一个 ReduceMean 调用序列的完整信息
struct ReduceMeanCallSequence {
  // 输入相关操作
  Value inputMemref;
  memref::ExtractAlignedPointerAsIndexOp inputExtract;
  arith::IndexCastOp inputIndexCast;
  LLVM::IntToPtrOp inputIntToPtr;
  
  // 输出相关操作
  memref::AllocOp outputAlloc;
  memref::ExtractAlignedPointerAsIndexOp outputExtract;
  arith::IndexCastOp outputIndexCast;
  LLVM::IntToPtrOp outputIntToPtr;
  
  // Stream 相关操作
  func::CallOp streamCreate;
  func::CallOp createHandles;
  func::CallOp reduceMeanCall;
  func::CallOp streamSync;
  func::CallOp streamDestroy;
  
  // 从 memref 类型中提取的维度信息
  std::vector<int64_t> inputShape;
  std::vector<int64_t> outputShape;
  int reduceAxis;  // 被reduce的维度索引
  
  Location loc;
  ReduceMeanCallSequence(Location l) : loc(l) {}
};

class ConvertReduceMeanToAffinePass
    : public PassWrapper<ConvertReduceMeanToAffinePass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { 
    return "convert-reduce-mean-to-affine"; 
  }
  
  StringRef getDescription() const final {
    return "Convert mgpuCudnnReduceMean calls to affine loop implementations";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Convert ReduceMean to Affine Pass ===\n");
    
    // 收集所有需要转换的 reduce mean 调用序列
    std::vector<ReduceMeanCallSequence> sequences;
    if (failed(collectReduceMeanSequences(funcOp, sequences))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect reduce mean sequences\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << sequences.size() 
               << " reduce mean call sequences\n");
    
    // 转换每个序列
    for (auto& seq : sequences) {
      if (failed(convertSequenceToAffineLoop(seq))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert a sequence\n");
        signalPassFailure();
        return;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted all sequences\n");
  }

private:
  // 收集所有 ReduceMean 调用序列
  LogicalResult collectReduceMeanSequences(
      func::FuncOp funcOp, 
      std::vector<ReduceMeanCallSequence>& sequences) {
    
    funcOp.walk([&](func::CallOp callOp) {
      // 查找 mgpuCudnnReduceMean 调用
      // 修复：正确获取 callee 名称
      if (callOp.getCallee() == "mgpuCudnnReduceMean") {
        LLVM_DEBUG(llvm::dbgs() << "Found mgpuCudnnReduceMean call\n");
        ReduceMeanCallSequence seq(callOp.getLoc());
        if (succeeded(extractSequenceInfo(callOp, seq))) {
          sequences.push_back(seq);
        }
      }
    });
    
    return success();
  }

  // 从 ReduceMean 调用中提取完整的操作序列信息
  LogicalResult extractSequenceInfo(
      func::CallOp reduceMeanCall, 
      ReduceMeanCallSequence& seq) {
    
    seq.reduceMeanCall = reduceMeanCall;
    seq.loc = reduceMeanCall.getLoc();
    
    auto operands = reduceMeanCall.getOperands();
    if (operands.size() < 15) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid number of operands: " 
                 << operands.size() << "\n");
      return failure();
    }
    
    // 获取输入和输出指针以及 stream
    Value inputPtr = operands[12];  // 第13个参数是输入指针
    Value outputPtr = operands[13]; // 第14个参数是输出指针
    Value streamPtr = operands[14]; // 第15个参数是stream
    
    // 回溯操作链
    if (failed(traceInputChain(inputPtr, seq)) ||
        failed(traceOutputChain(outputPtr, seq)) ||
        failed(traceStreamOps(streamPtr, reduceMeanCall, seq))) {
      return failure();
    }
    
    // 从 memref 类型中提取维度信息
    if (failed(extractDimensionsFromMemref(seq))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to extract dimensions from memref\n");
      return failure();
    }
    
    return success();
  }

  // 从 memref 类型中提取维度信息
  LogicalResult extractDimensionsFromMemref(ReduceMeanCallSequence& seq) {
    // 获取输入 memref 的类型
    auto inputMemrefType = seq.inputMemref.getType().dyn_cast<MemRefType>();
    if (!inputMemrefType) {
      LLVM_DEBUG(llvm::dbgs() << "Input is not a memref type\n");
      return failure();
    }
    
    // 获取输出 memref 的类型
    auto outputMemrefType = seq.outputAlloc.getType().dyn_cast<MemRefType>();
    if (!outputMemrefType) {
      LLVM_DEBUG(llvm::dbgs() << "Output is not a memref type\n");
      return failure();
    }
    
    // 检查维度必须是静态的
    if (!inputMemrefType.hasStaticShape() || !outputMemrefType.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Dynamic shapes not supported\n");
      return failure();
    }
    
    // 提取输入维度
    ArrayRef<int64_t> inputShapeRef = inputMemrefType.getShape();
    seq.inputShape.assign(inputShapeRef.begin(), inputShapeRef.end());
    
    // 提取输出维度
    ArrayRef<int64_t> outputShapeRef = outputMemrefType.getShape();
    seq.outputShape.assign(outputShapeRef.begin(), outputShapeRef.end());
    
    // 检查维度数量是否匹配
    if (seq.inputShape.size() != seq.outputShape.size()) {
      LLVM_DEBUG(llvm::dbgs() << "Input and output rank mismatch\n");
      return failure();
    }
    
    // 找出哪个维度被 reduce 了（输出维度为1）
    seq.reduceAxis = -1;
    for (size_t i = 0; i < seq.inputShape.size(); ++i) {
      if (seq.outputShape[i] == 1 && seq.inputShape[i] != 1) {
        if (seq.reduceAxis != -1) {
          LLVM_DEBUG(llvm::dbgs() << "Multiple reduce axes not supported\n");
          return failure();
        }
        seq.reduceAxis = i;
      } else if (seq.inputShape[i] != seq.outputShape[i]) {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported dimension change\n");
        return failure();
      }
    }
    
    if (seq.reduceAxis == -1) {
      LLVM_DEBUG(llvm::dbgs() << "Could not identify reduce axis\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Extracted dimensions:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Input shape: [");
    for (size_t i = 0; i < seq.inputShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << seq.inputShape[i]);
      if (i < seq.inputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "  Output shape: [");
    for (size_t i = 0; i < seq.outputShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << seq.outputShape[i]);
      if (i < seq.outputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "  Reduce axis: " << seq.reduceAxis << "\n");
    
    return success();
  }

  // 从常量操作中提取整数值
  LogicalResult extractIntFromConstant(Value val, int& result) {
    if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
        result = intAttr.getInt();
        return success();
      }
    }
    return failure();
  }

  // 回溯输入操作链: memref -> extract -> index_cast -> inttoptr
  LogicalResult traceInputChain(Value inputPtr, ReduceMeanCallSequence& seq) {
    // inttoptr
    auto intToPtrOp = inputPtr.getDefiningOp<LLVM::IntToPtrOp>();
    if (!intToPtrOp) {
      LLVM_DEBUG(llvm::dbgs() << "Input: Failed to find IntToPtrOp\n");
      return failure();
    }
    seq.inputIntToPtr = intToPtrOp;
    
    // index_cast
    auto indexCastOp = intToPtrOp.getOperand().getDefiningOp<arith::IndexCastOp>();
    if (!indexCastOp) {
      LLVM_DEBUG(llvm::dbgs() << "Input: Failed to find IndexCastOp\n");
      return failure();
    }
    seq.inputIndexCast = indexCastOp;
    
    // extract_aligned_pointer
    auto extractOp = indexCastOp.getOperand()
        .getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
    if (!extractOp) {
      LLVM_DEBUG(llvm::dbgs() << "Input: Failed to find ExtractAlignedPointerOp\n");
      return failure();
    }
    seq.inputExtract = extractOp;
    
    // memref (可能来自 alloc 或函数参数)
    seq.inputMemref = extractOp.getOperand();
    
    LLVM_DEBUG(llvm::dbgs() << "Input chain traced successfully\n");
    return success();
  }

  // 回溯输出操作链: alloc -> extract -> index_cast -> inttoptr
  LogicalResult traceOutputChain(Value outputPtr, ReduceMeanCallSequence& seq) {
    // inttoptr
    auto intToPtrOp = outputPtr.getDefiningOp<LLVM::IntToPtrOp>();
    if (!intToPtrOp) {
      LLVM_DEBUG(llvm::dbgs() << "Output: Failed to find IntToPtrOp\n");
      return failure();
    }
    seq.outputIntToPtr = intToPtrOp;
    
    // index_cast
    auto indexCastOp = intToPtrOp.getOperand().getDefiningOp<arith::IndexCastOp>();
    if (!indexCastOp) {
      LLVM_DEBUG(llvm::dbgs() << "Output: Failed to find IndexCastOp\n");
      return failure();
    }
    seq.outputIndexCast = indexCastOp;
    
    // extract_aligned_pointer
    auto extractOp = indexCastOp.getOperand()
        .getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
    if (!extractOp) {
      LLVM_DEBUG(llvm::dbgs() << "Output: Failed to find ExtractAlignedPointerOp\n");
      return failure();
    }
    seq.outputExtract = extractOp;
    
    // alloc
    auto allocOp = extractOp.getOperand().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      LLVM_DEBUG(llvm::dbgs() << "Output: Failed to find AllocOp\n");
      return failure();
    }
    seq.outputAlloc = allocOp;
    
    LLVM_DEBUG(llvm::dbgs() << "Output chain traced successfully\n");
    return success();
  }

  // 回溯 stream 相关操作
  LogicalResult traceStreamOps(
      Value streamPtr, 
      func::CallOp reduceMeanCall,
      ReduceMeanCallSequence& seq) {
    
    // stream 来自 mgpuStreamCreate
    auto streamCreateOp = streamPtr.getDefiningOp<func::CallOp>();
    if (!streamCreateOp) {
      LLVM_DEBUG(llvm::dbgs() << "Stream: Failed to find CallOp for stream\n");
      return failure();
    }
    
    // 修复：正确检查 callee 名称
    if (streamCreateOp.getCallee() != "mgpuStreamCreate") {
      LLVM_DEBUG(llvm::dbgs() << "Stream: CallOp is not mgpuStreamCreate, got: " 
                 << streamCreateOp.getCallee() << "\n");
      return failure();
    }
    seq.streamCreate = streamCreateOp;
    
    LLVM_DEBUG(llvm::dbgs() << "Found mgpuStreamCreate\n");
    
    // 向后查找 createHandles, streamSync, streamDestroy
    Operation* nextOp = streamCreateOp.getOperation()->getNextNode();
    
    // mgpuCreateHandlesForStream
    if (auto callOp = dyn_cast_or_null<func::CallOp>(nextOp)) {
      // 修复：正确检查 callee 名称
      if (callOp.getCallee() == "mgpuCreateHandlesForStream") {
        seq.createHandles = callOp;
        LLVM_DEBUG(llvm::dbgs() << "Found mgpuCreateHandlesForStream\n");
        nextOp = nextOp->getNextNode();
      }
    }
    
    // 跳过 reduceMeanCall (已经有了)
    if (nextOp == reduceMeanCall.getOperation()) {
      nextOp = nextOp->getNextNode();
    }
    
    // mgpuStreamSynchronize
    if (auto callOp = dyn_cast_or_null<func::CallOp>(nextOp)) {
      // 修复：正确检查 callee 名称
      if (callOp.getCallee() == "mgpuStreamSynchronize") {
        seq.streamSync = callOp;
        LLVM_DEBUG(llvm::dbgs() << "Found mgpuStreamSynchronize\n");
        nextOp = nextOp->getNextNode();
      }
    }
    
    // mgpuStreamDestroy
    if (auto callOp = dyn_cast_or_null<func::CallOp>(nextOp)) {
      // 修复：正确检查 callee 名称
      if (callOp.getCallee() == "mgpuStreamDestroy") {
        seq.streamDestroy = callOp;
        LLVM_DEBUG(llvm::dbgs() << "Found mgpuStreamDestroy\n");
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Stream ops traced successfully\n");
    return success();
  }

  // 将序列转换为 affine loop
  LogicalResult convertSequenceToAffineLoop(ReduceMeanCallSequence& seq) {
    LLVM_DEBUG(llvm::dbgs() << "\nConverting sequence:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Input shape: [");
    for (size_t i = 0; i < seq.inputShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << seq.inputShape[i]);
      if (i < seq.inputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "  Output shape: [");
    for (size_t i = 0; i < seq.outputShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << seq.outputShape[i]);
      if (i < seq.outputShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "  Reduce axis: " << seq.reduceAxis << "\n");
    
    // 在 outputAlloc 之后插入 affine loop
    OpBuilder builder(seq.outputAlloc.getOperation()->getNextNode());
    
    // 创建常量
    Value zero_f32 = builder.create<arith::ConstantOp>(
        seq.loc, builder.getF32Type(), builder.getF32FloatAttr(0.0));
    
    // reduce 维度的大小（用于计算平均值）
    int64_t reduceSize = seq.inputShape[seq.reduceAxis];
    Value reduceSize_f32 = builder.create<arith::ConstantOp>(
        seq.loc, builder.getF32Type(), 
        builder.getF32FloatAttr(static_cast<float>(reduceSize)));
    
    Value inputMemref = seq.inputMemref;
    Value outputMemref = seq.outputAlloc.getResult();
    
    // 创建外层循环（遍历所有非reduce维度）
    std::vector<AffineForOp> outerLoops;
    std::vector<Value> outerIVs;
    
    for (size_t dim = 0; dim < seq.outputShape.size(); ++dim) {
      if (dim != seq.reduceAxis) {
        auto forOp = builder.create<AffineForOp>(seq.loc, 0, seq.outputShape[dim]);
        builder.setInsertionPointToStart(forOp.getBody());
        outerLoops.push_back(forOp);
        outerIVs.push_back(forOp.getInductionVar());
      }
    }
    
    // 构建输出的索引（对于reduce维度使用常量0）
    std::vector<Value> outputIndices;
    Value c0 = builder.create<arith::ConstantIndexOp>(seq.loc, 0);
    
    size_t outerIVIndex = 0;
    for (size_t dim = 0; dim < seq.outputShape.size(); ++dim) {
      if (dim == seq.reduceAxis) {
        outputIndices.push_back(c0);
      } else {
        outputIndices.push_back(outerIVs[outerIVIndex++]);
      }
    }
    
    // 初始化累加器为 0
    builder.create<AffineStoreOp>(
        seq.loc, zero_f32, outputMemref, outputIndices);
    
    // 内层循环：遍历需要 reduce 的维度
    auto reduceLoop = builder.create<AffineForOp>(seq.loc, 0, reduceSize);
    builder.setInsertionPointToStart(reduceLoop.getBody());
    Value reduceIV = reduceLoop.getInductionVar();
    
    // 构建输入的索引
    std::vector<Value> inputIndices;
    outerIVIndex = 0;
    for (size_t dim = 0; dim < seq.inputShape.size(); ++dim) {
      if (dim == seq.reduceAxis) {
        inputIndices.push_back(reduceIV);
      } else {
        inputIndices.push_back(outerIVs[outerIVIndex++]);
      }
    }
    
    // 加载输入值
    Value inputVal = builder.create<AffineLoadOp>(
        seq.loc, inputMemref, inputIndices);
    
    // 加载当前累加值
    Value accumVal = builder.create<AffineLoadOp>(
        seq.loc, outputMemref, outputIndices);
    
    // 累加
    Value sum = builder.create<arith::AddFOp>(seq.loc, accumVal, inputVal);
    
    // 存储累加结果
    builder.create<AffineStoreOp>(
        seq.loc, sum, outputMemref, outputIndices);
    
    // 在 reduce 循环之后，计算 mean (sum / count)
    builder.setInsertionPointAfter(reduceLoop);
    
    Value finalSum = builder.create<AffineLoadOp>(
        seq.loc, outputMemref, outputIndices);
    Value mean = builder.create<arith::DivFOp>(seq.loc, finalSum, reduceSize_f32);
    builder.create<AffineStoreOp>(
        seq.loc, mean, outputMemref, outputIndices);
    
    // 删除原始的操作序列
    eraseSequence(seq);
    
    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted to affine loop\n");
    
    return success();
  }

  // 删除原始的操作序列
  void eraseSequence(ReduceMeanCallSequence& seq) {
    // 按照相反的顺序删除操作,避免use-after-free
    
    // 删除 stream 相关操作
    if (seq.streamDestroy) seq.streamDestroy.erase();
    if (seq.streamSync) seq.streamSync.erase();
    seq.reduceMeanCall.erase();
    if (seq.createHandles) seq.createHandles.erase();
    seq.streamCreate.erase();
    
    // 删除输出指针转换链
    seq.outputIntToPtr.erase();
    seq.outputIndexCast.erase();
    seq.outputExtract.erase();
    
    // 删除输入指针转换链
    seq.inputIntToPtr.erase();
    seq.inputIndexCast.erase();
    seq.inputExtract.erase();
    
    // 注意: 不删除 inputMemref 和 outputAlloc,
    // 因为它们在 affine loop 中还会使用
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createConvertReduceMeanToAffinePass() {
  return std::make_unique<ConvertReduceMeanToAffinePass>();
}

}

// Pass 注册
static mlir::PassRegistration<ConvertReduceMeanToAffinePass> pass;