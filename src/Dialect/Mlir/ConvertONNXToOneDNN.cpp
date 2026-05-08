#include "mlir/Pass/Pass.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/APFloat.h"

// Include ONNX dialect headers
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "onnx-to-onednn"

namespace {

// Base class for oneDNN lowering patterns
class ONNXToOneDNNPatternBase {
protected:
  // Helper function: Get or create function declaration
  func::FuncOp getOrCreateFunction(ModuleOp moduleOp, PatternRewriter &rewriter, 
      Location loc, StringRef name, FunctionType funcType) const {
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(name);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      funcOp = rewriter.create<func::FuncOp>(loc, name, funcType);
      funcOp.setPrivate();
    }
    return funcOp;
  }
};

// Pattern to convert onnx.Conv to mgpuOneDnnConv2dForward
class ConvOpLowering : public OpRewritePattern<mlir::ONNXConvOp>, 
                       public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXConvOp convOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Conv to oneDNN at " << loc << "\n");

    // Get input, weight, and bias tensors
    Value input = convOp.getX();
    Value weights = convOp.getW();
    Value bias = convOp.getB();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, "Input must have static shape");
    }
    
    // Get weight type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, "Weights must have static shape");
    }
    
    // Extract input dimensions (N, C, H, W)
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(convOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // Extract kernel dimensions (K, C, R, S)
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 4) {
      return rewriter.notifyMatchFailure(convOp, "Weights must be 4D tensor");
    }
    int64_t k = weightShape[0];
    int64_t r = weightShape[2];
    int64_t s = weightShape[3];
    
    // Extract convolution parameters
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1};
    
    if (auto dilationsAttr = convOp.getDilations()) {
      dilations.clear();
      for (auto attr : dilationsAttr.value()) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
    
    if (auto padsAttr = convOp.getPads()) {
      pads.clear();
      for (auto attr : padsAttr.value()) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
    
    if (auto stridesAttr = convOp.getStrides()) {
      strides.clear();
      for (auto attr : stridesAttr.value()) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
    
    // Validate parameter dimensions
    if (dilations.size() < 2 || pads.size() < 4 || strides.size() < 2) {
      return rewriter.notifyMatchFailure(convOp, "Invalid convolution parameters");
    }
    
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    
    LLVM_DEBUG(llvm::dbgs() << "Conv params: dilation=" << dilation_h << "," << dilation_w 
               << " pad=" << pad_h << "," << pad_w 
               << " stride=" << stride_h << "," << stride_w << "\n");
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kValue = createI32Const(k);
    auto rValue = createI32Const(r);
    auto sValue = createI32Const(s);
    auto padHValue = createI32Const(pad_h);
    auto padWValue = createI32Const(pad_w);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    auto weightMemref = markForBufferization(weights);
    Value biasMemref;
    if (bias)
      biasMemref = markForBufferization(bias);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    Value biasPtr;
    if (bias)
      biasPtr = getPtr(biasMemref);
    else
      biasPtr = create.llvm.null(ptrType);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(convOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create or get function declaration
    auto moduleOp = convOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnConv2dForward";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type, i32Type,           // k, r, s
        i32Type, i32Type,                    // pad_h, pad_w
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        ptrType, ptrType, ptrType,           // x_data, w_data, bias_data
        ptrType                              // y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kValue, rValue, sValue,
      padHValue, padWValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      inputPtr, weightPtr, biasPtr,
      outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(convOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Conv to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.MatMul to mgpuOneDnnMatMul
class MatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, 
                         public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = matMulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul to oneDNN at " << loc << "\n");

    // Get input tensors
    Value inputA = matMulOp.getA();
    Value inputB = matMulOp.getB();
    
    // Get input types
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || 
        !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
    }
    
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // MatMul needs at least 2D tensors
    if (inputShapeA.size() < 2 || inputShapeB.size() < 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must be at least 2D");
    }
    
    // Determine batch size, m, k, n dimensions
    int64_t batch = 1;
    int64_t m, k, n;
    
    if (inputShapeA.size() == 2 && inputShapeB.size() == 2) {
      // Standard 2D matrix multiplication
      m = inputShapeA[0];
      k = inputShapeA[1];
      n = inputShapeB[1];
      
      if (k != inputShapeB[0]) {
        return rewriter.notifyMatchFailure(matMulOp, 
          "Inner dimensions must match for matrix multiplication");
      }
    } else if (inputShapeA.size() == 3 && inputShapeB.size() == 3) {
      // Batched matrix multiplication
      batch = inputShapeA[0];
      m = inputShapeA[1];
      k = inputShapeA[2];
      n = inputShapeB[2];
      
      if (batch != inputShapeB[0] || k != inputShapeB[1]) {
        return rewriter.notifyMatchFailure(matMulOp, 
          "Incompatible dimensions for batched matmul");
      }
    } else {
      return rewriter.notifyMatchFailure(matMulOp, 
        "Only 2D or 3D tensors supported");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "MatMul dimensions: batch=" << batch 
               << ", m=" << m << ", k=" << k << ", n=" << n << "\n");
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto batchValue = createI32Const(batch);
    auto mValue = createI32Const(m);
    auto kValue = createI32Const(k);
    auto nValue = createI32Const(n);
    auto transposeAValue = createI1Const(false);  // No transpose for MatMul
    auto transposeBValue = createI1Const(false);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create or get function declaration
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnMatMul";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch, m, k, n
        i1Type, i1Type,                      // transpose_a, transpose_b
        ptrType, ptrType, ptrType            // a_data, b_data, c_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchValue, mValue, kValue, nValue,
      transposeAValue, transposeBValue,
      inputPtrA, inputPtrB, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MatMul to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.MaxPoolSingleOut to mgpuOneDnnMaxPool2d
class MaxPoolOpLowering : public OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>, 
                          public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp maxPoolOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = maxPoolOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MaxPoolSingleOut to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = maxPoolOp.getX();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must have static shape");
    }
    
    // Extract input dimensions (NCHW)
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // Extract attributes
    std::vector<int64_t> kernelShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    
    if (auto kernelShapeAttr = maxPoolOp.getKernelShapeAttr()) {
      for (auto attr : kernelShapeAttr) {
        kernelShape.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(maxPoolOp, "Kernel shape required");
    }
    
    if (auto padsAttr = maxPoolOp.getPadsAttr()) {
      for (auto attr : padsAttr) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      pads = std::vector<int64_t>(kernelShape.size() * 2, 0);
    }
    
    if (auto stridesAttr = maxPoolOp.getStridesAttr()) {
      for (auto attr : stridesAttr) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      strides = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (auto dilationsAttr = maxPoolOp.getDilationsAttr()) {
      for (auto attr : dilationsAttr) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      dilations = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (kernelShape.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only 2D pooling supported");
    }
    
    int64_t kernel_h = kernelShape[0];
    int64_t kernel_w = kernelShape[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kernelHValue = createI32Const(kernel_h);
    auto kernelWValue = createI32Const(kernel_w);
    auto padHValue = createI32Const(pad_h);
    auto padWValue = createI32Const(pad_w);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(
      maxPoolOp.getO_Y().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = maxPoolOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnMaxPool2d";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // kernel_h, kernel_w
        i32Type, i32Type,                    // pad_h, pad_w
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kernelHValue, kernelWValue,
      padHValue, padWValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      inputPtr, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(maxPoolOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted MaxPool to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.AveragePool to mgpuOneDnnAvgPool2d
class AvgPoolOpLowering : public OpRewritePattern<mlir::ONNXAveragePoolOp>, 
                          public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXAveragePoolOp avgPoolOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = avgPoolOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.AveragePool to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = avgPoolOp.getX();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Input must have static shape");
    }
    
    // Extract input dimensions (NCHW)
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // Extract attributes
    std::vector<int64_t> kernelShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    
    if (auto kernelShapeAttr = avgPoolOp.getKernelShapeAttr()) {
      for (auto attr : kernelShapeAttr) {
        kernelShape.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(avgPoolOp, "Kernel shape required");
    }
    
    if (auto padsAttr = avgPoolOp.getPadsAttr()) {
      for (auto attr : padsAttr) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      pads = std::vector<int64_t>(kernelShape.size() * 2, 0);
    }
    
    if (auto stridesAttr = avgPoolOp.getStridesAttr()) {
      for (auto attr : stridesAttr) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      strides = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (auto dilationsAttr = avgPoolOp.getDilationsAttr()) {
      for (auto attr : dilationsAttr) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      dilations = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    // Get count_include_pad attribute
    int64_t countIncludePad = 0;
    if (auto countIncludePadAttr = avgPoolOp.getCountIncludePadAttr()) {
      countIncludePad = countIncludePadAttr.getValue().getSExtValue();
    }
    
    if (kernelShape.size() != 2) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Only 2D pooling supported");
    }
    
    int64_t kernel_h = kernelShape[0];
    int64_t kernel_w = kernelShape[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kernelHValue = createI32Const(kernel_h);
    auto kernelWValue = createI32Const(kernel_w);
    auto padHValue = createI32Const(pad_h);
    auto padWValue = createI32Const(pad_w);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    auto countIncludePadValue = createI1Const(countIncludePad != 0);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(avgPoolOp.getY().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = avgPoolOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnAvgPool2d";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // kernel_h, kernel_w
        i32Type, i32Type,                    // pad_h, pad_w
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        i1Type,                              // count_include_pad
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kernelHValue, kernelWValue,
      padHValue, padWValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      countIncludePadValue,
      inputPtr, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(avgPoolOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted AvgPool to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.ReduceMeanV13 to mgpuOneDnnReduceMean
class ReduceMeanOpLowering : public OpRewritePattern<mlir::ONNXReduceMeanV13Op>, 
                             public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXReduceMeanV13Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXReduceMeanV13Op reduceMeanOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = reduceMeanOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.ReduceMeanV13 to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = reduceMeanOp.getData();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(reduceMeanOp, "Input must have static shape");
    }
    
    // Extract input dimensions (assume NCHW for 4D)
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(reduceMeanOp, 
        "Currently only support 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // Get axes attribute
    std::vector<int64_t> axes;
    if (auto axesAttr = reduceMeanOp.getAxesAttr()) {
      for (auto attr : axesAttr) {
        axes.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(reduceMeanOp, "Axes attribute required");
    }
    
    // Get keepdims attribute
    int64_t keepdims = reduceMeanOp.getKeepdims();
    
    // Check if axes are [2, 3] (H and W dimensions)
    // This is the most common case for Global Average Pooling
    bool reduceH = false;
    bool reduceW = false;
    for (auto axis : axes) {
      if (axis == 2) reduceH = true;
      if (axis == 3) reduceW = true;
    }
    
    if (!reduceH || !reduceW) {
      return rewriter.notifyMatchFailure(reduceMeanOp, 
        "Currently only support reduce on axes [2, 3] (H and W dimensions)");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "ReduceMean params: input=[" << n << "," << c 
               << "," << h << "," << w << "], axes=[2,3], keepdims=" 
               << keepdims << "\n");
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto axisHValue = createI32Const(reduceH ? 1 : 0);
    auto axisWValue = createI32Const(reduceW ? 1 : 0);
    auto keepdimsValue = createI1Const(keepdims != 0);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(
      reduceMeanOp.getReduced().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = reduceMeanOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnReduceMean";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // axis_h, axis_w
        i1Type,                              // keepdims
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      axisHValue, axisWValue,
      keepdimsValue,
      inputPtr, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(reduceMeanOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted ReduceMean to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.MatMul (batch cases) to mgpuOneDnnBatchMatMul
class BatchMatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, 
                              public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = matMulOp.getLoc();

    // Get input tensors
    Value inputA = matMulOp.getA();
    Value inputB = matMulOp.getB();
    
    // Get input types
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || 
        !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
    }
    
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // This pattern handles mixed 2D/3D cases (3D x 2D or 2D x 3D)
    // Pure 2D x 2D is handled by MatMulOpLowering
    // Pure 3D x 3D is also handled by MatMulOpLowering
    bool is_a_3d = (inputShapeA.size() == 3);
    bool is_b_3d = (inputShapeB.size() == 3);
    
    // Only handle mixed dimensionality cases
    if ((is_a_3d && is_b_3d) || (!is_a_3d && !is_b_3d)) {
      return rewriter.notifyMatchFailure(matMulOp, 
        "This pattern only handles mixed 2D/3D batch matmul");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul (batch) to oneDNN at " 
               << loc << "\n");
    
    // Determine batch, m, k, n dimensions
    int64_t batch, m, k, n;
    
    if (is_a_3d && !is_b_3d) {
      // 3D x 2D: (batch, m, k) x (k, n) -> (batch, m, n)
      if (inputShapeA.size() != 3 || inputShapeB.size() != 2) {
        return rewriter.notifyMatchFailure(matMulOp, "Invalid dimensions for 3D x 2D");
      }
      batch = inputShapeA[0];
      m = inputShapeA[1];
      k = inputShapeA[2];
      n = inputShapeB[1];
      
      if (k != inputShapeB[0]) {
        return rewriter.notifyMatchFailure(matMulOp, 
          "Inner dimensions must match: A[batch,m,k] x B[k,n]");
      }
    } else if (!is_a_3d && is_b_3d) {
      // 2D x 3D: (m, k) x (batch, k, n) -> (batch, m, n)
      if (inputShapeA.size() != 2 || inputShapeB.size() != 3) {
        return rewriter.notifyMatchFailure(matMulOp, "Invalid dimensions for 2D x 3D");
      }
      batch = inputShapeB[0];
      m = inputShapeA[0];
      k = inputShapeA[1];
      n = inputShapeB[2];
      
      if (k != inputShapeB[1]) {
        return rewriter.notifyMatchFailure(matMulOp, 
          "Inner dimensions must match: A[m,k] x B[batch,k,n]");
      }
    } else {
      return rewriter.notifyMatchFailure(matMulOp, "Unsupported dimension combination");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "BatchMatMul dimensions: batch=" << batch 
               << ", m=" << m << ", k=" << k << ", n=" << n 
               << " (A:" << (is_a_3d ? "3D" : "2D") 
               << ", B:" << (is_b_3d ? "3D" : "2D") << ")\n");
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto batchValue = createI32Const(batch);
    auto mValue = createI32Const(m);
    auto kValue = createI32Const(k);
    auto nValue = createI32Const(n);
    auto isA3dValue = createI1Const(is_a_3d);
    auto isB3dValue = createI1Const(is_b_3d);
    auto transposeAValue = createI1Const(false);  // No transpose for MatMul
    auto transposeBValue = createI1Const(false);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create or get function declaration
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnBatchMatMul";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch, m, k, n
        i1Type, i1Type,                      // is_a_3d, is_b_3d
        i1Type, i1Type,                      // transpose_a, transpose_b
        ptrType, ptrType, ptrType            // a_data, b_data, c_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchValue, mValue, kValue, nValue,
      isA3dValue, isB3dValue,
      transposeAValue, transposeBValue,
      inputPtrA, inputPtrB, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted batch MatMul to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.ReduceMeanV13 (general case) to mgpuOneDnnReduceMeanGeneral
class ReduceMeanGeneralOpLowering : public OpRewritePattern<mlir::ONNXReduceMeanV13Op>, 
                                     public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXReduceMeanV13Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXReduceMeanV13Op reduceMeanOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = reduceMeanOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.ReduceMeanV13 (general) to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = reduceMeanOp.getData();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(reduceMeanOp, "Input must have static shape");
    }
    
    // Extract input dimensions
    auto inputShape = inputType.getShape();
    int64_t ndims = inputShape.size();
    
    if (ndims < 2 || ndims > 5) {
      return rewriter.notifyMatchFailure(reduceMeanOp, 
        "Only support 2D-5D tensors");
    }
    
    // Get axes attribute
    std::vector<int64_t> axes;
    if (auto axesAttr = reduceMeanOp.getAxesAttr()) {
      for (auto attr : axesAttr) {
        int64_t axis = attr.cast<IntegerAttr>().getInt();
        // 归一化负数索引
        if (axis < 0) {
          axis = ndims + axis;
        }
        if (axis < 0 || axis >= ndims) {
          return rewriter.notifyMatchFailure(reduceMeanOp, 
            "Axis out of range");
        }
        axes.push_back(axis);
      }
    } else {
      return rewriter.notifyMatchFailure(reduceMeanOp, "Axes attribute required");
    }
    
    // Get keepdims attribute
    int64_t keepdims = reduceMeanOp.getKeepdims();
    
    LLVM_DEBUG({
      llvm::dbgs() << "ReduceMean params: input=[";
      for (int64_t i = 0; i < ndims; i++) {
        llvm::dbgs() << inputShape[i];
        if (i < ndims - 1) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "], axes=[";
      for (size_t i = 0; i < axes.size(); i++) {
        llvm::dbgs() << axes[i];
        if (i < axes.size() - 1) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "], keepdims=" << keepdims << "\n";
    });
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    auto indexType = rewriter.getIndexType();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto createIndexConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, indexType, 
                                               rewriter.getIndexAttr(value));
    };
    
    // Create ndims constant
    auto ndimsValue = createI32Const(ndims);
    
    // Create input_dims array
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // 分配临时数组用于存储input_dims
    auto inputDimsMemrefType = MemRefType::get({ndims}, i32Type);
    auto inputDimsMemref = rewriter.create<memref::AllocaOp>(loc, inputDimsMemrefType);
    
    for (int64_t i = 0; i < ndims; i++) {
      auto idx = createIndexConst(i);
      auto dimValue = createI32Const(inputShape[i]);
      rewriter.create<memref::StoreOp>(loc, dimValue, inputDimsMemref, ValueRange{idx});
    }
    
    // 获取input_dims数组指针
    auto inputDimsPtr = [&]() -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, inputDimsMemref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    }();
    
    // Create num_axes constant
    auto numAxesValue = createI32Const(axes.size());
    
    // Create axes array
    auto axesMemrefType = MemRefType::get({static_cast<int64_t>(axes.size())}, i32Type);
    auto axesMemref = rewriter.create<memref::AllocaOp>(loc, axesMemrefType);
    
    for (size_t i = 0; i < axes.size(); i++) {
      auto idx = createIndexConst(i);
      auto axisValue = createI32Const(axes[i]);
      rewriter.create<memref::StoreOp>(loc, axisValue, axesMemref, ValueRange{idx});
    }
    
    // 获取axes数组指针
    auto axesPtr = [&]() -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, axesMemref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    }();
    
    // Create keepdims constant
    auto keepdimsValue = createI1Const(keepdims != 0);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(
      reduceMeanOp.getReduced().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = reduceMeanOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnReduceMeanGeneral";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type,                             // ndims
        ptrType,                             // input_dims
        i32Type,                             // num_axes
        ptrType,                             // axes
        i1Type,                              // keepdims
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      ndimsValue,
      inputDimsPtr,
      numAxesValue,
      axesPtr,
      keepdimsValue,
      inputPtr,
      outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(reduceMeanOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted ReduceMean to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.MatMul (4D x 4D) to mgpuOneDnnBatchMatMul4D
class BatchMatMul4DOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, 
                                 public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = matMulOp.getLoc();

    // Get input tensors
    Value inputA = matMulOp.getA();
    Value inputB = matMulOp.getB();
    
    // Get input types
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || 
        !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
    }
    
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // This pattern only handles 4D x 4D case (for Transformer attention)
    if (inputShapeA.size() != 4 || inputShapeB.size() != 4) {
      return rewriter.notifyMatchFailure(matMulOp, 
        "This pattern only handles 4D x 4D batch matmul");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul (4D x 4D) to oneDNN at " 
               << loc << "\n");
    
    // Extract dimensions
    // A: [dim0, dim1, m, k]
    // B: [dim0, dim1, k, n]
    // C: [dim0, dim1, m, n]
    int64_t dim0_a = inputShapeA[0];
    int64_t dim1_a = inputShapeA[1];
    int64_t m = inputShapeA[2];
    int64_t k_a = inputShapeA[3];
    
    int64_t dim0_b = inputShapeB[0];
    int64_t dim1_b = inputShapeB[1];
    int64_t k_b = inputShapeB[2];
    int64_t n = inputShapeB[3];
    
    // Check that batch dimensions match
    if (dim0_a != dim0_b || dim1_a != dim1_b) {
      return rewriter.notifyMatchFailure(matMulOp, 
        "Batch dimensions must match for 4D x 4D matmul");
    }
    
    // Check that inner dimensions match
    if (k_a != k_b) {
      return rewriter.notifyMatchFailure(matMulOp, 
        "Inner dimensions must match: A[..., m, k] x B[..., k, n]");
    }
    
    int64_t dim0 = dim0_a;
    int64_t dim1 = dim1_a;
    int64_t k = k_a;
    
    LLVM_DEBUG(llvm::dbgs() << "BatchMatMul4D dimensions: [" << dim0 << "," << dim1 
               << "," << m << "," << k << "] x [" << dim0 << "," << dim1 
               << "," << k << "," << n << "] -> [" << dim0 << "," << dim1 
               << "," << m << "," << n << "]\n");
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto dim0Value = createI32Const(dim0);
    auto dim1Value = createI32Const(dim1);
    auto mValue = createI32Const(m);
    auto kValue = createI32Const(k);
    auto nValue = createI32Const(n);
    auto transposeAValue = createI1Const(false);  // No transpose for MatMul
    auto transposeBValue = createI1Const(false);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create or get function declaration
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnBatchMatMul4D";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type, i32Type,  // dim0, dim1, m, k, n
        i1Type, i1Type,                                // transpose_a, transpose_b
        ptrType, ptrType, ptrType                      // a_data, b_data, c_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      dim0Value, dim1Value, mValue, kValue, nValue,
      transposeAValue, transposeBValue,
      inputPtrA, inputPtrB, outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted 4D x 4D MatMul to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.Softmax to mgpuOneDnnSoftmax
class SoftmaxOpLowering : public OpRewritePattern<mlir::ONNXSoftmaxOp>, 
                          public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXSoftmaxOp softmaxOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = softmaxOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Softmax to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = softmaxOp.getInput();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(softmaxOp, "Input must have static shape");
    }
    
    // Extract input dimensions
    auto inputShape = inputType.getShape();
    int64_t ndims = inputShape.size();
    
    if (ndims < 1 || ndims > 6) {
      return rewriter.notifyMatchFailure(softmaxOp, 
        "Only support 1D-6D tensors");
    }
    
    // Get axis attribute
    int64_t axis = softmaxOp.getAxis();
    
    // 归一化负数索引
    if (axis < 0) {
      axis = ndims + axis;
    }
    
    if (axis < 0 || axis >= ndims) {
      return rewriter.notifyMatchFailure(softmaxOp, "Axis out of range");
    }
    
    LLVM_DEBUG({
      llvm::dbgs() << "Softmax params: input=[";
      for (int64_t i = 0; i < ndims; i++) {
        llvm::dbgs() << inputShape[i];
        if (i < ndims - 1) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "], axis=" << axis << "\n";
    });
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto indexType = rewriter.getIndexType();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createIndexConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, indexType, 
                                               rewriter.getIndexAttr(value));
    };
    
    // Create ndims constant
    auto ndimsValue = createI32Const(ndims);
    
    // Create dims array
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // 分配临时数组用于存储dims
    auto dimsMemrefType = MemRefType::get({ndims}, i32Type);
    auto dimsMemref = rewriter.create<memref::AllocaOp>(loc, dimsMemrefType);
    
    for (int64_t i = 0; i < ndims; i++) {
      auto idx = createIndexConst(i);
      auto dimValue = createI32Const(inputShape[i]);
      rewriter.create<memref::StoreOp>(loc, dimValue, dimsMemref, ValueRange{idx});
    }
    
    // 获取dims数组指针
    auto dimsPtr = [&]() -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, dimsMemref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    }();
    
    // Create axis constant
    auto axisValue = createI32Const(axis);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(softmaxOp.getOutput().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = softmaxOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnSoftmax";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type,                             // ndims
        ptrType,                             // dims
        i32Type,                             // axis
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      ndimsValue,
      dimsPtr,
      axisValue,
      inputPtr,
      outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(softmaxOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted Softmax to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.Gemm to mgpuOneDnnGemm
class GemmOpLowering : public OpRewritePattern<mlir::ONNXGemmOp>, 
                       public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp gemmOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = gemmOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Gemm to oneDNN at " << loc << "\n");

    // Get input tensors
    Value inputA = gemmOp.getA();
    Value inputB = gemmOp.getB();
    Value inputC = gemmOp.getC();  // bias
    
    // Get input types
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || 
        !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gemmOp, "Inputs must have static shapes");
    }
    
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // Gemm only supports 2D matrices
    if (inputShapeA.size() != 2 || inputShapeB.size() != 2) {
      return rewriter.notifyMatchFailure(gemmOp, "Gemm only supports 2D matrices");
    }
    
    // Get attributes
    auto transAAttr = gemmOp.getTransAAttr();
    auto transBAttr = gemmOp.getTransBAttr();
    auto alphaAttr = gemmOp.getAlphaAttr();
    auto betaAttr = gemmOp.getBetaAttr();
    
    int64_t transA = transAAttr ? transAAttr.getValue().getSExtValue() : 0;
    int64_t transB = transBAttr ? transBAttr.getValue().getSExtValue() : 0;
    float alpha = alphaAttr ? alphaAttr.getValue().convertToFloat() : 1.0f;
    float beta = betaAttr ? betaAttr.getValue().convertToFloat() : 1.0f;
    
    // 当前只支持 alpha=1.0 和 beta=1.0 的情况
    // 如果需要其他值，需要额外的scale操作
    if (std::abs(alpha - 1.0f) > 1e-6 || std::abs(beta - 1.0f) > 1e-6) {
      return rewriter.notifyMatchFailure(gemmOp, 
        "Currently only support alpha=1.0 and beta=1.0");
    }
    
    // 检查bias
    if (!inputC) {
      return rewriter.notifyMatchFailure(gemmOp, "Bias (C) is required");
    }
    
    // 计算维度
    // A: [m, k] or [k, m] if transposed
    // B: [k, n] or [n, k] if transposed
    int64_t m = transA ? inputShapeA[1] : inputShapeA[0];
    int64_t k_a = transA ? inputShapeA[0] : inputShapeA[1];
    int64_t k_b = transB ? inputShapeB[1] : inputShapeB[0];
    int64_t n = transB ? inputShapeB[0] : inputShapeB[1];
    
    if (k_a != k_b) {
      return rewriter.notifyMatchFailure(gemmOp, 
        "Inner dimensions must match");
    }
    
    int64_t k = k_a;
    
    LLVM_DEBUG(llvm::dbgs() << "Gemm dimensions: [" << m << "," << k 
               << "] x [" << k << "," << n << "]^" << (transB ? "T" : "") 
               << " + [" << n << "] -> [" << m << "," << n << "]\n");
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    auto f32Type = rewriter.getF32Type();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto createF32Const = [&](float value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, f32Type, 
                                               rewriter.getF32FloatAttr(value));
    };
    
    auto mValue = createI32Const(m);
    auto kValue = createI32Const(k);
    auto nValue = createI32Const(n);
    auto transposeAValue = createI1Const(transA != 0);
    auto transposeBValue = createI1Const(transB != 0);
    auto alphaValue = createF32Const(alpha);
    auto betaValue = createF32Const(beta);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    auto inputMemrefC = markForBufferization(inputC);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    auto inputPtrC = getPtr(inputMemrefC);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create or get function declaration
    auto moduleOp = gemmOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnGemm";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type,           // m, k, n
        i1Type, i1Type,                      // transpose_a, transpose_b
        f32Type, f32Type,                    // alpha, beta
        ptrType, ptrType, ptrType,           // a_data, b_data, c_data (bias)
        ptrType                              // y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      mValue, kValue, nValue,
      transposeAValue, transposeBValue,
      alphaValue, betaValue,
      inputPtrA, inputPtrB, inputPtrC,
      outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gemmOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted Gemm to oneDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.ReduceSumV11 to mgpuOneDnnReduceSum
class ReduceSumOpLowering : public OpRewritePattern<mlir::ONNXReduceSumV11Op>, 
                            public ONNXToOneDNNPatternBase {
public:
  using OpRewritePattern<mlir::ONNXReduceSumV11Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXReduceSumV11Op reduceSumOp, 
                                PatternRewriter &rewriter) const override {
    Location loc = reduceSumOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.ReduceSumV11 to oneDNN at " 
               << loc << "\n");

    // Get input tensor
    Value input = reduceSumOp.getData();
    
    // Get input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(reduceSumOp, "Input must have static shape");
    }
    
    // Extract input dimensions
    auto inputShape = inputType.getShape();
    int64_t ndims = inputShape.size();
    
    if (ndims < 2 || ndims > 5) {
      return rewriter.notifyMatchFailure(reduceSumOp, 
        "Only support 2D-5D tensors");
    }
    
    // Get axes attribute
    std::vector<int64_t> axes;
    if (auto axesAttr = reduceSumOp.getAxesAttr()) {
      for (auto attr : axesAttr) {
        int64_t axis = attr.cast<IntegerAttr>().getInt();
        // 归一化负数索引
        if (axis < 0) {
          axis = ndims + axis;
        }
        if (axis < 0 || axis >= ndims) {
          return rewriter.notifyMatchFailure(reduceSumOp, 
            "Axis out of range");
        }
        axes.push_back(axis);
      }
    } else {
      return rewriter.notifyMatchFailure(reduceSumOp, "Axes attribute required");
    }
    
    // Get keepdims attribute
    int64_t keepdims = reduceSumOp.getKeepdims();
    
    LLVM_DEBUG({
      llvm::dbgs() << "ReduceSum params: input=[";
      for (int64_t i = 0; i < ndims; i++) {
        llvm::dbgs() << inputShape[i];
        if (i < ndims - 1) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "], axes=[";
      for (size_t i = 0; i < axes.size(); i++) {
        llvm::dbgs() << axes[i];
        if (i < axes.size() - 1) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "], keepdims=" << keepdims << "\n";
    });
    
    // Create constants
    auto i32Type = rewriter.getI32Type();
    auto i1Type = rewriter.getI1Type();
    auto indexType = rewriter.getIndexType();
    
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, 
                                               rewriter.getI32IntegerAttr(value));
    };
    
    auto createI1Const = [&](bool value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i1Type, 
                                               rewriter.getBoolAttr(value));
    };
    
    auto createIndexConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, indexType, 
                                               rewriter.getIndexAttr(value));
    };
    
    // Create ndims constant
    auto ndimsValue = createI32Const(ndims);
    
    // Create input_dims array
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // 分配临时数组用于存储input_dims
    auto inputDimsMemrefType = MemRefType::get({ndims}, i32Type);
    auto inputDimsMemref = rewriter.create<memref::AllocaOp>(loc, inputDimsMemrefType);
    
    for (int64_t i = 0; i < ndims; i++) {
      auto idx = createIndexConst(i);
      auto dimValue = createI32Const(inputShape[i]);
      rewriter.create<memref::StoreOp>(loc, dimValue, inputDimsMemref, ValueRange{idx});
    }
    
    // 获取input_dims数组指针
    auto inputDimsPtr = [&]() -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, inputDimsMemref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    }();
    
    // Create num_axes constant
    auto numAxesValue = createI32Const(axes.size());
    
    // Create axes array
    auto axesMemrefType = MemRefType::get({static_cast<int64_t>(axes.size())}, i32Type);
    auto axesMemref = rewriter.create<memref::AllocaOp>(loc, axesMemrefType);
    
    for (size_t i = 0; i < axes.size(); i++) {
      auto idx = createIndexConst(i);
      auto axisValue = createI32Const(axes[i]);
      rewriter.create<memref::StoreOp>(loc, axisValue, axesMemref, ValueRange{idx});
    }
    
    // 获取axes数组指针
    auto axesPtr = [&]() -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, axesMemref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    }();
    
    // Create keepdims constant
    auto keepdimsValue = createI1Const(keepdims != 0);
    
    // Mark for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert to pointer
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output
    auto outputType = mlir::dyn_cast<RankedTensorType>(
      reduceSumOp.getReduced().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), 
                                           outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create function declaration
    auto moduleOp = reduceSumOp->getParentOfType<ModuleOp>();
    std::string functionName = "mgpuOneDnnReduceSum";
    
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type,                             // ndims
        ptrType,                             // input_dims
        i32Type,                             // num_axes
        ptrType,                             // axes
        i1Type,                              // keepdims
        ptrType, ptrType                     // x_data, y_data
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    std::vector<Value> args = {
      ndimsValue,
      inputDimsPtr,
      numAxesValue,
      axesPtr,
      keepdimsValue,
      inputPtr,
      outputPtr
    };
    
    rewriter.create<func::CallOp>(loc, TypeRange(), funcOp.getName(), 
                                  ValueRange(args));
    
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(reduceSumOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted ReduceSum to oneDNN call\n");
    return success();
  }
};

// Pass to convert ONNX operations to oneDNN calls
struct ONNXToOneDNNPass
    : public PassWrapper<ONNXToOneDNNPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-onnx-to-onednn"; }
  
  StringRef getDescription() const final {
    return "Convert ONNX operations to oneDNN runtime calls";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    // Define conversion patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvOpLowering>(context);
    patterns.add<BatchMatMul4DOpLowering>(context);
    patterns.add<BatchMatMulOpLowering>(context);
    patterns.add<MatMulOpLowering>(context);
    patterns.add<MaxPoolOpLowering>(context);
    patterns.add<AvgPoolOpLowering>(context);
    // patterns.add<ReduceMeanOpLowering>(context);
    patterns.add<ReduceMeanGeneralOpLowering>(context);
    patterns.add<SoftmaxOpLowering>(context);
    patterns.add<GemmOpLowering>(context);
    patterns.add<ReduceSumOpLowering>(context);
    
    // Apply patterns
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect, 
                          arith::ArithDialect, memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<arith::IndexCastOp>();
    
    target.addIllegalOp<mlir::ONNXConvOp>();
    target.addIllegalOp<mlir::ONNXMatMulOp>();
    target.addIllegalOp<mlir::ONNXMaxPoolSingleOutOp>();
    target.addIllegalOp<mlir::ONNXAveragePoolOp>();
    target.addIllegalOp<mlir::ONNXReduceMeanV13Op>();
    target.addIllegalOp<mlir::ONNXSoftmaxOp>();
    target.addIllegalOp<mlir::ONNXGemmOp>();
    target.addIllegalOp<mlir::ONNXReduceSumV11Op>();
    
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Pass registration
namespace onnx_mlir {
std::unique_ptr<Pass> createONNXToOneDNNPass() {
  return std::make_unique<ONNXToOneDNNPass>();
}
} // namespace onnx_mlir

static mlir::PassRegistration<ONNXToOneDNNPass> pass;