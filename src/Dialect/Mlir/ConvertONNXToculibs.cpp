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
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// Include ONNX dialect headers
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "onnx-to-culibs"

namespace {

class ONNXToCuLibsPatternBase {
protected:
  // 辅助函数：确保Handle Pool已初始化
void ensureHandlePoolInitialization(ModuleOp moduleOp, PatternRewriter &rewriter, Location loc, Type ptrType, func::FuncOp currentFunc) const {
  // 检查当前函数中是否已有初始化调用
  bool hasInitCall = false;
  bool hasDestroyCall = false;
  
  currentFunc.walk([&](func::CallOp callOp) {
    StringRef calleeName = callOp.getCallee();
    if (calleeName == "mgpuInitHandlePool") {
      hasInitCall = true;
    }
    if (calleeName == "mgpuDestroyHandlePool") {
      hasDestroyCall = true;
    }
    return WalkResult::advance();
  });
  
  // 如果已经有初始化和销毁调用，就不需要再添加
  if (hasInitCall && hasDestroyCall) {
    return;
  }
  
  auto i32Type = rewriter.getI32Type();
  
  // 创建初始化函数声明
  func::FuncOp initFunc = getOrCreateFunction(moduleOp, rewriter, loc, 
      "mgpuInitHandlePool", rewriter.getFunctionType({i32Type}, {}));
  
  // 创建销毁函数声明
  func::FuncOp destroyFunc = getOrCreateFunction(moduleOp, rewriter, loc, 
      "mgpuDestroyHandlePool", rewriter.getFunctionType({}, {}));
  
  // 确保当前函数有函数体
  if (currentFunc.getBody().empty()) {
    // 如果函数体为空，创建一个基本的函数体
    OpBuilder::InsertionGuard guard(rewriter);
    Block *entryBlock = rewriter.createBlock(&currentFunc.getBody());
    
    // 添加一个返回语句
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<func::ReturnOp>(loc);
  }
  
  // 插入初始化和销毁调用
  if (!hasInitCall) {
    // 在函数开头插入初始化调用
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&currentFunc.getBody().front());
    
    // 创建池大小常量（默认15个handle）
    auto poolSizeValue = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(15));
    rewriter.create<func::CallOp>(
        loc, TypeRange{}, initFunc.getName(), ValueRange{poolSizeValue});
    
    LLVM_DEBUG(llvm::dbgs() << "Inserted mgpuInitHandlePool call in current function: " 
               << currentFunc.getName() << "\n");
  }
  
  if (!hasDestroyCall) {
    // 在函数结尾插入销毁调用
    OpBuilder::InsertionGuard guard(rewriter);
    
    // 找到所有的返回语句，在每个返回语句前插入销毁调用
    SmallVector<func::ReturnOp> returnOps;
    currentFunc.walk([&](func::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
    });
    
    for (auto returnOp : returnOps) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<func::CallOp>(
          loc, TypeRange{}, destroyFunc.getName(), ValueRange{});
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Inserted mgpuDestroyHandlePool call(s) in current function: " 
               << currentFunc.getName() << "\n");
  }
}
  // 辅助函数：获取或创建函数声明
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

// 通用辅助函数：根据元素类型获取函数名后缀
std::string getFunctionSuffix(Type elementType) {
  if (elementType.isF16()) {
    return "_fp16";
  } else if (elementType.isF32()) {
    return "";
  } else {
    // 其他类型保持原函数名
    return "";
  }
}

// Pattern to convert onnx.Conv to a call to mgpuCudnnConv2dForward[_fp16]
class ConvOpLowering : public OpRewritePattern<mlir::ONNXConvOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    // Get the location for error reporting
    Location loc = convOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Conv at " << loc << "\n");

    // Get the input, weight, and bias tensors
    Value input = convOp.getX();
    Value weights = convOp.getW();
    Value bias = convOp.getB();
    
    // Get the input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, "Input must have static shape");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnConv2dForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // Get the weight type
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
      return rewriter.notifyMatchFailure(convOp, "Weights must be 4D tensor (KCHW)");
    }
    int64_t k = weightShape[0]; // Output channels
    int64_t r = weightShape[2]; // Kernel height
    int64_t s = weightShape[3]; // Kernel width
    
    // Extract convolution parameters
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1};
    
    // Extract from attributes if available
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
    
    // Extract specific parameter values
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
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
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
        // Extract the aligned pointer as index
        auto indexType = rewriter.getIndexType();
        auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
        
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
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a null CUDA stream (or get from context if available)
  auto moduleOp = convOp->getParentOfType<ModuleOp>();

  func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
  
  if (!streamCreateFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
    streamCreateFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamCreate", streamCreateType);
    streamCreateFunc.setPrivate();
  }
  
  func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

  if (!handleCreateFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
    handleCreateFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuCreateHandlesForStream", handleCreateType);
    handleCreateFunc.setPrivate();
  }

  auto streamCallOp = rewriter.create<func::CallOp>(
    loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
  auto streamPtr = streamCallOp.getResult(0);

  rewriter.create<func::CallOp>(
    loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
  
  func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
  
  if (!funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = rewriter.getFunctionType({
      i32Type, i32Type, i32Type, i32Type,  // n, c, h, w_in
      i32Type, i32Type, i32Type,           // k, r, s
      i32Type, i32Type,                    // pad_h, pad_w
      i32Type, i32Type,                    // stride_h, stride_w
      i32Type, i32Type,                    // dilation_h, dilation_w
      ptrType, ptrType, ptrType,           // x_data, w_data, bias_data
      ptrType,                             // y_data
      ptrType                              // stream
    }, {});
    
    funcOp = rewriter.create<func::FuncOp>(
      loc, functionName, funcType);
    funcOp.setPrivate();
  }
  
  std::vector<Value> args = {
    nValue, cValue, hValue, wValue,
    kValue, rValue, sValue,
    padHValue, padWValue,
    strideHValue, strideWValue,
    dilationHValue, dilationWValue,
    inputPtr, weightPtr, biasPtr,
    outputPtr, streamPtr
  };
  
  // 在创建func.CallOp之前，获取原始的onnx_node_name属性
  Attribute onnxNodeNameAttr = convOp->getAttr("onnx_node_name");

  auto callOp = rewriter.create<func::CallOp>(
    loc, TypeRange(), funcOp.getName(), ValueRange(args));
  
  // 如果原始操作有onnx_node_name属性，则传递给新的调用
  if (onnxNodeNameAttr) {
    callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
    LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
              << onnxNodeNameAttr << " to cuDNN call\n");
  }

  func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
  
  if (!streamSyncFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
    streamSyncFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamSynchronize", streamSyncType);
    streamSyncFunc.setPrivate();
  }

  rewriter.create<func::CallOp>(
    loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});

  func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
  
  if (!streamDestroyFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
    streamDestroyFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamDestroy", streamDestroyType);
    streamDestroyFunc.setPrivate();
  }
  
  rewriter.create<func::CallOp>(
    loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
  
  auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
      loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
  
  rewriter.replaceOp(convOp, resultTensor);
  
  LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Conv to cuDNN call\n");
  return success();
}
};

// Pattern to convert onnx.Add to a call to mgpuCudnnAdd[_fp16]
class AddOpLowering : public OpRewritePattern<mlir::ONNXAddOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = addOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Add at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = addOp.getA();
    Value inputB = addOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(addOp, "Inputs must have static shapes");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputTypeA.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar addition\n");
    }
    
    // 根据操作类型确定函数名
    std::string baseFunctionName = isScalarOperation ? "mgpuCudnnAddScalar" : "mgpuCudnnAdd";
    std::string functionName = baseFunctionName + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(addOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
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
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(addOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = addOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 查找或创建函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType, ptrType,  // input, scalar/inputB, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(addOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Add to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Flatten + onnx.Add to a single call to mgpuCudnnAddWithFlatten
class FlattenAddOpLowering : public OpRewritePattern<mlir::ONNXAddOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    
    // 检查是否有一个输入来自 Flatten 操作
    Value inputA = addOp.getA();
    Value inputB = addOp.getB();
    
    mlir::ONNXFlattenOp flattenOp = nullptr;
    Value flattenInput = nullptr;
    Value otherInput = nullptr;
    bool isInputAFlattened = false;
    
    // 检查 inputA 是否来自 Flatten
    if (auto flattenOpA = inputA.getDefiningOp<mlir::ONNXFlattenOp>()) {
      flattenOp = flattenOpA;
      flattenInput = flattenOpA.getInput();
      otherInput = inputB;
      isInputAFlattened = true;
    }
    // 检查 inputB 是否来自 Flatten
    else if (auto flattenOpB = inputB.getDefiningOp<mlir::ONNXFlattenOp>()) {
      flattenOp = flattenOpB;
      flattenInput = flattenOpB.getInput();
      otherInput = inputA;
      isInputAFlattened = false;
    }
    
    // 如果没有找到 Flatten 操作，不匹配此模式
    if (!flattenOp) {
      return rewriter.notifyMatchFailure(addOp, "No flatten operation found in inputs");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found Flatten+Add pattern at " << loc << "\n");
    
    // 获取 flatten 输入的原始形状
    auto flattenInputType = mlir::dyn_cast<RankedTensorType>(flattenInput.getType());
    auto otherInputType = mlir::dyn_cast<RankedTensorType>(otherInput.getType());
    auto outputType = mlir::dyn_cast<RankedTensorType>(addOp.getResult().getType());
    
    if (!flattenInputType || !flattenInputType.hasStaticShape() || 
        !otherInputType || !otherInputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(addOp, "Inputs must have static shapes");
    }
    
    // 验证 flatten 的轴参数
    auto axisAttr = flattenOp.getAxisAttr();
    int64_t axis = axisAttr ? axisAttr.getValue().getSExtValue() : 1;
    
    auto flattenInputShape = flattenInputType.getShape();
    auto otherInputShape = otherInputType.getShape();
    
    // 检查是否为支持的 flatten 模式 (axis=1, 4D->2D)
    if (axis != 1 || flattenInputShape.size() != 4) {
      return rewriter.notifyMatchFailure(addOp, "Only supports flatten with axis=1 from 4D to 2D");
    }
    
    // 验证维度兼容性
    int64_t batch_size = flattenInputShape[0];
    int64_t flattened_size = 1;
    for (size_t i = 1; i < flattenInputShape.size(); ++i) {
      flattened_size *= flattenInputShape[i];
    }
    
    if (otherInputShape.size() != 2 || 
        otherInputShape[0] != batch_size || 
        otherInputShape[1] != flattened_size) {
      return rewriter.notifyMatchFailure(addOp, "Incompatible shapes for flatten+add optimization");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Shape validation passed. Original shape: " 
               << flattenInputShape[0] << "x" << flattenInputShape[1] 
               << "x" << flattenInputShape[2] << "x" << flattenInputShape[3] << "\n");
    
    // 获取元素类型并确定函数名
    Type elementType = flattenInputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnAddWithFlatten" + functionSuffix;
    
    // 提取原始 4D 维度
    int64_t n = flattenInputShape[0];
    int64_t c = flattenInputShape[1];
    int64_t h = flattenInputShape[2];
    int64_t w = flattenInputShape[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto isAFlattenValue = createI32Const(isInputAFlattened ? 1 : 0);
    
    // 准备输入和输出缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto flattenInputMemref = markForBufferization(flattenInput);
    auto otherInputMemref = markForBufferization(otherInput);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto flattenInputPtr = getPtr(flattenInputMemref);
    auto otherInputPtr = getPtr(otherInputMemref);
    
    // 分配输出 memref (应该是 2D 形状)
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流 (复用现有逻辑)
    auto moduleOp = addOp->getParentOfType<ModuleOp>();
    
    // 创建流和句柄 (复用现有代码)
    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");
    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 查找或创建新的包装函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType, ptrType,  // flattenInput, otherInput, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type,  // isInputAFlattened
        ptrType   // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用新的包装函数
    std::vector<Value> args = {
      flattenInputPtr, otherInputPtr, outputPtr,
      nValue, cValue, hValue, wValue,
      isAFlattenValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流 (复用现有逻辑)
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(addOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted Flatten+Add to optimized cuDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.Sub to a call to mgpuCudnnSub[_fp16]
class SubOpLowering : public OpRewritePattern<mlir::ONNXSubOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXSubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXSubOp subOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = subOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Sub at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = subOp.getA();
    Value inputB = subOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(subOp, "Inputs must have static shapes");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputTypeA.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar subtraction\n");
    }
    
    // 根据操作类型确定函数名
    std::string baseFunctionName = isScalarOperation ? "mgpuCudnnSubScalar" : "mgpuCudnnSub";
    std::string functionName = baseFunctionName + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(subOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
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
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(subOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = subOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 查找或创建函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType, ptrType,  // inputA, inputB/scalar, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(subOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Sub to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Mul to a call to mgpuCudnnMul[_fp16]
class MulOpLowering : public OpRewritePattern<mlir::ONNXMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = mulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Mul at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = mulOp.getA();
    Value inputB = mulOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mulOp, "Inputs must have static shapes");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputTypeA.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar multiplication\n");
    }
    
    // 根据操作类型确定函数名
    std::string baseFunctionName = isScalarOperation ? "mgpuCudnnMulScalar" : "mgpuCudnnMul";
    std::string functionName = baseFunctionName + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(mulOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
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
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(mulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = mulOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 查找或创建函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType, ptrType,  // inputA, inputB/scalar, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(mulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Mul to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Neg to a call to mgpuCudnnNeg[_fp16]
class NegOpLowering : public OpRewritePattern<mlir::ONNXNegOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXNegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXNegOp negOp, PatternRewriter &rewriter) const override {
    // Get the location for error reporting
    Location loc = negOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Neg at " << loc << "\n");

    // Get the input tensor
    Value input = negOp.getX();
    
    // Get the input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(negOp, "Input must have static shape");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnNeg" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // Extract input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() < 1 || inputShape.size() > 4) {
      return rewriter.notifyMatchFailure(negOp, "Input must be 1D to 4D tensor");
    }
    
    // Pad shape to 4D (NCHW) if needed
    std::vector<int64_t> paddedShape(4, 1);
    int offset = 4 - inputShape.size();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      paddedShape[i + offset] = inputShape[i];
    }
    
    int64_t n = paddedShape[0];
    int64_t c = paddedShape[1];
    int64_t h = paddedShape[2];
    int64_t w = paddedShape[3];
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // Prepare input and output buffers
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(negOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = negOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // Look up or create the function
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType,  // input, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      inputPtr, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // Synchronize and destroy the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(negOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Neg to cuDNN call\n");
    return success();
  }
};

// // Pattern to convert onnx.MatMul to a call to mgpuCulibsFullyConnectedForward[_fp16]
// class MatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, public ONNXToCuLibsPatternBase {
// public:
//   using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
//     // 获取位置信息用于错误报告
//     Location loc = matMulOp.getLoc();
//     LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul at " << loc << "\n");

//     // 获取输入张量
//     Value inputA = matMulOp.getA();
//     Value inputB = matMulOp.getB();
    
//     // 获取输入类型
//     auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
//     auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
//     if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
//       return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
//     }
    
//     // 获取元素类型并确定函数名后缀
//     Type elementType = inputTypeA.getElementType();
//     std::string functionSuffix = getFunctionSuffix(elementType);
//     std::string functionName = "mgpuCulibsFullyConnectedForward" + functionSuffix;
    
//     LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
//     // 提取输入维度
//     auto inputShapeA = inputTypeA.getShape();
//     auto inputShapeB = inputTypeB.getShape();
    
//     // MatMul需要至少2D张量
//     if (inputShapeA.size() < 2 || inputShapeB.size() < 2) {
//       return rewriter.notifyMatchFailure(matMulOp, "Inputs must be at least 2D tensors");
//     }
    
//     // 我们只处理2D矩阵乘法（像全连接层那样）
//     if (inputShapeA.size() != 2 || inputShapeB.size() != 2) {
//       return rewriter.notifyMatchFailure(matMulOp, "Only 2D matrix multiplication is supported");
//     }
    
//     // 对于全连接，inputA形状为[batch_size, input_features]，inputB形状为[input_features, output_features]
//     int64_t batch_size = inputShapeA[0];
//     int64_t input_features = inputShapeA[1];
//     int64_t output_features = inputShapeB[1];
    
//     // 验证维度匹配
//     if (input_features != inputShapeB[0]) {
//       return rewriter.notifyMatchFailure(matMulOp, "Inner dimensions must match for matrix multiplication");
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "MatMul dimensions: batch_size=" << batch_size 
//                << ", input_features=" << input_features 
//                << ", output_features=" << output_features << "\n");
    
//     // 创建常量用于整数参数
//     auto i32Type = rewriter.getI32Type();
//     auto createI32Const = [&](int64_t value) -> Value {
//       return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
//     };
    
//     auto batchSizeValue = createI32Const(batch_size);
//     auto inputFeaturesValue = createI32Const(input_features);
//     auto outputFeaturesValue = createI32Const(output_features);
    
//     // 将输入张量标记为缓冲区
//     auto markForBufferization = [&](Value tensor) -> Value {
//       auto tensorType = tensor.getType().cast<RankedTensorType>();
//       auto memrefType = MemRefType::get(
//         tensorType.getShape(),
//         tensorType.getElementType());
//       return rewriter.create<UnrealizedConversionCastOp>(
//         loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
//     };
    
//     auto inputMemrefA = markForBufferization(inputA);
//     auto inputMemrefB = markForBufferization(inputB);
    
//     // 将memref转为void指针
//     auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
//     auto getPtr = [&](Value memref) -> Value {
//       // 提取对齐的指针为索引
//       auto indexType = rewriter.getIndexType();
//       auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
//       auto i64Type = rewriter.getIntegerType(64);
//       auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
//       return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
//     };
    
//     auto inputPtrA = getPtr(inputMemrefA);
//     auto weightPtrB = getPtr(inputMemrefB);
    
//     // 分配输出memref
//     auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
//     auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
//     auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
//     auto outputPtr = getPtr(outputMemref);
    
//     // 创建CUDA流
//     auto moduleOp = matMulOp->getParentOfType<ModuleOp>();

//     func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
//     if (!streamCreateFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
//       streamCreateFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamCreate", streamCreateType);
//       streamCreateFunc.setPrivate();
//     }

//     func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

//     if (!handleCreateFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
//       handleCreateFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuCreateHandlesForStream", handleCreateType);
//       handleCreateFunc.setPrivate();
//     }
    
//     auto streamCallOp = rewriter.create<func::CallOp>(
//       loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
//     auto streamPtr = streamCallOp.getResult(0);
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
  
//     // 创建transB标志（MatMul操作默认transB=0）
//     auto transBValue = createI32Const(0);

//     // 创建或查找FC函数声明
//     func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
//     if (!fcFunc) {
//       LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       auto fcFuncType = rewriter.getFunctionType({
//         i32Type, i32Type, i32Type,  // batch_size, input_features, output_features
//         i32Type,                    // transB 标志
//         ptrType, ptrType, ptrType,  // input_data, weight_data, bias_data
//         ptrType,                    // output_data
//         ptrType                     // stream
//       }, {});
      
//       fcFunc = rewriter.create<func::FuncOp>(
//         loc, functionName, fcFuncType);
//       fcFunc.setPrivate();
//     }
    
//     // 创建null指针用于偏置（MatMul没有偏置）
//     MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
//     auto nullBiasPtr = create.llvm.null(ptrType);
    
//     // 调用FC函数
//     std::vector<Value> args = {
//       batchSizeValue, inputFeaturesValue, outputFeaturesValue,
//       transBValue,   // transB = 0 for MatMul
//       inputPtrA, weightPtrB, nullBiasPtr,
//       outputPtr, streamPtr
//     };
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
//     // 同步流
//     func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
//     if (!streamSyncFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
//       streamSyncFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamSynchronize", streamSyncType);
//       streamSyncFunc.setPrivate();
//     }
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
//     // 销毁流
//     func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
//     if (!streamDestroyFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
//       streamDestroyFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamDestroy", streamDestroyType);
//       streamDestroyFunc.setPrivate();
//     }
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
//     // 将memref转回tensor
//     auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
//         loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
//     rewriter.replaceOp(matMulOp, resultTensor);
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MatMul to FC call\n");
//     return success();
//   }
// };

// Pattern to convert onnx.MatMul to a call to mgpuCulibsFullyConnectedForward[_fp16] or batched version
class MatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = matMulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul at " << loc << "\n");

    // 获取输入张量
    Value inputA = matMulOp.getA();
    Value inputB = matMulOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputTypeA.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // MatMul需要至少2D张量
    if (inputShapeA.size() < 2 || inputShapeB.size() < 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must be at least 2D tensors");
    }
    
    // 检查是否是标准2D矩阵乘法（保持原有逻辑不变）
    if (inputShapeA.size() == 2 && inputShapeB.size() == 2) {
      return handle2DMatMul(matMulOp, rewriter, inputA, inputB, inputTypeA, inputTypeB, functionSuffix);
    }
    
    // 处理带广播的批量矩阵乘法（新增逻辑）
    return handleBatchedMatMul(matMulOp, rewriter, inputA, inputB, inputTypeA, inputTypeB, functionSuffix);
  }

private:
  // 原有的2D MatMul处理逻辑（保持不变）
  LogicalResult handle2DMatMul(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter,
                               Value inputA, Value inputB, 
                               RankedTensorType inputTypeA, RankedTensorType inputTypeB,
                               const std::string &functionSuffix) const {
    Location loc = matMulOp.getLoc();
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    std::string functionName = "mgpuCulibsFullyConnectedForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << inputTypeA.getElementType() << "\n");
    
    // 对于全连接，inputA形状为[batch_size, input_features]，inputB形状为[input_features, output_features]
    int64_t batch_size = inputShapeA[0];
    int64_t input_features = inputShapeA[1];
    int64_t output_features = inputShapeB[1];
    
    // 验证维度匹配
    if (input_features != inputShapeB[0]) {
      return rewriter.notifyMatchFailure(matMulOp, "Inner dimensions must match for matrix multiplication");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "MatMul dimensions: batch_size=" << batch_size 
               << ", input_features=" << input_features 
               << ", output_features=" << output_features << "\n");
    
    // 创建常量用于整数参数
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batch_size);
    auto inputFeaturesValue = createI32Const(input_features);
    auto outputFeaturesValue = createI32Const(output_features);
    
    // 将输入张量标记为缓冲区
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
    
    // 将memref转为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto weightPtrB = getPtr(inputMemrefB);
    
    // 分配输出memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto streamPtr = createCudaStreamAndHandles(rewriter, loc, matMulOp);
  
    // 创建transB标志（MatMul操作默认transB=0）
    auto transBValue = createI32Const(0);

    // 创建或查找FC函数声明
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type,  // batch_size, input_features, output_features
        i32Type,                    // transB 标志
        ptrType, ptrType, ptrType,  // input_data, weight_data, bias_data
        ptrType,                    // output_data
        ptrType                     // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, functionName, fcFuncType);
      fcFunc.setPrivate();
    }
    
    // 创建null指针用于偏置（MatMul没有偏置）
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto nullBiasPtr = create.llvm.null(ptrType);
    
    // 调用FC函数
    std::vector<Value> args = {
      batchSizeValue, inputFeaturesValue, outputFeaturesValue,
      transBValue,   // transB = 0 for MatMul
      inputPtrA, weightPtrB, nullBiasPtr,
      outputPtr, streamPtr
    };
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = matMulOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }
    // 同步和清理流
    synchronizeAndCleanupStream(rewriter, loc, matMulOp, streamPtr);
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MatMul to FC call\n");
    return success();
  }
  
  // 新增的批量MatMul处理逻辑
  LogicalResult handleBatchedMatMul(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter,
                                   Value inputA, Value inputB,
                                   RankedTensorType inputTypeA, RankedTensorType inputTypeB,
                                   const std::string &functionSuffix) const {
    Location loc = matMulOp.getLoc();
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // 分析批量MatMul类型
    BatchedMatMulParams params;
    if (!analyzeBatchedMatMul(inputShapeA, inputShapeB, params)) {
      return rewriter.notifyMatchFailure(matMulOp, "Unsupported batched MatMul pattern");
    }
    
    std::string functionName = "mgpuCulibsBatchedMatMulForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using batched function: " << functionName 
               << " batch_size=" << params.batch_size
               << " m=" << params.m << " n=" << params.n << " k=" << params.k << "\n");
    
    // 创建常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    // 准备输入
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto streamPtr = createCudaStreamAndHandles(rewriter, loc, matMulOp);
    
    // 创建或查找批量MatMul函数声明
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    func::FuncOp batchedFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!batchedFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch_size, m, n, k
        i32Type, i32Type, i32Type,           // stride_a, stride_b, stride_c
        ptrType, ptrType, ptrType,           // input_a, input_b, output
        ptrType                              // stream
      }, {});
      
      batchedFunc = rewriter.create<func::FuncOp>(loc, functionName, funcType);
      batchedFunc.setPrivate();
    }
    
    // 调用批量MatMul函数
    std::vector<Value> args = {
      createI32Const(params.batch_size), createI32Const(params.m), 
      createI32Const(params.n), createI32Const(params.k),
      createI32Const(params.stride_a), createI32Const(params.stride_b), createI32Const(params.stride_c),
      inputPtrA, inputPtrB, outputPtr, streamPtr
    };
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = matMulOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(loc, TypeRange(), batchedFunc.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }
    // 同步和清理流
    synchronizeAndCleanupStream(rewriter, loc, matMulOp, streamPtr);
    
    // 转换回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    return success();
  }
  
  struct BatchedMatMulParams {
    int64_t batch_size;
    int64_t m, n, k;
    int64_t stride_a, stride_b, stride_c;
  };
  
  bool analyzeBatchedMatMul(ArrayRef<int64_t> shapeA, ArrayRef<int64_t> shapeB, 
                           BatchedMatMulParams &params) const {
    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();
    
    if (rankA == 2 && rankB == 3) {
      // 2D @ 3D广播: (m, k) @ (batch, k, n) = (batch, m, n)
      params.batch_size = shapeB[0];
      params.m = shapeA[0];
      params.k = shapeA[1];
      params.n = shapeB[2];
      params.stride_a = 0;  // A广播，stride为0
      params.stride_b = shapeB[1] * shapeB[2];
      params.stride_c = params.m * params.n;
      return shapeA[1] == shapeB[1]; // 检查k维度匹配
    }
    
    if (rankA == 3 && rankB == 2) {
      // 3D @ 2D广播: (batch, m, k) @ (k, n) = (batch, m, n)
      params.batch_size = shapeA[0];
      params.m = shapeA[1];
      params.k = shapeA[2];
      params.n = shapeB[1];
      params.stride_a = shapeA[1] * shapeA[2];
      params.stride_b = 0;  // B广播，stride为0
      params.stride_c = params.m * params.n;
      return shapeA[2] == shapeB[0]; // 检查k维度匹配
    }
    
    if (rankA == 3 && rankB == 3) {
      // 3D @ 3D: (batch, m, k) @ (batch, k, n) = (batch, m, n)
      params.batch_size = shapeA[0];
      params.m = shapeA[1];
      params.k = shapeA[2];
      params.n = shapeB[2];
      params.stride_a = shapeA[1] * shapeA[2];
      params.stride_b = shapeB[1] * shapeB[2];
      params.stride_c = params.m * params.n;
      return shapeA[0] == shapeB[0] && shapeA[2] == shapeB[1]; // 检查batch和k维度
    }
    
    return false; // 不支持的模式
  }
  
  // 辅助函数：创建CUDA流和句柄
  Value createCudaStreamAndHandles(PatternRewriter &rewriter, Location loc, mlir::ONNXMatMulOp matMulOp) const {
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
      
    return streamPtr;
  }
  
  // 辅助函数：同步和清理流
  void synchronizeAndCleanupStream(PatternRewriter &rewriter, Location loc, mlir::ONNXMatMulOp matMulOp, Value streamPtr) const {
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // 同步流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // 销毁流
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
  }
};

// Pattern to convert onnx.Flatten+onnx.Gemm to a call to mgpuCulibsFlattenFullyConnectedForward[_fp16]
class FlattenGemmOpLowering : public OpRewritePattern<mlir::ONNXGemmOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    // First, check if the input to the Gemm is from a Flatten operation
    auto flattenOp = gemmOp.getA().getDefiningOp<mlir::ONNXFlattenOp>();
    if (!flattenOp) {
      // Not a Flatten->Gemm pattern, skip this operation
      return failure();
    }

    // Get the location for error reporting
    Location loc = gemmOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Flatten+onnx.Gemm at " << loc << "\n");

    // Get the original input (before flattening)
    Value originalInput = flattenOp.getInput();
    Value weights = gemmOp.getB();
    Value bias = gemmOp.getC();
    
    // 获取元素类型并确定函数名后缀
    auto inputType = mlir::dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must be a ranked tensor");
    }
    
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCulibsFlattenFullyConnectedForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // Check that the Flatten has axis=1 (we only support that case)
    int64_t flattenAxis = 1;
    if (auto axisAttr = flattenOp.getAxisAttr()) {
      flattenAxis = axisAttr.getValue().getSExtValue();
    }
    
    if (flattenAxis != 1) {
      return rewriter.notifyMatchFailure(flattenOp, "Only support flattening with axis=1");
    }
    
    // Get the original input type
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must have static shape");
    }
    
    // Get input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(flattenOp, "Only 4D NCHW inputs are supported for now");
    }
    
    // Extract NCHW dimensions
    int64_t batchSize = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t height = inputShape[2];
    int64_t width = inputShape[3];
    
    // Calculate flattened features
    int64_t flattenedFeatures = channels * height * width;
    
    // Get the weights type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gemmOp, "Weights must have static shape");
    }
    
    // Extract weight dimensions
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 2) {
      return rewriter.notifyMatchFailure(gemmOp, "Weights must be 2D");
    }
    
    // Get attributes from Gemm op
    float alpha = 1.0f;
    if (auto alphaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getAlphaAttr()))
      alpha = alphaAttr.getValueAsDouble();
      
    float beta = 1.0f;
    if (auto betaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getBetaAttr()))
      beta = betaAttr.getValueAsDouble();
    
    bool transA = false;
    if (auto transAAttr = gemmOp.getTransAAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transAAttr)) {
        transA = intAttr.getValue().getSExtValue() != 0;
      }
    }

    bool transB = false;
    if (auto transBAttr = gemmOp.getTransBAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transBAttr)) {
        transB = intAttr.getValue().getSExtValue() != 0;
      }
    }
    
    // Check for unsupported attributes
    if (alpha != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Alpha != 1.0 not supported yet");
    }
    
    if (beta != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Beta != 1.0 not supported yet");
    }
    
    if (transA) {
      return rewriter.notifyMatchFailure(gemmOp, "TransA=1 not supported for flattened FC");
    }
    
    // 根据transB标志正确处理权重维度
    int64_t weight_input_dim, weight_output_dim;
    if (transB) {
      // transB=1时，权重会被转置，所以原始权重格式为 [output_features, input_features]
      weight_input_dim = weightShape[1];    // 转置后的输入维度
      weight_output_dim = weightShape[0];   // 转置后的输出维度
    } else {
      // transB=0时，权重格式为 [input_features, output_features]
      weight_input_dim = weightShape[0];
      weight_output_dim = weightShape[1];
    }
    
    // Check that the flattened dimension matches the weight input dimension
    if (flattenedFeatures != weight_input_dim) {
      return rewriter.notifyMatchFailure(gemmOp, "Flattened dimension doesn't match weight input dimension");
    }
    
    int64_t outputFeatures = weight_output_dim;
    
    LLVM_DEBUG(llvm::dbgs() << "Flatten+Gemm dimensions: batchSize=" << batchSize 
               << ", flattenedFeatures=" << flattenedFeatures 
               << ", outputFeatures=" << outputFeatures 
               << ", transB=" << transB << "\n");
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batchSize);
    auto channelsValue = createI32Const(channels);
    auto heightValue = createI32Const(height);
    auto widthValue = createI32Const(width);
    auto outputFeaturesValue = createI32Const(outputFeatures);
    
    // 创建transB标志
    auto transBValue = createI32Const(transB ? 1 : 0);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(originalInput);
    auto weightMemref = markForBufferization(weights);
    auto biasMemref = markForBufferization(bias);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    Value biasPtr;
    if (biasMemref)
      biasPtr = getPtr(biasMemref);
    else {
      MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
      biasPtr = create.llvm.null(ptrType);
    }
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = gemmOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // Create or locate the flatten-FC function declaration
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch_size, channels, height, width
        i32Type,                             // output_features
        i32Type,                             // transB 标志
        ptrType, ptrType, ptrType,           // input_data, weight_data, bias_data
        ptrType,                             // output_data
        ptrType                              // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, functionName, fcFuncType);
      fcFunc.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchSizeValue, channelsValue, heightValue, widthValue,
      outputFeaturesValue,
      transBValue,  // 传递transB标志
      inputPtr, weightPtr, biasPtr,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // Synchronize the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // Destroy the stream
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gemmOp, resultTensor);
    
    // The Flatten op will be removed automatically if it has no other uses
    if (flattenOp.use_empty())
      rewriter.eraseOp(flattenOp);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Flatten+onnx.Gemm to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.Gemm to a call to mgpuCulibsFullyConnectedForward[_fp16]
class GemmOpLowering : public OpRewritePattern<mlir::ONNXGemmOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = gemmOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Gemm at " << loc << "\n");

    // 获取输入张量
    Value inputA = gemmOp.getA();
    Value inputB = gemmOp.getB();
    Value inputC = gemmOp.getC(); // 这是偏置
    
    // 获取输入类型并确定函数名后缀
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    if (!inputTypeA) {
      return rewriter.notifyMatchFailure(gemmOp, "Input A must be a ranked tensor");
    }
    
    Type elementType = inputTypeA.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCulibsFullyConnectedForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 获取属性
    float alpha = 1.0f;
    if (auto alphaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getAlphaAttr()))
      alpha = alphaAttr.getValueAsDouble();
      
    float beta = 1.0f;
    if (auto betaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getBetaAttr()))
      beta = betaAttr.getValueAsDouble();

    bool transA = false;
    if (auto transAAttr = gemmOp.getTransAAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transAAttr)) {
        transA = intAttr.getValue().getSExtValue() != 0;
      }
    }

    bool transB = false;
    if (auto transBAttr = gemmOp.getTransBAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transBAttr)) {
        transB = intAttr.getValue().getSExtValue() != 0;
      }
    }
    
    // 检查是否有需要特殊处理的alpha和beta值
    if (alpha != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Alpha != 1.0 not supported yet");
    }
    
    if (beta != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Beta != 1.0 not supported yet");
    }
    
    // 获取输入类型
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gemmOp, "Inputs must have static shapes");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // Gemm需要2D张量
    if (inputShapeA.size() != 2 || inputShapeB.size() != 2) {
      return rewriter.notifyMatchFailure(gemmOp, "Gemm inputs must be 2D tensors");
    }
    
    // 根据转置标志确定实际维度
    int64_t batch_size, input_features;
    if (transA) {
      batch_size = inputShapeA[1];
      input_features = inputShapeA[0];
    } else {
      batch_size = inputShapeA[0];
      input_features = inputShapeA[1];
    }
    
    int64_t weight_rows, weight_cols;
    if (transB) {
      // transB=1时，B矩阵在ONNX中会被转置，所以原始B的维度为(weight_cols, weight_rows)
      weight_rows = inputShapeB[1];  // 转置后的行数
      weight_cols = inputShapeB[0];  // 转置后的列数
    } else {
      weight_rows = inputShapeB[0];
      weight_cols = inputShapeB[1];
    }
    
    // 验证内部维度匹配
    if (input_features != weight_rows) {
      return rewriter.notifyMatchFailure(gemmOp, "Inner dimensions must match for Gemm");
    }
    
    int64_t output_features = weight_cols;
    
    LLVM_DEBUG(llvm::dbgs() << "Gemm dimensions: batch_size=" << batch_size 
               << ", input_features=" << input_features 
               << ", output_features=" << output_features 
               << ", transA=" << transA << ", transB=" << transB << "\n");
    
    // 如果transA=1，我们暂时不支持
    if (transA) {
      return rewriter.notifyMatchFailure(gemmOp, "TransA=1 not supported for FC conversion");
    }
    
    // 创建常量用于整数参数
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batch_size);
    auto inputFeaturesValue = createI32Const(input_features);
    auto outputFeaturesValue = createI32Const(output_features);
    
    // 创建transB标志（int类型）
    auto transBValue = createI32Const(transB ? 1 : 0);
    
    // 将输入张量标记为缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    auto biasMemref = markForBufferization(inputC);
    
    // 将memref转为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto weightPtrB = getPtr(inputMemrefB);
    Value biasPtrC;
    if (biasMemref) {
        biasPtrC = getPtr(biasMemref);
    } else {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        biasPtrC = create.llvm.null(ptrType);
    }
    
    // 分配输出memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = gemmOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 创建或查找FC函数声明 - 现在需要支持transB参数和FP16/FP32后缀
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type,  // batch_size, input_features, output_features
        i32Type,                    // transB 标志 (int类型)
        ptrType, ptrType, ptrType,  // input_data, weight_data, bias_data
        ptrType,                    // output_data
        ptrType                     // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, functionName, fcFuncType);
      fcFunc.setPrivate();
    }
    
    // 调用FC函数
    std::vector<Value> args = {
      batchSizeValue, inputFeaturesValue, outputFeaturesValue,
      transBValue,  // 传递transB标志
      inputPtrA, weightPtrB, biasPtrC,
      outputPtr, streamPtr
    };
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = gemmOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }
    // 同步流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // 销毁流
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gemmOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Gemm to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.Flatten+onnx.MatMul to a call to mgpuCulibsFlattenFullyConnectedForward[_fp16]
class FlattenMatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
    // First, check if the input to the MatMul is from a Flatten operation
    auto flattenOp = matMulOp.getA().getDefiningOp<mlir::ONNXFlattenOp>();
    if (!flattenOp) {
      // Not a Flatten->MatMul pattern, skip this operation
      return failure();
    }

    // Get the location for error reporting
    Location loc = matMulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Flatten+onnx.MatMul at " << loc << "\n");

    // Get the original input (before flattening)
    Value originalInput = flattenOp.getInput();
    Value weights = matMulOp.getB();
    
    // 获取元素类型并确定函数名后缀
    auto inputType = mlir::dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must be a ranked tensor");
    }
    
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCulibsFlattenFullyConnectedForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // Check that the Flatten has axis=1 (we only support that case)
    int64_t flattenAxis = 1;
    if (auto axisAttr = flattenOp.getAxisAttr()) {
      flattenAxis = axisAttr.getValue().getSExtValue();
    }
    
    if (flattenAxis != 1) {
      return rewriter.notifyMatchFailure(flattenOp, "Only support flattening with axis=1");
    }
    
    // Get the original input type
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must have static shape");
    }
    
    // Get input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(flattenOp, "Only 4D NCHW inputs are supported for now");
    }
    
    // Extract NCHW dimensions
    int64_t batchSize = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t height = inputShape[2];
    int64_t width = inputShape[3];
    
    // Calculate flattened features
    int64_t flattenedFeatures = channels * height * width;
    
    // Get the weights type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Weights must have static shape");
    }
    
    // Extract weight dimensions
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Weights must be 2D");
    }
    
    // Check that the flattened dimension matches the weight input dimension
    if (flattenedFeatures != weightShape[0]) {
      return rewriter.notifyMatchFailure(matMulOp, "Flattened dimension doesn't match weight dimension");
    }
    
    int64_t outputFeatures = weightShape[1];
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batchSize);
    auto channelsValue = createI32Const(channels);
    auto heightValue = createI32Const(height);
    auto widthValue = createI32Const(width);
    auto outputFeaturesValue = createI32Const(outputFeatures);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(originalInput);
    auto weightMemref = markForBufferization(weights);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    
    // Create null bias pointer (MatMul doesn't have bias)
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto nullBiasPtr = create.llvm.null(ptrType);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);

    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
    
    auto transBValue = createI32Const(0);

    // Create or locate the flatten-FC function declaration
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch_size, channels, height, width
        i32Type,                             // output_features
        i32Type,                             // transB 标志
        ptrType, ptrType, ptrType,           // input_data, weight_data, bias_data
        ptrType,                             // output_data
        ptrType                              // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, functionName, fcFuncType);
      fcFunc.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchSizeValue, channelsValue, heightValue, widthValue,
      outputFeaturesValue,
      transBValue,  // transB = 0 for MatMul
      inputPtr, weightPtr, nullBiasPtr,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // Synchronize the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // Destroy the stream
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    // The Flatten op will be removed automatically if it has no other uses
    if (flattenOp.use_empty())
      rewriter.eraseOp(flattenOp);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Flatten+onnx.MatMul to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.MaxPoolSingleOut to a call to mgpuCudnnMaxPoolForward[_fp16]
class MaxPoolOpLowering : public OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp maxPoolOp, PatternRewriter &rewriter) const override {
    // 获取位置用于错误报告
    Location loc = maxPoolOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MaxPoolSingleOut at " << loc << "\n");

    // 获取输入张量
    Value input = maxPoolOp.getX();
    
    // 获取输入类型
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must have static shape");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnMaxPoolForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 获取输出类型
    auto outputType = mlir::dyn_cast<RankedTensorType>(maxPoolOp.getO_Y().getType());
    if (!outputType || !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Output must have static shape");
    }
    
    // 提取属性
    std::vector<int64_t> kernelShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    
    // 获取核形状（必需）
    if (auto kernelShapeAttr = maxPoolOp.getKernelShapeAttr()) {
      for (auto attr : kernelShapeAttr) {
        kernelShape.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(maxPoolOp, "Kernel shape attribute is required");
    }
    
    // 获取填充、步长和膨胀（可选，有默认值）
    if (auto padsAttr = maxPoolOp.getPadsAttr()) {
      for (auto attr : padsAttr) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无填充
      pads = std::vector<int64_t>(kernelShape.size() * 2, 0);
    }
    
    if (auto stridesAttr = maxPoolOp.getStridesAttr()) {
      for (auto attr : stridesAttr) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：步长为1
      strides = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (auto dilationsAttr = maxPoolOp.getDilationsAttr()) {
      for (auto attr : dilationsAttr) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无膨胀
      dilations = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    // 检查auto_pad属性
    std::string autoPad = "NOTSET";
    if (auto autoPadAttr = maxPoolOp.getAutoPadAttr()) {
      autoPad = autoPadAttr.getValue().str();
    }
    
    // 当前仅支持"NOTSET"自动填充模式
    if (autoPad != "NOTSET") {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only NOTSET auto_pad mode is supported");
    }
    
    // 检查ceil_mode属性
    int64_t ceilMode = 0;
    if (auto ceilModeAttr = maxPoolOp.getCeilModeAttr()) {
      ceilMode = ceilModeAttr.getValue().getSExtValue();
    }
    
    // // 当前仅支持ceil_mode == 0
    // if (ceilMode != 0) {
    //   return rewriter.notifyMatchFailure(maxPoolOp, "Only ceil_mode=0 is supported");
    // }
    
    // 提取输入维度（NCHW）
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // 提取池化参数
    if (kernelShape.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only 2D pooling is supported");
    }
    int64_t kernel_h = kernelShape[0];
    int64_t kernel_w = kernelShape[1];
    
    if (pads.size() != 4) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Pads must have 4 values");
    }
    int64_t pad_h_begin = pads[0]; // 顶部填充
    int64_t pad_w_begin = pads[1]; // 左侧填充
    int64_t pad_h_end = pads[2];   // 底部填充
    int64_t pad_w_end = pads[3];   // 右侧填充
    
    if (strides.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Strides must have 2 values");
    }
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    
    if (dilations.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Dilations must have 2 values");
    }
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kernelHValue = createI32Const(kernel_h);
    auto kernelWValue = createI32Const(kernel_w);
    auto padHBeginValue = createI32Const(pad_h_begin);
    auto padWBeginValue = createI32Const(pad_w_begin);
    auto padHEndValue = createI32Const(pad_h_end);
    auto padWEndValue = createI32Const(pad_w_end);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    auto ceilModeValue = createI32Const(ceilMode);
    
    // 准备bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // 转换memrefs为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针作为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // 分配输出memref
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = maxPoolOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);

    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
    
    // 查找或创建函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // kernel_h, kernel_w
        i32Type, i32Type,                    // pad_h_begin, pad_w_begin
        i32Type, i32Type,                    // pad_h_end, pad_w_end
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        i32Type,                             // ceil_mode
        ptrType, ptrType,                    // input_data, output_data
        ptrType                              // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kernelHValue, kernelWValue,
      padHBeginValue, padWBeginValue,
      padHEndValue, padWEndValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      ceilModeValue,
      inputPtr, outputPtr,
      streamPtr
    };
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = maxPoolOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }

    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(maxPoolOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MaxPoolSingleOut to cuDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.Div to a call to mgpuCudnnMulScalar[_fp16] (with reciprocal) for both fp32 and fp16
class DivOpLowering : public OpRewritePattern<mlir::ONNXDivOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXDivOp divOp, PatternRewriter &rewriter) const override {
    Location loc = divOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Div at " << loc << "\n");

    // 获取输入张量
    Value inputA = divOp.getA();
    Value inputB = divOp.getB();
    
    LLVM_DEBUG(llvm::dbgs() << "Input A: " << inputA << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Input A type: " << inputA.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Input B: " << inputB << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Input B type: " << inputB.getType() << "\n");
    
    // 获取输入类型 - 修复：同时检查两个输入的类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Input A must have static shape\n");
      return rewriter.notifyMatchFailure(divOp, "Input A must have static shape");
    }
    
    // 修复：添加对 inputB 类型的检查
    if (!inputTypeB || !inputTypeB.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Input B must have static shape\n");
      return rewriter.notifyMatchFailure(divOp, "Input B must have static shape");
    }
    
    // 检查元素类型是否为 FP32 或 FP16
    Type elementType = inputTypeA.getElementType();
    bool isFP32 = elementType.isF32();
    bool isFP16 = elementType.isF16();
    
    if (!isFP32 && !isFP16) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Only FP32 and FP16 element types supported\n");
      return rewriter.notifyMatchFailure(divOp, "Only FP32 and FP16 element types supported");
    }
    
    // 获取函数名后缀
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnMulScalar" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Element type: " << (isFP32 ? "FP32" : "FP16") << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << "\n");
    
    // 检查 inputB 是否为标量
    auto inputShapeB = inputTypeB.getShape();
    bool isScalar = inputShapeB.empty() || 
                   (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
                   (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }));
    
    if (!isScalar) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Only scalar division is supported\n");
      return rewriter.notifyMatchFailure(divOp, "Only scalar division is supported");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Detected scalar division\n");
    
    // 查找 inputB 的定义操作并提取标量值
    auto inputBDefOp = inputB.getDefiningOp();
    LLVM_DEBUG(llvm::dbgs() << "Input B defining op: " << (inputBDefOp ? inputBDefOp->getName().getStringRef() : "null") << "\n");
    
    APFloat scalarValue(APFloat::IEEEsingle());
    bool foundScalarValue = false;
    
    // 尝试多种方式提取标量值
    if (inputBDefOp) {
      StringRef opName = inputBDefOp->getName().getStringRef();
      LLVM_DEBUG(llvm::dbgs() << "Input B defining operation: " << opName << "\n");
      
      // 方法1: 直接从 onnx.Constant 提取
      if (auto constOp = mlir::dyn_cast<mlir::ONNXConstantOp>(inputBDefOp)) {
        auto valueOpt = constOp.getValue();
        if (valueOpt.has_value()) {
          if (auto valueAttr = valueOpt.value().dyn_cast<DenseElementsAttr>()) {
            if (valueAttr.getNumElements() == 1) {
              if (isFP32) {
                auto values = valueAttr.getValues<float>();
                float floatValue = *values.begin();
                scalarValue = APFloat(floatValue);
                foundScalarValue = true;
                LLVM_DEBUG(llvm::dbgs() << "Extracted FP32 value from onnx.Constant: " << floatValue << "\n");
              } else if (isFP16) {
                auto values = valueAttr.getValues<APFloat>();
                APFloat fp16Value = *values.begin();
                scalarValue = fp16Value;
                foundScalarValue = true;
                LLVM_DEBUG(llvm::dbgs() << "Extracted FP16 value from onnx.Constant: " << fp16Value.convertToFloat() << "\n");
              }
            }
          }
        }
      }
      
      // 方法2: 递归查找 krnl.global 操作
      if (!foundScalarValue) {
        std::function<Operation*(Value)> findKrnlGlobal = [&](Value val) -> Operation* {
          auto defOp = val.getDefiningOp();
          if (!defOp) {
            LLVM_DEBUG(llvm::dbgs() << "No defining operation for value\n");
            return nullptr;
          }
          
          StringRef opName = defOp->getName().getStringRef();
          LLVM_DEBUG(llvm::dbgs() << "Checking operation: " << opName << "\n");
          
          if (opName == "krnl.global") {
            return defOp;
          } else if (opName == "builtin.unrealized_conversion_cast") {
            auto castOp = mlir::dyn_cast<UnrealizedConversionCastOp>(defOp);
            if (castOp && castOp.getNumOperands() == 1) {
              LLVM_DEBUG(llvm::dbgs() << "Following unrealized_conversion_cast\n");
              return findKrnlGlobal(castOp.getOperand(0));
            }
          }
          LLVM_DEBUG(llvm::dbgs() << "Operation " << opName << " not recognized\n");
          return nullptr;
        };
        
        Operation* krnlGlobalOp = findKrnlGlobal(inputB);
        
        if (krnlGlobalOp) {
          LLVM_DEBUG(llvm::dbgs() << "Found krnl.global operation: " << *krnlGlobalOp << "\n");
          
          // 获取 krnl.global 的属性
          auto shapeAttr = krnlGlobalOp->getAttr("shape").dyn_cast<ArrayAttr>();
          auto valueAttr = krnlGlobalOp->getAttr("value").dyn_cast<DenseElementsAttr>();
          
          if (shapeAttr && valueAttr && shapeAttr.size() == 0 && valueAttr.getNumElements() == 1) {
            if (isFP32) {
              auto values = valueAttr.getValues<float>();
              float floatValue = *values.begin();
              scalarValue = APFloat(floatValue);
              foundScalarValue = true;
              LLVM_DEBUG(llvm::dbgs() << "Extracted FP32 value from krnl.global: " << floatValue << "\n");
            } else if (isFP16) {
              auto values = valueAttr.getValues<APFloat>();
              APFloat fp16Value = *values.begin();
              scalarValue = fp16Value;
              foundScalarValue = true;
              LLVM_DEBUG(llvm::dbgs() << "Extracted FP16 value from krnl.global: " << fp16Value.convertToFloat() << "\n");
            }
          }
        }
      }
      
      // 方法3: 尝试从 arith.constant 提取
      if (!foundScalarValue) {
        if (auto arithConstOp = mlir::dyn_cast<arith::ConstantOp>(inputBDefOp)) {
          if (auto floatAttr = arithConstOp.getValue().dyn_cast<FloatAttr>()) {
            scalarValue = floatAttr.getValue();
            foundScalarValue = true;
            LLVM_DEBUG(llvm::dbgs() << "Extracted value from arith.constant: " << scalarValue.convertToFloat() << "\n");
          }
        }
      }
    }
    
    if (!foundScalarValue) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Could not extract scalar value from Input B\n");
      return rewriter.notifyMatchFailure(divOp, "Could not extract scalar value from Input B");
    }
    
    // 检查除零错误并计算倒数
    if (scalarValue.isZero()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Division by zero\n");
      return rewriter.notifyMatchFailure(divOp, "Division by zero");
    }
    
    APFloat reciprocalValue(APFloat::IEEEsingle());
    
    if (isFP32) {
      // FP32 处理 - 计算倒数
      float floatValue = scalarValue.convertToFloat();
      float reciprocalFloat = 1.0f / floatValue;
      reciprocalValue = APFloat(reciprocalFloat);
      LLVM_DEBUG(llvm::dbgs() << "Calculated FP32 reciprocal: " << reciprocalFloat << "\n");
    } else {
      // FP16 处理 - 在 FP16 精度下计算倒数
      reciprocalValue = APFloat::getOne(APFloat::IEEEhalf());
      auto divResult = reciprocalValue.divide(scalarValue, APFloat::rmNearestTiesToEven);
      if (divResult != APFloat::opOK) {
        LLVM_DEBUG(llvm::dbgs() << "Failed: Division operation failed\n");
        return rewriter.notifyMatchFailure(divOp, "Reciprocal calculation failed");
      }
      LLVM_DEBUG(llvm::dbgs() << "Calculated FP16 reciprocal: " << reciprocalValue.convertToFloat() << "\n");
    }
    
    // 检查输入A的维度
    auto inputShapeA = inputTypeA.getShape();
    LLVM_DEBUG(llvm::dbgs() << "Input A shape dimensions: " << inputShapeA.size() << "\n");
    
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: Input A dimensions must be 1D to 4D\n");
      return rewriter.notifyMatchFailure(divOp, "Input A must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    LLVM_DEBUG(llvm::dbgs() << "Padded dimensions: [" << n << "," << c << "," << h << "," << w << "]\n");
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 使用 memref.global 创建倒数常量 - 根据类型选择
    Type scalarType = isFP32 ? rewriter.getF32Type() : rewriter.getF16Type();
    auto moduleOp = divOp->getParentOfType<ModuleOp>();
    
    // 生成唯一的全局变量名
    std::string globalName = "reciprocal_" + 
                            std::to_string(std::hash<float>{}(reciprocalValue.convertToFloat())) + 
                            (isFP32 ? "_fp32" : "_fp16");
    auto nameAttr = rewriter.getStringAttr(globalName);
    
    // 检查是否已经存在相同的全局变量
    memref::GlobalOp existingGlobal = moduleOp.lookupSymbol<memref::GlobalOp>(globalName);
    memref::GlobalOp reciprocalGlobalOp;
    
    if (!existingGlobal) {
      // 创建标量 memref 类型
      auto scalarMemrefType = MemRefType::get({}, scalarType);
      
      // 创建标量张量类型和 DenseElementsAttr (memref.global 需要 DenseElementsAttr)
      auto scalarTensorType = RankedTensorType::get({}, scalarType);
      DenseElementsAttr reciprocalAttr;
      
      if (isFP32) {
        reciprocalAttr = DenseElementsAttr::get(scalarTensorType, reciprocalValue.convertToFloat());
      } else {
        reciprocalAttr = DenseElementsAttr::get(scalarTensorType, reciprocalValue);
      }
      
      // 在模块开始处创建 memref.global
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      reciprocalGlobalOp = rewriter.create<memref::GlobalOp>(
          loc,
          nameAttr,                    // sym_name
          rewriter.getStringAttr("private"), // sym_visibility  
          scalarMemrefType,            // type
          reciprocalAttr,              // initial_value (must be DenseElementsAttr)
          /*constant=*/true,           // constant
          /*alignment=*/nullptr        // alignment
      );
      
      LLVM_DEBUG(llvm::dbgs() << "Created memref.global for reciprocal: " << *reciprocalGlobalOp << "\n");
    } else {
      reciprocalGlobalOp = existingGlobal;
      LLVM_DEBUG(llvm::dbgs() << "Reusing existing memref.global for reciprocal\n");
    }
    
    // 获取全局变量的引用 - 修改：插入到当前函数的最开始位置
    auto scalarMemrefType = MemRefType::get({}, scalarType);
    Value reciprocalRef;
    {
      // 获取当前函数
      auto funcOp = divOp->getParentOfType<func::FuncOp>();
      
      // 临时改变插入位置到函数开始
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      
      reciprocalRef = rewriter.create<memref::GetGlobalOp>(
          loc, scalarMemrefType, reciprocalGlobalOp.getSymName());
      
      LLVM_DEBUG(llvm::dbgs() << "Created memref.get_global at function start\n");
    }
    
    // 标记输入为缓冲区化
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto reciprocalPtr = getPtr(reciprocalRef);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(divOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");
    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

    // 使用标量乘法函数（用倒数实现除法）
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType, ptrType,  // input, scalar, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数 (inputA * reciprocal)
    std::vector<Value> args = {
      inputPtrA, reciprocalPtr, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(divOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Div to cuDNN MulScalar call (" 
               << (isFP32 ? "FP32" : "FP16") << ")\n");
    return success();
  }
};

// Pattern to convert onnx.AveragePool to a call to mgpuCudnnAveragePoolForward[_fp16]
class AveragePoolOpLowering : public OpRewritePattern<mlir::ONNXAveragePoolOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXAveragePoolOp avgPoolOp, PatternRewriter &rewriter) const override {
    // 获取位置用于错误报告
    Location loc = avgPoolOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.AveragePool at " << loc << "\n");

    // 获取输入张量
    Value input = avgPoolOp.getX();
    
    // 获取输入类型
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Input must have static shape");
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    std::string functionName = "mgpuCudnnAveragePoolForward" + functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << " for element type: " << elementType << "\n");
    
    // 获取输出类型
    auto outputType = mlir::dyn_cast<RankedTensorType>(avgPoolOp.getY().getType());
    if (!outputType || !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Output must have static shape");
    }
    
    // 提取属性
    std::vector<int64_t> kernelShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    
    // 获取核形状（必需）
    if (auto kernelShapeAttr = avgPoolOp.getKernelShapeAttr()) {
      for (auto attr : kernelShapeAttr) {
        kernelShape.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(avgPoolOp, "Kernel shape attribute is required");
    }
    
    // 获取填充、步长和膨胀（可选，有默认值）
    if (auto padsAttr = avgPoolOp.getPadsAttr()) {
      for (auto attr : padsAttr) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无填充
      pads = std::vector<int64_t>(kernelShape.size() * 2, 0);
    }
    
    if (auto stridesAttr = avgPoolOp.getStridesAttr()) {
      for (auto attr : stridesAttr) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：步长为1
      strides = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (auto dilationsAttr = avgPoolOp.getDilationsAttr()) {
      for (auto attr : dilationsAttr) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无膨胀
      dilations = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    // 检查auto_pad属性
    std::string autoPad = "NOTSET";
    if (auto autoPadAttr = avgPoolOp.getAutoPadAttr()) {
      autoPad = autoPadAttr.getValue().str();
    }
    
    // 当前仅支持"NOTSET"自动填充模式
    if (autoPad != "NOTSET") {
      return rewriter.notifyMatchFailure(avgPoolOp, "Only NOTSET auto_pad mode is supported");
    }
    
    // 检查ceil_mode属性
    int64_t ceilMode = 0;
    if (auto ceilModeAttr = avgPoolOp.getCeilModeAttr()) {
      ceilMode = ceilModeAttr.getValue().getSExtValue();
    }
    
    // 检查count_include_pad属性
    int64_t countIncludePad = 0;
    if (auto countIncludePadAttr = avgPoolOp.getCountIncludePadAttr()) {
      countIncludePad = countIncludePadAttr.getValue().getSExtValue();
    }
    
    // 提取输入维度（NCHW）
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // 提取池化参数
    if (kernelShape.size() != 2) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Only 2D pooling is supported");
    }
    int64_t kernel_h = kernelShape[0];
    int64_t kernel_w = kernelShape[1];
    
    if (pads.size() != 4) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Pads must have 4 values");
    }
    int64_t pad_h_begin = pads[0]; // 顶部填充
    int64_t pad_w_begin = pads[1]; // 左侧填充
    int64_t pad_h_end = pads[2];   // 底部填充
    int64_t pad_w_end = pads[3];   // 右侧填充
    
    if (strides.size() != 2) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Strides must have 2 values");
    }
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    
    if (dilations.size() != 2) {
      return rewriter.notifyMatchFailure(avgPoolOp, "Dilations must have 2 values");
    }
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kernelHValue = createI32Const(kernel_h);
    auto kernelWValue = createI32Const(kernel_w);
    auto padHBeginValue = createI32Const(pad_h_begin);
    auto padWBeginValue = createI32Const(pad_w_begin);
    auto padHEndValue = createI32Const(pad_h_end);
    auto padWEndValue = createI32Const(pad_w_end);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    auto ceilModeValue = createI32Const(ceilMode);
    auto countIncludePadValue = createI32Const(countIncludePad);
    
    // 准备bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // 转换memrefs为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针作为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // 分配输出memref
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = avgPoolOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCreateHandlesForStream", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);

    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});
    
    // 查找或创建函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // kernel_h, kernel_w
        i32Type, i32Type,                    // pad_h_begin, pad_w_begin
        i32Type, i32Type,                    // pad_h_end, pad_w_end
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        i32Type,                             // ceil_mode
        i32Type,                             // count_include_pad
        ptrType, ptrType,                    // input_data, output_data
        ptrType                              // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, functionName, funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kernelHValue, kernelWValue,
      padHBeginValue, padWBeginValue,
      padHEndValue, padWEndValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      ceilModeValue,
      countIncludePadValue,
      inputPtr, outputPtr,
      streamPtr
    };
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = avgPoolOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(avgPoolOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.AveragePool to cuDNN call\n");
    return success();
  }
};

// // Pattern to convert onnx.Transpose to specific cuTENSOR calls
// class TransposeOpLowering : public OpRewritePattern<mlir::ONNXTransposeOp>, public ONNXToCuLibsPatternBase {
// public:
//   using OpRewritePattern<mlir::ONNXTransposeOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(mlir::ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
//     Location loc = transposeOp.getLoc();
//     LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Transpose at " << loc << "\n");

//     // 获取输入张量
//     Value input = transposeOp.getData();
    
//     // 获取输入类型
//     auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
//     if (!inputType || !inputType.hasStaticShape()) {
//       return rewriter.notifyMatchFailure(transposeOp, "Input must have static shape");
//     }
    
//     // 检查是否为4D张量
//     auto inputShape = inputType.getShape();
//     if (inputShape.size() != 4) {
//       return rewriter.notifyMatchFailure(transposeOp, "Only 4D tensors are supported");
//     }
    
//     // 获取置换参数
//     std::vector<int64_t> perm;
//     if (auto permAttr = transposeOp.getPermAttr()) {
//       for (auto attr : permAttr) {
//         perm.push_back(attr.cast<IntegerAttr>().getInt());
//       }
//     } else {
//       return rewriter.notifyMatchFailure(transposeOp, "Permutation attribute is required");
//     }
    
//     if (perm.size() != 4) {
//       return rewriter.notifyMatchFailure(transposeOp, "Permutation must have 4 elements");
//     }
    
//     // 检查是否为支持的置换模式
//     std::string functionName;
//     bool isSupported = false;
    
//     if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
//       // [0,2,1,3] 模式
//       functionName = "mgpuCulibsTranspose_0213";
//       isSupported = true;
//       LLVM_DEBUG(llvm::dbgs() << "Detected [0,2,1,3] transpose pattern\n");
//     } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1) {
//       // [0,2,3,1] 模式
//       functionName = "mgpuCulibsTranspose_0231";
//       isSupported = true;
//       LLVM_DEBUG(llvm::dbgs() << "Detected [0,2,3,1] transpose pattern\n");
//     } else {
//       return rewriter.notifyMatchFailure(transposeOp, "Unsupported permutation pattern. Only [0,2,1,3] and [0,2,3,1] are supported");
//     }
    
//     // 获取元素类型并确定函数名后缀
//     Type elementType = inputType.getElementType();
//     std::string functionSuffix = getFunctionSuffix(elementType);
//     functionName += functionSuffix;
    
//     LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << "\n");
    
//     // 提取维度
//     int64_t batch_size = inputShape[0];
//     int64_t dim1 = inputShape[1];
//     int64_t dim2 = inputShape[2];
//     int64_t dim3 = inputShape[3];
    
//     // 创建整数参数常量
//     auto i32Type = rewriter.getI32Type();
//     auto createI32Const = [&](int64_t value) -> Value {
//       return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
//     };
    
//     auto batchSizeValue = createI32Const(batch_size);
//     auto dim1Value = createI32Const(dim1);
//     auto dim2Value = createI32Const(dim2);
//     auto dim3Value = createI32Const(dim3);
    
//     // 准备输入和输出缓冲区
//     auto markForBufferization = [&](Value tensor) -> Value {
//       auto tensorType = tensor.getType().cast<RankedTensorType>();
//       auto memrefType = MemRefType::get(
//         tensorType.getShape(),
//         tensorType.getElementType());
//       return rewriter.create<UnrealizedConversionCastOp>(
//         loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
//     };
    
//     auto inputMemref = markForBufferization(input);
    
//     // 转换 memrefs 为 void pointers
//     auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
//     auto getPtr = [&](Value memref) -> Value {
//       auto indexType = rewriter.getIndexType();
//       auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
//       auto i64Type = rewriter.getIntegerType(64);
//       auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
//       return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
//     };
    
//     auto inputPtr = getPtr(inputMemref);
    
//     // 分配输出 memref
//     auto outputType = mlir::dyn_cast<RankedTensorType>(transposeOp.getResult().getType());
//     auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
//     auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
//     auto outputPtr = getPtr(outputMemref);
    
//     // 创建 CUDA 流
//     auto moduleOp = transposeOp->getParentOfType<ModuleOp>();

//     func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
//     if (!streamCreateFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
//       auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
//       streamCreateFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamCreate", streamCreateType);
//       streamCreateFunc.setPrivate();
//     }

//     // use global handle
//     // func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCreateHandlesForStream");
//     // if (!handleCreateFunc) {
//     //   OpBuilder::InsertionGuard guard(rewriter);
//     //   rewriter.setInsertionPointToStart(moduleOp.getBody());
//     //   auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
//     //   handleCreateFunc = rewriter.create<func::FuncOp>(
//     //     loc, "mgpuCreateHandlesForStream", handleCreateType);
//     //   handleCreateFunc.setPrivate();
//     // }
    
//     auto streamCallOp = rewriter.create<func::CallOp>(
//       loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
//     auto streamPtr = streamCallOp.getResult(0);
    
//     // rewriter.create<func::CallOp>(
//     //   loc, TypeRange{}, "mgpuCreateHandlesForStream", ValueRange{streamPtr});

//     // 查找或创建函数声明
//     func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
//     if (!funcOp) {
//       LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
      
//       // 所有函数都使用相同的签名
//       auto funcType = rewriter.getFunctionType({
//         ptrType, ptrType,                       // input_data, output_data
//         i32Type, i32Type, i32Type, i32Type,     // batch_size, dim1, dim2, dim3
//         ptrType                                 // stream
//       }, {});
      
//       funcOp = rewriter.create<func::FuncOp>(
//         loc, functionName, funcType);
//       funcOp.setPrivate();
//     }
    
//     // 调用函数
//     std::vector<Value> args = {
//       inputPtr, outputPtr,
//       batchSizeValue, dim1Value, dim2Value, dim3Value,
//       streamPtr
//     };
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
//     // 同步并销毁流
//     func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
//     if (!streamSyncFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
//       auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
//       streamSyncFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamSynchronize", streamSyncType);
//       streamSyncFunc.setPrivate();
//     }
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
//     func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
//     if (!streamDestroyFunc) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(moduleOp.getBody());
//       auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
//       streamDestroyFunc = rewriter.create<func::FuncOp>(
//         loc, "mgpuStreamDestroy", streamDestroyType);
//       streamDestroyFunc.setPrivate();
//     }
    
//     rewriter.create<func::CallOp>(
//       loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
//     // 将 memref 转换回 tensor
//     auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
//         loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
//     rewriter.replaceOp(transposeOp, resultTensor);
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Transpose to " << functionName << "\n");
//     return success();
//   }
// };

// Pattern to convert onnx.Transpose to specific cuTENSOR calls (supports 3D and 4D tensors)
class TransposeOpLowering : public OpRewritePattern<mlir::ONNXTransposeOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    Location loc = transposeOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Transpose at " << loc << "\n");

    // 获取输入张量
    Value input = transposeOp.getData();
    
    // 获取输入类型
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(transposeOp, "Input must have static shape");
    }
    
    // 检查是否为3D或4D张量
    auto inputShape = inputType.getShape();
    size_t numDims = inputShape.size();
    if (numDims != 3 && numDims != 4) {
      return rewriter.notifyMatchFailure(transposeOp, "Only 3D and 4D tensors are supported");
    }
    
    // 获取置换参数
    std::vector<int64_t> perm;
    if (auto permAttr = transposeOp.getPermAttr()) {
      for (auto attr : permAttr) {
        perm.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(transposeOp, "Permutation attribute is required");
    }
    
    if (perm.size() != numDims) {
      return rewriter.notifyMatchFailure(transposeOp, "Permutation size must match tensor dimensions");
    }
    
    // 检查是否为支持的置换模式
    std::string functionName;
    bool isSupported = false;
    bool is3D = (numDims == 3);
    
    if (is3D) {
      // 3D tensor patterns
      if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
        // [1,0,2] 模式
        functionName = "mgpuCulibsTranspose_102";
        isSupported = true;
        LLVM_DEBUG(llvm::dbgs() << "Detected 3D [1,0,2] transpose pattern\n");
      } else {
        return rewriter.notifyMatchFailure(transposeOp, "Unsupported 3D permutation pattern. Only [1,0,2] is supported");
      }
    } else {
      // 4D tensor patterns (existing code)
      if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
        // [0,2,1,3] 模式
        functionName = "mgpuCulibsTranspose_0213";
        isSupported = true;
        LLVM_DEBUG(llvm::dbgs() << "Detected 4D [0,2,1,3] transpose pattern\n");
      } else if (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1) {
        // [0,2,3,1] 模式
        functionName = "mgpuCulibsTranspose_0231";
        isSupported = true;
        LLVM_DEBUG(llvm::dbgs() << "Detected 4D [0,2,3,1] transpose pattern\n");
      } else {
        return rewriter.notifyMatchFailure(transposeOp, "Unsupported 4D permutation pattern. Only [0,2,1,3] and [0,2,3,1] are supported");
      }
    }
    
    // 获取元素类型并确定函数名后缀
    Type elementType = inputType.getElementType();
    std::string functionSuffix = getFunctionSuffix(elementType);
    functionName += functionSuffix;
    
    LLVM_DEBUG(llvm::dbgs() << "Using function: " << functionName << "\n");
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    // 准备维度参数
    std::vector<Value> dimensionValues;
    for (size_t i = 0; i < numDims; ++i) {
      dimensionValues.push_back(createI32Const(inputShape[i]));
    }
    
    // 准备输入和输出缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(transposeOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = transposeOp->getParentOfType<ModuleOp>();

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    // 查找或创建函数声明
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(functionName);
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating " << functionName << " declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      // 根据维度创建不同的函数签名
      FunctionType funcType;
      if (is3D) {
        // 3D函数签名: (input_ptr, output_ptr, dim0, dim1, dim2, stream)
        funcType = rewriter.getFunctionType({
          ptrType, ptrType,                       // input_data, output_data
          i32Type, i32Type, i32Type,              // dim0, dim1, dim2
          ptrType                                 // stream
        }, {});
      } else {
        // 4D函数签名: (input_ptr, output_ptr, batch_size, dim1, dim2, dim3, stream)
        funcType = rewriter.getFunctionType({
          ptrType, ptrType,                       // input_data, output_data
          i32Type, i32Type, i32Type, i32Type,     // batch_size, dim1, dim2, dim3
          ptrType                                 // stream
        }, {});
      }
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, StringRef(functionName), funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {inputPtr, outputPtr};
    args.insert(args.end(), dimensionValues.begin(), dimensionValues.end());
    args.push_back(streamPtr);
    
    // 在创建func.CallOp之前，获取原始的onnx_node_name属性
    Attribute onnxNodeNameAttr = transposeOp->getAttr("onnx_node_name");

    auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 如果原始操作有onnx_node_name属性，则传递给新的调用
    if (onnxNodeNameAttr) {
      callOp->setAttr("onnx_node_name", onnxNodeNameAttr);
      LLVM_DEBUG(llvm::dbgs() << "Transferred onnx_node_name: " 
                << onnxNodeNameAttr << " to cuDNN call\n");
    }
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(transposeOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Transpose to " << functionName << "\n");
    return success();
  }
};

// Pattern to convert onnx.Gather to separated affine.for loops for better parallelization
class GatherOpLowering : public OpRewritePattern<mlir::ONNXGatherOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Gather at " << loc << "\n");

    // 获取输入
    Value input = gatherOp.getData();       // 嵌入矩阵，形状如 [vocab_size, embed_dim]
    Value indices = gatherOp.getIndices();  // 索引张量，形状如 [batch_size, seq_len]
    
    // 获取axis属性（默认为0）
    int64_t axis = 0;
    if (auto axisAttr = gatherOp.getAxisAttr()) {
      axis = axisAttr.getValue().getSExtValue();
    }
    
    // 当前只支持axis=0的gather操作（词嵌入查找的典型情况）
    if (axis != 0) {
      return rewriter.notifyMatchFailure(gatherOp, "Only axis=0 gather operations are supported");
    }

    // 获取输入类型
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto indicesType = mlir::dyn_cast<RankedTensorType>(indices.getType());
    auto outputType = mlir::dyn_cast<RankedTensorType>(gatherOp.getResult().getType());
    
    if (!inputType || !inputType.hasStaticShape() ||
        !indicesType || !indicesType.hasStaticShape() ||
        !outputType || !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gatherOp, "All tensors must have static shapes");
    }

    // 验证输入形状
    auto inputShape = inputType.getShape();
    auto indicesShape = indicesType.getShape();
    auto outputShape = outputType.getShape();
    
    if (inputShape.size() != 2) {
      return rewriter.notifyMatchFailure(gatherOp, "Input must be 2D tensor (typical embedding matrix)");
    }
    
    if (indicesShape.size() != 2) {
      return rewriter.notifyMatchFailure(gatherOp, "Indices must be 2D tensor (batch_size, seq_len)");
    }

    // 提取维度
    int64_t vocab_size = inputShape[0];     // 词汇表大小
    int64_t embed_dim = inputShape[1];      // 嵌入维度
    int64_t batch_size = indicesShape[0];   // 批大小
    int64_t seq_len = indicesShape[1];      // 序列长度
    
    LLVM_DEBUG(llvm::dbgs() << "Gather dimensions: vocab_size=" << vocab_size 
               << ", embed_dim=" << embed_dim 
               << ", batch_size=" << batch_size 
               << ", seq_len=" << seq_len << "\n");
    
    // 验证输出形状是否正确
    if (outputShape.size() != 3 || 
        outputShape[0] != batch_size || 
        outputShape[1] != seq_len || 
        outputShape[2] != embed_dim) {
      return rewriter.notifyMatchFailure(gatherOp, "Output shape mismatch");
    }

    // 将输入张量转换为memref
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);     // memref<vocab_size x embed_dim x elementType>
    auto indicesMemref = markForBufferization(indices); // memref<batch_size x seq_len x i64>
    
    // 分配输出memref
    auto outputMemrefType = MemRefType::get(outputShape, outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    
    // 分配临时索引存储memref
    auto indexType = rewriter.getIndexType();
    auto indicesBufferType = MemRefType::get({batch_size, seq_len}, indexType);
    auto indicesBuffer = rewriter.create<memref::AllocOp>(loc, indicesBufferType);
    
    // 创建常量
    auto createIndexConst = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, indexType, rewriter.getIndexAttr(value));
    };
    
    auto vocabSizeConst = createIndexConst(vocab_size);
    auto zeroConst = createIndexConst(0);
    
    LLVM_DEBUG(llvm::dbgs() << "Creating separated affine loops for gather operation\n");
    
    // 第一步：预计算所有索引（与原始逻辑完全一致）
    // affine.for %i = 0 to batch_size {
    //   affine.for %j = 0 to seq_len {
    //     %token_id = affine.load %indicesMemref[%i, %j] : memref<batch_size x seq_len x i64>
    //     %idx = arith.index_cast %token_id : i64 to index
    //     %adjusted_idx = arith.addi %idx, %vocab_size : index
    //     %is_negative = arith.cmpi slt, %idx, %zero : index
    //     %final_idx = arith.select %is_negative, %adjusted_idx, %idx : index
    //     affine.store %final_idx, %indicesBuffer[%i, %j] : memref<batch_size x seq_len x index>
    //   }
    // }
    
    auto outerLoop1 = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/0, /*upperBound=*/batch_size, /*step=*/1);
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerLoop1.getBody());
      Value i = outerLoop1.getInductionVar();
      
      auto innerLoop1 = rewriter.create<affine::AffineForOp>(
          loc, /*lowerBound=*/0, /*upperBound=*/seq_len, /*step=*/1);
      
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop1.getBody());
        Value j = innerLoop1.getInductionVar();
        
        // 加载原始token ID
        auto tokenId = rewriter.create<affine::AffineLoadOp>(
            loc, indicesMemref, ValueRange{i, j});
        
        // 转换为index类型
        auto idx = rewriter.create<arith::IndexCastOp>(
            loc, indexType, tokenId);
        
        // 处理负索引：adjusted_idx = idx + vocab_size
        auto adjustedIdx = rewriter.create<arith::AddIOp>(
            loc, idx, vocabSizeConst);
        
        // 检查是否为负数
        auto isNegative = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, idx, zeroConst);
        
        // 选择最终索引
        auto finalIdx = rewriter.create<arith::SelectOp>(
            loc, isNegative, adjustedIdx, idx);
        
        // 存储计算好的索引
        rewriter.create<affine::AffineStoreOp>(
            loc, finalIdx, indicesBuffer, ValueRange{i, j});
      }
    }
    
    // 第二步：使用预计算的索引进行内存复制
    // affine.for %i = 0 to batch_size {
    //   affine.for %j = 0 to seq_len {
    //     affine.for %k = 0 to embed_dim {
    //       %idx = affine.load %indicesBuffer[%i, %j] : memref<batch_size x seq_len x index>
    //       %value = memref.load %inputMemref[%idx, %k] : memref<vocab_size x embed_dim x elementType>
    //       affine.store %value, %outputMemref[%i, %j, %k] : memref<batch_size x seq_len x embed_dim x elementType>
    //     }
    //   }
    // }
    
    auto outerLoop2 = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/0, /*upperBound=*/batch_size, /*step=*/1);
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerLoop2.getBody());
      Value i = outerLoop2.getInductionVar();
      
      auto innerLoop2 = rewriter.create<affine::AffineForOp>(
          loc, /*lowerBound=*/0, /*upperBound=*/seq_len, /*step=*/1);
      
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop2.getBody());
        Value j = innerLoop2.getInductionVar();
        
        auto embedLoop = rewriter.create<affine::AffineForOp>(
            loc, /*lowerBound=*/0, /*upperBound=*/embed_dim, /*step=*/1);
        
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(embedLoop.getBody());
          Value k = embedLoop.getInductionVar();
          
          // 加载预计算的索引
          auto precomputedIdx = rewriter.create<affine::AffineLoadOp>(
              loc, indicesBuffer, ValueRange{i, j});
          
          // 从输入矩阵加载值
          auto value = rewriter.create<memref::LoadOp>(
              loc, inputMemref, ValueRange{precomputedIdx, k});
          
          // 存储到输出
          rewriter.create<affine::AffineStoreOp>(
              loc, value, outputMemref, ValueRange{i, j, k});
        }
      }
    }
    
    // 将输出memref转换回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gatherOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Gather to separated affine loops\n");
    return success();
  }
};

// Pass to convert ONNX operations to cuDNN calls
struct ONNXToCuDNNPass
    : public PassWrapper<ONNXToCuDNNPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-onnx-to-culibs"; }
  StringRef getDescription() const final {
    return "Convert ONNX operations to cuDNN runtime calls";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<affine::AffineDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    // Define the conversion patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvOpLowering>(context);
    patterns.add<AddOpLowering>(context);
    patterns.add<SubOpLowering>(context);
    patterns.add<MulOpLowering>(context);
    patterns.add<DivOpLowering>(context);
    patterns.add<NegOpLowering>(context);
    patterns.add<MatMulOpLowering>(context);
    patterns.add<GemmOpLowering>(context);
    patterns.add<FlattenAddOpLowering>(context, /*benefit=*/2);
    patterns.add<FlattenGemmOpLowering>(context, /*benefit=*/2);
    patterns.add<FlattenMatMulOpLowering>(context, /*benefit=*/2);
    patterns.add<MaxPoolOpLowering>(context);
    patterns.add<AveragePoolOpLowering>(context);
    patterns.add<TransposeOpLowering>(context);
    patterns.add<GatherOpLowering>(context);
    

    // Apply patterns
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect, 
                           memref::MemRefDialect, affine::AffineDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<arith::IndexCastOp>();                       
    target.addIllegalOp<mlir::ONNXConvOp>();
    target.addIllegalOp<mlir::ONNXAddOp>();
    target.addIllegalOp<mlir::ONNXSubOp>();
    target.addIllegalOp<mlir::ONNXMulOp>();
    target.addIllegalOp<mlir::ONNXDivOp>();
    target.addIllegalOp<mlir::ONNXNegOp>();
    target.addIllegalOp<mlir::ONNXMatMulOp>();
    target.addIllegalOp<mlir::ONNXGemmOp>();
    target.addIllegalOp<mlir::ONNXMaxPoolSingleOutOp>();
    target.addIllegalOp<mlir::ONNXAveragePoolOp>();
    target.addIllegalOp<mlir::ONNXTransposeOp>();
    target.addIllegalOp<mlir::ONNXGatherOp>();
    
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Pass registration
namespace onnx_mlir {
    std::unique_ptr<Pass> createONNXToCuDNNPass() {
      return std::make_unique<ONNXToCuDNNPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<ONNXToCuDNNPass> pass;