#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "convert-concat-split-to-affine"

namespace {

// 存储 Concat 操作信息
struct ConcatOpInfo {
  Operation* concatOp;
  std::vector<Value> inputs;
  Value output;
  int64_t axis;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<int64_t> outputShape;
  Location loc;
  
  ConcatOpInfo(Operation* op) : concatOp(op), loc(op->getLoc()) {}
};

// 存储 Split 操作信息
struct SplitOpInfo {
  Operation* splitOp;
  Value input;
  std::vector<Value> outputs;
  int64_t axis;
  std::vector<int64_t> inputShape;
  std::vector<std::vector<int64_t>> outputShapes;
  Location loc;
  
  SplitOpInfo(Operation* op) : splitOp(op), loc(op->getLoc()) {}
};

class ConvertConcatSplitToAffinePass
    : public PassWrapper<ConvertConcatSplitToAffinePass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { 
    return "convert-concat-split-to-affine"; 
  }
  
  StringRef getDescription() const final {
    return "Convert onnx.Concat and onnx.Split to affine loop implementations";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Convert Concat/Split to Affine Pass ===\n");
    
    // 收集所有 Concat 和 Split 操作
    std::vector<ConcatOpInfo> concatOps;
    std::vector<SplitOpInfo> splitOps;
    
    if (failed(collectConcatSplitOps(funcOp, concatOps, splitOps))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << concatOps.size() 
               << " concat ops and " << splitOps.size() << " split ops\n");
    
    // 转换 Concat 操作
    for (auto& info : concatOps) {
      if (failed(convertConcatToAffineLoop(info))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert a concat op\n");
        signalPassFailure();
        return;
      }
    }
    
    // 转换 Split 操作
    for (auto& info : splitOps) {
      if (failed(convertSplitToAffineLoop(info))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert a split op\n");
        signalPassFailure();
        return;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted all operations\n");
  }

private:
  // 收集所有 Concat 和 Split 操作
  LogicalResult collectConcatSplitOps(
      func::FuncOp funcOp,
      std::vector<ConcatOpInfo>& concatOps,
      std::vector<SplitOpInfo>& splitOps) {
    
    funcOp.walk([&](Operation* op) {
      if (op->getName().getStringRef() == "onnx.Concat") {
        ConcatOpInfo info(op);
        if (succeeded(extractConcatInfo(op, info))) {
          concatOps.push_back(info);
        }
      } else if (op->getName().getStringRef() == "onnx.Split") {
        SplitOpInfo info(op);
        if (succeeded(extractSplitInfo(op, info))) {
          splitOps.push_back(info);
        }
      }
    });
    
    return success();
  }

  // 提取 Concat 操作信息
  LogicalResult extractConcatInfo(Operation* op, ConcatOpInfo& info) {
    // 获取 axis 属性
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis")) {
      // 使用 getValue().getSExtValue() 来支持 si64 等类型
      info.axis = axisAttr.getValue().getSExtValue();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Concat: no axis attribute\n");
      return failure();
    }
    
    // 获取输入
    for (auto operand : op->getOperands()) {
      info.inputs.push_back(operand);
      
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (!tensorType || !tensorType.hasStaticShape()) {
        LLVM_DEBUG(llvm::dbgs() << "Concat: input not static tensor\n");
        return failure();
      }
      
      std::vector<int64_t> shape(tensorType.getShape().begin(), 
                                  tensorType.getShape().end());
      info.inputShapes.push_back(shape);
    }
    
    // 获取输出
    if (op->getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Concat: expected 1 result\n");
      return failure();
    }
    
    info.output = op->getResult(0);
    auto outputType = info.output.getType().dyn_cast<RankedTensorType>();
    if (!outputType || !outputType.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Concat: output not static tensor\n");
      return failure();
    }
    
    info.outputShape.assign(outputType.getShape().begin(), 
                           outputType.getShape().end());
    
    LLVM_DEBUG(llvm::dbgs() << "Concat: axis=" << info.axis 
               << ", num_inputs=" << info.inputs.size() << "\n");
    
    return success();
  }

  // 提取 Split 操作信息
  LogicalResult extractSplitInfo(Operation* op, SplitOpInfo& info) {
    // 获取 axis 属性
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis")) {
      // 使用 getValue().getSExtValue() 来支持 si64 等类型
      info.axis = axisAttr.getValue().getSExtValue();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Split: no axis attribute\n");
      return failure();
    }
    
    // 获取输入
    if (op->getNumOperands() < 1) {
      LLVM_DEBUG(llvm::dbgs() << "Split: no input\n");
      return failure();
    }
    
    info.input = op->getOperand(0);
    auto inputType = info.input.getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Split: input not static tensor\n");
      return failure();
    }
    
    info.inputShape.assign(inputType.getShape().begin(), 
                          inputType.getShape().end());
    
    // 获取输出
    for (auto result : op->getResults()) {
      info.outputs.push_back(result);
      
      auto tensorType = result.getType().dyn_cast<RankedTensorType>();
      if (!tensorType || !tensorType.hasStaticShape()) {
        LLVM_DEBUG(llvm::dbgs() << "Split: output not static tensor\n");
        return failure();
      }
      
      std::vector<int64_t> shape(tensorType.getShape().begin(), 
                                  tensorType.getShape().end());
      info.outputShapes.push_back(shape);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Split: axis=" << info.axis 
               << ", num_outputs=" << info.outputs.size() << "\n");
    
    return success();
  }

  // 将 Concat 转换为 affine loop
  LogicalResult convertConcatToAffineLoop(ConcatOpInfo& info) {
    LLVM_DEBUG(llvm::dbgs() << "\nConverting Concat operation\n");
    
    // 在原始操作之前插入
    OpBuilder builder(info.concatOp);
    builder.setInsertionPoint(info.concatOp);
    
    // 创建输出 memref
    auto outputType = MemRefType::get(info.outputShape, builder.getF32Type());
    auto allocOp = builder.create<memref::AllocOp>(info.loc, outputType);
    allocOp->setAttr("alignment", builder.getI64IntegerAttr(16));
    Value outputMemref = allocOp.getResult();
    
    // 处理输入：检查每个输入的实际类型
    std::vector<Value> inputMemrefs;
    for (size_t i = 0; i < info.inputs.size(); ++i) {
      Value input = info.inputs[i];
      Value memrefInput;
      
      // 检查输入的实际运行时类型
      Type inputType = input.getType();
      if (inputType.isa<MemRefType>()) {
        // 已经是 memref，直接使用
        memrefInput = input;
        LLVM_DEBUG(llvm::dbgs() << "  Input " << i << " is already memref\n");
      } else if (auto tensorType = inputType.dyn_cast<RankedTensorType>()) {
        // 是 tensor，需要通过 bufferization.to_memref 获取对应的 memref
        // 或者假设在运行这个 pass 之前，已经做了 bufferization
        // 这里我们创建一个 bufferization.to_memref 操作
        LLVM_DEBUG(llvm::dbgs() << "  Input " << i << " is tensor, needs conversion\n");
        
        auto memrefType = MemRefType::get(info.inputShapes[i], builder.getF32Type());
        
        // 使用 unrealized_conversion_cast 作为临时方案
        // 在实际使用中，应该在此 pass 之前运行 bufferization
        auto castOp = builder.create<UnrealizedConversionCastOp>(
            info.loc, memrefType, input);
        memrefInput = castOp.getResult(0);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Unsupported input type\n");
        return failure();
      }
      
      inputMemrefs.push_back(memrefInput);
    }
    
    // 计算每个输入在 concat 轴上的偏移
    std::vector<int64_t> axisOffsets;
    int64_t currentOffset = 0;
    for (const auto& shape : info.inputShapes) {
      axisOffsets.push_back(currentOffset);
      currentOffset += shape[info.axis];
    }
    
    // 获取维度数量
    size_t rank = info.outputShape.size();
    
    // 创建嵌套循环来遍历第一个输入的所有维度
    // 外层循环对应非 axis 维度，内层对应所有输入的复制
    std::vector<int64_t> loopBounds = info.inputShapes[0];
    
    // 创建循环嵌套
    std::function<void(OpBuilder&, size_t, std::vector<Value>&)> buildLoops;
    buildLoops = [&](OpBuilder& b, size_t dim, std::vector<Value>& ivs) {
      if (dim == rank) {
        // 在最内层，为每个输入生成 load 和 store
        for (size_t i = 0; i < inputMemrefs.size(); ++i) {
          // 构建输入索引
          std::vector<Value> inputIndices = ivs;
          
          // 构建输出索引（需要调整 axis 维度）
          std::vector<Value> outputIndices = ivs;
          
          // 在 axis 维度上加上偏移
          if (axisOffsets[i] > 0) {
            // 使用 affine.apply 来计算偏移后的索引
            // 创建 affine map: (d0) -> (d0 + offset)
            AffineExpr d0 = b.getAffineDimExpr(0);
            AffineExpr offsetExpr = b.getAffineConstantExpr(axisOffsets[i]);
            AffineMap map = AffineMap::get(1, 0, d0 + offsetExpr, b.getContext());
            Value axisIV = ivs[info.axis];
            Value newAxisIdx = b.create<AffineApplyOp>(info.loc, map, ValueRange{axisIV});
            outputIndices[info.axis] = newAxisIdx;
          }
          
          // Load from input
          Value val = b.create<AffineLoadOp>(info.loc, inputMemrefs[i], inputIndices);
          
          // Store to output
          b.create<AffineStoreOp>(info.loc, val, outputMemref, outputIndices);
        }
        return;
      }
      
      // 创建当前维度的循环
      auto forOp = b.create<AffineForOp>(info.loc, 0, loopBounds[dim]);
      ivs.push_back(forOp.getInductionVar());
      
      // 在循环体中插入操作
      b.setInsertionPointToStart(forOp.getBody());
      buildLoops(b, dim + 1, ivs);
      
      ivs.pop_back();
    };
    
    std::vector<Value> ivs;
    buildLoops(builder, 0, ivs);
    
    // 重置插入点到原始操作之后（循环之后）
    builder.setInsertionPoint(info.concatOp);
    
    // 替换原始操作的使用
    // 如果原始输出是 tensor 类型，需要转换回 tensor
    Value replacementValue;
    if (info.output.getType().isa<RankedTensorType>()) {
      auto castOp = builder.create<UnrealizedConversionCastOp>(
          info.loc, info.output.getType(), outputMemref);
      replacementValue = castOp.getResult(0);
    } else {
      replacementValue = outputMemref;
    }
    
    // 替换所有使用
    info.output.replaceAllUsesWith(replacementValue);
    
    // 删除原始操作
    info.concatOp->erase();
    
    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted Concat\n");
    return success();
  }

  // 将 Split 转换为 affine loop
  LogicalResult convertSplitToAffineLoop(SplitOpInfo& info) {
    LLVM_DEBUG(llvm::dbgs() << "\nConverting Split operation\n");
    
    // 在原始操作之前插入
    OpBuilder builder(info.splitOp);
    builder.setInsertionPoint(info.splitOp);
    
    // 检查输入类型并获取 memref
    Value inputMemref;
    Type inputType = info.input.getType();
    
    if (inputType.isa<MemRefType>()) {
      inputMemref = info.input;
      LLVM_DEBUG(llvm::dbgs() << "  Input is already memref\n");
    } else if (auto tensorType = inputType.dyn_cast<RankedTensorType>()) {
      LLVM_DEBUG(llvm::dbgs() << "  Input is tensor, needs conversion\n");
      auto memrefType = MemRefType::get(info.inputShape, builder.getF32Type());
      auto castOp = builder.create<UnrealizedConversionCastOp>(
          info.loc, memrefType, info.input);
      inputMemref = castOp.getResult(0);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  Unsupported input type\n");
      return failure();
    }
    
    // 为每个输出创建 memref
    std::vector<Value> outputMemrefs;
    for (const auto& shape : info.outputShapes) {
      auto memrefType = MemRefType::get(shape, builder.getF32Type());
      auto allocOp = builder.create<memref::AllocOp>(info.loc, memrefType);
      allocOp->setAttr("alignment", builder.getI64IntegerAttr(16));
      outputMemrefs.push_back(allocOp.getResult());
    }
    
    // 计算每个输出在 split 轴上的偏移
    std::vector<int64_t> axisOffsets;
    int64_t currentOffset = 0;
    for (const auto& shape : info.outputShapes) {
      axisOffsets.push_back(currentOffset);
      currentOffset += shape[info.axis];
    }
    
    // 获取维度数量
    size_t rank = info.outputShapes[0].size();
    std::vector<int64_t> loopBounds = info.outputShapes[0];
    
    // 创建循环嵌套
    std::function<void(OpBuilder&, size_t, std::vector<Value>&)> buildLoops;
    buildLoops = [&](OpBuilder& b, size_t dim, std::vector<Value>& ivs) {
      if (dim == rank) {
        // 在最内层，为每个输出生成 load 和 store
        for (size_t i = 0; i < outputMemrefs.size(); ++i) {
          // 构建输入索引（需要调整 axis 维度）
          std::vector<Value> inputIndices = ivs;
          
          // 在 axis 维度上加上偏移
          if (axisOffsets[i] > 0) {
            // 使用 affine.apply 来计算偏移后的索引
            // 创建 affine map: (d0) -> (d0 + offset)
            AffineExpr d0 = b.getAffineDimExpr(0);
            AffineExpr offsetExpr = b.getAffineConstantExpr(axisOffsets[i]);
            AffineMap map = AffineMap::get(1, 0, d0 + offsetExpr, b.getContext());
            Value axisIV = ivs[info.axis];
            Value newAxisIdx = b.create<AffineApplyOp>(info.loc, map, ValueRange{axisIV});
            inputIndices[info.axis] = newAxisIdx;
          }
          
          // 构建输出索引
          std::vector<Value> outputIndices = ivs;
          
          // Load from input
          Value val = b.create<AffineLoadOp>(info.loc, inputMemref, inputIndices);
          
          // Store to output
          b.create<AffineStoreOp>(info.loc, val, outputMemrefs[i], outputIndices);
        }
        return;
      }
      
      // 创建当前维度的循环
      auto forOp = b.create<AffineForOp>(info.loc, 0, loopBounds[dim]);
      ivs.push_back(forOp.getInductionVar());
      
      // 在循环体中插入操作
      b.setInsertionPointToStart(forOp.getBody());
      buildLoops(b, dim + 1, ivs);
      
      ivs.pop_back();
    };
    
    std::vector<Value> ivs;
    buildLoops(builder, 0, ivs);
    
    // 重置插入点到原始操作之后（循环之后）
    builder.setInsertionPoint(info.splitOp);
    
    // 替换原始操作的使用
    std::vector<Value> replacementValues;
    for (size_t i = 0; i < info.outputs.size(); ++i) {
      // 如果原始输出是 tensor 类型，需要转换回 tensor
      if (info.outputs[i].getType().isa<RankedTensorType>()) {
        auto castOp = builder.create<UnrealizedConversionCastOp>(
            info.loc, info.outputs[i].getType(), outputMemrefs[i]);
        replacementValues.push_back(castOp.getResult(0));
      } else {
        replacementValues.push_back(outputMemrefs[i]);
      }
    }
    
    // 替换所有使用
    for (size_t i = 0; i < info.outputs.size(); ++i) {
      info.outputs[i].replaceAllUsesWith(replacementValues[i]);
    }
    
    // 删除原始操作
    info.splitOp->erase();
    
    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted Split\n");
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createConvertConcatSplitToAffinePass() {
  return std::make_unique<ConvertConcatSplitToAffinePass>();
}

} // namespace onnx_mlir

// Pass 注册
static mlir::PassRegistration<ConvertConcatSplitToAffinePass> pass;