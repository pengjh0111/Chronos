#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "convert-mha-pattern-to-op"

namespace {

// Pattern to convert Multi-Head Attention operation sequence to onnx.MHA
class MultiHeadAttentionPatternToOp : public OpRewritePattern<mlir::ONNXMatMulOp> {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  // 辅助函数：从Value中提取常量属性
  static std::optional<Attribute> getConstantValue(Value value) {
    auto defOp = value.getDefiningOp();
    if (!defOp) return std::nullopt;
    
    if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(defOp)) {
      return constOp.getValue();
    }
    
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp outputProjMatMul, 
                                PatternRewriter &rewriter) const override {
    Location loc = outputProjMatMul.getLoc();
    
    LLVM_DEBUG(llvm::dbgs() << "Attempting to match MHA pattern at " << loc << "\n");
    
    llvm::SmallVector<Operation*> opsToErase;
    
    // Step 1: 验证这是输出投影MatMul并获取其输入
    Value reshapedAttnOutput = outputProjMatMul.getA();
    Value outputProjWeight = outputProjMatMul.getB();
    
    // Step 2: 匹配final Reshape
    auto finalReshapeOp = reshapedAttnOutput.getDefiningOp<mlir::ONNXReshapeOp>();
    if (!finalReshapeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Expected Reshape before output projection");
    }
    opsToErase.push_back(finalReshapeOp);
    
    // Step 3: 匹配final Transpose
    Value transposedAttnOutput = finalReshapeOp.getData();
    auto finalTransposeOp = transposedAttnOutput.getDefiningOp<mlir::ONNXTransposeOp>();
    if (!finalTransposeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Expected Transpose before final Reshape");
    }
    
    // 验证Transpose的perm属性 [0, 2, 1, 3]
    auto finalTransposePerm = finalTransposeOp.getPerm();
    if (!finalTransposePerm || finalTransposePerm->size() != 4) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Invalid transpose permutation");
    }
    opsToErase.push_back(finalTransposeOp);
    
    // Step 4: 匹配attention output MatMul (Softmax @ V)
    Value attnOutputMatMulResult = finalTransposeOp.getData();
    auto attnOutputMatMul = attnOutputMatMulResult.getDefiningOp<mlir::ONNXMatMulOp>();
    if (!attnOutputMatMul) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Expected MatMul (attention @ V)");
    }
    opsToErase.push_back(attnOutputMatMul);
    
    Value attnWeights = attnOutputMatMul.getA();
    Value vTransposed = attnOutputMatMul.getB();
    
    // Step 5: 匹配Softmax
    auto softmaxOp = attnWeights.getDefiningOp<mlir::ONNXSoftmaxOp>();
    if (!softmaxOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected Softmax");
    }
    opsToErase.push_back(softmaxOp);
    
    // Step 6: 匹配scale Mul
    Value scaledScores = softmaxOp.getInput();
    auto scaleMulOp = scaledScores.getDefiningOp<mlir::ONNXMulOp>();
    if (!scaleMulOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected scale Mul");
    }
    opsToErase.push_back(scaleMulOp);
    
    Value qkScores = scaleMulOp.getA();
    Value scaleValue = scaleMulOp.getB();
    
    // Step 7: 匹配QK MatMul
    auto qkMatMul = qkScores.getDefiningOp<mlir::ONNXMatMulOp>();
    if (!qkMatMul) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected Q*K^T MatMul");
    }
    opsToErase.push_back(qkMatMul);
    
    Value qTransposed = qkMatMul.getA();
    Value kTransposed = qkMatMul.getB();
    
    // Step 8-10: 匹配Q, K, V Transpose
    auto qTransposeOp = qTransposed.getDefiningOp<mlir::ONNXTransposeOp>();
    if (!qTransposeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected Q Transpose");
    }
    opsToErase.push_back(qTransposeOp);
    
    auto kTransposeOp = kTransposed.getDefiningOp<mlir::ONNXTransposeOp>();
    if (!kTransposeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected K Transpose");
    }
    opsToErase.push_back(kTransposeOp);
    
    auto vTransposeOp = vTransposed.getDefiningOp<mlir::ONNXTransposeOp>();
    if (!vTransposeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected V Transpose");
    }
    opsToErase.push_back(vTransposeOp);
    
    // Step 11-13: 匹配Q, K, V Reshape
    Value qReshaped = qTransposeOp.getData();
    auto qReshapeOp = qReshaped.getDefiningOp<mlir::ONNXReshapeOp>();
    if (!qReshapeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected Q Reshape");
    }
    opsToErase.push_back(qReshapeOp);
    
    Value kReshaped = kTransposeOp.getData();
    auto kReshapeOp = kReshaped.getDefiningOp<mlir::ONNXReshapeOp>();
    if (!kReshapeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected K Reshape");
    }
    opsToErase.push_back(kReshapeOp);
    
    Value vReshaped = vTransposeOp.getData();
    auto vReshapeOp = vReshaped.getDefiningOp<mlir::ONNXReshapeOp>();
    if (!vReshapeOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected V Reshape");
    }
    opsToErase.push_back(vReshapeOp);
    
    // Step 14-16: 匹配Q, K, V Slice
    Value qSliced = qReshapeOp.getData();
    auto qSliceOp = qSliced.getDefiningOp<mlir::ONNXSliceOp>();
    if (!qSliceOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected Q Slice");
    }
    opsToErase.push_back(qSliceOp);
    
    Value kSliced = kReshapeOp.getData();
    auto kSliceOp = kSliced.getDefiningOp<mlir::ONNXSliceOp>();
    if (!kSliceOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected K Slice");
    }
    opsToErase.push_back(kSliceOp);
    
    Value vSliced = vReshapeOp.getData();
    auto vSliceOp = vSliced.getDefiningOp<mlir::ONNXSliceOp>();
    if (!vSliceOp) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected V Slice");
    }
    opsToErase.push_back(vSliceOp);
    
    // Step 17: 验证三个Slice来自同一个QKV投影
    Value qkvProjected = qSliceOp.getData();
    if (qkvProjected != kSliceOp.getData() || qkvProjected != vSliceOp.getData()) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Q, K, V must be sliced from same QKV projection");
    }
    
    // Step 18: 匹配QKV投影MatMul
    auto qkvProjMatMul = qkvProjected.getDefiningOp<mlir::ONNXMatMulOp>();
    if (!qkvProjMatMul) {
      return rewriter.notifyMatchFailure(outputProjMatMul, "Expected QKV projection MatMul");
    }
    opsToErase.push_back(qkvProjMatMul);
    
    Value originalInput = qkvProjMatMul.getA();
    Value qkvWeight = qkvProjMatMul.getB();
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully matched complete MHA pattern\n");
    
    // ========== 提取参数 ==========
    
    // 从输入提取维度
    auto inputType = mlir::dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Input must have static shape");
    }
    
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 3) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Expected 3D input [batch, seq_len, hidden_dim]");
    }
    
    // 从QKV权重提取维度
    auto qkvWeightType = mlir::dyn_cast<RankedTensorType>(qkvWeight.getType());
    if (!qkvWeightType || !qkvWeightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "QKV weight must have static shape");
    }
    
    auto qkvWeightShape = qkvWeightType.getShape();
    int64_t qkv_total_dim = qkvWeightShape[1];
    int64_t single_proj_dim = qkv_total_dim / 3;
    
    // 从Q Reshape提取num_heads和head_dim
    Value qReshapeShape = qReshapeOp.getShape();
    auto qReshapeShapeAttrOpt = getConstantValue(qReshapeShape);
    
    if (!qReshapeShapeAttrOpt.has_value()) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Q Reshape shape must be constant");
    }
    
    auto qReshapeShapeDense = mlir::dyn_cast<DenseElementsAttr>(qReshapeShapeAttrOpt.value());
    if (!qReshapeShapeDense) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Q Reshape shape must be dense");
    }
    
    auto qReshapeValues = qReshapeShapeDense.getValues<int64_t>();
    if (qReshapeValues.size() != 4) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Q Reshape must be 4D");
    }
    
    int64_t num_heads = *(qReshapeValues.begin() + 2);
    int64_t head_dim = *(qReshapeValues.begin() + 3);
    
    // 验证维度一致性
    if (single_proj_dim != num_heads * head_dim) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Projection dimension mismatch");
    }
    
    int64_t q_proj_size = head_dim;
    int64_t k_proj_size = head_dim;
    int64_t v_proj_size = head_dim;
    
    // 从输出投影权重提取输出维度
    auto outputProjWeightType = mlir::dyn_cast<RankedTensorType>(
      outputProjWeight.getType());
    if (!outputProjWeightType || !outputProjWeightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(outputProjMatMul, 
        "Output projection weight must have static shape");
    }
    
    auto outputProjWeightShape = outputProjWeightType.getShape();
    int64_t o_proj_size = outputProjWeightShape[1];
    
    // 提取scale值
    double sm_scaler = 1.0;
    auto scaleAttrOpt = getConstantValue(scaleValue);
    if (scaleAttrOpt.has_value()) {
      if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(scaleAttrOpt.value())) {
        if (denseAttr.isSplat()) {
          auto splatValue = denseAttr.getSplatValue<APFloat>();
          sm_scaler = splatValue.convertToDouble();
        } else if (denseAttr.getNumElements() > 0) {
          auto values = denseAttr.getValues<APFloat>();
          sm_scaler = (*values.begin()).convertToDouble();
        }
      } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(scaleAttrOpt.value())) {
        sm_scaler = floatAttr.getValueAsDouble();
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "MHA parameters:\n"
               << "  num_heads=" << num_heads << "\n"
               << "  q_proj_size=" << q_proj_size << "\n"
               << "  k_proj_size=" << k_proj_size << "\n"
               << "  v_proj_size=" << v_proj_size << "\n"
               << "  o_proj_size=" << o_proj_size << "\n"
               << "  sm_scaler=" << sm_scaler << "\n");
    
    // ========== 合并权重 ==========
    
    // 使用onnx.Concat在第二维度(axis=1)上连接QKV权重和输出投影权重
    // 形状: [input_dim, 3*proj_dim] + [input_dim, output_dim] -> [input_dim, 3*proj_dim + output_dim]
    auto axisAttr = rewriter.getIntegerAttr(
    rewriter.getIntegerType(64, /*isSigned=*/true), 
    1
    );

    auto qkvWeightShape_int = qkvWeightType.getShape();
    auto outputWeightShape_int = outputProjWeightType.getShape();
    int64_t input_dim = qkvWeightShape_int[0];
    int64_t combined_dim = qkvWeightShape_int[1] + outputWeightShape_int[1];

    auto combinedWeightType = RankedTensorType::get(
    {input_dim, combined_dim},
    qkvWeightType.getElementType()
    );

    // 使用通用的create方法创建Concat操作
    OperationState concatState(loc, "onnx.Concat");
    concatState.addOperands({qkvWeight, outputProjWeight});
    concatState.addTypes(combinedWeightType);
    concatState.addAttribute("axis", axisAttr);
    Operation* concatOpRaw = rewriter.create(concatState);

    Value mergedWeights = concatOpRaw->getResult(0);
    
    // ========== 创建onnx.MHA操作 ==========
    
    auto outputType = outputProjMatMul.getResult().getType();
    
    // 创建None值用于lo_win_idx和hi_win_idx
    Value noneValue = rewriter.create<ONNXNoneOp>(loc).getResult();
    
    // 创建属性
    auto numHeadsAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true), num_heads);
    auto qProjSizeAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true), q_proj_size);
    auto kProjSizeAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true), k_proj_size);
    auto vProjSizeAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true), v_proj_size);
    auto oProjSizeAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true), o_proj_size);
    auto smScalerAttr = rewriter.getF64FloatAttr(sm_scaler);
    
    // 创建MHA操作 (self-attention: Q=K=V=originalInput)
    auto mhaOp = rewriter.create<ONNXMHAOp>(
      loc,
      outputType,
      originalInput,  // Q
      originalInput,  // K  
      originalInput,  // V
      mergedWeights,  // 合并的权重
      noneValue,      // lo_win_idx
      noneValue,      // hi_win_idx
      numHeadsAttr,
      qProjSizeAttr,
      kProjSizeAttr,
      vProjSizeAttr,
      oProjSizeAttr,
      smScalerAttr
    );
    
    // 保留onnx_node_name属性
    if (auto onnxNodeNameAttr = outputProjMatMul->getAttr("onnx_node_name")) {
      mhaOp->setAttr("onnx_node_name", onnxNodeNameAttr);
    }
    
    // 替换输出投影MatMul
    rewriter.replaceOp(outputProjMatMul, mhaOp.getResult());
    
    // 清理中间操作（逆序删除避免use-after-free）
    for (Operation* op : llvm::reverse(opsToErase)) {
      if (op && op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted MHA pattern to onnx.MHA op\n");
    return success();
  }
};

class ConvertMHAPatternToOpPass
    : public PassWrapper<ConvertMHAPatternToOpPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-mha-pattern-to-op"; 
  }
  
  StringRef getDescription() const final {
    return "Convert MHA operation patterns to onnx.MHA op";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Convert MHA Pattern to Op Pass ===\n");
    
    RewritePatternSet patterns(&getContext());
    patterns.add<MultiHeadAttentionPatternToOp>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to apply patterns\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully applied patterns\n");
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createConvertMHAPatternToOpPass() {
  return std::make_unique<ConvertMHAPatternToOpPass>();
}

} // namespace onnx_mlir

// Pass注册
static mlir::PassRegistration<ConvertMHAPatternToOpPass> pass;