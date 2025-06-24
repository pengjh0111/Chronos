#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

#define DEBUG_TYPE "f16-threshold-optimization"

namespace {

class F16ThresholdOptimizationPattern : public OpRewritePattern<ONNXCastOp> {
public:
  using OpRewritePattern<ONNXCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCastOp castOp, 
                               PatternRewriter &rewriter) const override {
    // 检查第一个Cast操作：f16 -> f32
    Value castInput = castOp.getOperand();
    Type inputType = castInput.getType();
    Type outputType = castOp.getResult().getType();
    
    // 检查是否是f16到f32的转换
    if (!isF16ToF32Cast(inputType, outputType))
      return failure();
    
    // 查找使用这个cast结果的GreaterOrEqual操作
    if (!castOp->hasOneUse())
      return failure();
      
    Operation *user = *castOp->getUsers().begin();
    auto greaterOrEqualOp = dyn_cast<ONNXGreaterOrEqualOp>(user);
    if (!greaterOrEqualOp)
      return failure();
    
    // 检查GreaterOrEqual的第二个操作数是否是f32常量
    Value thresholdValue = greaterOrEqualOp.getOperand(1);
    auto constantOp = thresholdValue.getDefiningOp<ONNXConstantOp>();
    if (!constantOp)
      return failure();
    
    // 检查常量是否是f32类型
    Type thresholdType = thresholdValue.getType();
    if (!isF32Scalar(thresholdType))
      return failure();
    
    // 检查GreaterOrEqual的输出是否只被一个Cast使用（i1 -> f16）
    if (!greaterOrEqualOp->hasOneUse())
      return failure();
      
    Operation *finalUser = *greaterOrEqualOp->getUsers().begin();
    auto finalCastOp = dyn_cast<ONNXCastOp>(finalUser);
    if (!finalCastOp)
      return failure();
    
    // 检查最后的Cast是否是i1到f16的转换
    Type finalInputType = finalCastOp.getOperand().getType();
    Type finalOutputType = finalCastOp.getResult().getType();
    if (!isI1ToF16Cast(finalInputType, finalOutputType))
      return failure();
    
    LLVM_DEBUG(llvm::dbgs() << "Found f16 threshold optimization pattern\n");
    
    // 执行优化
    // 1. 创建f16版本的阈值常量
    Value f16ThresholdConstant = createF16Threshold(rewriter, constantOp, castOp.getLoc());
    if (!f16ThresholdConstant)
      return failure();
    
    // 2. 创建新的GreaterOrEqual操作，直接使用f16输入和f16阈值
    auto newGreaterOrEqualOp = rewriter.create<ONNXGreaterOrEqualOp>(
        greaterOrEqualOp.getLoc(),
        greaterOrEqualOp.getResult().getType(),
        castInput,  // 直接使用原始f16输入
        f16ThresholdConstant
    );
    
    // 3. 替换GreaterOrEqual操作的结果
    rewriter.replaceOp(greaterOrEqualOp, newGreaterOrEqualOp.getResult());
    
    // 4. 现在finalCastOp的输入已经更新，保持最后的Cast操作不变
    
    return success();
  }

private:
  bool isF16ToF32Cast(Type inputType, Type outputType) const {
    auto inputTensorType = inputType.dyn_cast<TensorType>();
    auto outputTensorType = outputType.dyn_cast<TensorType>();
    
    if (!inputTensorType || !outputTensorType)
      return false;
    
    return inputTensorType.getElementType().isF16() &&
           outputTensorType.getElementType().isF32();
  }
  
  bool isI1ToF16Cast(Type inputType, Type outputType) const {
    auto inputTensorType = inputType.dyn_cast<TensorType>();
    auto outputTensorType = outputType.dyn_cast<TensorType>();
    
    if (!inputTensorType || !outputTensorType)
      return false;
    
    return inputTensorType.getElementType().isInteger(1) &&
           outputTensorType.getElementType().isF16();
  }
  
  bool isF32Scalar(Type type) const {
    auto tensorType = type.dyn_cast<TensorType>();
    if (!tensorType)
      return false;
    
    return tensorType.getElementType().isF32();
  }
  
  Value createF16Threshold(PatternRewriter &rewriter, ONNXConstantOp f32Constant, Location loc) const {
    // 获取f32常量值
    auto valueAttr = f32Constant.getValue();
    if (!valueAttr.has_value())
      return nullptr;
      
    auto denseAttr = valueAttr->dyn_cast<DenseElementsAttr>();
    if (!denseAttr)
      return nullptr;
    
    // 获取f32值并转换为f16
    if (!denseAttr.isSplat())
      return nullptr;
      
    auto f32APFloat = denseAttr.getSplatValue<APFloat>();
    auto f16Type = rewriter.getF16Type();
    auto f16TensorType = RankedTensorType::get({}, f16Type);
    
    // 创建f16的APFloat
    APFloat f16APFloat = f32APFloat;
    bool losesInfo = false;
    f16APFloat.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &losesInfo);
    
    auto f16Attr = DenseElementsAttr::get(f16TensorType, f16APFloat);
    
    // 创建新的常量操作
    return rewriter.create<ONNXConstantOp>(
        loc,
        /*sparse_value=*/Attribute(), // 空的sparse_value
        /*value=*/f16Attr             // f16的DenseElementsAttr
    ).getResult();
  }
};

struct F16ThresholdOptimizationPass
    : public PassWrapper<F16ThresholdOptimizationPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "fp16-threshold-optimization"; }
  StringRef getDescription() const final {
    return "Optimize f16 threshold comparisons by eliminating unnecessary f32 casts";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ONNXDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM_DEBUG(llvm::dbgs() << "Running F16ThresholdOptimizationPass\n");
    
    RewritePatternSet patterns(context);
    patterns.add<F16ThresholdOptimizationPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed F16ThresholdOptimizationPass\n");
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createF16ThresholdOptimizationPass() {
      return std::make_unique<F16ThresholdOptimizationPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<F16ThresholdOptimizationPass> pass;