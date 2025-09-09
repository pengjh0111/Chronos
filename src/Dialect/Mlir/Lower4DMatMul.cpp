// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/PatternMatch.h"

// // 引入 ONNX Dialect 的头文件
// #include "src/Dialect/ONNX/ONNXOps.hpp"

// using namespace mlir;

// #define DEBUG_TYPE "lower-4d-matmul"

// namespace {

// // 定义一个新的 Pass，专门用于处理 4D MatMul 降维
// class Lower4DMatMulPass
//     : public PassWrapper<Lower4DMatMulPass, OperationPass<func::FuncOp>> {

// public:
//   StringRef getArgument() const final { return "lower-4d-matmul"; }
//   StringRef getDescription() const final {
//     return "Lowers 4D onnx.MatMul operations to 3D by inserting reshape operations.";
//   }

//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
//     MLIRContext *context = &getContext();
//     OpBuilder builder(context);

//     // 1. 首先收集所有符合条件的 4D MatMul 操作
//     // 这样做可以避免在遍历时修改 IR 导致迭代器失效
//     std::vector<ONNXMatMulOp> matMulOpsToProcess;
//     funcOp.walk([&](ONNXMatMulOp matMulOp) {
//       // 获取第一个操作数 A 的类型
//       auto operandAType = matMulOp.getA().getType().dyn_cast<RankedTensorType>();

//       // 检查类型是否存在，并且维度是否为 4
//       if (operandAType && operandAType.getRank() == 4) {
//         matMulOpsToProcess.push_back(matMulOp);
//       }
//     });

//     if (matMulOpsToProcess.empty()) {
//         return; // 没有需要处理的操作
//     }
    
//     // 2. 遍历收集到的 MatMul 列表并进行转换
//     for (ONNXMatMulOp matMulOp : matMulOpsToProcess) {
//       // 设置插入点，新的操作将会在原始 MatMul 之前创建
//       builder.setInsertionPoint(matMulOp);

//       Value A = matMulOp.getA();
//       Value B = matMulOp.getB();
//       Location loc = matMulOp.getLoc();

//       auto aType = A.getType().cast<RankedTensorType>();
//       auto bType = B.getType().cast<RankedTensorType>();
//       auto resultType = matMulOp.getResult().getType().cast<RankedTensorType>();

//       // 确保形状是静态的，以获取维度信息
//       if (!aType.hasStaticShape() || !bType.hasStaticShape()) {
//         matMulOp.emitWarning("Skipping MatMul with dynamic shapes.");
//         continue;
//       }

//       ArrayRef<int64_t> aShape = aType.getShape();
//       ArrayRef<int64_t> bShape = bType.getShape();
//       Type elementType = aType.getElementType();

//       // 3. 创建降维的 Reshape 操作 (A -> A_3D)
//       // 计算新的 3D 形状 [d0*d1, d2, d3]
//       int64_t aNewDim0 = aShape[0] * aShape[1];
//       SmallVector<int64_t, 3> aNewShapeVec = {aNewDim0, aShape[2], aShape[3]};
      
//       // 创建用于 Reshape 的形状常量
//       auto aShapeConst = createShapeConstant(builder, loc, aNewShapeVec);
      
//       // 定义新的 3D 张量类型并创建 Reshape 操作
//       auto aNewType = RankedTensorType::get(aNewShapeVec, elementType);
//       Value aReshaped = builder.create<ONNXReshapeOp>(loc, aNewType, A, aShapeConst);

//       // 4. 创建降维的 Reshape 操作 (B -> B_3D)
//       // 计算新的 3D 形状 [d0*d1, d2, d3]
//       int64_t bNewDim0 = bShape[0] * bShape[1];
//       SmallVector<int64_t, 3> bNewShapeVec = {bNewDim0, bShape[2], bShape[3]};
      
//       // 创建用于 Reshape 的形状常量
//       auto bShapeConst = createShapeConstant(builder, loc, bNewShapeVec);

//       // 定义新的 3D 张量类型并创建 Reshape 操作
//       auto bNewType = RankedTensorType::get(bNewShapeVec, elementType);
//       Value bReshaped = builder.create<ONNXReshapeOp>(loc, bNewType, B, bShapeConst);

//       // 5. 创建新的 3D MatMul 操作
//       // 新 MatMul 的结果类型也需要是 3D 的
//       SmallVector<int64_t, 3> newResultShapeVec = {aNewDim0, aShape[2], bShape[3]};
//       auto newResultType = RankedTensorType::get(newResultShapeVec, elementType);
//       Value newMatMulResult = builder.create<ONNXMatMulOp>(loc, newResultType, aReshaped, bReshaped);

//       // 6. 创建恢复维度的 Reshape 操作 (Result_3D -> Result_4D)
//       // 获取原始 MatMul 的输出形状，作为恢复的目标形状
//       ArrayRef<int64_t> originalResultShape = resultType.getShape();
//       auto originalShapeConst = createShapeConstant(builder, loc, originalResultShape);

//       Value finalResult = builder.create<ONNXReshapeOp>(loc, resultType, newMatMulResult, originalShapeConst);
      
//       // 7. 替换原始 MatMul 的所有用途，并将其删除
//       matMulOp.getResult().replaceAllUsesWith(finalResult);
//       matMulOp.erase();
//     }
//   }

// private:
//   // 辅助函数，用于创建一个 onnx.Constant 来表示形状
//   Value createShapeConstant(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape) {
//       // 形状本身是一个一维张量，类型为 i64
//       auto shapeTensorType = RankedTensorType::get({(int64_t)shape.size()}, builder.getI64Type());
//       // 使用 DenseElementsAttr 来存储形状的值
//       auto shapeAttribute = DenseElementsAttr::get(shapeTensorType, shape);
      
//       return builder.create<ONNXConstantOp>(loc, nullptr, shapeAttribute);
// }
// };

// } // end anonymous namespace

// // 注册 Pass
// namespace onnx_mlir {
// std::unique_ptr<Pass> createLower4DMatMulPass() {
//   return std::make_unique<Lower4DMatMulPass>();
// }
// } // namespace onnx_mlir

// static mlir::PassRegistration<Lower4DMatMulPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

// 引入 ONNX Dialect 的头文件
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

#define DEBUG_TYPE "lower-4d-matmul"

namespace {

// 定义一个新的 Pass，专门用于处理 4D MatMul 降维
class Lower4DMatMulPass
    : public PassWrapper<Lower4DMatMulPass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { return "lower-4d-matmul"; }
  StringRef getDescription() const final {
    return "Lowers 4D onnx.MatMul operations to 3D by inserting reshape operations.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // 1. 首先收集所有符合条件的 4D MatMul 操作
    std::vector<ONNXMatMulOp> matMulOpsToProcess;
    funcOp.walk([&](ONNXMatMulOp matMulOp) {
      auto operandAType = matMulOp.getA().getType().dyn_cast<RankedTensorType>();
      if (operandAType && operandAType.getRank() == 4) {
        matMulOpsToProcess.push_back(matMulOp);
      }
    });

    if (matMulOpsToProcess.empty()) {
        return;
    }
    
    // 2. 遍历收集到的 MatMul 列表并进行转换
    for (ONNXMatMulOp matMulOp : matMulOpsToProcess) {
      builder.setInsertionPoint(matMulOp);

      Value A = matMulOp.getA();
      Value B = matMulOp.getB();
      Location loc = matMulOp.getLoc();

      auto aType = A.getType().cast<RankedTensorType>();
      auto bType = B.getType().cast<RankedTensorType>();
      auto resultType = matMulOp.getResult().getType().cast<RankedTensorType>();

      if (!aType.hasStaticShape() || !bType.hasStaticShape()) {
        matMulOp.emitWarning("Skipping MatMul with dynamic shapes.");
        continue;
      }

      ArrayRef<int64_t> aShape = aType.getShape();
      ArrayRef<int64_t> bShape = bType.getShape();
      Type elementType = aType.getElementType();

      // 3. 创建降维的 Reshape 操作 (A -> A_3D)
      int64_t aNewDim0 = aShape[0] * aShape[1];
      SmallVector<int64_t, 3> aNewShapeVec = {aNewDim0, aShape[2], aShape[3]};
      auto aShapeConst = createShapeConstant(builder, loc, aNewShapeVec);
      auto aNewType = RankedTensorType::get(aNewShapeVec, elementType);
      auto aReshaped = builder.create<ONNXReshapeOp>(loc, aNewType, A, aShapeConst);
      aReshaped->setAttr("onnx_node_name", builder.getStringAttr("reshape")); // ++ 添加属性

      // 4. 创建降维的 Reshape 操作 (B -> B_3D)
      int64_t bNewDim0 = bShape[0] * bShape[1];
      SmallVector<int64_t, 3> bNewShapeVec = {bNewDim0, bShape[2], bShape[3]};
      auto bShapeConst = createShapeConstant(builder, loc, bNewShapeVec);
      auto bNewType = RankedTensorType::get(bNewShapeVec, elementType);
      auto bReshaped = builder.create<ONNXReshapeOp>(loc, bNewType, B, bShapeConst);
      bReshaped->setAttr("onnx_node_name", builder.getStringAttr("reshape")); // ++ 添加属性

      // 5. 创建新的 3D MatMul 操作
      SmallVector<int64_t, 3> newResultShapeVec = {aNewDim0, aShape[2], bShape[3]};
      auto newResultType = RankedTensorType::get(newResultShapeVec, elementType);
      auto newMatMulResult = builder.create<ONNXMatMulOp>(loc, newResultType, aReshaped, bReshaped);
      newMatMulResult->setAttr("onnx_node_name", builder.getStringAttr("matmul")); // ++ 添加属性

      // 6. 创建恢复维度的 Reshape 操作 (Result_3D -> Result_4D)
      ArrayRef<int64_t> originalResultShape = resultType.getShape();
      auto originalShapeConst = createShapeConstant(builder, loc, originalResultShape);
      auto finalResult = builder.create<ONNXReshapeOp>(loc, resultType, newMatMulResult, originalShapeConst);
      finalResult->setAttr("onnx_node_name", builder.getStringAttr("reshape")); // ++ 添加属性
      
      // 7. 替换原始 MatMul 的所有用途，并将其删除
      matMulOp.getResult().replaceAllUsesWith(finalResult);

      // 遍历新结果的用户，如果没有 onnx_node_name 属性则补上

      for (Operation *user : matMulOp.getResult().getUsers()) {
        if (!user->hasAttr("onnx_node_name")) {
          user->setAttr("onnx_node_name", builder.getStringAttr("autogen"));
        }
      }

      for (Operation *user : finalResult.getResult().getUsers()) {
        if (!user->hasAttr("onnx_node_name")) {
          user->setAttr("onnx_node_name", builder.getStringAttr("autogen"));
        }
      }

      for (Operation *user : newMatMulResult.getResult().getUsers()) {
        if (!user->hasAttr("onnx_node_name")) {
          user->setAttr("onnx_node_name", builder.getStringAttr("autogen"));
        }
      }

      for (Operation *user : bReshaped.getResult().getUsers()) {
        if (!user->hasAttr("onnx_node_name")) {
          user->setAttr("onnx_node_name", builder.getStringAttr("autogen"));
        }
      }

      for (Operation *user : aReshaped.getResult().getUsers()) {
        if (!user->hasAttr("onnx_node_name")) {
          user->setAttr("onnx_node_name", builder.getStringAttr("autogen"));
        }
      }

      matMulOp.erase();
    }
  }

private:
  // 辅助函数，用于创建一个 onnx.Constant 来表示形状
  Value createShapeConstant(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape) {
    auto shapeTensorType = RankedTensorType::get({(int64_t)shape.size()}, builder.getI64Type());
    auto shapeAttribute = DenseElementsAttr::get(shapeTensorType, shape);
    return builder.create<ONNXConstantOp>(loc, nullptr, shapeAttribute);
  }
};

} // end anonymous namespace

// 注册 Pass
namespace onnx_mlir {
std::unique_ptr<Pass> createLower4DMatMulPass() {
  return std::make_unique<Lower4DMatMulPass>();
}
} // namespace onnx_mlir

static mlir::PassRegistration<Lower4DMatMulPass> pass;