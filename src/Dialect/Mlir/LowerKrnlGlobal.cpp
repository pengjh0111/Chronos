#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

#define DEBUG_TYPE "lower-krnl-global"

namespace {

// 定义一个新的 Pass，用于将 krnl.global 转换为 memref.global + memref.get_global
// 同时处理标量 memref.alloca
class LowerKrnlGlobalPass
    : public PassWrapper<LowerKrnlGlobalPass, OperationPass<ModuleOp>> {

public:
  StringRef getArgument() const final { return "lower-krnl-global"; }
  StringRef getDescription() const final {
    return "Lowers krnl.global operations to memref.global and memref.get_global, "
           "and converts scalar memref to rank-1 memref.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    SymbolTable symbolTable(moduleOp);

    // 存储需要处理的 krnl.global 操作及其所在的 func
    SmallVector<std::pair<KrnlGlobalOp, func::FuncOp>> krnlGlobalOps;

    // 1. 收集所有 krnl.global 操作
    moduleOp.walk([&](func::FuncOp funcOp) {
      funcOp.walk([&](KrnlGlobalOp globalOp) {
        krnlGlobalOps.push_back({globalOp, funcOp});
      });
    });

    if (!krnlGlobalOps.empty()) {
      // 用于跟踪标量 memref 的映射：原始值 -> 新值
      DenseMap<Value, Value> scalarMemrefMapping;

      // 2. 遍历每个 krnl.global 并进行转换
      for (auto &pair : krnlGlobalOps) {
        KrnlGlobalOp krnlGlobalOp = pair.first;
        func::FuncOp funcOp = pair.second;

        // 获取 krnl.global 的属性
        StringRef name = krnlGlobalOp.getName();
        Attribute valueAttr = krnlGlobalOp.getValueAttr();
        Type resultType = krnlGlobalOp.getResult().getType();

        // 确保 resultType 是 MemRefType
        auto memrefType = resultType.dyn_cast<MemRefType>();
        if (!memrefType) {
          krnlGlobalOp.emitWarning("Skipping krnl.global with non-memref type.");
          continue;
        }

        // 处理标量情况：将 rank 0 转换为 rank 1
        Type actualMemrefType = memrefType;
        Attribute actualValueAttr = valueAttr;
        bool isScalar = (memrefType.getRank() == 0);
        
        if (isScalar) {
          // 将 memref<T> 转换为 memref<1xT>
          actualMemrefType = MemRefType::get({1}, memrefType.getElementType());
          
          // 调整 valueAttr：将 tensor<T> 转换为 tensor<1xT>
          if (auto denseAttr = valueAttr.dyn_cast_or_null<DenseElementsAttr>()) {
            auto newTensorType = RankedTensorType::get({1}, memrefType.getElementType());
            actualValueAttr = denseAttr.reshape(newTensorType);
          }
        }

        // 生成唯一的符号名
        std::string symbolName = name.str();
        // 确保符号名唯一
        int suffix = 0;
        std::string uniqueName = symbolName;
        while (symbolTable.lookup(uniqueName)) {
          uniqueName = symbolName + "_" + std::to_string(suffix++);
        }
        symbolName = uniqueName;

        // 3. 在 module 级别创建 memref.global
        builder.setInsertionPointToStart(moduleOp.getBody());

        auto memrefGlobal = builder.create<memref::GlobalOp>(
            krnlGlobalOp.getLoc(),
            StringRef(symbolName),
            builder.getStringAttr("private"),
            actualMemrefType.cast<MemRefType>(),
            actualValueAttr,
            true,
            nullptr);

        // 4. 在 func 的开始位置创建 memref.get_global
        Block &entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        auto getGlobal = builder.create<memref::GetGlobalOp>(
            krnlGlobalOp.getLoc(),
            actualMemrefType.cast<MemRefType>(),
            FlatSymbolRefAttr::get(context, symbolName));

        // 5. 如果是标量，记录映射关系，稍后修复使用者
        if (isScalar) {
          scalarMemrefMapping[krnlGlobalOp.getResult()] = getGlobal.getResult();
        }

        // 6. 替换 krnl.global 的所有使用
        krnlGlobalOp.getResult().replaceAllUsesWith(getGlobal.getResult());

        // 7. 删除原始的 krnl.global 操作
        krnlGlobalOp.erase();
      }

      // 8. 修复所有使用标量 memref 的 load/store 操作
      fixScalarMemrefUsers(builder, scalarMemrefMapping);
    }

    // 9. 处理所有标量 memref.alloca
    SmallVector<memref::AllocaOp> scalarAllocas;
    moduleOp.walk([&](memref::AllocaOp allocaOp) {
      auto memrefType = allocaOp.getType().dyn_cast<MemRefType>();
      if (memrefType && memrefType.getRank() == 0) {
        scalarAllocas.push_back(allocaOp);
      }
    });

    DenseMap<Value, Value> allocaScalarMapping;
    for (auto allocaOp : scalarAllocas) {
      auto memrefType = allocaOp.getType().cast<MemRefType>();
      builder.setInsertionPoint(allocaOp);

      // 将 memref<T> 转换为 memref<1xT>
      auto newType = MemRefType::get({1}, memrefType.getElementType());
      auto newAlloca = builder.create<memref::AllocaOp>(
          allocaOp.getLoc(),
          newType);

      // 记录映射
      allocaScalarMapping[allocaOp.getResult()] = newAlloca.getResult();

      // 替换所有使用
      allocaOp.getResult().replaceAllUsesWith(newAlloca.getResult());

      // 删除原始 alloca
      allocaOp.erase();
    }

    // 10. 修复 alloca 的使用者
    fixScalarMemrefUsers(builder, allocaScalarMapping);
  }

private:
  void fixScalarMemrefUsers(OpBuilder &builder,
                            const DenseMap<Value, Value> &scalarMapping) {
    for (auto &entry : scalarMapping) {
      Value newMemref = entry.second;

      SmallVector<Operation *> usersToFix;
      for (Operation *user : newMemref.getUsers())
        usersToFix.push_back(user);

      for (Operation *user : usersToFix) {
        builder.setInsertionPoint(user);
        Location loc = user->getLoc();

        // 公共：构造 [0] 索引
        auto c0idx = builder.create<arith::ConstantIndexOp>(loc, 0);
        // affine map：() -> (0)，即固定返回 0 的一维 map
        auto map0 = AffineMap::get(/*dimCount=*/0, /*symCount=*/0,
                                   builder.getAffineConstantExpr(0),
                                   builder.getContext());

        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          if (loadOp.getIndices().empty()) {
            auto newLoad = builder.create<memref::LoadOp>(
                loc, newMemref, ValueRange{c0idx});
            loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
            loadOp.erase();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          if (storeOp.getIndices().empty()) {
            builder.create<memref::StoreOp>(
                loc, storeOp.getValue(), newMemref, ValueRange{c0idx});
            storeOp.erase();
          }
        } else if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(user)) {
          // 将 affine_map<() -> ()> 替换为 affine_map<() -> (0)>
          if (affineLoad.getMap().getNumResults() == 0) {
            auto newLoad = builder.create<affine::AffineLoadOp>(
                loc, newMemref, map0, ValueRange{});
            affineLoad.getResult().replaceAllUsesWith(newLoad.getResult());
            affineLoad.erase();
          }
        } else if (auto affineStore = dyn_cast<affine::AffineStoreOp>(user)) {
          if (affineStore.getMap().getNumResults() == 0) {
            builder.create<affine::AffineStoreOp>(
                loc, affineStore.getValue(), newMemref, map0, ValueRange{});
            affineStore.erase();
          }
        } else if (auto krnlLoad = dyn_cast<KrnlLoadOp>(user)) {
          // krnl.load %x[] -> krnl.load %x[c0]
          if (krnlLoad.getIndices().empty()) {
            auto newLoad = builder.create<KrnlLoadOp>(
                loc, newMemref, ValueRange{c0idx});
            krnlLoad.getResult().replaceAllUsesWith(newLoad.getResult());
            krnlLoad.erase();
          }
        } else if (auto krnlStore = dyn_cast<KrnlStoreOp>(user)) {
          if (krnlStore.getIndices().empty()) {
            builder.create<KrnlStoreOp>(
                loc, krnlStore.getValue(), newMemref, ValueRange{c0idx});
            krnlStore.erase();
          }
        }
      }
    }
  }
};

} // end anonymous namespace

// 注册 Pass
namespace onnx_mlir {
std::unique_ptr<Pass> createLowerKrnlGlobalPass() {
  return std::make_unique<LowerKrnlGlobalPass>();
}
} // namespace onnx_mlir

static mlir::PassRegistration<LowerKrnlGlobalPass> pass;