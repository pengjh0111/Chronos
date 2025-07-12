#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

struct KrnlMemcpyToAffinePass 
    : public PassWrapper<KrnlMemcpyToAffinePass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KrnlMemcpyToAffinePass)
    
    void runOnOperation() override;
    StringRef getArgument() const final { return "krnl-memcpy-to-affine"; }
    StringRef getDescription() const final { 
        return "Convert krnl.memcpy operations to explicit affine.for loops with affine.load and affine.store operations."; 
    }

private:
    // 检查memref是否具有静态形状
    bool hasStaticShape(MemRefType memrefType) {
        return memrefType.hasStaticShape();
    }
    
    // 检查两个memref类型是否兼容（形状和元素类型相同）
    bool areMemrefTypesCompatible(MemRefType srcType, MemRefType dstType) {
        return srcType.getShape() == dstType.getShape() &&
               srcType.getElementType() == dstType.getElementType();
    }
    
    // 生成嵌套的affine.for循环来实现memcpy
    void generateAffineLoops(OpBuilder &builder, Location loc, 
                           Value srcMemref, Value dstMemref,
                           MemRefType memrefType) {
        
        auto shape = memrefType.getShape();
        unsigned rank = memrefType.getRank();
        
        if (rank == 0) {
            // 标量情况：直接load和store
            Value loadedValue = builder.create<affine::AffineLoadOp>(loc, srcMemref, ValueRange{});
            builder.create<affine::AffineStoreOp>(loc, loadedValue, dstMemref, ValueRange{});
            return;
        }
        
        // 递归生成嵌套循环
        generateNestedLoops(builder, loc, srcMemref, dstMemref, shape, 0, ValueRange{});
    }
    
    // 递归生成嵌套的affine.for循环
    void generateNestedLoops(OpBuilder &builder, Location loc,
                           Value srcMemref, Value dstMemref,
                           ArrayRef<int64_t> shape, unsigned currentDim,
                           ValueRange currentIndices) {
        
        if (currentDim == shape.size()) {
            // 到达最内层，执行load和store
            Value loadedValue = builder.create<affine::AffineLoadOp>(loc, srcMemref, currentIndices);
            builder.create<affine::AffineStoreOp>(loc, loadedValue, dstMemref, currentIndices);
            return;
        }
        
        // 创建当前维度的循环
        int64_t dimSize = shape[currentDim];
        auto forOp = builder.create<affine::AffineForOp>(loc, 0, dimSize);
        
        // 在循环体内继续处理下一个维度
        builder.setInsertionPointToStart(forOp.getBody());
        
        // 构建新的索引列表
        SmallVector<Value> newIndices(currentIndices.begin(), currentIndices.end());
        newIndices.push_back(forOp.getInductionVar());
        
        generateNestedLoops(builder, loc, srcMemref, dstMemref, shape, 
                          currentDim + 1, newIndices);
        
        // 恢复插入点到循环后面
        builder.setInsertionPointAfter(forOp);
    }
    
    // 处理单个krnl.memcpy操作
    LogicalResult processKrnlMemcpy(Operation *memcpyOp) {
        OpBuilder builder(memcpyOp);
        Location loc = memcpyOp->getLoc();
        
        // 获取krnl.memcpy的操作数
        // krnl.memcpy的签名通常是：(dst, src, size, offset1, offset2)
        if (memcpyOp->getNumOperands() < 2) {
            return memcpyOp->emitError("krnl.memcpy operation requires at least 2 operands");
        }
        
        Value dstMemref = memcpyOp->getOperand(0);
        Value srcMemref = memcpyOp->getOperand(1);
        
        // 获取memref类型
        auto dstType = dstMemref.getType().dyn_cast<MemRefType>();
        auto srcType = srcMemref.getType().dyn_cast<MemRefType>();
        
        if (!dstType || !srcType) {
            return memcpyOp->emitError("krnl.memcpy operands must be memref types");
        }
        
        // 检查是否具有静态形状
        if (!hasStaticShape(dstType) || !hasStaticShape(srcType)) {
            return memcpyOp->emitError("krnl.memcpy conversion only supports static shapes");
        }
        
        // 检查源和目标类型是否兼容
        if (!areMemrefTypesCompatible(srcType, dstType)) {
            return memcpyOp->emitError("krnl.memcpy source and destination memref types are incompatible");
        }
        
        // 检查布局是否为identity（连续内存布局）
        if (!srcType.getLayout().isIdentity() || !dstType.getLayout().isIdentity()) {
            return memcpyOp->emitError("krnl.memcpy conversion only supports identity layout (contiguous memory)");
        }
        
        // 设置插入点在memcpy操作之前
        builder.setInsertionPoint(memcpyOp);
        
        // 生成affine循环
        generateAffineLoops(builder, loc, srcMemref, dstMemref, srcType);
        
        return success();
    }
};

void KrnlMemcpyToAffinePass::runOnOperation() {
    ModuleOp module = getOperation();
    
    // 收集所有的krnl.memcpy操作
    SmallVector<Operation*, 8> memcpyOps;
    
    module.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "krnl.memcpy") {
            memcpyOps.push_back(op);
        }
    });
    
    // 处理每个krnl.memcpy操作
    for (Operation *memcpyOp : memcpyOps) {
        if (failed(processKrnlMemcpy(memcpyOp))) {
            signalPassFailure();
            return;
        }
        
        // 删除原始的krnl.memcpy操作
        memcpyOp->erase();
    }
}

} // end anonymous namespace

namespace onnx_mlir {
namespace krnl {

std::unique_ptr<mlir::Pass> createKrnlMemcpyToAffinePass() {
    return std::make_unique<KrnlMemcpyToAffinePass>();
}

} // namespace krnl
} // namespace onnx_mlir

static mlir::PassRegistration<KrnlMemcpyToAffinePass> pass;