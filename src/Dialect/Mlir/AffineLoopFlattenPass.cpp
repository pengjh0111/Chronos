// #include "mlir/Pass/Pass.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "llvm/Support/Debug.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/IR/AffineExpr.h"
// #include "mlir/IR/AffineMap.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/IRMapping.h"

// using namespace mlir;

// #define DEBUG_TYPE "for-loop-flatten"

// namespace {

// class ForLoopFlattenPattern : public OpRewritePattern<affine::AffineForOp> {
// public:
//   using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(affine::AffineForOp outerForOp,
//                                PatternRewriter &rewriter) const override {
//     // 检查是否是4层完美嵌套的affine.for
//     auto nestedStructure = analyzeNestedForStructure(outerForOp);
//     if (!nestedStructure.has_value())
//       return failure();

//     auto [dim0, dim1, dim2, dim3, innerMostOp] = *nestedStructure;
    
//     LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten: " 
//                << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
//                << " -> " << (dim0 * dim1) << "x" << dim2 << "x" << dim3 << "\n");

//     // 收集需要重塑的memref
//     SmallVector<Value> memrefsToReshape;
//     SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
//     if (!collectMemRefs(innerMostOp, memrefsToReshape, reshapeInfo, dim0, dim1, dim2, dim3))
//       return failure();

//     Location loc = outerForOp.getLoc();
    
//     // 在外层循环前插入reinterpret_cast操作进行降维
//     rewriter.setInsertionPoint(outerForOp);
//     SmallVector<Value> reshapedMemrefs;
    
//     for (auto [memref, originalType] : reshapeInfo) {
//       Value reshaped = createReshapedMemref(rewriter, memref, originalType, 
//                                            dim0, dim1, dim2, dim3, loc);
//       if (!reshaped)
//         return failure();
//       reshapedMemrefs.push_back(reshaped);
//     }

//     // 首先获取原始循环体内容
//     Block *originalInnerBody = &innerMostOp.getRegion().front();
//     SmallVector<Operation*> opsToClone;
//     for (auto &op : originalInnerBody->getOperations()) {
//       if (!isa<affine::AffineYieldOp>(op)) {
//         opsToClone.push_back(&op);
//       }
//     }

//     // 创建三层嵌套的for循环
//     int64_t flattenedSize = dim0 * dim1;
//     MLIRContext *ctx = rewriter.getContext();
    
//     // 创建第一层循环 (flattened dimension: 0 to flattenedSize)
//     auto newOuterFor = createSingleDimForOp(rewriter, loc, 0, flattenedSize);
//     Block *outerBlock = &newOuterFor.getRegion().front();
//     BlockArgument flattenedIdx = outerBlock->getArgument(0);
    
//     // 创建第二层循环 (dim2: 0 to dim2)
//     rewriter.setInsertionPointToStart(outerBlock);
//     auto newMiddleFor = createSingleDimForOp(rewriter, loc, 0, dim2);
//     Block *middleBlock = &newMiddleFor.getRegion().front();
//     BlockArgument idx2 = middleBlock->getArgument(0);
    
//     // 创建第三层循环 (dim3: 0 to dim3)
//     rewriter.setInsertionPointToStart(middleBlock);
//     auto newInnerFor = createSingleDimForOp(rewriter, loc, 0, dim3);
//     Block *innerBlock = &newInnerFor.getRegion().front();
//     BlockArgument idx3 = innerBlock->getArgument(0);

//     // 在最内层循环体中处理逻辑
//     rewriter.setInsertionPointToStart(innerBlock);
    
//     // 创建memref映射
//     DenseMap<Value, Value> memrefMap;
//     for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
//       memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
//     }
    
//     // 在循环体开始处计算原始索引
//     Value dim1Value = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
//     Value originalIdx0 = rewriter.create<arith::DivSIOp>(loc, flattenedIdx, dim1Value);
//     Value originalIdx1 = rewriter.create<arith::RemSIOp>(loc, flattenedIdx, dim1Value);
    
//     // 建立索引映射
//     IRMapping globalMapping;
//     globalMapping.map(outerForOp.getInductionVar(), originalIdx0);
    
//     // 找到其他层的索引参数并建立映射
//     auto level2Op = findNestedFor(outerForOp);
//     if (level2Op) {
//       globalMapping.map(level2Op->getInductionVar(), originalIdx1);
//       auto level3Op = findNestedFor(*level2Op);
//       if (level3Op) {
//         globalMapping.map(level3Op->getInductionVar(), idx2);
//         auto level4Op = findNestedFor(*level3Op);
//         if (level4Op) {
//           globalMapping.map(level4Op->getInductionVar(), idx3);
//         }
//       }
//     }

//     // 克隆循环体操作
//     for (Operation *op : opsToClone) {
//       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
//         Value memref = loadOp.getMemRef();
//         if (memrefMap.count(memref)) {
//           // 使用重塑的memref
//           Value newMemref = memrefMap.lookup(memref);
//           auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
//               op->getLoc(), newMemref, ValueRange{flattenedIdx, idx2, idx3});
//           globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
//         } else {
//           // 正常克隆
//           auto clonedOp = rewriter.clone(*op, globalMapping);
//           globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
//         }
//       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
//         Value memref = storeOp.getMemRef();
//         if (memrefMap.count(memref)) {
//           // 使用重塑的memref
//           Value newMemref = memrefMap.lookup(memref);
//           Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
//           rewriter.create<affine::AffineStoreOp>(
//               op->getLoc(), valueToStore, newMemref, ValueRange{flattenedIdx, idx2, idx3});
//         } else {
//           // 正常克隆
//           rewriter.clone(*op, globalMapping);
//         }
//       } else {
//         // 对于其他操作，正常克隆
//         auto clonedOp = rewriter.clone(*op, globalMapping);
//         // 映射结果
//         for (unsigned i = 0; i < op->getNumResults(); ++i) {
//           globalMapping.map(op->getResult(i), clonedOp->getResult(i));
//         }
//       }
//     }

//     // 添加yield操作 (从内到外)
//     rewriter.create<affine::AffineYieldOp>(loc);
//     rewriter.setInsertionPointAfter(newInnerFor);
//     rewriter.create<affine::AffineYieldOp>(loc);
//     rewriter.setInsertionPointAfter(newMiddleFor);
//     rewriter.create<affine::AffineYieldOp>(loc);

//     // 删除原来的嵌套循环
//     rewriter.eraseOp(outerForOp);

//     return success();
//   }

// private:
//   struct NestedForInfo {
//     int64_t dim0, dim1, dim2, dim3;
//     affine::AffineForOp innerMostOp;
//   };

//   // 创建单维度的for循环
//   affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
//                                            Location loc, 
//                                            int64_t lowerBound, 
//                                            int64_t upperBound) const {
//     MLIRContext *ctx = rewriter.getContext();
    
//     // 创建常量边界的 AffineMap
//     AffineMap lowerBoundMap = AffineMap::getConstantMap(lowerBound, ctx);
//     AffineMap upperBoundMap = AffineMap::getConstantMap(upperBound, ctx);
    
//     // 创建AffineForOp，使用正确的构造函数
//     auto forOp = rewriter.create<affine::AffineForOp>(
//         loc,
//         ValueRange{},        // lowerBoundOperands (空，因为使用常量map)
//         lowerBoundMap,       // lowerBoundMap
//         ValueRange{},        // upperBoundOperands (空，因为使用常量map)  
//         upperBoundMap,       // upperBoundMap
//         /*step=*/1,          // step
//         ValueRange{},        // iterArgs (空)
//         /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {
//           // 空的body builder，稍后会手动添加内容
//         });
    
//     return forOp;
//   }

//   // 检查是否为完美的4层嵌套循环结构
//   std::optional<NestedForInfo> analyzeNestedForStructure(
//       affine::AffineForOp outerOp) const {
    
//     // 获取第一层维度大小
//     auto dim0 = getLoopBound(outerOp);
//     if (!dim0.has_value())
//       return std::nullopt;

//     // 检查第一层是否为完美嵌套（只能包含一个嵌套for循环和yield）
//     if (!isPerfectlyNestedLevel(outerOp))
//       return std::nullopt;

//     // 查找第二层嵌套
//     auto secondFor = findNestedFor(outerOp);
//     if (!secondFor)
//       return std::nullopt;
    
//     auto dim1 = getLoopBound(*secondFor);
//     if (!dim1.has_value())
//       return std::nullopt;

//     // 检查第二层是否为完美嵌套
//     if (!isPerfectlyNestedLevel(*secondFor))
//       return std::nullopt;

//     // 查找第三层嵌套
//     auto thirdFor = findNestedFor(*secondFor);
//     if (!thirdFor)
//       return std::nullopt;
    
//     auto dim2 = getLoopBound(*thirdFor);
//     if (!dim2.has_value())
//       return std::nullopt;

//     // 检查第三层是否为完美嵌套
//     if (!isPerfectlyNestedLevel(*thirdFor))
//       return std::nullopt;

//     // 查找第四层嵌套
//     auto fourthFor = findNestedFor(*thirdFor);
//     if (!fourthFor)
//       return std::nullopt;
    
//     auto dim3 = getLoopBound(*fourthFor);
//     if (!dim3.has_value())
//       return std::nullopt;

//     // 检查第四层（最内层）是否只包含简单操作，不能有额外的嵌套循环
//     if (!isSimpleLoopBody(*fourthFor))
//       return std::nullopt;

//     LLVM_DEBUG(llvm::dbgs() << "Validated perfect 4-layer nesting: " 
//                << *dim0 << "x" << *dim1 << "x" << *dim2 << "x" << *dim3 << "\n");

//     return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *fourthFor};
//   }

//   // 检查一个循环层是否为完美嵌套（只包含一个嵌套for循环和yield操作）
//   bool isPerfectlyNestedLevel(affine::AffineForOp forOp) const {
//     Block &body = forOp.getRegion().front();
    
//     int nestedForCount = 0;
//     int yieldCount = 0;
//     int totalOps = 0;
    
//     for (auto &op : body.getOperations()) {
//       totalOps++;
//       if (isa<affine::AffineForOp>(&op)) {
//         nestedForCount++;
//       } else if (isa<affine::AffineYieldOp>(&op)) {
//         yieldCount++;
//       } else {
//         // 存在其他操作，不是完美嵌套
//         LLVM_DEBUG(llvm::dbgs() << "Found non-nested operation in loop body: " 
//                    << op.getName() << ", rejecting perfect nesting\n");
//         return false;
//       }
//     }
    
//     // 完美嵌套应该只包含一个嵌套for循环和一个yield操作
//     bool isPerfect = (nestedForCount == 1 && yieldCount == 1 && totalOps == 2);
    
//     if (!isPerfect) {
//       LLVM_DEBUG(llvm::dbgs() << "Level validation failed: nestedForCount=" 
//                  << nestedForCount << ", yieldCount=" << yieldCount 
//                  << ", totalOps=" << totalOps << "\n");
//     }
    
//     return isPerfect;
//   }

//   // 检查最内层循环体是否只包含简单操作（不能有额外的嵌套循环）
//   bool isSimpleLoopBody(affine::AffineForOp forOp) const {
//     Block &body = forOp.getRegion().front();
    
//     for (auto &op : body.getOperations()) {
//       if (isa<affine::AffineForOp>(&op)) {
//         // 最内层不能再有嵌套循环
//         LLVM_DEBUG(llvm::dbgs() << "Found nested for loop in innermost body, rejecting\n");
//         return false;
//       } else if (isa<affine::AffineLoadOp, affine::AffineStoreOp, 
//                      arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
//                      arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
//                      arith::ConstantOp, affine::AffineYieldOp>(&op)) {
//         // 这些是允许的简单操作
//         continue;
//       } else {
//         // 对于其他操作，我们保守地允许，但记录日志
//         LLVM_DEBUG(llvm::dbgs() << "Found operation in innermost body: " 
//                    << op.getName() << " (allowing)\n");
//       }
//     }
    
//     return true;
//   }

//   std::optional<affine::AffineForOp> findNestedFor(
//       affine::AffineForOp parentOp) const {
//     for (auto &op : parentOp.getRegion().front()) {
//       if (auto nestedFor = dyn_cast<affine::AffineForOp>(&op))
//         return nestedFor;
//     }
//     return std::nullopt;
//   }

//   std::optional<int64_t> getLoopBound(affine::AffineForOp forOp) const {
//     // 检查下界是否为0
//     auto lowerBound = forOp.getLowerBound();
//     auto lowerMap = lowerBound.getMap();
//     if (lowerMap.getNumResults() != 1)
//       return std::nullopt;
    
//     auto lowerExpr = lowerMap.getResult(0);
//     if (auto lowerConst = lowerExpr.dyn_cast<AffineConstantExpr>()) {
//       if (lowerConst.getValue() != 0)
//         return std::nullopt;
//     } else {
//       return std::nullopt;
//     }

//     // 获取上界常量
//     auto upperBound = forOp.getUpperBound();
//     auto upperMap = upperBound.getMap();
//     if (upperMap.getNumResults() != 1)
//       return std::nullopt;
      
//     auto upperExpr = upperMap.getResult(0);
//     if (auto upperConst = upperExpr.dyn_cast<AffineConstantExpr>()) {
//       return upperConst.getValue();
//     }
    
//     return std::nullopt;
//   }

//   bool collectMemRefs(affine::AffineForOp innerMostOp,
//                      SmallVector<Value> &memrefsToReshape,
//                      SmallVector<std::pair<Value, MemRefType>> &reshapeInfo,
//                      int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    
//     innerMostOp.getRegion().walk([&](Operation *op) {
//       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
//         Value memref = loadOp.getMemRef();
//         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
//         if (memrefType && memrefType.getRank() == 4 && 
//             isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
//           if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
//               == memrefsToReshape.end()) {
//             memrefsToReshape.push_back(memref);
//             reshapeInfo.push_back({memref, memrefType});
//           }
//         }
//       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
//         Value memref = storeOp.getMemRef();
//         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
//         if (memrefType && memrefType.getRank() == 4 && 
//             isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
//           if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
//               == memrefsToReshape.end()) {
//             memrefsToReshape.push_back(memref);
//             reshapeInfo.push_back({memref, memrefType});
//           }
//         }
//       }
//     });

//     return !memrefsToReshape.empty();
//   }

//   bool isCompatibleShape(MemRefType memrefType, int64_t dim0, int64_t dim1, 
//                         int64_t dim2, int64_t dim3) const {
//     auto shape = memrefType.getShape();
//     return shape.size() == 4 && 
//            (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
//            (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
//            (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
//            (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
//   }

//   Value createReshapedMemref(PatternRewriter &rewriter, Value originalMemref,
//                             MemRefType originalType, int64_t dim0, int64_t dim1,
//                             int64_t dim2, int64_t dim3, Location loc) const {
    
//     Type elementType = originalType.getElementType();
//     int64_t flattenedSize = dim0 * dim1;
    
//     // 创建新的3D shape: [dim0*dim1, dim2, dim3]
//     SmallVector<int64_t> newShape = {flattenedSize, dim2, dim3};
//     auto newMemrefType = MemRefType::get(newShape, elementType);

//     // 计算strides (假设原始memref是连续的row-major布局)
//     SmallVector<int64_t> strides = {dim2 * dim3, dim3, 1};
//     SmallVector<int64_t> offset = {0};

//     return rewriter.create<memref::ReinterpretCastOp>(
//         loc, newMemrefType, originalMemref, 
//         /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
//         /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
//     ).getResult();
//   }
// };

// struct ForLoopFlattenPass
//     : public PassWrapper<ForLoopFlattenPass, OperationPass<ModuleOp>> {
  
//   StringRef getArgument() const final { return "for-loop-flatten"; }
//   StringRef getDescription() const final {
//     return "Flatten 4D nested for loops to 3D by merging first two dimensions";
//   }
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<affine::AffineDialect, memref::MemRefDialect, arith::ArithDialect>();
//   }
  
//   void runOnOperation() override {
//     ModuleOp moduleOp = getOperation();
//     MLIRContext *context = &getContext();
    
//     LLVM_DEBUG(llvm::dbgs() << "Running ForLoopFlattenPass\n");
    
//     RewritePatternSet patterns(context);
//     patterns.add<ForLoopFlattenPattern>(context);
    
//     if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
//       signalPassFailure();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed ForLoopFlattenPass\n");
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
//     std::unique_ptr<Pass> createForLoopFlattenPass() {
//       return std::make_unique<ForLoopFlattenPass>();
//     }
// } // namespace onnx_mlir

// static mlir::PassRegistration<ForLoopFlattenPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

#define DEBUG_TYPE "for-loop-flatten"

namespace {

class ForLoopFlattenPattern : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp outerForOp,
                               PatternRewriter &rewriter) const override {
    // 检查是否是4层完美嵌套的affine.for
    auto nestedStructure = analyzeNestedForStructure(outerForOp);
    if (!nestedStructure.has_value())
      return failure();

    auto [dim0, dim1, dim2, dim3, innerMostOp] = *nestedStructure;
    
    LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten: " 
               << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
               << " -> " << (dim0 * dim1) << "x" << dim2 << "x" << dim3 << "\n");

    // 收集需要重塑的memref
    SmallVector<Value> memrefsToReshape;
    SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
    if (!collectMemRefs(innerMostOp, memrefsToReshape, reshapeInfo, dim0, dim1, dim2, dim3))
      return failure();

    Location loc = outerForOp.getLoc();
    
    // 在外层循环前插入reinterpret_cast操作进行降维
    rewriter.setInsertionPoint(outerForOp);
    SmallVector<Value> reshapedMemrefs;
    
    for (auto [memref, originalType] : reshapeInfo) {
      Value reshaped = createReshapedMemref(rewriter, memref, originalType, 
                                           dim0, dim1, dim2, dim3, loc);
      if (!reshaped)
        return failure();
      reshapedMemrefs.push_back(reshaped);
    }

    // 首先获取原始循环体内容
    Block *originalInnerBody = &innerMostOp.getRegion().front();
    SmallVector<Operation*> opsToClone;
    for (auto &op : originalInnerBody->getOperations()) {
      if (!isa<affine::AffineYieldOp>(op)) {
        opsToClone.push_back(&op);
      }
    }

    // 创建三层嵌套的for循环
    int64_t flattenedSize = dim0 * dim1;
    MLIRContext *ctx = rewriter.getContext();
    
    // 创建第一层循环 (flattened dimension: 0 to flattenedSize)
    auto newOuterFor = createSingleDimForOp(rewriter, loc, 0, flattenedSize);
    Block *outerBlock = &newOuterFor.getRegion().front();
    BlockArgument flattenedIdx = outerBlock->getArgument(0);
    
    // 创建第二层循环 (dim2: 0 to dim2)
    rewriter.setInsertionPointToStart(outerBlock);
    auto newMiddleFor = createSingleDimForOp(rewriter, loc, 0, dim2);
    Block *middleBlock = &newMiddleFor.getRegion().front();
    BlockArgument idx2 = middleBlock->getArgument(0);
    
    // 创建第三层循环 (dim3: 0 to dim3)
    rewriter.setInsertionPointToStart(middleBlock);
    auto newInnerFor = createSingleDimForOp(rewriter, loc, 0, dim3);
    Block *innerBlock = &newInnerFor.getRegion().front();
    BlockArgument idx3 = innerBlock->getArgument(0);

    // 在最内层循环体中处理逻辑
    rewriter.setInsertionPointToStart(innerBlock);
    
    // 创建memref映射
    DenseMap<Value, Value> memrefMap;
    for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
      memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
    }
    
    // 在循环体开始处计算原始索引
    Value dim1Value = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
    Value originalIdx0 = rewriter.create<arith::DivSIOp>(loc, flattenedIdx, dim1Value);
    Value originalIdx1 = rewriter.create<arith::RemSIOp>(loc, flattenedIdx, dim1Value);
    
    // 建立索引映射
    IRMapping globalMapping;
    globalMapping.map(outerForOp.getInductionVar(), originalIdx0);
    
    // 找到其他层的索引参数并建立映射
    auto level2Op = findNestedFor(outerForOp);
    if (level2Op) {
      globalMapping.map(level2Op->getInductionVar(), originalIdx1);
      auto level3Op = findNestedFor(*level2Op);
      if (level3Op) {
        globalMapping.map(level3Op->getInductionVar(), idx2);
        auto level4Op = findNestedFor(*level3Op);
        if (level4Op) {
          globalMapping.map(level4Op->getInductionVar(), idx3);
        }
      }
    }

    // 克隆循环体操作
    for (Operation *op : opsToClone) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        Value memref = loadOp.getMemRef();
        if (memrefMap.count(memref)) {
          // 使用重塑的memref
          Value newMemref = memrefMap.lookup(memref);
          auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
              op->getLoc(), newMemref, ValueRange{flattenedIdx, idx2, idx3});
          globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
        } else {
          // 正常克隆
          auto clonedOp = rewriter.clone(*op, globalMapping);
          globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        Value memref = storeOp.getMemRef();
        if (memrefMap.count(memref)) {
          // 使用重塑的memref
          Value newMemref = memrefMap.lookup(memref);
          Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
          rewriter.create<affine::AffineStoreOp>(
              op->getLoc(), valueToStore, newMemref, ValueRange{flattenedIdx, idx2, idx3});
        } else {
          // 正常克隆
          rewriter.clone(*op, globalMapping);
        }
      } else {
        // 对于其他操作，正常克隆
        auto clonedOp = rewriter.clone(*op, globalMapping);
        // 映射结果
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          globalMapping.map(op->getResult(i), clonedOp->getResult(i));
        }
      }
    }

    // 添加yield操作 (从内到外)
    rewriter.create<affine::AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(newInnerFor);
    rewriter.create<affine::AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(newMiddleFor);
    rewriter.create<affine::AffineYieldOp>(loc);

    // 删除原来的嵌套循环
    rewriter.eraseOp(outerForOp);

    return success();
  }

private:
  struct NestedForInfo {
    int64_t dim0, dim1, dim2, dim3;
    affine::AffineForOp innerMostOp;
  };

  // 创建单维度的for循环
  affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
                                           Location loc, 
                                           int64_t lowerBound, 
                                           int64_t upperBound) const {
    MLIRContext *ctx = rewriter.getContext();
    
    // 创建常量边界的 AffineMap
    AffineMap lowerBoundMap = AffineMap::getConstantMap(lowerBound, ctx);
    AffineMap upperBoundMap = AffineMap::getConstantMap(upperBound, ctx);
    
    // 创建AffineForOp，使用正确的构造函数
    auto forOp = rewriter.create<affine::AffineForOp>(
        loc,
        ValueRange{},        // lowerBoundOperands (空，因为使用常量map)
        lowerBoundMap,       // lowerBoundMap
        ValueRange{},        // upperBoundOperands (空，因为使用常量map)  
        upperBoundMap,       // upperBoundMap
        /*step=*/1,          // step
        ValueRange{},        // iterArgs (空)
        /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {
          // 空的body builder，稍后会手动添加内容
        });
    
    return forOp;
  }

  // 检查是否为完美的4层嵌套循环结构
  std::optional<NestedForInfo> analyzeNestedForStructure(
      affine::AffineForOp outerOp) const {
    
    // 获取第一层维度大小
    auto dim0 = getLoopBound(outerOp);
    if (!dim0.has_value())
      return std::nullopt;

    // 检查第一层是否为完美嵌套（只能包含一个嵌套for循环和yield）
    if (!isPerfectlyNestedLevel(outerOp))
      return std::nullopt;

    // 查找第二层嵌套
    auto secondFor = findNestedFor(outerOp);
    if (!secondFor)
      return std::nullopt;
    
    auto dim1 = getLoopBound(*secondFor);
    if (!dim1.has_value())
      return std::nullopt;

    // 检查第二层是否为完美嵌套
    if (!isPerfectlyNestedLevel(*secondFor))
      return std::nullopt;

    // 查找第三层嵌套
    auto thirdFor = findNestedFor(*secondFor);
    if (!thirdFor)
      return std::nullopt;
    
    auto dim2 = getLoopBound(*thirdFor);
    if (!dim2.has_value())
      return std::nullopt;

    // 检查第三层是否为完美嵌套
    if (!isPerfectlyNestedLevel(*thirdFor))
      return std::nullopt;

    // 查找第四层嵌套
    auto fourthFor = findNestedFor(*thirdFor);
    if (!fourthFor)
      return std::nullopt;
    
    auto dim3 = getLoopBound(*fourthFor);
    if (!dim3.has_value())
      return std::nullopt;

    // 检查第四层（最内层）是否只包含简单操作，不能有额外的嵌套循环
    if (!isSimpleLoopBody(*fourthFor))
      return std::nullopt;

    LLVM_DEBUG(llvm::dbgs() << "Validated perfect 4-layer nesting: " 
               << *dim0 << "x" << *dim1 << "x" << *dim2 << "x" << *dim3 << "\n");

    return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *fourthFor};
  }

  // 检查一个循环层是否为完美嵌套（只包含一个嵌套for循环和yield操作）
  bool isPerfectlyNestedLevel(affine::AffineForOp forOp) const {
    Block &body = forOp.getRegion().front();
    
    int nestedForCount = 0;
    int yieldCount = 0;
    int totalOps = 0;
    
    for (auto &op : body.getOperations()) {
      totalOps++;
      if (isa<affine::AffineForOp>(&op)) {
        nestedForCount++;
      } else if (isa<affine::AffineYieldOp>(&op)) {
        yieldCount++;
      } else {
        // 存在其他操作，不是完美嵌套
        LLVM_DEBUG(llvm::dbgs() << "Found non-nested operation in loop body: " 
                   << op.getName() << ", rejecting perfect nesting\n");
        return false;
      }
    }
    
    // 完美嵌套应该只包含一个嵌套for循环和一个yield操作
    bool isPerfect = (nestedForCount == 1 && yieldCount == 1 && totalOps == 2);
    
    if (!isPerfect) {
      LLVM_DEBUG(llvm::dbgs() << "Level validation failed: nestedForCount=" 
                 << nestedForCount << ", yieldCount=" << yieldCount 
                 << ", totalOps=" << totalOps << "\n");
    }
    
    return isPerfect;
  }

  // 检查最内层循环体是否只包含简单操作（不能有额外的嵌套循环）
  bool isSimpleLoopBody(affine::AffineForOp forOp) const {
    Block &body = forOp.getRegion().front();
    
    for (auto &op : body.getOperations()) {
      if (isa<affine::AffineForOp>(&op)) {
        // 最内层不能再有嵌套循环
        LLVM_DEBUG(llvm::dbgs() << "Found nested for loop in innermost body, rejecting\n");
        return false;
      } else if (isa<affine::AffineLoadOp, affine::AffineStoreOp, 
                     arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                     arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                     arith::ConstantOp, affine::AffineYieldOp>(&op)) {
        // 这些是允许的简单操作
        continue;
      } else {
        // 对于其他操作，我们保守地允许，但记录日志
        LLVM_DEBUG(llvm::dbgs() << "Found operation in innermost body: " 
                   << op.getName() << " (allowing)\n");
      }
    }
    
    return true;
  }

  std::optional<affine::AffineForOp> findNestedFor(
      affine::AffineForOp parentOp) const {
    for (auto &op : parentOp.getRegion().front()) {
      if (auto nestedFor = dyn_cast<affine::AffineForOp>(&op))
        return nestedFor;
    }
    return std::nullopt;
  }

  std::optional<int64_t> getLoopBound(affine::AffineForOp forOp) const {
    // 检查下界是否为0
    auto lowerBound = forOp.getLowerBound();
    auto lowerMap = lowerBound.getMap();
    if (lowerMap.getNumResults() != 1)
      return std::nullopt;
    
    auto lowerExpr = lowerMap.getResult(0);
    if (auto lowerConst = lowerExpr.dyn_cast<AffineConstantExpr>()) {
      if (lowerConst.getValue() != 0)
        return std::nullopt;
    } else {
      return std::nullopt;
    }

    // 获取上界常量
    auto upperBound = forOp.getUpperBound();
    auto upperMap = upperBound.getMap();
    if (upperMap.getNumResults() != 1)
      return std::nullopt;
      
    auto upperExpr = upperMap.getResult(0);
    if (auto upperConst = upperExpr.dyn_cast<AffineConstantExpr>()) {
      return upperConst.getValue();
    }
    
    return std::nullopt;
  }

  bool collectMemRefs(affine::AffineForOp innerMostOp,
                     SmallVector<Value> &memrefsToReshape,
                     SmallVector<std::pair<Value, MemRefType>> &reshapeInfo,
                     int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    
    // 首先收集所有访问的memref，检查是否存在维度不匹配的情况
    SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
    bool hasIncompatibleAccess = false;
    
    innerMostOp.getRegion().walk([&](Operation *op) {
      if (hasIncompatibleAccess) return; // 已经发现不兼容，提前退出
      
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        Value memref = loadOp.getMemRef();
        auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
        if (memrefType && memrefType.getRank() == 4) {
          // 检查访问的索引是否与循环变量匹配（简化检查：假设使用标准的4D索引模式）
          if (!isMemrefAccessCompatibleWithLoopBounds(loadOp, dim0, dim1, dim2, dim3)) {
            LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in load, rejecting flatten\n");
            hasIncompatibleAccess = true;
            return;
          }
          
          if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
            if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
                            [memref](const auto& pair) { return pair.first == memref; }) 
                == allAccessedMemrefs.end()) {
              allAccessedMemrefs.push_back({memref, memrefType});
            }
          }
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        Value memref = storeOp.getMemRef();
        auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
        if (memrefType && memrefType.getRank() == 4) {
          // 检查访问的索引是否与循环变量匹配
          if (!isMemrefAccessCompatibleWithLoopBounds(storeOp, dim0, dim1, dim2, dim3)) {
            LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in store, rejecting flatten\n");
            hasIncompatibleAccess = true;
            return;
          }
          
          if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
            if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
                            [memref](const auto& pair) { return pair.first == memref; }) 
                == allAccessedMemrefs.end()) {
              allAccessedMemrefs.push_back({memref, memrefType});
            }
          }
        }
      }
    });

    // 如果发现不兼容的访问，拒绝flatten
    if (hasIncompatibleAccess) {
      return false;
    }

    // 只有当所有访问的4D memref都与循环边界兼容时，才进行收集
    for (auto [memref, memrefType] : allAccessedMemrefs) {
      if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
          == memrefsToReshape.end()) {
        memrefsToReshape.push_back(memref);
        reshapeInfo.push_back({memref, memrefType});
      }
    }

    return !memrefsToReshape.empty();
  }

  bool isCompatibleShape(MemRefType memrefType, int64_t dim0, int64_t dim1, 
                        int64_t dim2, int64_t dim3) const {
    auto shape = memrefType.getShape();
    return shape.size() == 4 && 
           (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
           (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
           (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
           (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
  }

  // 检查memref访问模式是否与循环边界兼容
  bool isMemrefAccessCompatibleWithLoopBounds(Operation* op, int64_t dim0, int64_t dim1, 
                                              int64_t dim2, int64_t dim3) const {
    Value memref;
    AffineMap accessMap;
    SmallVector<Value> indices;
    
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      memref = loadOp.getMemRef();
      accessMap = loadOp.getAffineMap();
      indices = loadOp.getMapOperands();
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = storeOp.getMemRef();
      accessMap = storeOp.getAffineMap();
      indices = storeOp.getMapOperands();
    } else {
      return true; // 非访问操作，认为兼容
    }
    
    auto memrefType = memref.getType().dyn_cast<MemRefType>();
    if (!memrefType || memrefType.getRank() != 4) {
      return true; // 非4D memref，不影响我们的判断
    }
    
    auto shape = memrefType.getShape();
    
    // 检查memref的实际维度是否严格匹配循环迭代范围
    // 这是最严格的检查：要求tensor的每个维度都精确等于循环的迭代范围
    bool strictMatch = (shape[0] == dim0 && shape[1] == dim1 && 
                       shape[2] == dim2 && shape[3] == dim3);
    
    if (!strictMatch) {
      LLVM_DEBUG(llvm::dbgs() << "Memref shape mismatch with loop bounds: "
                 << "memref=" << shape[0] << "x" << shape[1] << "x" 
                 << shape[2] << "x" << shape[3] 
                 << ", loops=" << dim0 << "x" << dim1 << "x" 
                 << dim2 << "x" << dim3 << "\n");
      return false;
    }
    
    // 进一步检查访问模式是否是标准的4D索引模式 [d0, d1, d2, d3]
    if (accessMap.getNumResults() != 4) {
      LLVM_DEBUG(llvm::dbgs() << "Non-4D access pattern, rejecting\n");
      return false;
    }
    
    // 检查是否是恒等映射 (d0, d1, d2, d3) -> (d0, d1, d2, d3)
    for (unsigned i = 0; i < 4; ++i) {
      auto expr = accessMap.getResult(i);
      if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() != i) {
          LLVM_DEBUG(llvm::dbgs() << "Non-identity access pattern at dimension " 
                     << i << ", rejecting\n");
          return false;
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Non-dimension expression in access pattern at dimension " 
                   << i << ", rejecting\n");
        return false;
      }
    }
    
    return true;
  }

  Value createReshapedMemref(PatternRewriter &rewriter, Value originalMemref,
                            MemRefType originalType, int64_t dim0, int64_t dim1,
                            int64_t dim2, int64_t dim3, Location loc) const {
    
    Type elementType = originalType.getElementType();
    int64_t flattenedSize = dim0 * dim1;
    
    // 创建新的3D shape: [dim0*dim1, dim2, dim3]
    SmallVector<int64_t> newShape = {flattenedSize, dim2, dim3};
    auto newMemrefType = MemRefType::get(newShape, elementType);

    // 计算strides (假设原始memref是连续的row-major布局)
    SmallVector<int64_t> strides = {dim2 * dim3, dim3, 1};
    SmallVector<int64_t> offset = {0};

    return rewriter.create<memref::ReinterpretCastOp>(
        loc, newMemrefType, originalMemref, 
        /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
        /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
    ).getResult();
  }
};

struct ForLoopFlattenPass
    : public PassWrapper<ForLoopFlattenPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "for-loop-flatten"; }
  StringRef getDescription() const final {
    return "Flatten 4D nested for loops to 3D by merging first two dimensions";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect, arith::ArithDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM_DEBUG(llvm::dbgs() << "Running ForLoopFlattenPass\n");
    
    RewritePatternSet patterns(context);
    patterns.add<ForLoopFlattenPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed ForLoopFlattenPass\n");
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createForLoopFlattenPass() {
      return std::make_unique<ForLoopFlattenPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<ForLoopFlattenPass> pass;