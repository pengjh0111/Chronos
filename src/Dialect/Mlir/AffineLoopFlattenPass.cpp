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
    
//     // 首先收集所有访问的memref，检查是否存在维度不匹配的情况
//     SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
//     bool hasIncompatibleAccess = false;
    
//     innerMostOp.getRegion().walk([&](Operation *op) {
//       if (hasIncompatibleAccess) return; // 已经发现不兼容，提前退出
      
//       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
//         Value memref = loadOp.getMemRef();
//         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
//         if (memrefType && memrefType.getRank() == 4) {
//           // 检查访问的索引是否与循环变量匹配（简化检查：假设使用标准的4D索引模式）
//           if (!isMemrefAccessCompatibleWithLoopBounds(loadOp, dim0, dim1, dim2, dim3)) {
//             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in load, rejecting flatten\n");
//             hasIncompatibleAccess = true;
//             return;
//           }
          
//           if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
//             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
//                             [memref](const auto& pair) { return pair.first == memref; }) 
//                 == allAccessedMemrefs.end()) {
//               allAccessedMemrefs.push_back({memref, memrefType});
//             }
//           }
//         }
//       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
//         Value memref = storeOp.getMemRef();
//         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
//         if (memrefType && memrefType.getRank() == 4) {
//           // 检查访问的索引是否与循环变量匹配
//           if (!isMemrefAccessCompatibleWithLoopBounds(storeOp, dim0, dim1, dim2, dim3)) {
//             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in store, rejecting flatten\n");
//             hasIncompatibleAccess = true;
//             return;
//           }
          
//           if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
//             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
//                             [memref](const auto& pair) { return pair.first == memref; }) 
//                 == allAccessedMemrefs.end()) {
//               allAccessedMemrefs.push_back({memref, memrefType});
//             }
//           }
//         }
//       }
//     });

//     // 如果发现不兼容的访问，拒绝flatten
//     if (hasIncompatibleAccess) {
//       return false;
//     }

//     // 只有当所有访问的4D memref都与循环边界兼容时，才进行收集
//     for (auto [memref, memrefType] : allAccessedMemrefs) {
//       if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
//           == memrefsToReshape.end()) {
//         memrefsToReshape.push_back(memref);
//         reshapeInfo.push_back({memref, memrefType});
//       }
//     }

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

//   // 检查memref访问模式是否与循环边界兼容
//   bool isMemrefAccessCompatibleWithLoopBounds(Operation* op, int64_t dim0, int64_t dim1, 
//                                               int64_t dim2, int64_t dim3) const {
//     Value memref;
//     AffineMap accessMap;
//     SmallVector<Value> indices;
    
//     if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
//       memref = loadOp.getMemRef();
//       accessMap = loadOp.getAffineMap();
//       indices = loadOp.getMapOperands();
//     } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
//       memref = storeOp.getMemRef();
//       accessMap = storeOp.getAffineMap();
//       indices = storeOp.getMapOperands();
//     } else {
//       return true; // 非访问操作，认为兼容
//     }
    
//     auto memrefType = memref.getType().dyn_cast<MemRefType>();
//     if (!memrefType || memrefType.getRank() != 4) {
//       return true; // 非4D memref，不影响我们的判断
//     }
    
//     auto shape = memrefType.getShape();
    
//     // 检查memref的实际维度是否严格匹配循环迭代范围
//     // 这是最严格的检查：要求tensor的每个维度都精确等于循环的迭代范围
//     bool strictMatch = (shape[0] == dim0 && shape[1] == dim1 && 
//                        shape[2] == dim2 && shape[3] == dim3);
    
//     if (!strictMatch) {
//       LLVM_DEBUG(llvm::dbgs() << "Memref shape mismatch with loop bounds: "
//                  << "memref=" << shape[0] << "x" << shape[1] << "x" 
//                  << shape[2] << "x" << shape[3] 
//                  << ", loops=" << dim0 << "x" << dim1 << "x" 
//                  << dim2 << "x" << dim3 << "\n");
//       return false;
//     }
    
//     // 进一步检查访问模式是否是标准的4D索引模式 [d0, d1, d2, d3]
//     if (accessMap.getNumResults() != 4) {
//       LLVM_DEBUG(llvm::dbgs() << "Non-4D access pattern, rejecting\n");
//       return false;
//     }
    
//     // 检查是否是恒等映射 (d0, d1, d2, d3) -> (d0, d1, d2, d3)
//     for (unsigned i = 0; i < 4; ++i) {
//       auto expr = accessMap.getResult(i);
//       if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
//         if (dimExpr.getPosition() != i) {
//           LLVM_DEBUG(llvm::dbgs() << "Non-identity access pattern at dimension " 
//                      << i << ", rejecting\n");
//           return false;
//         }
//       } else {
//         LLVM_DEBUG(llvm::dbgs() << "Non-dimension expression in access pattern at dimension " 
//                    << i << ", rejecting\n");
//         return false;
//       }
//     }
    
//     return true;
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

// // namespace onnx_mlir {
// //     std::unique_ptr<Pass> createForLoopFlattenPass() {
// //       return std::make_unique<ForLoopFlattenPass>();
// //     }
// // } // namespace onnx_mlir

// // static mlir::PassRegistration<ForLoopFlattenPass> pass;


// // // // // 添加了选择相邻最小两个维度进行flatten的逻辑
// // // // #include "mlir/Pass/Pass.h"
// // // // #include "mlir/IR/PatternMatch.h"
// // // // #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// // // // #include "llvm/Support/Debug.h"
// // // // #include "mlir/Dialect/Affine/IR/AffineOps.h"
// // // // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // // // #include "mlir/Dialect/Arith/IR/Arith.h"
// // // // #include "mlir/IR/AffineExpr.h"
// // // // #include "mlir/IR/AffineMap.h"
// // // // #include "mlir/IR/Builders.h"
// // // // #include "mlir/IR/IRMapping.h"

// // // // using namespace mlir;

// // // // #define DEBUG_TYPE "for-loop-flatten"

// // // // namespace {

// // // // class ForLoopFlattenPattern : public OpRewritePattern<affine::AffineForOp> {
// // // // public:
// // // //   using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

// // // //   LogicalResult matchAndRewrite(affine::AffineForOp outerForOp,
// // // //                                PatternRewriter &rewriter) const override {
// // // //     // 检查是否是4层完美嵌套的affine.for
// // // //     auto nestedStructure = analyzeNestedForStructure(outerForOp);
// // // //     if (!nestedStructure.has_value())
// // // //       return failure();

// // // //     auto [dim0, dim1, dim2, dim3, innerMostOp, mergeIdx0, mergeIdx1] = *nestedStructure;
    
// // // //     LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten: " 
// // // //                << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
// // // //                << " -> merging dimensions " << mergeIdx0 << " and " << mergeIdx1 
// // // //                << " (sizes " << getDimension(dim0, dim1, dim2, dim3, mergeIdx0) 
// // // //                << "x" << getDimension(dim0, dim1, dim2, dim3, mergeIdx1) << ")\n");

// // // //     // 收集需要重塑的memref
// // // //     SmallVector<Value> memrefsToReshape;
// // // //     SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
// // // //     if (!collectMemRefs(innerMostOp, memrefsToReshape, reshapeInfo, dim0, dim1, dim2, dim3))
// // // //       return failure();

// // // //     Location loc = outerForOp.getLoc();
    
// // // //     // 在外层循环前插入reinterpret_cast操作进行降维
// // // //     rewriter.setInsertionPoint(outerForOp);
// // // //     SmallVector<Value> reshapedMemrefs;
    
// // // //     for (auto [memref, originalType] : reshapeInfo) {
// // // //       Value reshaped = createReshapedMemref(rewriter, memref, originalType, 
// // // //                                            dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1, loc);
// // // //       if (!reshaped)
// // // //         return failure();
// // // //       reshapedMemrefs.push_back(reshaped);
// // // //     }

// // // //     // 首先获取原始循环体内容
// // // //     Block *originalInnerBody = &innerMostOp.getRegion().front();
// // // //     SmallVector<Operation*> opsToClone;
// // // //     for (auto &op : originalInnerBody->getOperations()) {
// // // //       if (!isa<affine::AffineYieldOp>(op)) {
// // // //         opsToClone.push_back(&op);
// // // //       }
// // // //     }

// // // //     // 计算合并后的维度大小和新的维度配置
// // // //     auto [newDims, mergedDimIdx] = calculateNewDimensions(dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1);
    
// // // //     // 创建三层嵌套的for循环
// // // //     auto [newFor0, newFor1, newFor2] = createThreeLayerLoops(rewriter, loc, newDims);
    
// // // //     // 获取循环索引
// // // //     Block *innerBlock = &newFor2.getRegion().front();
// // // //     auto loopIndices = getLoopIndices(newFor0, newFor1, newFor2);
    
// // // //     // 在最内层循环体中处理逻辑
// // // //     rewriter.setInsertionPointToStart(innerBlock);
    
// // // //     // 创建memref映射
// // // //     DenseMap<Value, Value> memrefMap;
// // // //     for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
// // // //       memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
// // // //     }
    
// // // //     // 计算原始4D索引
// // // //     auto originalIndices = calculateOriginalIndices(rewriter, loc, loopIndices, 
// // // //                                                    dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1);
    
// // // //     // 建立索引映射
// // // //     IRMapping globalMapping = buildIndexMapping(outerForOp, originalIndices);

// // // //     // 克隆循环体操作
// // // //     cloneLoopBody(rewriter, opsToClone, globalMapping, memrefMap, 
// // // //                   loopIndices, mergeIdx0, mergeIdx1);

// // // //     // 添加yield操作 (从内到外)
// // // //     addYieldOperations(rewriter, newFor0, newFor1, newFor2);

// // // //     // 删除原来的嵌套循环
// // // //     rewriter.eraseOp(outerForOp);

// // // //     return success();
// // // //   }

// // // // private:
// // // //   struct NestedForInfo {
// // // //     int64_t dim0, dim1, dim2, dim3;
// // // //     affine::AffineForOp innerMostOp;
// // // //     int mergeIdx0, mergeIdx1;  // 要合并的两个相邻维度的索引
// // // //   };

// // // //   // 获取指定索引的维度大小
// // // //   int64_t getDimension(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int idx) const {
// // // //     switch (idx) {
// // // //       case 0: return dim0;
// // // //       case 1: return dim1;
// // // //       case 2: return dim2;
// // // //       case 3: return dim3;
// // // //       default: return -1;
// // // //     }
// // // //   }

// // // //   // 创单维度的for循环
// // // //   affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
// // // //                                            Location loc, 
// // // //                                            int64_t lowerBound, 
// // // //                                            int64_t upperBound) const {
// // // //     MLIRContext *ctx = rewriter.getContext();
    
// // // //     AffineMap lowerBoundMap = AffineMap::getConstantMap(lowerBound, ctx);
// // // //     AffineMap upperBoundMap = AffineMap::getConstantMap(upperBound, ctx);
    
// // // //     auto forOp = rewriter.create<affine::AffineForOp>(
// // // //         loc, ValueRange{}, lowerBoundMap, ValueRange{}, upperBoundMap,
// // // //         /*step=*/1, ValueRange{},
// // // //         /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {});
    
// // // //     return forOp;
// // // //   }

// // // //   // 检查是否为完美的4层嵌套循环结构，并找到最小的相邻维度对
// // // //   std::optional<NestedForInfo> analyzeNestedForStructure(
// // // //       affine::AffineForOp outerOp) const {
    
// // // //     // 获取所有4层的维度大小
// // // //     auto dim0 = getLoopBound(outerOp);
// // // //     if (!dim0.has_value()) return std::nullopt;
// // // //     if (!isPerfectlyNestedLevel(outerOp)) return std::nullopt;

// // // //     auto secondFor = findNestedFor(outerOp);
// // // //     if (!secondFor) return std::nullopt;
// // // //     auto dim1 = getLoopBound(*secondFor);
// // // //     if (!dim1.has_value()) return std::nullopt;
// // // //     if (!isPerfectlyNestedLevel(*secondFor)) return std::nullopt;

// // // //     auto thirdFor = findNestedFor(*secondFor);
// // // //     if (!thirdFor) return std::nullopt;
// // // //     auto dim2 = getLoopBound(*thirdFor);
// // // //     if (!dim2.has_value()) return std::nullopt;
// // // //     if (!isPerfectlyNestedLevel(*thirdFor)) return std::nullopt;

// // // //     auto fourthFor = findNestedFor(*thirdFor);
// // // //     if (!fourthFor) return std::nullopt;
// // // //     auto dim3 = getLoopBound(*fourthFor);
// // // //     if (!dim3.has_value()) return std::nullopt;
// // // //     if (!isSimpleLoopBody(*fourthFor)) return std::nullopt;

// // // //     // 找到最小的相邻维度对
// // // //     std::array<int64_t, 4> dims = {*dim0, *dim1, *dim2, *dim3};
// // // //     int bestIdx0 = 0, bestIdx1 = 1;
// // // //     int64_t minProduct = dims[0] * dims[1];
    
// // // //     for (int i = 1; i < 3; ++i) {
// // // //       int64_t product = dims[i] * dims[i + 1];
// // // //       if (product < minProduct) {
// // // //         minProduct = product;
// // // //         bestIdx0 = i;
// // // //         bestIdx1 = i + 1;
// // // //       }
// // // //     }

// // // //     LLVM_DEBUG(llvm::dbgs() << "Validated perfect 4-layer nesting: " 
// // // //                << *dim0 << "x" << *dim1 << "x" << *dim2 << "x" << *dim3 
// // // //                << ", choosing to merge dimensions " << bestIdx0 << " and " << bestIdx1 
// // // //                << " (product: " << minProduct << ")\n");

// // // //     return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *fourthFor, bestIdx0, bestIdx1};
// // // //   }

// // // //   // 计算合并后的新维度配置
// // // //   std::pair<std::array<int64_t, 3>, int> calculateNewDimensions(
// // // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
// // // //       int mergeIdx0, int mergeIdx1) const {
    
// // // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
// // // //     std::array<int64_t, 3> newDims;
    
// // // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1];
    
// // // //     // 创建新的3D维度数组
// // // //     int newIdx = 0;
// // // //     int mergedDimIdx = -1;
    
// // // //     for (int i = 0; i < 4; ++i) {
// // // //       if (i == mergeIdx0) {
// // // //         newDims[newIdx] = mergedSize;
// // // //         mergedDimIdx = newIdx;
// // // //         newIdx++;
// // // //         ++i; // 跳过mergeIdx1
// // // //       } else {
// // // //         newDims[newIdx] = originalDims[i];
// // // //         newIdx++;
// // // //       }
// // // //     }
    
// // // //     return {newDims, mergedDimIdx};
// // // //   }

// // // //   // 创建三层嵌套循环
// // // //   std::tuple<affine::AffineForOp, affine::AffineForOp, affine::AffineForOp> 
// // // //   createThreeLayerLoops(PatternRewriter &rewriter, Location loc, 
// // // //                        const std::array<int64_t, 3>& newDims) const {
    
// // // //     auto outerFor = createSingleDimForOp(rewriter, loc, 0, newDims[0]);
// // // //     Block *outerBlock = &outerFor.getRegion().front();
    
// // // //     rewriter.setInsertionPointToStart(outerBlock);
// // // //     auto middleFor = createSingleDimForOp(rewriter, loc, 0, newDims[1]);
// // // //     Block *middleBlock = &middleFor.getRegion().front();
    
// // // //     rewriter.setInsertionPointToStart(middleBlock);
// // // //     auto innerFor = createSingleDimForOp(rewriter, loc, 0, newDims[2]);
    
// // // //     return {outerFor, middleFor, innerFor};
// // // //   }

// // // //   // 获取三层循环的索引
// // // //   std::array<BlockArgument, 3> getLoopIndices(
// // // //       affine::AffineForOp for0, affine::AffineForOp for1, affine::AffineForOp for2) const {
// // // //     return {
// // // //       for0.getRegion().front().getArgument(0),
// // // //       for1.getRegion().front().getArgument(0), 
// // // //       for2.getRegion().front().getArgument(0)
// // // //     };
// // // //   }

// // // //   // 计算原始4D索引
// // // //   std::array<Value, 4> calculateOriginalIndices(
// // // //       PatternRewriter &rewriter, Location loc,
// // // //       const std::array<BlockArgument, 3>& loopIndices,
// // // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
// // // //       int mergeIdx0, int mergeIdx1) const {
    
// // // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
// // // //     std::array<Value, 4> originalIndices;
    
// // // //     // 首先将新索引映射回原始索引位置
// // // //     int newIdx = 0;
// // // //     for (int i = 0; i < 4; ++i) {
// // // //       if (i == mergeIdx0) {
// // // //         // 对于合并的维度，需要计算分解
// // // //         Value mergedIdx = loopIndices[newIdx];
// // // //         Value dim1Size = rewriter.create<arith::ConstantIndexOp>(loc, originalDims[mergeIdx1]);
// // // //         originalIndices[i] = rewriter.create<arith::DivSIOp>(loc, mergedIdx, dim1Size);
// // // //         originalIndices[i + 1] = rewriter.create<arith::RemSIOp>(loc, mergedIdx, dim1Size);
// // // //         ++i; // 跳过下一个索引
// // // //         ++newIdx;
// // // //       } else {
// // // //         originalIndices[i] = loopIndices[newIdx];
// // // //         ++newIdx;
// // // //       }
// // // //     }
    
// // // //     return originalIndices;
// // // //   }

// // // //   // 建立索引映射
// // // //   IRMapping buildIndexMapping(affine::AffineForOp outerForOp, 
// // // //                              const std::array<Value, 4>& originalIndices) const {
// // // //     IRMapping globalMapping;
    
// // // //     // 映射所有4层循环的索引变量
// // // //     globalMapping.map(outerForOp.getInductionVar(), originalIndices[0]);
    
// // // //     auto level2Op = findNestedFor(outerForOp);
// // // //     if (level2Op) {
// // // //       globalMapping.map(level2Op->getInductionVar(), originalIndices[1]);
// // // //       auto level3Op = findNestedFor(*level2Op);
// // // //       if (level3Op) {
// // // //         globalMapping.map(level3Op->getInductionVar(), originalIndices[2]);
// // // //         auto level4Op = findNestedFor(*level3Op);
// // // //         if (level4Op) {
// // // //           globalMapping.map(level4Op->getInductionVar(), originalIndices[3]);
// // // //         }
// // // //       }
// // // //     }
    
// // // //     return globalMapping;
// // // //   }

// // // //   // 克隆循环体操作
// // // //   void cloneLoopBody(PatternRewriter &rewriter, const SmallVector<Operation*>& opsToClone,
// // // //                     IRMapping& globalMapping, const DenseMap<Value, Value>& memrefMap,
// // // //                     const std::array<BlockArgument, 3>& loopIndices,
// // // //                     int mergeIdx0, int mergeIdx1) const {
    
// // // //     for (Operation *op : opsToClone) {
// // // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // // //         Value memref = loadOp.getMemRef();
// // // //         if (memrefMap.count(memref)) {
// // // //           Value newMemref = memrefMap.lookup(memref);
// // // //           auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
// // // //               op->getLoc(), newMemref, ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // // //           globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
// // // //         } else {
// // // //           auto clonedOp = rewriter.clone(*op, globalMapping);
// // // //           globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
// // // //         }
// // // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // // //         Value memref = storeOp.getMemRef();
// // // //         if (memrefMap.count(memref)) {
// // // //           Value newMemref = memrefMap.lookup(memref);
// // // //           Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
// // // //           rewriter.create<affine::AffineStoreOp>(
// // // //               op->getLoc(), valueToStore, newMemref, 
// // // //               ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // // //         } else {
// // // //           rewriter.clone(*op, globalMapping);
// // // //         }
// // // //       } else {
// // // //         auto clonedOp = rewriter.clone(*op, globalMapping);
// // // //         for (unsigned i = 0; i < op->getNumResults(); ++i) {
// // // //           globalMapping.map(op->getResult(i), clonedOp->getResult(i));
// // // //         }
// // // //       }
// // // //     }
// // // //   }

// // // //   // 添加yield操作
// // // //   void addYieldOperations(PatternRewriter &rewriter,
// // // //                          affine::AffineForOp for0, affine::AffineForOp for1, 
// // // //                          affine::AffineForOp for2) const {
// // // //     Location loc = for0.getLoc();
    
// // // //     rewriter.create<affine::AffineYieldOp>(loc);
// // // //     rewriter.setInsertionPointAfter(for2);
// // // //     rewriter.create<affine::AffineYieldOp>(loc);
// // // //     rewriter.setInsertionPointAfter(for1);
// // // //     rewriter.create<affine::AffineYieldOp>(loc);
// // // //   }

// // // //   // 检查一个循环层是否为完美嵌套
// // // //   bool isPerfectlyNestedLevel(affine::AffineForOp forOp) const {
// // // //     Block &body = forOp.getRegion().front();
    
// // // //     int nestedForCount = 0;
// // // //     int yieldCount = 0;
// // // //     int totalOps = 0;
    
// // // //     for (auto &op : body.getOperations()) {
// // // //       totalOps++;
// // // //       if (isa<affine::AffineForOp>(&op)) {
// // // //         nestedForCount++;
// // // //       } else if (isa<affine::AffineYieldOp>(&op)) {
// // // //         yieldCount++;
// // // //       } else {
// // // //         LLVM_DEBUG(llvm::dbgs() << "Found non-nested operation in loop body: " 
// // // //                    << op.getName() << ", rejecting perfect nesting\n");
// // // //         return false;
// // // //       }
// // // //     }
    
// // // //     bool isPerfect = (nestedForCount == 1 && yieldCount == 1 && totalOps == 2);
    
// // // //     if (!isPerfect) {
// // // //       LLVM_DEBUG(llvm::dbgs() << "Level validation failed: nestedForCount=" 
// // // //                  << nestedForCount << ", yieldCount=" << yieldCount 
// // // //                  << ", totalOps=" << totalOps << "\n");
// // // //     }
    
// // // //     return isPerfect;
// // // //   }

// // // //   // 检查最内层循环体是否只包含简单操作
// // // //   bool isSimpleLoopBody(affine::AffineForOp forOp) const {
// // // //     Block &body = forOp.getRegion().front();
    
// // // //     for (auto &op : body.getOperations()) {
// // // //       if (isa<affine::AffineForOp>(&op)) {
// // // //         LLVM_DEBUG(llvm::dbgs() << "Found nested for loop in innermost body, rejecting\n");
// // // //         return false;
// // // //       } else if (isa<affine::AffineLoadOp, affine::AffineStoreOp, 
// // // //                      arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
// // // //                      arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
// // // //                      arith::ConstantOp, affine::AffineYieldOp>(&op)) {
// // // //         continue;
// // // //       } else {
// // // //         LLVM_DEBUG(llvm::dbgs() << "Found operation in innermost body: " 
// // // //                    << op.getName() << " (allowing)\n");
// // // //       }
// // // //     }
    
// // // //     return true;
// // // //   }

// // // //   std::optional<affine::AffineForOp> findNestedFor(
// // // //       affine::AffineForOp parentOp) const {
// // // //     for (auto &op : parentOp.getRegion().front()) {
// // // //       if (auto nestedFor = dyn_cast<affine::AffineForOp>(&op))
// // // //         return nestedFor;
// // // //     }
// // // //     return std::nullopt;
// // // //   }

// // // //   std::optional<int64_t> getLoopBound(affine::AffineForOp forOp) const {
// // // //     auto lowerBound = forOp.getLowerBound();
// // // //     auto lowerMap = lowerBound.getMap();
// // // //     if (lowerMap.getNumResults() != 1)
// // // //       return std::nullopt;
    
// // // //     auto lowerExpr = lowerMap.getResult(0);
// // // //     if (auto lowerConst = lowerExpr.dyn_cast<AffineConstantExpr>()) {
// // // //       if (lowerConst.getValue() != 0)
// // // //         return std::nullopt;
// // // //     } else {
// // // //       return std::nullopt;
// // // //     }

// // // //     auto upperBound = forOp.getUpperBound();
// // // //     auto upperMap = upperBound.getMap();
// // // //     if (upperMap.getNumResults() != 1)
// // // //       return std::nullopt;
      
// // // //     auto upperExpr = upperMap.getResult(0);
// // // //     if (auto upperConst = upperExpr.dyn_cast<AffineConstantExpr>()) {
// // // //       return upperConst.getValue();
// // // //     }
    
// // // //     return std::nullopt;
// // // //   }

// // // //   bool collectMemRefs(affine::AffineForOp innerMostOp,
// // // //                      SmallVector<Value> &memrefsToReshape,
// // // //                      SmallVector<std::pair<Value, MemRefType>> &reshapeInfo,
// // // //                      int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    
// // // //     SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
// // // //     bool hasIncompatibleAccess = false;
    
// // // //     innerMostOp.getRegion().walk([&](Operation *op) {
// // // //       if (hasIncompatibleAccess) return;
      
// // // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // // //         Value memref = loadOp.getMemRef();
// // // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // // //         if (memrefType && memrefType.getRank() == 4) {
// // // //           if (!isMemrefAccessCompatibleWithLoopBounds(loadOp, dim0, dim1, dim2, dim3)) {
// // // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in load, rejecting flatten\n");
// // // //             hasIncompatibleAccess = true;
// // // //             return;
// // // //           }
          
// // // //           if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
// // // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // // //                 == allAccessedMemrefs.end()) {
// // // //               allAccessedMemrefs.push_back({memref, memrefType});
// // // //             }
// // // //           }
// // // //         }
// // // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // // //         Value memref = storeOp.getMemRef();
// // // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // // //         if (memrefType && memrefType.getRank() == 4) {
// // // //           if (!isMemrefAccessCompatibleWithLoopBounds(storeOp, dim0, dim1, dim2, dim3)) {
// // // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in store, rejecting flatten\n");
// // // //             hasIncompatibleAccess = true;
// // // //             return;
// // // //           }
          
// // // //           if (isCompatibleShape(memrefType, dim0, dim1, dim2, dim3)) {
// // // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // // //                 == allAccessedMemrefs.end()) {
// // // //               allAccessedMemrefs.push_back({memref, memrefType});
// // // //             }
// // // //           }
// // // //         }
// // // //       }
// // // //     });

// // // //     if (hasIncompatibleAccess) {
// // // //       return false;
// // // //     }

// // // //     for (auto [memref, memrefType] : allAccessedMemrefs) {
// // // //       if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
// // // //           == memrefsToReshape.end()) {
// // // //         memrefsToReshape.push_back(memref);
// // // //         reshapeInfo.push_back({memref, memrefType});
// // // //       }
// // // //     }

// // // //     return !memrefsToReshape.empty();
// // // //   }

// // // //   bool isCompatibleShape(MemRefType memrefType, int64_t dim0, int64_t dim1, 
// // // //                         int64_t dim2, int64_t dim3) const {
// // // //     auto shape = memrefType.getShape();
// // // //     return shape.size() == 4 && 
// // // //            (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
// // // //            (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
// // // //            (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
// // // //            (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
// // // //   }

// // // //   bool isMemrefAccessCompatibleWithLoopBounds(Operation* op, int64_t dim0, int64_t dim1, 
// // // //                                               int64_t dim2, int64_t dim3) const {
// // // //     Value memref;
// // // //     AffineMap accessMap;
// // // //     SmallVector<Value> indices;
    
// // // //     if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // // //       memref = loadOp.getMemRef();
// // // //       accessMap = loadOp.getAffineMap();
// // // //       indices = loadOp.getMapOperands();
// // // //     } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // // //       memref = storeOp.getMemRef();
// // // //       accessMap = storeOp.getAffineMap();
// // // //       indices = storeOp.getMapOperands();
// // // //     } else {
// // // //       return true;
// // // //     }
    
// // // //     auto memrefType = memref.getType().dyn_cast<MemRefType>();
// // // //     if (!memrefType || memrefType.getRank() != 4) {
// // // //       return true;
// // // //     }
    
// // // //     auto shape = memrefType.getShape();
    
// // // //     bool strictMatch = (shape[0] == dim0 && shape[1] == dim1 && 
// // // //                        shape[2] == dim2 && shape[3] == dim3);
    
// // // //     if (!strictMatch) {
// // // //       LLVM_DEBUG(llvm::dbgs() << "Memref shape mismatch with loop bounds: "
// // // //                  << "memref=" << shape[0] << "x" << shape[1] << "x" 
// // // //                  << shape[2] << "x" << shape[3] 
// // // //                  << ", loops=" << dim0 << "x" << dim1 << "x" 
// // // //                  << dim2 << "x" << dim3 << "\n");
// // // //       return false;
// // // //     }
    
// // // //     if (accessMap.getNumResults() != 4) {
// // // //       LLVM_DEBUG(llvm::dbgs() << "Non-4D access pattern, rejecting\n");
// // // //       return false;
// // // //     }
    
// // // //     for (unsigned i = 0; i < 4; ++i) {
// // // //       auto expr = accessMap.getResult(i);
// // // //       if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
// // // //         if (dimExpr.getPosition() != i) {
// // // //           LLVM_DEBUG(llvm::dbgs() << "Non-identity access pattern at dimension " 
// // // //                      << i << ", rejecting\n");
// // // //           return false;
// // // //         }
// // // //       } else {
// // // //         LLVM_DEBUG(llvm::dbgs() << "Non-dimension expression in access pattern at dimension " 
// // // //                    << i << ", rejecting\n");
// // // //         return false;
// // // //       }
// // // //     }
    
// // // //     return true;
// // // //   }

// // // //   // 修改后的创建重塑memref方法，支持任意相邻维度合并
// // // //   Value createReshapedMemref(PatternRewriter &rewriter, Value originalMemref,
// // // //                             MemRefType originalType, int64_t dim0, int64_t dim1,
// // // //                             int64_t dim2, int64_t dim3, int mergeIdx0, int mergeIdx1,
// // // //                             Location loc) const {
    
// // // //     Type elementType = originalType.getElementType();
// // // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
    
// // // //     // 计算合并后的维度
// // // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1];
    
// // // //     // 创建新的3D shape
// // // //     SmallVector<int64_t> newShape;
// // // //     for (int i = 0; i < 4; ++i) {
// // // //       if (i == mergeIdx0) {
// // // //         newShape.push_back(mergedSize);
// // // //         ++i; // 跳过mergeIdx1
// // // //       } else {
// // // //         newShape.push_back(originalDims[i]);
// // // //       }
// // // //     }
    
// // // //     auto newMemrefType = MemRefType::get(newShape, elementType);

// // // //     // 计算strides
// // // //     SmallVector<int64_t> strides;
// // // //     SmallVector<int64_t> offset = {0};
    
// // // //     // 根据合并的维度位置计算strides
// // // //     if (mergeIdx0 == 0) { // 合并前两个维度
// // // //       strides = {dim2 * dim3, dim3, 1};
// // // //     } else if (mergeIdx0 == 1) { // 合并中间两个维度  
// // // //       strides = {(dim1 * dim2) * dim3, dim3, 1};
// // // //     } else { // 合并后两个维度
// // // //       strides = {dim1 * (dim2 * dim3), (dim2 * dim3), 1};
// // // //     }

// // // //     return rewriter.create<memref::ReinterpretCastOp>(
// // // //         loc, newMemrefType, originalMemref, 
// // // //         /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
// // // //         /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
// // // //     ).getResult();
// // // //   }
// // // // };

// // // // struct ForLoopFlattenPass
// // // //     : public PassWrapper<ForLoopFlattenPass, OperationPass<ModuleOp>> {
  
// // // //   StringRef getArgument() const final { return "for-loop-flatten"; }
// // // //   StringRef getDescription() const final {
// // // //     return "Flatten 4D nested for loops to 3D by merging the smallest adjacent dimensions";
// // // //   }
  
// // // //   void getDependentDialects(DialectRegistry &registry) const override {
// // // //     registry.insert<affine::AffineDialect, memref::MemRefDialect, arith::ArithDialect>();
// // // //   }
  
// // // //   void runOnOperation() override {
// // // //     ModuleOp moduleOp = getOperation();
// // // //     MLIRContext *context = &getContext();
    
// // // //     LLVM_DEBUG(llvm::dbgs() << "Running ForLoopFlattenPass\n");
    
// // // //     RewritePatternSet patterns(context);
// // // //     patterns.add<ForLoopFlattenPattern>(context);
    
// // // //     if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
// // // //       signalPassFailure();
// // // //     }
    
// // // //     LLVM_DEBUG(llvm::dbgs() << "Completed ForLoopFlattenPass\n");
// // // //   }
// // // // };

// // // // } // end anonymous namespace

// // // // namespace onnx_mlir {
// // // //     std::unique_ptr<Pass> createForLoopFlattenPass() {
// // // //       return std::make_unique<ForLoopFlattenPass>();
// // // //     }
// // // // } // namespace onnx_mlir

// // // // static mlir::PassRegistration<ForLoopFlattenPass> pass;


// // // #include "mlir/Pass/Pass.h"
// // // #include "mlir/IR/PatternMatch.h"
// // // #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// // // #include "llvm/Support/Debug.h"
// // // #include "mlir/Dialect/Affine/IR/AffineOps.h"
// // // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // // #include "mlir/Dialect/Arith/IR/Arith.h"
// // // #include "mlir/IR/AffineExpr.h"
// // // #include "mlir/IR/AffineMap.h"
// // // #include "mlir/IR/Builders.h"
// // // #include "mlir/IR/IRMapping.h"

// // // using namespace mlir;

// // // #define DEBUG_TYPE "for-loop-flatten"

// // // namespace {

// // // class ForLoopFlattenPattern : public OpRewritePattern<affine::AffineForOp> {
// // // public:
// // //   using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

// // //   LogicalResult matchAndRewrite(affine::AffineForOp outerForOp,
// // //                                PatternRewriter &rewriter) const override {
// // //     // 首先检查是否是5层完美嵌套的affine.for
// // //     auto nested5DStructure = analyze5DNestedForStructure(outerForOp);
// // //     if (nested5DStructure.has_value()) {
// // //       auto info = *nested5DStructure;
// // //       return handle5DNestedFor(outerForOp, rewriter, info);
// // //     }

// // //     // 检查是否是4层完美嵌套的affine.for (原有逻辑)
// // //     auto nestedStructure = analyzeNestedForStructure(outerForOp);
// // //     if (!nestedStructure.has_value())
// // //       return failure();

// // //     auto [dim0, dim1, dim2, dim3, innerMostOp, mergeIdx0, mergeIdx1] = *nestedStructure;
    
// // //     LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten: " 
// // //                << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
// // //                << " -> merging dimensions " << mergeIdx0 << " and " << mergeIdx1 
// // //                << " (sizes " << getDimension4D(dim0, dim1, dim2, dim3, mergeIdx0) 
// // //                << "x" << getDimension4D(dim0, dim1, dim2, dim3, mergeIdx1) << ")\n");

// // //     // 收集需要重塑的memref
// // //     SmallVector<Value> memrefsToReshape;
// // //     SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
// // //     if (!collectMemRefs4D(innerMostOp, memrefsToReshape, reshapeInfo, dim0, dim1, dim2, dim3))
// // //       return failure();

// // //     Location loc = outerForOp.getLoc();
    
// // //     // 在外层循环前插入reinterpret_cast操作进行降维
// // //     rewriter.setInsertionPoint(outerForOp);
// // //     SmallVector<Value> reshapedMemrefs;
    
// // //     for (auto [memref, originalType] : reshapeInfo) {
// // //       Value reshaped = createReshapedMemref4D(rewriter, memref, originalType, 
// // //                                            dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1, loc);
// // //       if (!reshaped)
// // //         return failure();
// // //       reshapedMemrefs.push_back(reshaped);
// // //     }

// // //     // 首先获取原始循环体内容
// // //     Block *originalInnerBody = &innerMostOp.getRegion().front();
// // //     SmallVector<Operation*> opsToClone;
// // //     for (auto &op : originalInnerBody->getOperations()) {
// // //       if (!isa<affine::AffineYieldOp>(op)) {
// // //         opsToClone.push_back(&op);
// // //       }
// // //     }

// // //     // 计算合并后的维度大小和新的维度配置
// // //     auto [newDims, mergedDimIdx] = calculateNewDimensions4D(dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1);
    
// // //     // 创建三层嵌套的for循环
// // //     auto [newFor0, newFor1, newFor2] = createThreeLayerLoops(rewriter, loc, newDims);
    
// // //     // 获取循环索引
// // //     Block *innerBlock = &newFor2.getRegion().front();
// // //     auto loopIndices = getLoopIndices(newFor0, newFor1, newFor2);
    
// // //     // 在最内层循环体中处理逻辑
// // //     rewriter.setInsertionPointToStart(innerBlock);
    
// // //     // 创建memref映射
// // //     DenseMap<Value, Value> memrefMap;
// // //     for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
// // //       memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
// // //     }
    
// // //     // 计算原始4D索引
// // //     auto originalIndices = calculateOriginalIndices4D(rewriter, loc, loopIndices, 
// // //                                                    dim0, dim1, dim2, dim3, mergeIdx0, mergeIdx1);
    
// // //     // 建立索引映射
// // //     IRMapping globalMapping = buildIndexMapping4D(outerForOp, originalIndices);

// // //     // 克隆循环体操作
// // //     cloneLoopBody4D(rewriter, opsToClone, globalMapping, memrefMap, 
// // //                   loopIndices, mergeIdx0, mergeIdx1);

// // //     // 添加yield操作 (从内到外)
// // //     addYieldOperations(rewriter, newFor0, newFor1, newFor2);

// // //     // 删除原来的嵌套循环
// // //     rewriter.eraseOp(outerForOp);

// // //     return success();
// // //   }

// // // private:
// // //   struct NestedForInfo {
// // //     int64_t dim0, dim1, dim2, dim3;
// // //     affine::AffineForOp innerMostOp;
// // //     int mergeIdx0, mergeIdx1;  // 要合并的两个相邻维度的索引
// // //   };

// // //   struct Nested5DForInfo {
// // //     int64_t dim0, dim1, dim2, dim3, dim4;
// // //     affine::AffineForOp innerMostOp;
// // //     int mergeIdx0, mergeIdx1, mergeIdx2;  // 要合并的三个相邻维度的索引
// // //   };

// // //   // 获取指定索引的4D维度大小
// // //   int64_t getDimension4D(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int idx) const {
// // //     switch (idx) {
// // //       case 0: return dim0;
// // //       case 1: return dim1;
// // //       case 2: return dim2;
// // //       case 3: return dim3;
// // //       default: return -1;
// // //     }
// // //   }

// // //   // 获取指定索引的5D维度大小
// // //   int64_t getDimension5D(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4, int idx) const {
// // //     switch (idx) {
// // //       case 0: return dim0;
// // //       case 1: return dim1;
// // //       case 2: return dim2;
// // //       case 3: return dim3;
// // //       case 4: return dim4;
// // //       default: return -1;
// // //     }
// // //   }

// // //   // 检查是否为完美的5层嵌套循环结构，并找到最小的相邻三个维度
// // //   std::optional<Nested5DForInfo> analyze5DNestedForStructure(
// // //       affine::AffineForOp outerOp) const {
    
// // //     // 获取所有5层的维度大小
// // //     auto dim0 = getLoopBound(outerOp);
// // //     if (!dim0.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(outerOp)) return std::nullopt;

// // //     auto secondFor = findNestedFor(outerOp);
// // //     if (!secondFor) return std::nullopt;
// // //     auto dim1 = getLoopBound(*secondFor);
// // //     if (!dim1.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(*secondFor)) return std::nullopt;

// // //     auto thirdFor = findNestedFor(*secondFor);
// // //     if (!thirdFor) return std::nullopt;
// // //     auto dim2 = getLoopBound(*thirdFor);
// // //     if (!dim2.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(*thirdFor)) return std::nullopt;

// // //     auto fourthFor = findNestedFor(*thirdFor);
// // //     if (!fourthFor) return std::nullopt;
// // //     auto dim3 = getLoopBound(*fourthFor);
// // //     if (!dim3.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(*fourthFor)) return std::nullopt;

// // //     auto fifthFor = findNestedFor(*fourthFor);
// // //     if (!fifthFor) return std::nullopt;
// // //     auto dim4 = getLoopBound(*fifthFor);
// // //     if (!dim4.has_value()) return std::nullopt;
// // //     if (!isSimpleLoopBody(*fifthFor)) return std::nullopt;

// // //     // 找到最小的相邻三个维度
// // //     std::array<int64_t, 5> dims = {*dim0, *dim1, *dim2, *dim3, *dim4};
// // //     int bestIdx0 = 0, bestIdx1 = 1, bestIdx2 = 2;
// // //     int64_t minProduct = dims[0] * dims[1] * dims[2];
    
// // //     for (int i = 1; i <= 2; ++i) {
// // //       int64_t product = dims[i] * dims[i + 1] * dims[i + 2];
// // //       if (product < minProduct) {
// // //         minProduct = product;
// // //         bestIdx0 = i;
// // //         bestIdx1 = i + 1;
// // //         bestIdx2 = i + 2;
// // //       }
// // //     }

// // //     LLVM_DEBUG(llvm::dbgs() << "Validated perfect 5-layer nesting: " 
// // //                << *dim0 << "x" << *dim1 << "x" << *dim2 << "x" << *dim3 << "x" << *dim4
// // //                << ", choosing to merge dimensions " << bestIdx0 << ", " << bestIdx1 << " and " << bestIdx2
// // //                << " (product: " << minProduct << ")\n");

// // //     return Nested5DForInfo{*dim0, *dim1, *dim2, *dim3, *dim4, *fifthFor, bestIdx0, bestIdx1, bestIdx2};
// // //   }

// // //   // 处理5D嵌套循环
// // //   LogicalResult handle5DNestedFor(affine::AffineForOp outerForOp,
// // //                                  PatternRewriter &rewriter,
// // //                                  Nested5DForInfo& info) const {
    
// // //     LLVM_DEBUG(llvm::dbgs() << "Processing 5D nested for loop: " 
// // //                << info.dim0 << "x" << info.dim1 << "x" << info.dim2 << "x" << info.dim3 << "x" << info.dim4
// // //                << " -> merging dimensions " << info.mergeIdx0 << ", " << info.mergeIdx1 << ", " << info.mergeIdx2 << "\n");

// // //     // 收集需要重塑的memref
// // //     SmallVector<Value> memrefsToReshape;
// // //     SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
// // //     if (!collectMemRefs5D(info.innerMostOp, memrefsToReshape, reshapeInfo, 
// // //                           info.dim0, info.dim1, info.dim2, info.dim3, info.dim4))
// // //       return failure();

// // //     Location loc = outerForOp.getLoc();
    
// // //     // 在外层循环前插入reinterpret_cast操作进行降维
// // //     rewriter.setInsertionPoint(outerForOp);
// // //     SmallVector<Value> reshapedMemrefs;
    
// // //     for (auto [memref, originalType] : reshapeInfo) {
// // //       Value reshaped = createReshapedMemref5D(rewriter, memref, originalType, 
// // //                                              info.dim0, info.dim1, info.dim2, info.dim3, info.dim4,
// // //                                              info.mergeIdx0, info.mergeIdx1, info.mergeIdx2, loc);
// // //       if (!reshaped)
// // //         return failure();
// // //       reshapedMemrefs.push_back(reshaped);
// // //     }

// // //     // 获取原始循环体内容
// // //     Block *originalInnerBody = &info.innerMostOp.getRegion().front();
// // //     SmallVector<Operation*> opsToClone;
// // //     for (auto &op : originalInnerBody->getOperations()) {
// // //       if (!isa<affine::AffineYieldOp>(op)) {
// // //         opsToClone.push_back(&op);
// // //       }
// // //     }

// // //     // 计算合并后的维度大小
// // //     auto newDims = calculateNewDimensions5D(info.dim0, info.dim1, info.dim2, info.dim3, info.dim4,
// // //                                            info.mergeIdx0, info.mergeIdx1, info.mergeIdx2);
    
// // //     // 创建三层嵌套的for循环
// // //     auto [newFor0, newFor1, newFor2] = createThreeLayerLoops(rewriter, loc, newDims);
    
// // //     // 获取循环索引
// // //     Block *innerBlock = &newFor2.getRegion().front();
// // //     auto loopIndices = getLoopIndices(newFor0, newFor1, newFor2);
    
// // //     // 在最内层循环体中处理逻辑
// // //     rewriter.setInsertionPointToStart(innerBlock);
    
// // //     // 创建memref映射
// // //     DenseMap<Value, Value> memrefMap;
// // //     for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
// // //       memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
// // //     }
    
// // //     // 计算原始5D索引
// // //     auto originalIndices = calculateOriginalIndices5D(rewriter, loc, loopIndices, 
// // //                                                      info.dim0, info.dim1, info.dim2, info.dim3, info.dim4,
// // //                                                      info.mergeIdx0, info.mergeIdx1, info.mergeIdx2);
    
// // //     // 建立索引映射
// // //     IRMapping globalMapping = buildIndexMapping5D(outerForOp, originalIndices);

// // //     // 克隆循环体操作
// // //     cloneLoopBody5D(rewriter, opsToClone, globalMapping, memrefMap, loopIndices);

// // //     // 添加yield操作
// // //     addYieldOperations(rewriter, newFor0, newFor1, newFor2);

// // //     // 删除原来的嵌套循环
// // //     rewriter.eraseOp(outerForOp);

// // //     return success();
// // //   }

// // //   // 计算5D合并后的新维度配置
// // //   std::array<int64_t, 3> calculateNewDimensions5D(
// // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
// // //       int mergeIdx0, int mergeIdx1, int mergeIdx2) const {
    
// // //     std::array<int64_t, 5> originalDims = {dim0, dim1, dim2, dim3, dim4};
// // //     std::array<int64_t, 3> newDims;
    
// // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1] * originalDims[mergeIdx2];
    
// // //     // 创建新的3D维度数组
// // //     int newIdx = 0;
// // //     for (int i = 0; i < 5; ++i) {
// // //       if (i == mergeIdx0) {
// // //         newDims[newIdx] = mergedSize;
// // //         newIdx++;
// // //         i += 2; // 跳过mergeIdx1和mergeIdx2
// // //       } else {
// // //         newDims[newIdx] = originalDims[i];
// // //         newIdx++;
// // //       }
// // //     }
    
// // //     return newDims;
// // //   }

// // //   // 计算5D原始索引
// // //   std::array<Value, 5> calculateOriginalIndices5D(
// // //       PatternRewriter &rewriter, Location loc,
// // //       const std::array<BlockArgument, 3>& loopIndices,
// // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
// // //       int mergeIdx0, int mergeIdx1, int mergeIdx2) const {
    
// // //     std::array<int64_t, 5> originalDims = {dim0, dim1, dim2, dim3, dim4};
// // //     std::array<Value, 5> originalIndices;
    
// // //     // 将新索引映射回原始索引位置
// // //     int newIdx = 0;
// // //     for (int i = 0; i < 5; ++i) {
// // //       if (i == mergeIdx0) {
// // //         // 对于合并的三个维度，需要计算分解
// // //         Value mergedIdx = loopIndices[newIdx];
// // //         Value dim1Size = rewriter.create<arith::ConstantIndexOp>(loc, originalDims[mergeIdx1]);
// // //         Value dim2Size = rewriter.create<arith::ConstantIndexOp>(loc, originalDims[mergeIdx2]);
// // //         Value dim12Size = rewriter.create<arith::MulIOp>(loc, dim1Size, dim2Size);
        
// // //         // 计算第一个维度索引: mergedIdx / (dim1 * dim2)
// // //         originalIndices[i] = rewriter.create<arith::DivSIOp>(loc, mergedIdx, dim12Size);
        
// // //         // 计算第二个维度索引: (mergedIdx % (dim1 * dim2)) / dim2
// // //         Value temp = rewriter.create<arith::RemSIOp>(loc, mergedIdx, dim12Size);
// // //         originalIndices[i + 1] = rewriter.create<arith::DivSIOp>(loc, temp, dim2Size);
        
// // //         // 计算第三个维度索引: mergedIdx % dim2
// // //         originalIndices[i + 2] = rewriter.create<arith::RemSIOp>(loc, mergedIdx, dim2Size);
        
// // //         i += 2; // 跳过下两个索引
// // //         ++newIdx;
// // //       } else {
// // //         originalIndices[i] = loopIndices[newIdx];
// // //         ++newIdx;
// // //       }
// // //     }
    
// // //     return originalIndices;
// // //   }

// // //   // 建立5D索引映射
// // //   IRMapping buildIndexMapping5D(affine::AffineForOp outerForOp, 
// // //                                const std::array<Value, 5>& originalIndices) const {
// // //     IRMapping globalMapping;
    
// // //     // 映射所有5层循环的索引变量
// // //     globalMapping.map(outerForOp.getInductionVar(), originalIndices[0]);
    
// // //     auto level2Op = findNestedFor(outerForOp);
// // //     if (level2Op) {
// // //       globalMapping.map(level2Op->getInductionVar(), originalIndices[1]);
// // //       auto level3Op = findNestedFor(*level2Op);
// // //       if (level3Op) {
// // //         globalMapping.map(level3Op->getInductionVar(), originalIndices[2]);
// // //         auto level4Op = findNestedFor(*level3Op);
// // //         if (level4Op) {
// // //           globalMapping.map(level4Op->getInductionVar(), originalIndices[3]);
// // //           auto level5Op = findNestedFor(*level4Op);
// // //           if (level5Op) {
// // //             globalMapping.map(level5Op->getInductionVar(), originalIndices[4]);
// // //           }
// // //         }
// // //       }
// // //     }
    
// // //     return globalMapping;
// // //   }

// // //   // 克隆5D循环体操作
// // //   void cloneLoopBody5D(PatternRewriter &rewriter, const SmallVector<Operation*>& opsToClone,
// // //                       IRMapping& globalMapping, const DenseMap<Value, Value>& memrefMap,
// // //                       const std::array<BlockArgument, 3>& loopIndices) const {
    
// // //     for (Operation *op : opsToClone) {
// // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //         Value memref = loadOp.getMemRef();
// // //         if (memrefMap.count(memref)) {
// // //           Value newMemref = memrefMap.lookup(memref);
// // //           auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
// // //               op->getLoc(), newMemref, ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // //           globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
// // //         } else {
// // //           auto clonedOp = rewriter.clone(*op, globalMapping);
// // //           globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
// // //         }
// // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //         Value memref = storeOp.getMemRef();
// // //         if (memrefMap.count(memref)) {
// // //           Value newMemref = memrefMap.lookup(memref);
// // //           Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
// // //           rewriter.create<affine::AffineStoreOp>(
// // //               op->getLoc(), valueToStore, newMemref, 
// // //               ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // //         } else {
// // //           rewriter.clone(*op, globalMapping);
// // //         }
// // //       } else {
// // //         auto clonedOp = rewriter.clone(*op, globalMapping);
// // //         for (unsigned i = 0; i < op->getNumResults(); ++i) {
// // //           globalMapping.map(op->getResult(i), clonedOp->getResult(i));
// // //         }
// // //       }
// // //     }
// // //   }

// // //   // 收集5D memref
// // //   bool collectMemRefs5D(affine::AffineForOp innerMostOp,
// // //                        SmallVector<Value> &memrefsToReshape,
// // //                        SmallVector<std::pair<Value, MemRefType>> &reshapeInfo,
// // //                        int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    
// // //     SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
// // //     bool hasIncompatibleAccess = false;
    
// // //     innerMostOp.getRegion().walk([&](Operation *op) {
// // //       if (hasIncompatibleAccess) return;
      
// // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //         Value memref = loadOp.getMemRef();
// // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // //         if (memrefType && memrefType.getRank() == 5) {
// // //           if (!isMemrefAccessCompatibleWithLoopBounds5D(loadOp, dim0, dim1, dim2, dim3, dim4)) {
// // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in load, rejecting flatten\n");
// // //             hasIncompatibleAccess = true;
// // //             return;
// // //           }
          
// // //           if (isCompatibleShape5D(memrefType, dim0, dim1, dim2, dim3, dim4)) {
// // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // //                 == allAccessedMemrefs.end()) {
// // //               allAccessedMemrefs.push_back({memref, memrefType});
// // //             }
// // //           }
// // //         }
// // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //         Value memref = storeOp.getMemRef();
// // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // //         if (memrefType && memrefType.getRank() == 5) {
// // //           if (!isMemrefAccessCompatibleWithLoopBounds5D(storeOp, dim0, dim1, dim2, dim3, dim4)) {
// // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in store, rejecting flatten\n");
// // //             hasIncompatibleAccess = true;
// // //             return;
// // //           }
          
// // //           if (isCompatibleShape5D(memrefType, dim0, dim1, dim2, dim3, dim4)) {
// // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // //                 == allAccessedMemrefs.end()) {
// // //               allAccessedMemrefs.push_back({memref, memrefType});
// // //             }
// // //           }
// // //         }
// // //       }
// // //     });

// // //     if (hasIncompatibleAccess) {
// // //       return false;
// // //     }

// // //     for (auto [memref, memrefType] : allAccessedMemrefs) {
// // //       if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
// // //           == memrefsToReshape.end()) {
// // //         memrefsToReshape.push_back(memref);
// // //         reshapeInfo.push_back({memref, memrefType});
// // //       }
// // //     }

// // //     return !memrefsToReshape.empty();
// // //   }

// // //   bool isCompatibleShape5D(MemRefType memrefType, int64_t dim0, int64_t dim1, 
// // //                           int64_t dim2, int64_t dim3, int64_t dim4) const {
// // //     auto shape = memrefType.getShape();
// // //     return shape.size() == 5 && 
// // //            (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
// // //            (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
// // //            (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
// // //            (shape[3] == dim3 || shape[3] == ShapedType::kDynamic) &&
// // //            (shape[4] == dim4 || shape[4] == ShapedType::kDynamic);
// // //   }

// // //   bool isMemrefAccessCompatibleWithLoopBounds5D(Operation* op, int64_t dim0, int64_t dim1, 
// // //                                                 int64_t dim2, int64_t dim3, int64_t dim4) const {
// // //     Value memref;
// // //     AffineMap accessMap;
// // //     SmallVector<Value> indices;
    
// // //     if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //       memref = loadOp.getMemRef();
// // //       accessMap = loadOp.getAffineMap();
// // //       indices = loadOp.getMapOperands();
// // //     } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //       memref = storeOp.getMemRef();
// // //       accessMap = storeOp.getAffineMap();
// // //       indices = storeOp.getMapOperands();
// // //     } else {
// // //       return true;
// // //     }
    
// // //     auto memrefType = memref.getType().dyn_cast<MemRefType>();
// // //     if (!memrefType || memrefType.getRank() != 5) {
// // //       return true;
// // //     }
    
// // //     auto shape = memrefType.getShape();
    
// // //     bool strictMatch = (shape[0] == dim0 && shape[1] == dim1 && 
// // //                        shape[2] == dim2 && shape[3] == dim3 && shape[4] == dim4);
    
// // //     if (!strictMatch) {
// // //       LLVM_DEBUG(llvm::dbgs() << "Memref shape mismatch with loop bounds: "
// // //                  << "memref=" << shape[0] << "x" << shape[1] << "x" 
// // //                  << shape[2] << "x" << shape[3] << "x" << shape[4]
// // //                  << ", loops=" << dim0 << "x" << dim1 << "x" 
// // //                  << dim2 << "x" << dim3 << "x" << dim4 << "\n");
// // //       return false;
// // //     }
    
// // //     if (accessMap.getNumResults() != 5) {
// // //       LLVM_DEBUG(llvm::dbgs() << "Non-5D access pattern, rejecting\n");
// // //       return false;
// // //     }
    
// // //     for (unsigned i = 0; i < 5; ++i) {
// // //       auto expr = accessMap.getResult(i);
// // //       if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
// // //         if (dimExpr.getPosition() != i) {
// // //           LLVM_DEBUG(llvm::dbgs() << "Non-identity access pattern at dimension " 
// // //                      << i << ", rejecting\n");
// // //           return false;
// // //         }
// // //       } else {
// // //         LLVM_DEBUG(llvm::dbgs() << "Non-dimension expression in access pattern at dimension " 
// // //                    << i << ", rejecting\n");
// // //         return false;
// // //       }
// // //     }
    
// // //     return true;
// // //   }

// // //   // 创建5D重塑memref方法
// // //   Value createReshapedMemref5D(PatternRewriter &rewriter, Value originalMemref,
// // //                               MemRefType originalType, int64_t dim0, int64_t dim1,
// // //                               int64_t dim2, int64_t dim3, int64_t dim4, 
// // //                               int mergeIdx0, int mergeIdx1, int mergeIdx2,
// // //                               Location loc) const {
    
// // //     Type elementType = originalType.getElementType();
// // //     std::array<int64_t, 5> originalDims = {dim0, dim1, dim2, dim3, dim4};
    
// // //     // 计算合并后的维度
// // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1] * originalDims[mergeIdx2];
    
// // //     // 创建新的3D shape
// // //     SmallVector<int64_t> newShape;
// // //     for (int i = 0; i < 5; ++i) {
// // //       if (i == mergeIdx0) {
// // //         newShape.push_back(mergedSize);
// // //         i += 2; // 跳过mergeIdx1和mergeIdx2
// // //       } else {
// // //         newShape.push_back(originalDims[i]);
// // //       }
// // //     }
    
// // //     auto newMemrefType = MemRefType::get(newShape, elementType);

// // //     // 计算strides
// // //     SmallVector<int64_t> strides;
// // //     SmallVector<int64_t> offset = {0};
    
// // //     // 根据合并的维度位置计算strides
// // //     if (mergeIdx0 == 0) { // 合并前三个维度: (0,1,2)
// // //       strides = {dim3 * dim4, 1};
// // //     } else if (mergeIdx0 == 1) { // 合并中间三个维度: (1,2,3)
// // //       strides = {(dim1 * dim2 * dim3) * dim4, 1};
// // //     } else { // 合并后三个维度: (2,3,4)
// // //       strides = {dim1 * (dim2 * dim3 * dim4), 1};
// // //     }

// // //     return rewriter.create<memref::ReinterpretCastOp>(
// // //         loc, newMemrefType, originalMemref, 
// // //         /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
// // //         /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
// // //     ).getResult();
// // //   }

// // //   // ========== 原有4D处理方法 ==========

// // //   // 创单维度的for循环
// // //   affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
// // //                                            Location loc, 
// // //                                            int64_t lowerBound, 
// // //                                            int64_t upperBound) const {
// // //     MLIRContext *ctx = rewriter.getContext();
    
// // //     AffineMap lowerBoundMap = AffineMap::getConstantMap(lowerBound, ctx);
// // //     AffineMap upperBoundMap = AffineMap::getConstantMap(upperBound, ctx);
    
// // //     auto forOp = rewriter.create<affine::AffineForOp>(
// // //         loc, ValueRange{}, lowerBoundMap, ValueRange{}, upperBoundMap,
// // //         /*step=*/1, ValueRange{},
// // //         /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {});
    
// // //     return forOp;
// // //   }

// // //   // 检查是否为完美的4层嵌套循环结构，并找到最小的相邻维度对
// // //   std::optional<NestedForInfo> analyzeNestedForStructure(
// // //       affine::AffineForOp outerOp) const {
    
// // //     // 获取所有4层的维度大小
// // //     auto dim0 = getLoopBound(outerOp);
// // //     if (!dim0.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(outerOp)) return std::nullopt;

// // //     auto secondFor = findNestedFor(outerOp);
// // //     if (!secondFor) return std::nullopt;
// // //     auto dim1 = getLoopBound(*secondFor);
// // //     if (!dim1.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(*secondFor)) return std::nullopt;

// // //     auto thirdFor = findNestedFor(*secondFor);
// // //     if (!thirdFor) return std::nullopt;
// // //     auto dim2 = getLoopBound(*thirdFor);
// // //     if (!dim2.has_value()) return std::nullopt;
// // //     if (!isPerfectlyNestedLevel(*thirdFor)) return std::nullopt;

// // //     auto fourthFor = findNestedFor(*thirdFor);
// // //     if (!fourthFor) return std::nullopt;
// // //     auto dim3 = getLoopBound(*fourthFor);
// // //     if (!dim3.has_value()) return std::nullopt;
// // //     if (!isSimpleLoopBody(*fourthFor)) return std::nullopt;

// // //     // 找到最小的相邻维度对
// // //     std::array<int64_t, 4> dims = {*dim0, *dim1, *dim2, *dim3};
// // //     int bestIdx0 = 0, bestIdx1 = 1;
// // //     int64_t minProduct = dims[0] * dims[1];
    
// // //     for (int i = 1; i < 3; ++i) {
// // //       int64_t product = dims[i] * dims[i + 1];
// // //       if (product < minProduct) {
// // //         minProduct = product;
// // //         bestIdx0 = i;
// // //         bestIdx1 = i + 1;
// // //       }
// // //     }

// // //     LLVM_DEBUG(llvm::dbgs() << "Validated perfect 4-layer nesting: " 
// // //                << *dim0 << "x" << *dim1 << "x" << *dim2 << "x" << *dim3 
// // //                << ", choosing to merge dimensions " << bestIdx0 << " and " << bestIdx1 
// // //                << " (product: " << minProduct << ")\n");

// // //     return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *fourthFor, bestIdx0, bestIdx1};
// // //   }

// // //   // 计算4D合并后的新维度配置
// // //   std::pair<std::array<int64_t, 3>, int> calculateNewDimensions4D(
// // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
// // //       int mergeIdx0, int mergeIdx1) const {
    
// // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
// // //     std::array<int64_t, 3> newDims;
    
// // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1];
    
// // //     // 创建新的3D维度数组
// // //     int newIdx = 0;
// // //     int mergedDimIdx = -1;
    
// // //     for (int i = 0; i < 4; ++i) {
// // //       if (i == mergeIdx0) {
// // //         newDims[newIdx] = mergedSize;
// // //         mergedDimIdx = newIdx;
// // //         newIdx++;
// // //         ++i; // 跳过mergeIdx1
// // //       } else {
// // //         newDims[newIdx] = originalDims[i];
// // //         newIdx++;
// // //       }
// // //     }
    
// // //     return {newDims, mergedDimIdx};
// // //   }

// // //   // 创建三层嵌套循环
// // //   std::tuple<affine::AffineForOp, affine::AffineForOp, affine::AffineForOp> 
// // //   createThreeLayerLoops(PatternRewriter &rewriter, Location loc, 
// // //                        const std::array<int64_t, 3>& newDims) const {
    
// // //     auto outerFor = createSingleDimForOp(rewriter, loc, 0, newDims[0]);
// // //     Block *outerBlock = &outerFor.getRegion().front();
    
// // //     rewriter.setInsertionPointToStart(outerBlock);
// // //     auto middleFor = createSingleDimForOp(rewriter, loc, 0, newDims[1]);
// // //     Block *middleBlock = &middleFor.getRegion().front();
    
// // //     rewriter.setInsertionPointToStart(middleBlock);
// // //     auto innerFor = createSingleDimForOp(rewriter, loc, 0, newDims[2]);
    
// // //     return {outerFor, middleFor, innerFor};
// // //   }

// // //   // 获取三层循环的索引
// // //   std::array<BlockArgument, 3> getLoopIndices(
// // //       affine::AffineForOp for0, affine::AffineForOp for1, affine::AffineForOp for2) const {
// // //     return {
// // //       for0.getRegion().front().getArgument(0),
// // //       for1.getRegion().front().getArgument(0), 
// // //       for2.getRegion().front().getArgument(0)
// // //     };
// // //   }

// // //   // 计算4D原始索引
// // //   std::array<Value, 4> calculateOriginalIndices4D(
// // //       PatternRewriter &rewriter, Location loc,
// // //       const std::array<BlockArgument, 3>& loopIndices,
// // //       int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
// // //       int mergeIdx0, int mergeIdx1) const {
    
// // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
// // //     std::array<Value, 4> originalIndices;
    
// // //     // 首先将新索引映射回原始索引位置
// // //     int newIdx = 0;
// // //     for (int i = 0; i < 4; ++i) {
// // //       if (i == mergeIdx0) {
// // //         // 对于合并的维度，需要计算分解
// // //         Value mergedIdx = loopIndices[newIdx];
// // //         Value dim1Size = rewriter.create<arith::ConstantIndexOp>(loc, originalDims[mergeIdx1]);
// // //         originalIndices[i] = rewriter.create<arith::DivSIOp>(loc, mergedIdx, dim1Size);
// // //         originalIndices[i + 1] = rewriter.create<arith::RemSIOp>(loc, mergedIdx, dim1Size);
// // //         ++i; // 跳过下一个索引
// // //         ++newIdx;
// // //       } else {
// // //         originalIndices[i] = loopIndices[newIdx];
// // //         ++newIdx;
// // //       }
// // //     }
    
// // //     return originalIndices;
// // //   }

// // //   // 建立4D索引映射
// // //   IRMapping buildIndexMapping4D(affine::AffineForOp outerForOp, 
// // //                                const std::array<Value, 4>& originalIndices) const {
// // //     IRMapping globalMapping;
    
// // //     // 映射所有4层循环的索引变量
// // //     globalMapping.map(outerForOp.getInductionVar(), originalIndices[0]);
    
// // //     auto level2Op = findNestedFor(outerForOp);
// // //     if (level2Op) {
// // //       globalMapping.map(level2Op->getInductionVar(), originalIndices[1]);
// // //       auto level3Op = findNestedFor(*level2Op);
// // //       if (level3Op) {
// // //         globalMapping.map(level3Op->getInductionVar(), originalIndices[2]);
// // //         auto level4Op = findNestedFor(*level3Op);
// // //         if (level4Op) {
// // //           globalMapping.map(level4Op->getInductionVar(), originalIndices[3]);
// // //         }
// // //       }
// // //     }
    
// // //     return globalMapping;
// // //   }

// // //   // 克隆4D循环体操作
// // //   void cloneLoopBody4D(PatternRewriter &rewriter, const SmallVector<Operation*>& opsToClone,
// // //                       IRMapping& globalMapping, const DenseMap<Value, Value>& memrefMap,
// // //                       const std::array<BlockArgument, 3>& loopIndices,
// // //                       int mergeIdx0, int mergeIdx1) const {
    
// // //     for (Operation *op : opsToClone) {
// // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //         Value memref = loadOp.getMemRef();
// // //         if (memrefMap.count(memref)) {
// // //           Value newMemref = memrefMap.lookup(memref);
// // //           auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
// // //               op->getLoc(), newMemref, ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // //           globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
// // //         } else {
// // //           auto clonedOp = rewriter.clone(*op, globalMapping);
// // //           globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
// // //         }
// // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //         Value memref = storeOp.getMemRef();
// // //         if (memrefMap.count(memref)) {
// // //           Value newMemref = memrefMap.lookup(memref);
// // //           Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
// // //           rewriter.create<affine::AffineStoreOp>(
// // //               op->getLoc(), valueToStore, newMemref, 
// // //               ValueRange{loopIndices[0], loopIndices[1], loopIndices[2]});
// // //         } else {
// // //           rewriter.clone(*op, globalMapping);
// // //         }
// // //       } else {
// // //         auto clonedOp = rewriter.clone(*op, globalMapping);
// // //         for (unsigned i = 0; i < op->getNumResults(); ++i) {
// // //           globalMapping.map(op->getResult(i), clonedOp->getResult(i));
// // //         }
// // //       }
// // //     }
// // //   }

// // //   // 添加yield操作
// // //   void addYieldOperations(PatternRewriter &rewriter,
// // //                          affine::AffineForOp for0, affine::AffineForOp for1, 
// // //                          affine::AffineForOp for2) const {
// // //     Location loc = for0.getLoc();
    
// // //     rewriter.create<affine::AffineYieldOp>(loc);
// // //     rewriter.setInsertionPointAfter(for2);
// // //     rewriter.create<affine::AffineYieldOp>(loc);
// // //     rewriter.setInsertionPointAfter(for1);
// // //     rewriter.create<affine::AffineYieldOp>(loc);
// // //   }

// // //   // 检查一个循环层是否为完美嵌套
// // //   bool isPerfectlyNestedLevel(affine::AffineForOp forOp) const {
// // //     Block &body = forOp.getRegion().front();
    
// // //     int nestedForCount = 0;
// // //     int yieldCount = 0;
// // //     int totalOps = 0;
    
// // //     for (auto &op : body.getOperations()) {
// // //       totalOps++;
// // //       if (isa<affine::AffineForOp>(&op)) {
// // //         nestedForCount++;
// // //       } else if (isa<affine::AffineYieldOp>(&op)) {
// // //         yieldCount++;
// // //       } else {
// // //         LLVM_DEBUG(llvm::dbgs() << "Found non-nested operation in loop body: " 
// // //                    << op.getName() << ", rejecting perfect nesting\n");
// // //         return false;
// // //       }
// // //     }
    
// // //     bool isPerfect = (nestedForCount == 1 && yieldCount == 1 && totalOps == 2);
    
// // //     if (!isPerfect) {
// // //       LLVM_DEBUG(llvm::dbgs() << "Level validation failed: nestedForCount=" 
// // //                  << nestedForCount << ", yieldCount=" << yieldCount 
// // //                  << ", totalOps=" << totalOps << "\n");
// // //     }
    
// // //     return isPerfect;
// // //   }

// // //   // 检查最内层循环体是否只包含简单操作
// // //   bool isSimpleLoopBody(affine::AffineForOp forOp) const {
// // //     Block &body = forOp.getRegion().front();
    
// // //     for (auto &op : body.getOperations()) {
// // //       if (isa<affine::AffineForOp>(&op)) {
// // //         LLVM_DEBUG(llvm::dbgs() << "Found nested for loop in innermost body, rejecting\n");
// // //         return false;
// // //       } else if (isa<affine::AffineLoadOp, affine::AffineStoreOp, 
// // //                      arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
// // //                      arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
// // //                      arith::ConstantOp, affine::AffineYieldOp>(&op)) {
// // //         continue;
// // //       } else {
// // //         LLVM_DEBUG(llvm::dbgs() << "Found operation in innermost body: " 
// // //                    << op.getName() << " (allowing)\n");
// // //       }
// // //     }
    
// // //     return true;
// // //   }

// // //   std::optional<affine::AffineForOp> findNestedFor(
// // //       affine::AffineForOp parentOp) const {
// // //     for (auto &op : parentOp.getRegion().front()) {
// // //       if (auto nestedFor = dyn_cast<affine::AffineForOp>(&op))
// // //         return nestedFor;
// // //     }
// // //     return std::nullopt;
// // //   }

// // //   std::optional<int64_t> getLoopBound(affine::AffineForOp forOp) const {
// // //     auto lowerBound = forOp.getLowerBound();
// // //     auto lowerMap = lowerBound.getMap();
// // //     if (lowerMap.getNumResults() != 1)
// // //       return std::nullopt;
    
// // //     auto lowerExpr = lowerMap.getResult(0);
// // //     if (auto lowerConst = lowerExpr.dyn_cast<AffineConstantExpr>()) {
// // //       if (lowerConst.getValue() != 0)
// // //         return std::nullopt;
// // //     } else {
// // //       return std::nullopt;
// // //     }

// // //     auto upperBound = forOp.getUpperBound();
// // //     auto upperMap = upperBound.getMap();
// // //     if (upperMap.getNumResults() != 1)
// // //       return std::nullopt;
      
// // //     auto upperExpr = upperMap.getResult(0);
// // //     if (auto upperConst = upperExpr.dyn_cast<AffineConstantExpr>()) {
// // //       return upperConst.getValue();
// // //     }
    
// // //     return std::nullopt;
// // //   }

// // //   bool collectMemRefs4D(affine::AffineForOp innerMostOp,
// // //                        SmallVector<Value> &memrefsToReshape,
// // //                        SmallVector<std::pair<Value, MemRefType>> &reshapeInfo,
// // //                        int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    
// // //     SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
// // //     bool hasIncompatibleAccess = false;
    
// // //     innerMostOp.getRegion().walk([&](Operation *op) {
// // //       if (hasIncompatibleAccess) return;
      
// // //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //         Value memref = loadOp.getMemRef();
// // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // //         if (memrefType && memrefType.getRank() == 4) {
// // //           if (!isMemrefAccessCompatibleWithLoopBounds4D(loadOp, dim0, dim1, dim2, dim3)) {
// // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in load, rejecting flatten\n");
// // //             hasIncompatibleAccess = true;
// // //             return;
// // //           }
          
// // //           if (isCompatibleShape4D(memrefType, dim0, dim1, dim2, dim3)) {
// // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // //                 == allAccessedMemrefs.end()) {
// // //               allAccessedMemrefs.push_back({memref, memrefType});
// // //             }
// // //           }
// // //         }
// // //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //         Value memref = storeOp.getMemRef();
// // //         auto memrefType = memref.getType().dyn_cast<MemRefType>();
        
// // //         if (memrefType && memrefType.getRank() == 4) {
// // //           if (!isMemrefAccessCompatibleWithLoopBounds4D(storeOp, dim0, dim1, dim2, dim3)) {
// // //             LLVM_DEBUG(llvm::dbgs() << "Found memref with incompatible access pattern in store, rejecting flatten\n");
// // //             hasIncompatibleAccess = true;
// // //             return;
// // //           }
          
// // //           if (isCompatibleShape4D(memrefType, dim0, dim1, dim2, dim3)) {
// // //             if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
// // //                             [memref](const auto& pair) { return pair.first == memref; }) 
// // //                 == allAccessedMemrefs.end()) {
// // //               allAccessedMemrefs.push_back({memref, memrefType});
// // //             }
// // //           }
// // //         }
// // //       }
// // //     });

// // //     if (hasIncompatibleAccess) {
// // //       return false;
// // //     }

// // //     for (auto [memref, memrefType] : allAccessedMemrefs) {
// // //       if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
// // //           == memrefsToReshape.end()) {
// // //         memrefsToReshape.push_back(memref);
// // //         reshapeInfo.push_back({memref, memrefType});
// // //       }
// // //     }

// // //     return !memrefsToReshape.empty();
// // //   }

// // //   bool isCompatibleShape4D(MemRefType memrefType, int64_t dim0, int64_t dim1, 
// // //                           int64_t dim2, int64_t dim3) const {
// // //     auto shape = memrefType.getShape();
// // //     return shape.size() == 4 && 
// // //            (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
// // //            (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
// // //            (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
// // //            (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
// // //   }

// // //   bool isMemrefAccessCompatibleWithLoopBounds4D(Operation* op, int64_t dim0, int64_t dim1, 
// // //                                                 int64_t dim2, int64_t dim3) const {
// // //     Value memref;
// // //     AffineMap accessMap;
// // //     SmallVector<Value> indices;
    
// // //     if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// // //       memref = loadOp.getMemRef();
// // //       accessMap = loadOp.getAffineMap();
// // //       indices = loadOp.getMapOperands();
// // //     } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// // //       memref = storeOp.getMemRef();
// // //       accessMap = storeOp.getAffineMap();
// // //       indices = storeOp.getMapOperands();
// // //     } else {
// // //       return true;
// // //     }
    
// // //     auto memrefType = memref.getType().dyn_cast<MemRefType>();
// // //     if (!memrefType || memrefType.getRank() != 4) {
// // //       return true;
// // //     }
    
// // //     auto shape = memrefType.getShape();
    
// // //     bool strictMatch = (shape[0] == dim0 && shape[1] == dim1 && 
// // //                        shape[2] == dim2 && shape[3] == dim3);
    
// // //     if (!strictMatch) {
// // //       LLVM_DEBUG(llvm::dbgs() << "Memref shape mismatch with loop bounds: "
// // //                  << "memref=" << shape[0] << "x" << shape[1] << "x" 
// // //                  << shape[2] << "x" << shape[3] 
// // //                  << ", loops=" << dim0 << "x" << dim1 << "x" 
// // //                  << dim2 << "x" << dim3 << "\n");
// // //       return false;
// // //     }
    
// // //     if (accessMap.getNumResults() != 4) {
// // //       LLVM_DEBUG(llvm::dbgs() << "Non-4D access pattern, rejecting\n");
// // //       return false;
// // //     }
    
// // //     for (unsigned i = 0; i < 4; ++i) {
// // //       auto expr = accessMap.getResult(i);
// // //       if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
// // //         if (dimExpr.getPosition() != i) {
// // //           LLVM_DEBUG(llvm::dbgs() << "Non-identity access pattern at dimension " 
// // //                      << i << ", rejecting\n");
// // //           return false;
// // //         }
// // //       } else {
// // //         LLVM_DEBUG(llvm::dbgs() << "Non-dimension expression in access pattern at dimension " 
// // //                    << i << ", rejecting\n");
// // //         return false;
// // //       }
// // //     }
    
// // //     return true;
// // //   }

// // //   // 修改后的创建重塑memref方法，支持任意相邻维度合并
// // //   Value createReshapedMemref4D(PatternRewriter &rewriter, Value originalMemref,
// // //                               MemRefType originalType, int64_t dim0, int64_t dim1,
// // //                               int64_t dim2, int64_t dim3, int mergeIdx0, int mergeIdx1,
// // //                               Location loc) const {
    
// // //     Type elementType = originalType.getElementType();
// // //     std::array<int64_t, 4> originalDims = {dim0, dim1, dim2, dim3};
    
// // //     // 计算合并后的维度
// // //     int64_t mergedSize = originalDims[mergeIdx0] * originalDims[mergeIdx1];
    
// // //     // 创建新的3D shape
// // //     SmallVector<int64_t> newShape;
// // //     for (int i = 0; i < 4; ++i) {
// // //       if (i == mergeIdx0) {
// // //         newShape.push_back(mergedSize);
// // //         ++i; // 跳过mergeIdx1
// // //       } else {
// // //         newShape.push_back(originalDims[i]);
// // //       }
// // //     }
    
// // //     auto newMemrefType = MemRefType::get(newShape, elementType);

// // //     // 计算strides
// // //     SmallVector<int64_t> strides;
// // //     SmallVector<int64_t> offset = {0};
    
// // //     // 根据合并的维度位置计算strides
// // //     if (mergeIdx0 == 0) { // 合并前两个维度
// // //       strides = {dim2 * dim3, dim3, 1};
// // //     } else if (mergeIdx0 == 1) { // 合并中间两个维度  
// // //       strides = {(dim1 * dim2) * dim3, dim3, 1};
// // //     } else { // 合并后两个维度
// // //       strides = {dim1 * (dim2 * dim3), (dim2 * dim3), 1};
// // //     }

// // //     return rewriter.create<memref::ReinterpretCastOp>(
// // //         loc, newMemrefType, originalMemref, 
// // //         /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
// // //         /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
// // //     ).getResult();
// // //   }
// // // };

// // // struct ForLoopFlattenPass
// // //     : public PassWrapper<ForLoopFlattenPass, OperationPass<ModuleOp>> {
  
// // //   StringRef getArgument() const final { return "for-loop-flatten"; }
// // //   StringRef getDescription() const final {
// // //     return "Flatten 4D or 5D nested for loops to 3D by merging adjacent dimensions with smallest product";
// // //   }
  
// // //   void getDependentDialects(DialectRegistry &registry) const override {
// // //     registry.insert<affine::AffineDialect, memref::MemRefDialect, arith::ArithDialect>();
// // //   }
  
// // //   void runOnOperation() override {
// // //     ModuleOp moduleOp = getOperation();
// // //     MLIRContext *context = &getContext();
    
// // //     LLVM_DEBUG(llvm::dbgs() << "Running ForLoopFlattenPass\n");
    
// // //     RewritePatternSet patterns(context);
// // //     patterns.add<ForLoopFlattenPattern>(context);
    
// // //     if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
// // //       signalPassFailure();
// // //     }
    
// // //     LLVM_DEBUG(llvm::dbgs() << "Completed ForLoopFlattenPass\n");
// // //   }
// // // };

// // // } // end anonymous namespace

// // // namespace onnx_mlir {
// // //     std::unique_ptr<Pass> createForLoopFlattenPass() {
// // //       return std::make_unique<ForLoopFlattenPass>();
// // //     }
// // // } // namespace onnx_mlir

// // // static mlir::PassRegistration<ForLoopFlattenPass> pass;


// // // 取消shape检查，支持loop维度与memref维度不严格匹配的情况
// // #include "mlir/Pass/Pass.h"
// // #include "mlir/IR/PatternMatch.h"
// // #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// // #include "llvm/Support/Debug.h"
// // #include "mlir/Dialect/Affine/IR/AffineOps.h"
// // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // #include "mlir/Dialect/Arith/IR/Arith.h"
// // #include "mlir/IR/AffineExpr.h"
// // #include "mlir/IR/AffineMap.h"
// // #include "mlir/IR/Builders.h"
// // #include "mlir/IR/IRMapping.h"

// // using namespace mlir;

// // #define DEBUG_TYPE "for-loop-flatten"

// // namespace {

// // // 存储 memref 访问信息的结构
// // struct MemrefAccessInfo {
// //   Value memref;
// //   MemRefType memrefType;
// //   // 记录第一维和第二维可能的最大索引值
// //   int64_t maxDim0Index;  
// //   int64_t maxDim1Index;
// // };

// // // 增强的循环 Flatten Pattern
// // // 
// // // 关键设计决策：使用 memref.load/store 而不是 affine.load/store
// // // 
// // // 原因：
// // // - affine.load/store 要求索引必须是 affine dimensions 或 symbols
// // // - affine dimension 必须是循环归纳变量或 affine.apply 的**直接**结果
// // // - 在我们的 flatten 转换中，需要先用 arith.divsi/remsi 恢复原始索引
// // // - 然后用 affine.apply 计算偏移后的索引
// // // - 这个间接计算链会"打断" affine dimension 的属性
// // // - 导致 affine.load 无法接受这样的索引
// // // 
// // // 解决方案：
// // // - 对于需要 reshape 的 memref，使用 memref.load/store
// // // - memref 操作接受任何 index 类型的值，没有 affine 的限制
// // // - 仍然保留 affine 循环结构，只是内部的 load/store 使用 memref 版本
// // // - 这样既支持了复杂的索引计算，又保持了循环优化的可能性
// // //
// // class ForLoopFlattenPattern : public OpRewritePattern<affine::AffineForOp> {
// // public:
// //   using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

// //   LogicalResult matchAndRewrite(affine::AffineForOp outerForOp,
// //                                PatternRewriter &rewriter) const override {
// //     // 检查是否是4层完美嵌套的affine.for
// //     auto nestedStructure = analyzeNestedForStructure(outerForOp);
// //     if (!nestedStructure.has_value())
// //       return failure();

// //     auto [dim0, dim1, dim2, dim3, innerMostOp] = *nestedStructure;
    
// //     LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten: " 
// //                << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
// //                << " -> " << (dim0 * dim1) << "x" << dim2 << "x" << dim3 << "\n");

// //     // 收集需要处理的memref及其访问信息
// //     SmallVector<MemrefAccessInfo> memrefAccessInfos;
    
// //     if (!analyzeMemrefAccesses(innerMostOp, memrefAccessInfos, dim0, dim1, dim2, dim3))
// //       return failure();

// //     if (memrefAccessInfos.empty()) {
// //       LLVM_DEBUG(llvm::dbgs() << "No compatible memrefs found\n");
// //       return failure();
// //     }

// //     Location loc = outerForOp.getLoc();
    
// //     // 为每个memref创建reshape后的版本
// //     rewriter.setInsertionPoint(outerForOp);
// //     DenseMap<Value, Value> reshapedMemrefMap;
// //     DenseMap<Value, MemrefAccessInfo> accessInfoMap;
    
// //     for (auto &info : memrefAccessInfos) {
// //       Value reshaped = createReshapedMemrefForAccess(rewriter, info, loc);
// //       if (!reshaped)
// //         return failure();
// //       reshapedMemrefMap[info.memref] = reshaped;
// //       accessInfoMap[info.memref] = info;
// //     }

// //     // 创建新的三层嵌套循环
// //     int64_t flattenedSize = dim0 * dim1;
    
// //     // 第一层：flattened dimension (0 to flattenedSize)
// //     auto newOuterFor = createSingleDimForOp(rewriter, loc, 0, flattenedSize);
// //     Block *outerBlock = &newOuterFor.getRegion().front();
// //     BlockArgument flattenedIdx = outerBlock->getArgument(0);
    
// //     // 第二层：dim2 (0 to dim2)
// //     rewriter.setInsertionPointToStart(outerBlock);
// //     auto newMiddleFor = createSingleDimForOp(rewriter, loc, 0, dim2);
// //     Block *middleBlock = &newMiddleFor.getRegion().front();
// //     BlockArgument idx2 = middleBlock->getArgument(0);
    
// //     // 第三层：dim3 (0 to dim3)
// //     rewriter.setInsertionPointToStart(middleBlock);
// //     auto newInnerFor = createSingleDimForOp(rewriter, loc, 0, dim3);
// //     Block *innerBlock = &newInnerFor.getRegion().front();
// //     BlockArgument idx3 = innerBlock->getArgument(0);

// //     // 在最内层循环体中处理
// //     rewriter.setInsertionPointToStart(innerBlock);
    
// //     // 计算原始的两个循环索引
// //     Value dim1Value = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
// //     Value originalIdx0 = rewriter.create<arith::DivSIOp>(loc, flattenedIdx, dim1Value);
// //     Value originalIdx1 = rewriter.create<arith::RemSIOp>(loc, flattenedIdx, dim1Value);
    
// //     // 收集原始循环体的所有操作
// //     Block *originalInnerBody = &innerMostOp.getRegion().front();
// //     SmallVector<Operation*> opsToClone;
// //     for (auto &op : originalInnerBody->getOperations()) {
// //       if (!isa<affine::AffineYieldOp>(op)) {
// //         opsToClone.push_back(&op);
// //       }
// //     }
    
// //     // 建立基本的索引映射
// //     IRMapping globalMapping;
    
// //     // 映射四层循环的归纳变量
// //     globalMapping.map(outerForOp.getInductionVar(), originalIdx0);
    
// //     auto level2Op = findNestedFor(outerForOp);
// //     if (level2Op) {
// //       globalMapping.map(level2Op->getInductionVar(), originalIdx1);
// //       auto level3Op = findNestedFor(*level2Op);
// //       if (level3Op) {
// //         globalMapping.map(level3Op->getInductionVar(), idx2);
// //         auto level4Op = findNestedFor(*level3Op);
// //         if (level4Op) {
// //           globalMapping.map(level4Op->getInductionVar(), idx3);
// //         }
// //       }
// //     }

// //     // 克隆循环体操作，特殊处理 load/store
// //     for (Operation *op : opsToClone) {
// //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// //         Value newLoadResult = cloneLoadOp(loadOp, rewriter, globalMapping, 
// //                                          reshapedMemrefMap, accessInfoMap,
// //                                          flattenedIdx, idx2, idx3,
// //                                          originalIdx0, originalIdx1, dim1);
// //         if (newLoadResult) {
// //           globalMapping.map(loadOp.getResult(), newLoadResult);
// //         }
// //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// //         cloneStoreOp(storeOp, rewriter, globalMapping, 
// //                     reshapedMemrefMap, accessInfoMap,
// //                     flattenedIdx, idx2, idx3,
// //                     originalIdx0, originalIdx1, dim1);
// //       } else {
// //         // 其他操作正常克隆
// //         auto clonedOp = rewriter.clone(*op, globalMapping);
// //         for (unsigned i = 0; i < op->getNumResults(); ++i) {
// //           globalMapping.map(op->getResult(i), clonedOp->getResult(i));
// //         }
// //       }
// //     }

// //     // 添加yield操作
// //     rewriter.create<affine::AffineYieldOp>(loc);
// //     rewriter.setInsertionPointAfter(newInnerFor);
// //     rewriter.create<affine::AffineYieldOp>(loc);
// //     rewriter.setInsertionPointAfter(newMiddleFor);
// //     rewriter.create<affine::AffineYieldOp>(loc);

// //     // 删除原循环
// //     rewriter.eraseOp(outerForOp);

// //     return success();
// //   }

// // private:
// //   struct NestedForInfo {
// //     int64_t dim0, dim1, dim2, dim3;
// //     affine::AffineForOp innerMostOp;
// //   };

// //   // 分析memref的访问模式
// //   bool analyzeMemrefAccesses(affine::AffineForOp innerMostOp,
// //                             SmallVector<MemrefAccessInfo> &accessInfos,
// //                             int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    
// //     DenseMap<Value, MemrefAccessInfo> memrefInfoMap;
    
// //     auto collectAccessInfo = [&](Value memref, AffineMap accessMap, 
// //                                  SmallVector<Value> indices) -> bool {
// //       auto memrefType = memref.getType().dyn_cast<MemRefType>();
// //       if (!memrefType || memrefType.getRank() != 4)
// //         return true;
      
// //       // 检查后两维是否匹配
// //       auto shape = memrefType.getShape();
// //       if (shape[2] != dim2 || shape[3] != dim3) {
// //         LLVM_DEBUG(llvm::dbgs() << "Memref dimensions 2-3 don't match loop bounds\n");
// //         return false;
// //       }
      
// //       // 分析前两维的访问范围
// //       // 我们需要确保所有访问都在memref的范围内
// //       if (accessMap.getNumResults() != 4) {
// //         LLVM_DEBUG(llvm::dbgs() << "Non-4D access map\n");
// //         return false;
// //       }
      
// //       // 获取或创建此memref的访问信息
// //       if (memrefInfoMap.find(memref) == memrefInfoMap.end()) {
// //         MemrefAccessInfo info;
// //         info.memref = memref;
// //         info.memrefType = memrefType;
// //         info.maxDim0Index = 0;
// //         info.maxDim1Index = 0;
// //         memrefInfoMap[memref] = info;
// //       }
      
// //       auto &info = memrefInfoMap[memref];
      
// //       // 分析第一维的表达式以估算最大索引
// //       // 对于 %arg16 + c 这样的表达式，最大值是 (dim0-1) + c
// //       auto dim0Expr = accessMap.getResult(0);
// //       int64_t maxOffset0 = estimateMaxValue(dim0Expr, dim0 - 1);
// //       info.maxDim0Index = std::max(info.maxDim0Index, maxOffset0);
      
// //       // 分析第二维
// //       auto dim1Expr = accessMap.getResult(1);
// //       int64_t maxOffset1 = estimateMaxValue(dim1Expr, dim1 - 1);
// //       info.maxDim1Index = std::max(info.maxDim1Index, maxOffset1);
      
// //       // 检查访问是否越界
// //       if (info.maxDim0Index >= shape[0] || info.maxDim1Index >= shape[1]) {
// //         LLVM_DEBUG(llvm::dbgs() << "Access out of bounds: max indices (" 
// //                    << info.maxDim0Index << ", " << info.maxDim1Index 
// //                    << ") exceed shape (" << shape[0] << ", " << shape[1] << ")\n");
// //         return false;
// //       }
      
// //       return true;
// //     };
    
// //     // 遍历所有load/store操作
// //     bool success = true;
// //     innerMostOp.getRegion().walk([&](Operation *op) {
// //       if (!success) return;
      
// //       if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
// //         if (!collectAccessInfo(loadOp.getMemRef(), 
// //                               loadOp.getAffineMap(), 
// //                               loadOp.getMapOperands())) {
// //           success = false;
// //         }
// //       } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
// //         if (!collectAccessInfo(storeOp.getMemRef(), 
// //                               storeOp.getAffineMap(), 
// //                               storeOp.getMapOperands())) {
// //           success = false;
// //         }
// //       }
// //     });
    
// //     if (!success)
// //       return false;
    
// //     // 将收集的信息转移到输出向量
// //     for (auto &pair : memrefInfoMap) {
// //       accessInfos.push_back(pair.second);
// //       LLVM_DEBUG(llvm::dbgs() << "Memref access info: shape=" 
// //                  << pair.second.memrefType.getShape()[0] << "x"
// //                  << pair.second.memrefType.getShape()[1] << "x"
// //                  << pair.second.memrefType.getShape()[2] << "x"
// //                  << pair.second.memrefType.getShape()[3]
// //                  << ", max indices=(" << pair.second.maxDim0Index 
// //                  << ", " << pair.second.maxDim1Index << ")\n");
// //     }
    
// //     return !accessInfos.empty();
// //   }

// //   // 估算affine表达式的最大值
// //   int64_t estimateMaxValue(AffineExpr expr, int64_t maxLoopValue) const {
// //     if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
// //       return constExpr.getValue();
// //     } else if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
// //       return maxLoopValue;
// //     } else if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
// //       int64_t lhs = estimateMaxValue(binaryExpr.getLHS(), maxLoopValue);
// //       int64_t rhs = estimateMaxValue(binaryExpr.getRHS(), maxLoopValue);
      
// //       switch (binaryExpr.getKind()) {
// //         case AffineExprKind::Add:
// //           return lhs + rhs;
// //         case AffineExprKind::Mul:
// //           return lhs * rhs;
// //         case AffineExprKind::Mod:
// //           return rhs - 1;  // a % b 的最大值是 b-1
// //         case AffineExprKind::FloorDiv:
// //           return lhs / rhs;
// //         case AffineExprKind::CeilDiv:
// //           return (lhs + rhs - 1) / rhs;
// //         default:
// //           return maxLoopValue;
// //       }
// //     }
// //     return maxLoopValue;
// //   }

// //   // 克隆load操作，处理索引转换
// //   Value cloneLoadOp(affine::AffineLoadOp loadOp, PatternRewriter &rewriter,
// //                    IRMapping &mapping,
// //                    const DenseMap<Value, Value> &reshapedMemrefMap,
// //                    const DenseMap<Value, MemrefAccessInfo> &accessInfoMap,
// //                    Value flattenedIdx, Value idx2, Value idx3,
// //                    Value originalIdx0, Value originalIdx1, int64_t dim1) const {
    
// //     Value memref = loadOp.getMemRef();
    
// //     // 如果这个memref需要reshape
// //     if (reshapedMemrefMap.count(memref)) {
// //       Value newMemref = reshapedMemrefMap.lookup(memref);
// //       const auto &accessInfo = accessInfoMap.lookup(memref);
// //       int64_t memrefDim1 = accessInfo.memrefType.getShape()[1];
      
// //       // 计算flattened索引
// //       Value newIdx0 = computeFlattenedIndex(rewriter, loadOp.getLoc(),
// //                                            loadOp.getAffineMap(),
// //                                            loadOp.getMapOperands(),
// //                                            originalIdx0, originalIdx1,
// //                                            memrefDim1, mapping);
      
// //       // 由于索引是动态计算的，使用 memref.load 而不是 affine.load
// //       auto newLoadOp = rewriter.create<memref::LoadOp>(
// //           loadOp.getLoc(), newMemref, ValueRange{newIdx0, idx2, idx3});
      
// //       return newLoadOp.getResult();
// //     } else {
// //       // 正常克隆
// //       auto clonedOp = rewriter.clone(*loadOp.getOperation(), mapping);
// //       return clonedOp->getResult(0);
// //     }
// //   }

// //   // 克隆store操作，处理索引转换
// //   void cloneStoreOp(affine::AffineStoreOp storeOp, PatternRewriter &rewriter,
// //                    IRMapping &mapping,
// //                    const DenseMap<Value, Value> &reshapedMemrefMap,
// //                    const DenseMap<Value, MemrefAccessInfo> &accessInfoMap,
// //                    Value flattenedIdx, Value idx2, Value idx3,
// //                    Value originalIdx0, Value originalIdx1, int64_t dim1) const {
    
// //     Value memref = storeOp.getMemRef();
    
// //     // 如果这个memref需要reshape
// //     if (reshapedMemrefMap.count(memref)) {
// //       Value newMemref = reshapedMemrefMap.lookup(memref);
// //       const auto &accessInfo = accessInfoMap.lookup(memref);
// //       int64_t memrefDim1 = accessInfo.memrefType.getShape()[1];
      
// //       // 计算flattened索引
// //       Value newIdx0 = computeFlattenedIndex(rewriter, storeOp.getLoc(),
// //                                            storeOp.getAffineMap(),
// //                                            storeOp.getMapOperands(),
// //                                            originalIdx0, originalIdx1,
// //                                            memrefDim1, mapping);
      
// //       // 获取要存储的值
// //       Value valueToStore = mapping.lookupOrDefault(storeOp.getValueToStore());
      
// //       // 由于索引是动态计算的，使用 memref.store 而不是 affine.store
// //       rewriter.create<memref::StoreOp>(
// //           storeOp.getLoc(), valueToStore, newMemref, ValueRange{newIdx0, idx2, idx3});
// //     } else {
// //       // 正常克隆
// //       rewriter.clone(*storeOp.getOperation(), mapping);
// //     }
// //   }

// //   // 计算flatten后的索引
// //   // 对于访问 [expr0(d0, d1), expr1(d0, d1), d2, d3]
// //   // 转换为 [expr0_value * memref_dim1 + expr1_value, d2, d3]
// //   // 由于我们使用 memref.load/store，可以使用算术操作
// //   Value computeFlattenedIndex(PatternRewriter &rewriter, Location loc,
// //                              AffineMap accessMap, SmallVector<Value> indices,
// //                              Value originalIdx0, Value originalIdx1,
// //                              int64_t memrefDim1, IRMapping &mapping) const {
    
// //     MLIRContext *ctx = rewriter.getContext();
    
// //     // 获取前两维的affine表达式
// //     AffineExpr dim0Expr = accessMap.getResult(0);
// //     AffineExpr dim1Expr = accessMap.getResult(1);
    
// //     // 准备映射后的操作数
// //     SmallVector<Value> mappedOperands;
// //     for (auto idx : indices) {
// //       mappedOperands.push_back(mapping.lookupOrDefault(idx));
// //     }
    
// //     // 计算 dim0 表达式的值
// //     auto dim0Map = AffineMap::get(accessMap.getNumDims(), 0, {dim0Expr}, ctx);
// //     Value dim0Value = rewriter.create<affine::AffineApplyOp>(
// //         loc, dim0Map, mappedOperands).getResult();
    
// //     // 计算 dim1 表达式的值
// //     auto dim1Map = AffineMap::get(accessMap.getNumDims(), 0, {dim1Expr}, ctx);
// //     Value dim1Value = rewriter.create<affine::AffineApplyOp>(
// //         loc, dim1Map, mappedOperands).getResult();
    
// //     // 使用算术操作计算 flattenedIndex = dim0Value * memrefDim1 + dim1Value
// //     Value memrefDim1Value = rewriter.create<arith::ConstantIndexOp>(loc, memrefDim1);
// //     Value product = rewriter.create<arith::MulIOp>(loc, dim0Value, memrefDim1Value);
// //     Value flattenedIndex = rewriter.create<arith::AddIOp>(loc, product, dim1Value);
    
// //     return flattenedIndex;
// //   }

// //   // 为memref创建reshape后的版本
// //   Value createReshapedMemrefForAccess(PatternRewriter &rewriter, 
// //                                      const MemrefAccessInfo &info,
// //                                      Location loc) const {
    
// //     auto shape = info.memrefType.getShape();
// //     Type elementType = info.memrefType.getElementType();
    
// //     // 计算flatten后的第一维大小
// //     int64_t flattenedDim0 = shape[0] * shape[1];
    
// //     // 创建新的3D shape: [shape[0]*shape[1], shape[2], shape[3]]
// //     SmallVector<int64_t> newShape = {flattenedDim0, shape[2], shape[3]};
// //     auto newMemrefType = MemRefType::get(newShape, elementType);

// //     // 计算strides (假设原始memref是连续的row-major布局)
// //     SmallVector<int64_t> strides = {shape[2] * shape[3], shape[3], 1};
// //     SmallVector<int64_t> offset = {0};

// //     LLVM_DEBUG(llvm::dbgs() << "Creating reshaped memref: " 
// //                << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3]
// //                << " -> " << flattenedDim0 << "x" << shape[2] << "x" << shape[3] << "\n");

// //     return rewriter.create<memref::ReinterpretCastOp>(
// //         loc, newMemrefType, info.memref, 
// //         /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
// //         /*static_offsets=*/offset, /*static_sizes=*/newShape, /*static_strides=*/strides
// //     ).getResult();
// //   }

// //   // 创建单维度的for循环
// //   affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
// //                                            Location loc, 
// //                                            int64_t lowerBound, 
// //                                            int64_t upperBound) const {
// //     MLIRContext *ctx = rewriter.getContext();
    
// //     AffineMap lowerBoundMap = AffineMap::getConstantMap(lowerBound, ctx);
// //     AffineMap upperBoundMap = AffineMap::getConstantMap(upperBound, ctx);
    
// //     auto forOp = rewriter.create<affine::AffineForOp>(
// //         loc,
// //         ValueRange{}, lowerBoundMap,
// //         ValueRange{}, upperBoundMap,
// //         /*step=*/1,
// //         ValueRange{},
// //         [](OpBuilder &, Location, Value, ValueRange) {});
    
// //     return forOp;
// //   }

// //   // 分析嵌套循环结构
// //   std::optional<NestedForInfo> analyzeNestedForStructure(
// //       affine::AffineForOp outerOp) const {
    
// //     auto dim0 = getLoopBound(outerOp);
// //     if (!dim0.has_value())
// //       return std::nullopt;

// //     if (!isPerfectlyNestedLevel(outerOp))
// //       return std::nullopt;

// //     auto secondFor = findNestedFor(outerOp);
// //     if (!secondFor)
// //       return std::nullopt;
    
// //     auto dim1 = getLoopBound(*secondFor);
// //     if (!dim1.has_value())
// //       return std::nullopt;

// //     if (!isPerfectlyNestedLevel(*secondFor))
// //       return std::nullopt;

// //     auto thirdFor = findNestedFor(*secondFor);
// //     if (!thirdFor)
// //       return std::nullopt;
    
// //     auto dim2 = getLoopBound(*thirdFor);
// //     if (!dim2.has_value())
// //       return std::nullopt;

// //     if (!isPerfectlyNestedLevel(*thirdFor))
// //       return std::nullopt;

// //     auto fourthFor = findNestedFor(*thirdFor);
// //     if (!fourthFor)
// //       return std::nullopt;
    
// //     auto dim3 = getLoopBound(*fourthFor);
// //     if (!dim3.has_value())
// //       return std::nullopt;

// //     return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *fourthFor};
// //   }

// //   // 检查是否为完美嵌套的一层
// //   bool isPerfectlyNestedLevel(affine::AffineForOp forOp) const {
// //     Block &body = forOp.getRegion().front();
    
// //     int opCount = 0;
// //     bool hasNestedFor = false;
    
// //     for (auto &op : body.getOperations()) {
// //       if (isa<affine::AffineYieldOp>(op))
// //         continue;
      
// //       opCount++;
// //       if (isa<affine::AffineForOp>(op))
// //         hasNestedFor = true;
// //     }
    
// //     return opCount == 1 && hasNestedFor;
// //   }

// //   // 查找嵌套的for循环
// //   std::optional<affine::AffineForOp> findNestedFor(affine::AffineForOp forOp) const {
// //     Block &body = forOp.getRegion().front();
    
// //     for (auto &op : body.getOperations()) {
// //       if (auto nestedFor = dyn_cast<affine::AffineForOp>(op)) {
// //         return nestedFor;
// //       }
// //     }
    
// //     return std::nullopt;
// //   }

// //   // 获取循环的边界（假设是从0开始的常量边界）
// //   std::optional<int64_t> getLoopBound(affine::AffineForOp forOp) const {
// //     auto lowerBound = forOp.getLowerBoundMap();
// //     auto lowerMap = lowerBound;
// //     if (lowerMap.getNumResults() != 1)
// //       return std::nullopt;
      
// //     auto lowerExpr = lowerMap.getResult(0);
// //     if (auto lowerConst = lowerExpr.dyn_cast<AffineConstantExpr>()) {
// //       if (lowerConst.getValue() != 0)
// //         return std::nullopt;
// //     } else {
// //       return std::nullopt;
// //     }
    
// //     auto upperBound = forOp.getUpperBoundMap();
// //     auto upperMap = upperBound;
// //     if (upperMap.getNumResults() != 1)
// //       return std::nullopt;
      
// //     auto upperExpr = upperMap.getResult(0);
// //     if (auto upperConst = upperExpr.dyn_cast<AffineConstantExpr>()) {
// //       return upperConst.getValue();
// //     }
    
// //     return std::nullopt;
// //   }
// // };

// // struct ForLoopFlattenPass
// //     : public PassWrapper<ForLoopFlattenPass, OperationPass<ModuleOp>> {
  
// //   StringRef getArgument() const final { return "for-loop-flatten"; }
// //   StringRef getDescription() const final {
// //     return "Enhanced flatten of 4D nested for loops to 3D, supporting different tensor shapes";
// //   }
  
// //   void getDependentDialects(DialectRegistry &registry) const override {
// //     registry.insert<affine::AffineDialect, memref::MemRefDialect, arith::ArithDialect>();
// //   }
  
// //   void runOnOperation() override {
// //     ModuleOp moduleOp = getOperation();
// //     MLIRContext *context = &getContext();
    
// //     LLVM_DEBUG(llvm::dbgs() << "Running ForLoopFlattenPass\n");
    
// //     RewritePatternSet patterns(context);
// //     patterns.add<ForLoopFlattenPattern>(context);
    
// //     if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
// //       signalPassFailure();
// //     }
    
// //     LLVM_DEBUG(llvm::dbgs() << "Completed ForLoopFlattenPass\n");
// //   }
// // };

// // } // end anonymous namespace

// // namespace onnx_mlir {
// //     std::unique_ptr<Pass> createForLoopFlattenPass() {
// //       return std::make_unique<ForLoopFlattenPass>();
// //     }
// // } // namespace onnx_mlir

// // static mlir::PassRegistration<ForLoopFlattenPass> pass;


// 如果存在chanle维度不一致，则对h,w进行flatten
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

// Flatten模式枚举
enum class FlattenMode {
  FrontTwoDims,  // Flatten前两个维度 (N, C) -> (N*C, H, W)
  LastTwoDims    // Flatten后两个维度 (N, C, H, W) -> (N, C, H*W)
};

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
    
    // 确定使用哪种flatten模式
    FlattenMode mode = determineFlattenMode(innerMostOp, dim0, dim1, dim2, dim3);
    
    if (mode == FlattenMode::FrontTwoDims) {
      LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten (front two dims): " 
                 << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
                 << " -> " << (dim0 * dim1) << "x" << dim2 << "x" << dim3 << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Found 4D nested for loop to flatten (last two dims): " 
                 << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 
                 << " -> " << dim0 << "x" << dim1 << "x" << (dim2 * dim3) << "\n");
    }

    // 收集需要重塑的memref
    SmallVector<Value> memrefsToReshape;
    SmallVector<std::pair<Value, MemRefType>> reshapeInfo;
    
    if (!collectMemRefs(innerMostOp, memrefsToReshape, reshapeInfo, dim0, dim1, dim2, dim3, mode))
      return failure();

    Location loc = outerForOp.getLoc();
    
    // 在外层循环前插入reinterpret_cast操作进行降维
    rewriter.setInsertionPoint(outerForOp);
    SmallVector<Value> reshapedMemrefs;
    
    for (auto [memref, originalType] : reshapeInfo) {
      Value reshaped = createReshapedMemref(rewriter, memref, originalType, 
                                           dim0, dim1, dim2, dim3, loc, mode);
      if (!reshaped)
        return failure();
      reshapedMemrefs.push_back(reshaped);
    }

    // 根据模式创建新的循环结构
    if (mode == FlattenMode::FrontTwoDims) {
      return rewriteWithFrontTwoDimsFlattening(outerForOp, rewriter, dim0, dim1, dim2, dim3,
                                               innerMostOp, memrefsToReshape, reshapedMemrefs, loc);
    } else {
      return rewriteWithLastTwoDimsFlattening(outerForOp, rewriter, dim0, dim1, dim2, dim3,
                                              innerMostOp, memrefsToReshape, reshapedMemrefs, loc);
    }
  }

private:
  struct NestedForInfo {
    int64_t dim0, dim1, dim2, dim3;
    affine::AffineForOp innerMostOp;
  };

  // 确定使用哪种flatten模式
  FlattenMode determineFlattenMode(affine::AffineForOp innerMostOp,
                                   int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    bool allMemrefsMatchLoopBounds = true;
    
    innerMostOp.getRegion().walk([&](Operation *op) {
      Value memref;
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        memref = loadOp.getMemRef();
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        memref = storeOp.getMemRef();
      } else {
        return;
      }
      
      auto memrefType = memref.getType().dyn_cast<MemRefType>();
      if (!memrefType || memrefType.getRank() != 4) {
        return;
      }
      
      auto shape = memrefType.getShape();
      // 检查shape是否与循环边界完全匹配
      if (shape[0] != dim0 || shape[1] != dim1 || shape[2] != dim2 || shape[3] != dim3) {
        LLVM_DEBUG(llvm::dbgs() << "Found memref with mismatched shape: "
                   << "memref=" << shape[0] << "x" << shape[1] << "x" 
                   << shape[2] << "x" << shape[3] 
                   << ", loops=" << dim0 << "x" << dim1 << "x" 
                   << dim2 << "x" << dim3 
                   << " -> Using LastTwoDims mode\n");
        allMemrefsMatchLoopBounds = false;
      }
    });
    
    // 如果所有memref的shape都与循环边界匹配，使用前两维flatten
    // 否则使用后两维flatten（适用于CNN中channel维度不匹配的情况）
    return allMemrefsMatchLoopBounds ? FlattenMode::FrontTwoDims : FlattenMode::LastTwoDims;
  }

  // 前两个维度flatten的重写逻辑（原始逻辑）
  LogicalResult rewriteWithFrontTwoDimsFlattening(
      affine::AffineForOp outerForOp, PatternRewriter &rewriter,
      int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
      affine::AffineForOp innerMostOp,
      const SmallVector<Value> &memrefsToReshape,
      const SmallVector<Value> &reshapedMemrefs,
      Location loc) const {
    
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
        
        // 更新映射
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          globalMapping.map(op->getResult(i), clonedOp->getResult(i));
        }
      }
    }

    // 替换原始循环
    rewriter.eraseOp(outerForOp);
    
    return success();
  }

  // **修复后的**: 后两个维度flatten的重写逻辑
  LogicalResult rewriteWithLastTwoDimsFlattening(
      affine::AffineForOp outerForOp, PatternRewriter &rewriter,
      int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
      affine::AffineForOp innerMostOp,
      const SmallVector<Value> &memrefsToReshape,
      const SmallVector<Value> &reshapedMemrefs,
      Location loc) const {
    
    // 获取原始循环体内容
    Block *originalInnerBody = &innerMostOp.getRegion().front();
    SmallVector<Operation*> opsToClone;
    for (auto &op : originalInnerBody->getOperations()) {
      if (!isa<affine::AffineYieldOp>(op)) {
        opsToClone.push_back(&op);
      }
    }

    // 创建三层嵌套的for循环
    int64_t flattenedSize = dim2 * dim3;
    
    // 创建第一层循环 (dim0: 0 to dim0)
    auto newOuterFor = createSingleDimForOp(rewriter, loc, 0, dim0);
    Block *outerBlock = &newOuterFor.getRegion().front();
    BlockArgument idx0 = outerBlock->getArgument(0);
    
    // 创建第二层循环 (dim1: 0 to dim1)
    rewriter.setInsertionPointToStart(outerBlock);
    auto newMiddleFor = createSingleDimForOp(rewriter, loc, 0, dim1);
    Block *middleBlock = &newMiddleFor.getRegion().front();
    BlockArgument idx1 = middleBlock->getArgument(0);
    
    // 创建第三层循环 (flattened dimension: 0 to flattenedSize)
    rewriter.setInsertionPointToStart(middleBlock);
    auto newInnerFor = createSingleDimForOp(rewriter, loc, 0, flattenedSize);
    Block *innerBlock = &newInnerFor.getRegion().front();
    BlockArgument flattenedIdx = innerBlock->getArgument(0);

    // 在最内层循环体中处理逻辑
    rewriter.setInsertionPointToStart(innerBlock);
    
    // 创建memref映射
    DenseMap<Value, Value> memrefMap;
    for (size_t i = 0; i < memrefsToReshape.size(); ++i) {
      memrefMap[memrefsToReshape[i]] = reshapedMemrefs[i];
    }
    
    // 建立索引映射 - 注意这里要保留原始的4个循环变量的映射
    IRMapping globalMapping;
    
    // 获取原始的4层循环的归纳变量
    BlockArgument origIdx0 = outerForOp.getInductionVar();
    globalMapping.map(origIdx0, idx0);
    
    auto level2Op = findNestedFor(outerForOp);
    if (!level2Op)
      return failure();
    BlockArgument origIdx1 = level2Op->getInductionVar();
    globalMapping.map(origIdx1, idx1);
    
    auto level3Op = findNestedFor(*level2Op);
    if (!level3Op)
      return failure();
    BlockArgument origIdx2 = level3Op->getInductionVar();
    
    auto level4Op = findNestedFor(*level3Op);
    if (!level4Op)
      return failure();
    BlockArgument origIdx3 = level4Op->getInductionVar();

    // 计算原始的 idx2 和 idx3（从扁平化索引恢复）
    Value dim3Value = rewriter.create<arith::ConstantIndexOp>(loc, dim3);
    Value originalIdx2 = rewriter.create<arith::DivSIOp>(loc, flattenedIdx, dim3Value);
    Value originalIdx3 = rewriter.create<arith::RemSIOp>(loc, flattenedIdx, dim3Value);
    
    globalMapping.map(origIdx2, originalIdx2);
    globalMapping.map(origIdx3, originalIdx3);

    // **关键修复**: 跟踪所有的affine.apply操作及其依赖的索引
    DenseMap<Value, Value> affineApplyResults;
    
    // 克隆循环体操作
    for (Operation *op : opsToClone) {
      // **修复1**: 处理 affine.apply 操作
      if (auto applyOp = dyn_cast<affine::AffineApplyOp>(op)) {
        // 获取 affine map 和操作数
        AffineMap map = applyOp.getAffineMap();
        SmallVector<Value> newOperands;
        
        // 映射操作数
        for (Value operand : applyOp.getOperands()) {
          Value mappedOperand = globalMapping.lookupOrDefault(operand);
          newOperands.push_back(mappedOperand);
        }
        
        // 创建新的 affine.apply
        auto newApplyOp = rewriter.create<affine::AffineApplyOp>(
            op->getLoc(), map, newOperands);
        
        // 保存结果映射
        globalMapping.map(applyOp.getResult(), newApplyOp.getResult());
        affineApplyResults[applyOp.getResult()] = newApplyOp.getResult();
        
        continue;
      }
      
      // **修复2**: 处理 load 操作，特别是带有复杂索引的
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        Value memref = loadOp.getMemRef();
        
        if (memrefMap.count(memref)) {
          // 这是需要重塑的memref
          Value newMemref = memrefMap.lookup(memref);
          
          // 获取原始的索引
          SmallVector<Value> originalIndices;
          for (Value idx : loadOp.getIndices()) {
            originalIndices.push_back(idx);
          }
          
          // 构建新的索引：[idx0, mappedIdx1, flattenedIdx]
          // 其中 mappedIdx1 可能是 affine.apply 的结果
          SmallVector<Value> newIndices;
          
          // 第一个索引（batch）
          newIndices.push_back(idx0);
          
          // 第二个索引（channel）- 可能需要通过affine.apply计算
          Value channelIdx = globalMapping.lookupOrDefault(originalIndices[1]);
          newIndices.push_back(channelIdx);
          
          // 第三个索引（扁平化的空间维度）
          newIndices.push_back(flattenedIdx);
          
          // 创建新的load操作
          auto newLoadOp = rewriter.create<affine::AffineLoadOp>(
              op->getLoc(), newMemref, newIndices);
          globalMapping.map(loadOp.getResult(), newLoadOp.getResult());
        } else {
          // 不需要重塑的memref，正常克隆
          auto clonedOp = rewriter.clone(*op, globalMapping);
          globalMapping.map(loadOp.getResult(), clonedOp->getResult(0));
        }
        continue;
      }
      
      // **修复3**: 处理 store 操作
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        Value memref = storeOp.getMemRef();
        
        if (memrefMap.count(memref)) {
          // 这是需要重塑的memref
          Value newMemref = memrefMap.lookup(memref);
          Value valueToStore = globalMapping.lookupOrDefault(storeOp.getValueToStore());
          
          // 获取原始的索引
          SmallVector<Value> originalIndices;
          for (Value idx : storeOp.getIndices()) {
            originalIndices.push_back(idx);
          }
          
          // 构建新的索引：[idx0, mappedIdx1, flattenedIdx]
          SmallVector<Value> newIndices;
          
          // 第一个索引（batch）
          newIndices.push_back(idx0);
          
          // 第二个索引（channel）- 可能需要通过affine.apply计算
          Value channelIdx = globalMapping.lookupOrDefault(originalIndices[1]);
          newIndices.push_back(channelIdx);
          
          // 第三个索引（扁平化的空间维度）
          newIndices.push_back(flattenedIdx);
          
          // 创建新的store操作
          rewriter.create<affine::AffineStoreOp>(
              op->getLoc(), valueToStore, newMemref, newIndices);
        } else {
          // 不需要重塑的memref，正常克隆
          rewriter.clone(*op, globalMapping);
        }
        continue;
      }
      
      // 对于其他操作，正常克隆
      auto clonedOp = rewriter.clone(*op, globalMapping);
      
      // 更新映射
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        globalMapping.map(op->getResult(i), clonedOp->getResult(i));
      }
    }

    // 替换原始循环
    rewriter.eraseOp(outerForOp);
    
    return success();
  }

  affine::AffineForOp createSingleDimForOp(PatternRewriter &rewriter, 
                                          Location loc, 
                                          int64_t lowerBound, 
                                          int64_t upperBound) const {
    auto lbMap = AffineMap::getConstantMap(lowerBound, rewriter.getContext());
    auto ubMap = AffineMap::getConstantMap(upperBound, rewriter.getContext());
    return rewriter.create<affine::AffineForOp>(loc, ValueRange{}, lbMap, ValueRange{}, ubMap, 1);
  }

  std::optional<NestedForInfo> analyzeNestedForStructure(affine::AffineForOp outerForOp) const {
    // 检查第一层
    if (!isPerfectlyNested(outerForOp))
      return std::nullopt;
    
    auto dim0 = getLoopBound(outerForOp);
    if (!dim0.has_value())
      return std::nullopt;

    // 检查第二层
    auto level2 = findNestedFor(outerForOp);
    if (!level2.has_value() || !isPerfectlyNested(*level2))
      return std::nullopt;
    
    auto dim1 = getLoopBound(*level2);
    if (!dim1.has_value())
      return std::nullopt;

    // 检查第三层
    auto level3 = findNestedFor(*level2);
    if (!level3.has_value() || !isPerfectlyNested(*level3))
      return std::nullopt;
    
    auto dim2 = getLoopBound(*level3);
    if (!dim2.has_value())
      return std::nullopt;

    // 检查第四层（最内层）
    auto level4 = findNestedFor(*level3);
    if (!level4.has_value())
      return std::nullopt;
    
    auto dim3 = getLoopBound(*level4);
    if (!dim3.has_value())
      return std::nullopt;

    // 确保第四层是最内层（没有更多嵌套）
    auto level5 = findNestedFor(*level4);
    if (level5.has_value())
      return std::nullopt;

    return NestedForInfo{*dim0, *dim1, *dim2, *dim3, *level4};
  }

  bool isPerfectlyNested(affine::AffineForOp forOp) const {
    Block &body = forOp.getRegion().front();
    
    int forOpCount = 0;
    int yieldOpCount = 0;
    int otherOpCount = 0;
    
    for (auto &op : body.getOperations()) {
      if (isa<affine::AffineForOp>(op)) {
        forOpCount++;
      } else if (isa<affine::AffineYieldOp>(op)) {
        yieldOpCount++;
      } else {
        otherOpCount++;
      }
    }
    
    // 完美嵌套：恰好一个for循环，一个yield，没有其他操作
    return forOpCount == 1 && yieldOpCount == 1 && otherOpCount == 0;
  }

  // 查找嵌套在当前for循环内部的for循环
  std::optional<affine::AffineForOp> findNestedFor(affine::AffineForOp forOp) const {
    Block &body = forOp.getRegion().front();
    
    for (auto &op : body.getOperations()) {
      if (auto nestedFor = dyn_cast<affine::AffineForOp>(op)) {
        return nestedFor;
      }
    }
    
    return std::nullopt;
  }

  // 获取循环的边界（假设是从0到常量的简单边界）
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
                     int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                     FlattenMode mode) const {
    
    // 收集所有访问的4D memref
    SmallVector<std::pair<Value, MemRefType>> allAccessedMemrefs;
    
    innerMostOp.getRegion().walk([&](Operation *op) {
      Value memref;
      
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        memref = loadOp.getMemRef();
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        memref = storeOp.getMemRef();
      } else {
        return;
      }
      
      auto memrefType = memref.getType().dyn_cast<MemRefType>();
      
      if (memrefType && memrefType.getRank() == 4) {
        // **修复**: 不再严格要求标准访问模式，只检查兼容性
        // 这样可以处理带有 affine.apply 的访问
        if (isCompatibleForFlatten(memrefType, dim0, dim1, dim2, dim3, mode)) {
          if (std::find_if(allAccessedMemrefs.begin(), allAccessedMemrefs.end(),
                          [memref](const auto& pair) { return pair.first == memref; }) 
              == allAccessedMemrefs.end()) {
            allAccessedMemrefs.push_back({memref, memrefType});
          }
        }
      }
    });

    // 收集到reshapeInfo中
    for (auto [memref, memrefType] : allAccessedMemrefs) {
      if (std::find(memrefsToReshape.begin(), memrefsToReshape.end(), memref) 
          == memrefsToReshape.end()) {
        memrefsToReshape.push_back(memref);
        reshapeInfo.push_back({memref, memrefType});
      }
    }

    return !memrefsToReshape.empty();
  }

  // 根据flatten模式检查memref是否兼容
  bool isCompatibleForFlatten(MemRefType memrefType, int64_t dim0, int64_t dim1, 
                              int64_t dim2, int64_t dim3, FlattenMode mode) const {
    auto shape = memrefType.getShape();
    
    if (mode == FlattenMode::FrontTwoDims) {
      // 前两维flatten模式：要求shape与循环边界完全匹配
      return shape.size() == 4 && 
             (shape[0] == dim0 || shape[0] == ShapedType::kDynamic) &&
             (shape[1] == dim1 || shape[1] == ShapedType::kDynamic) &&
             (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
             (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
    } else {
      // 后两维flatten模式：只要求后两维与循环边界匹配，前两维可以不同
      return shape.size() == 4 && 
             (shape[2] == dim2 || shape[2] == ShapedType::kDynamic) &&
             (shape[3] == dim3 || shape[3] == ShapedType::kDynamic);
    }
  }

  Value createReshapedMemref(PatternRewriter &rewriter, Value originalMemref,
                            MemRefType originalType, int64_t dim0, int64_t dim1,
                            int64_t dim2, int64_t dim3, Location loc,
                            FlattenMode mode) const {
    
    Type elementType = originalType.getElementType();
    auto originalShape = originalType.getShape();
    
    SmallVector<int64_t> newShape;
    SmallVector<int64_t> strides;
    
    if (mode == FlattenMode::FrontTwoDims) {
      // Flatten前两维: [dim0, dim1, dim2, dim3] -> [dim0*dim1, dim2, dim3]
      int64_t flattenedSize = dim0 * dim1;
      newShape = {flattenedSize, dim2, dim3};
      strides = {dim2 * dim3, dim3, 1};
    } else {
      // Flatten后两维: [N, C, dim2, dim3] -> [N, C, dim2*dim3]
      // 使用memref的实际shape的前两维，而不是循环的dim0, dim1
      int64_t flattenedSize = dim2 * dim3;
      newShape = {originalShape[0], originalShape[1], flattenedSize};
      strides = {originalShape[1] * flattenedSize, flattenedSize, 1};
    }
    
    auto newMemrefType = MemRefType::get(newShape, elementType);
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
    return "Flatten 4D nested for loops to 3D by merging either first two or last two dimensions";
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