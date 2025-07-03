// /*
//  * SPDX-License-Identifier: Apache-2.0
//  */

// //===---------------- Flatten.cpp - Lowering Flatten Op -------------------===//
// //
// // Copyright 2019-2023 The IBM Research Authors.
// //
// // =============================================================================
// //
// // This file lowers the ONNX Flatten Operator to Krnl dialect.
// //
// //===----------------------------------------------------------------------===//

// #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

// using namespace mlir;

// namespace onnx_mlir {

// //===----------------------------------------------------------------------===//
// // Helper function to insert alloc and dealloc ops for memref of dynamic shape.
// //
// // Should namespace or static be used here?
// Value insertAllocForFlatten(MemRefType memRefType, Location loc,
//     ConversionPatternRewriter &rewriter, Value input, int64_t axisValue) {
//   MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
//   memref::AllocOp alloc;
//   auto inputShape = mlir::cast<MemRefType>(input.getType()).getShape();
//   int64_t inputRank = inputShape.size();

//   SmallVector<Value, 2> allocOperands;
//   // Compute size for the first dimension when not constant
//   if (memRefType.isDynamicDim(0)) {
//     Value dimVal = create.math.constantIndex(1);
//     for (int64_t i = 0; i < axisValue; i++)
//       dimVal = create.math.mul(dimVal, create.mem.dim(input, i));
//     allocOperands.emplace_back(dimVal);
//   }

//   // Compute size for the second dimension when not constant
//   if (memRefType.isDynamicDim(1)) {
//     Value dimVal = create.math.constantIndex(1);
//     for (int64_t i = axisValue; i < inputRank; i++)
//       dimVal = create.math.mul(dimVal, create.mem.dim(input, i));
//     allocOperands.emplace_back(dimVal);
//   }

//   return create.mem.alignedAlloc(memRefType, allocOperands);
// }

// struct ONNXFlattenOpLowering : public OpConversionPattern<ONNXFlattenOp> {
//   ONNXFlattenOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
//       : OpConversionPattern(typeConverter, ctx) {}

//   LogicalResult matchAndRewrite(ONNXFlattenOp flattenOp,
//       ONNXFlattenOpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const final {

//     // Gather info.
//     Operation *op = flattenOp.getOperation();
//     Location loc = ONNXLoc<ONNXFlattenOp>(op);

//     Value input = adaptor.getInput();
//     auto inputTy = mlir::cast<MemRefType>(input.getType());
//     auto inputShape = inputTy.getShape();
//     size_t inputRank = inputShape.size();
//     int64_t axisValue = flattenOp.getAxis();
//     if (axisValue < 0)
//       axisValue = inputRank + axisValue;
//     MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);

//     // Convert the output type to MemRefType.
//     Type convertedType = typeConverter->convertType(*op->result_type_begin());
//     assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
//            "Failed to convert type to MemRefType");
//     MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

//     // Insert alloc and dealloc
//     Value alloc = (hasAllConstantDimensions(outputMemRefType))
//                       ? create.mem.alignedAlloc(outputMemRefType)
//                       : insertAllocForFlatten(
//                             outputMemRefType, loc, rewriter, input, axisValue);

//     // Define loops and iteration trip counts (equivalent to size of input)
//     ValueRange indices;
//     std::vector<Value> originalLoops;
//     defineLoops(rewriter, loc, originalLoops, inputRank);
//     // TODO use new KrnlDialectBuilder.
//     krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
//     for (size_t i = 0; i < inputRank; ++i)
//       addDimensionToPack(rewriter, loc, pack, input, i);

//     // Create the loops
//     KrnlIterateOp iterateOp = create.krnl.iterate(pack);
//     Block &iterationBlock = iterateOp.getBodyRegion().front();

//     // Now perform the insertions into the body of the just generated loops.
//     // Insert instructions inside the KernelIterateOp body.
//     rewriter.setInsertionPointToStart(&iterationBlock);

//     // Generate the load of input
//     SmallVector<Value, 4> inputMemRefVal(iterationBlock.getArguments().begin(),
//         iterationBlock.getArguments().end());
//     Value inputVal = create.krnl.load(input, inputMemRefVal);

//     // Generate the store for output
//     // Define affine map for first dim of output
//     AffineExpr firstIndexAE = rewriter.getAffineConstantExpr(0);
//     AffineExpr firstAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
//     for (int64_t i = axisValue - 1; i >= 0; i--) {
//       AffineExpr dimIndexAE = rewriter.getAffineDimExpr(i);
//       firstIndexAE = firstIndexAE + dimIndexAE * firstAccumulatedDimSizeAE;
//       AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(i);
//       firstAccumulatedDimSizeAE = dimSizeAE * firstAccumulatedDimSizeAE;
//     }
//     AffineMap firstDimMap = AffineMap::get(axisValue, axisValue, firstIndexAE);

//     // Create the parameter lists for the affine map
//     MemRefBuilder createMemRef(rewriter, loc);
//     SmallVector<Value, 4> firstMapArgList;
//     for (int64_t i = 0; i < axisValue; i++)
//       firstMapArgList.emplace_back(iterationBlock.getArguments()[i]);

//     for (int64_t i = 0; i < axisValue; i++)
//       firstMapArgList.emplace_back(createMemRef.dim(input, i));

//     auto firstDimVal = rewriter.create<affine::AffineApplyOp>(
//         loc, firstDimMap, firstMapArgList);

//     // Generate index for second dim of output
//     AffineExpr secondIndexAE = rewriter.getAffineConstantExpr(0);
//     AffineExpr secondAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
//     // Can not use auto for i here because i may be negative
//     for (int64_t i = inputRank - 1; i >= axisValue; i--) {
//       int64_t idx = i - axisValue;
//       AffineExpr dimIndexAE = rewriter.getAffineDimExpr(idx);
//       secondIndexAE = secondIndexAE + dimIndexAE * secondAccumulatedDimSizeAE;
//       AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(idx);
//       secondAccumulatedDimSizeAE = dimSizeAE * secondAccumulatedDimSizeAE;
//     }
//     AffineMap secondDimMap = AffineMap::get(
//         inputRank - axisValue, inputRank - axisValue, secondIndexAE);

//     // Create the parameter lists for the affine map
//     SmallVector<Value, 4> secondMapArgList;
//     for (size_t i = axisValue; i < inputRank; i++)
//       secondMapArgList.emplace_back(iterationBlock.getArguments()[i]);
//     for (size_t i = axisValue; i < inputRank; i++)
//       secondMapArgList.emplace_back(createMemRef.dim(input, i));

//     auto secondDimVal = rewriter.create<affine::AffineApplyOp>(
//         loc, secondDimMap, secondMapArgList);

//     // Create the store
//     SmallVector<Value, 2> outputMemRefVal = {firstDimVal, secondDimVal};
//     if (hasAllConstantDimensions(outputMemRefType))
//       create.krnl.store(inputVal, alloc, outputMemRefVal);
//     else
//       create.krnl.store(inputVal, alloc, outputMemRefVal);

//     rewriter.replaceOp(op, alloc);
//     onnxToKrnlSimdReport(op);
//     return success();
//   }
// };

// void populateLoweringONNXFlattenOpPattern(RewritePatternSet &patterns,
//     TypeConverter &typeConverter, MLIRContext *ctx) {
//   patterns.insert<ONNXFlattenOpLowering>(typeConverter, ctx);
// }

// } // namespace onnx_mlir


/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Flatten.cpp - Lowering Flatten Op -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Krnl dialect.
// Optimized version that uses memref.reinterpret_cast when possible.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper function to check if input memref can use reinterpret_cast
//
bool canUseReinterpretCast(MemRefType inputType) {
  // Check if the memref has default (contiguous) layout
  auto layout = inputType.getLayout();
  
  // If layout is identity or affine map that represents contiguous layout
  if (layout.isIdentity()) {
    return true;
  }
  
  // For affine map layouts, check if it represents a contiguous layout
  if (auto affineMap = mlir::dyn_cast<AffineMapAttr>(layout)) {
    if (affineMap.getValue().isPermutation() && 
        affineMap.getValue().isMinorIdentity()) {
      return true;
    }
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// Helper function to compute flatten dimensions
//
std::pair<int64_t, int64_t> computeFlattenDims(
    ArrayRef<int64_t> inputShape, int64_t axisValue) {
  int64_t firstDim = 1;
  int64_t secondDim = 1;
  
  // Calculate first dimension (product of dimensions before axis)
  for (int64_t i = 0; i < axisValue; i++) {
    if (inputShape[i] == ShapedType::kDynamic) {
      firstDim = ShapedType::kDynamic;
      break;
    }
    firstDim *= inputShape[i];
  }
  
  // Calculate second dimension (product of dimensions from axis onwards)
  for (int64_t i = axisValue; i < static_cast<int64_t>(inputShape.size()); i++) {
    if (inputShape[i] == ShapedType::kDynamic) {
      secondDim = ShapedType::kDynamic;
      break;
    }
    secondDim *= inputShape[i];
  }
  
  return {firstDim, secondDim};
}

//===----------------------------------------------------------------------===//
// Helper function to compute dynamic dimension sizes at runtime
//
Value computeDynamicDimSize(ConversionPatternRewriter &rewriter, Location loc,
    Value input, int64_t startIdx, int64_t endIdx) {
  MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
  
  Value dimVal = create.math.constantIndex(1);
  for (int64_t i = startIdx; i < endIdx; i++) {
    dimVal = create.math.mul(dimVal, create.mem.dim(input, i));
  }
  return dimVal;
}

//===----------------------------------------------------------------------===//
// Helper function to create reinterpret_cast operation
//
Value createReinterpretCast(ConversionPatternRewriter &rewriter, Location loc,
    Value input, MemRefType outputType, int64_t axisValue) {
  MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
  
  auto inputType = mlir::cast<MemRefType>(input.getType());
  auto inputShape = inputType.getShape();
  
  // Compute output shape
  auto [firstDim, secondDim] = computeFlattenDims(inputShape, axisValue);
  
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  
  // First dimension
  if (firstDim == ShapedType::kDynamic) {
    Value dynamicSize = computeDynamicDimSize(rewriter, loc, input, 0, axisValue);
    sizes.push_back(dynamicSize);
  } else {
    sizes.push_back(rewriter.getIndexAttr(firstDim));
  }
  
  // Second dimension  
  if (secondDim == ShapedType::kDynamic) {
    Value dynamicSize = computeDynamicDimSize(rewriter, loc, input, 
        axisValue, static_cast<int64_t>(inputShape.size()));
    sizes.push_back(dynamicSize);
  } else {
    sizes.push_back(rewriter.getIndexAttr(secondDim));
  }
  
  // Compute strides for row-major layout
  strides.push_back(rewriter.getIndexAttr(secondDim == ShapedType::kDynamic ? 0 : secondDim));
  strides.push_back(rewriter.getIndexAttr(1));
  
  // Create reinterpret_cast
  return rewriter.create<memref::ReinterpretCastOp>(
      loc, outputType, input, /*offset=*/rewriter.getIndexAttr(0), sizes, strides);
}

//===----------------------------------------------------------------------===//
// Helper function to create reshape operation (fallback for static shapes)
//
Value createReshape(ConversionPatternRewriter &rewriter, Location loc,
    Value input, MemRefType outputType, int64_t axisValue) {
  MultiDialectBuilder<MemRefBuilder, MathBuilder> create(rewriter, loc);
  
  auto inputType = mlir::cast<MemRefType>(input.getType());
  auto inputShape = inputType.getShape();
  
  // Only use reshape for static shapes
  auto [firstDim, secondDim] = computeFlattenDims(inputShape, axisValue);
  
  if (firstDim != ShapedType::kDynamic && secondDim != ShapedType::kDynamic) {
    // Create a 1D memref to hold the new shape [firstDim, secondDim]
    auto shapeMemRefType = MemRefType::get({2}, rewriter.getIndexType());
    Value shapeMemRef = create.mem.alignedAlloc(shapeMemRefType);
    
    // Store the new dimensions into the shape memref
    Value firstDimValue = create.math.constantIndex(firstDim);
    Value secondDimValue = create.math.constantIndex(secondDim);
    Value zeroIndex = create.math.constantIndex(0);
    Value oneIndex = create.math.constantIndex(1);
    
    create.mem.store(firstDimValue, shapeMemRef, zeroIndex);
    create.mem.store(secondDimValue, shapeMemRef, oneIndex);
    
    return rewriter.create<memref::ReshapeOp>(loc, outputType, input, shapeMemRef);
  }
  
  return nullptr; // Cannot use reshape
}

//===----------------------------------------------------------------------===//
// Original helper function for krnl loop fallback
//
Value insertAllocForFlatten(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, Value input, int64_t axisValue) {
  MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
  auto inputShape = mlir::cast<MemRefType>(input.getType()).getShape();
  int64_t inputRank = inputShape.size();

  SmallVector<Value, 2> allocOperands;
  
  // Compute size for the first dimension when not constant
  if (memRefType.isDynamicDim(0)) {
    Value dimVal = computeDynamicDimSize(rewriter, loc, input, 0, axisValue);
    allocOperands.emplace_back(dimVal);
  }

  // Compute size for the second dimension when not constant
  if (memRefType.isDynamicDim(1)) {
    Value dimVal = computeDynamicDimSize(rewriter, loc, input, axisValue, inputRank);
    allocOperands.emplace_back(dimVal);
  }

  return create.mem.alignedAlloc(memRefType, allocOperands);
}

//===----------------------------------------------------------------------===//
// Fallback function using original krnl loop implementation
//
Value createKrnlLoopFlatten(ConversionPatternRewriter &rewriter, Location loc,
    Value input, MemRefType outputMemRefType, int64_t axisValue) {
  
  auto inputTy = mlir::cast<MemRefType>(input.getType());
  auto inputShape = inputTy.getShape();
  size_t inputRank = inputShape.size();
  
  MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);

  // Insert alloc and dealloc
  Value alloc = (hasAllConstantDimensions(outputMemRefType))
                    ? create.mem.alignedAlloc(outputMemRefType)
                    : insertAllocForFlatten(outputMemRefType, loc, rewriter, input, axisValue);

  // Define loops and iteration trip counts
  std::vector<Value> originalLoops;
  defineLoops(rewriter, loc, originalLoops, inputRank);
  krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
  for (size_t i = 0; i < inputRank; ++i)
    addDimensionToPack(rewriter, loc, pack, input, i);

  // Create the loops
  KrnlIterateOp iterateOp = create.krnl.iterate(pack);
  Block &iterationBlock = iterateOp.getBodyRegion().front();
  rewriter.setInsertionPointToStart(&iterationBlock);

  // Generate the load of input
  SmallVector<Value, 4> inputMemRefVal(iterationBlock.getArguments().begin(),
      iterationBlock.getArguments().end());
  Value inputVal = create.krnl.load(input, inputMemRefVal);

  // Generate index computation for output using affine maps
  AffineExpr firstIndexAE = rewriter.getAffineConstantExpr(0);
  AffineExpr firstAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
  for (int64_t i = axisValue - 1; i >= 0; i--) {
    AffineExpr dimIndexAE = rewriter.getAffineDimExpr(i);
    firstIndexAE = firstIndexAE + dimIndexAE * firstAccumulatedDimSizeAE;
    AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(i);
    firstAccumulatedDimSizeAE = dimSizeAE * firstAccumulatedDimSizeAE;
  }
  AffineMap firstDimMap = AffineMap::get(axisValue, axisValue, firstIndexAE);

  SmallVector<Value, 4> firstMapArgList;
  for (int64_t i = 0; i < axisValue; i++)
    firstMapArgList.emplace_back(iterationBlock.getArguments()[i]);
  for (int64_t i = 0; i < axisValue; i++)
    firstMapArgList.emplace_back(create.mem.dim(input, i));

  auto firstDimVal = rewriter.create<affine::AffineApplyOp>(
      loc, firstDimMap, firstMapArgList);

  // Generate index for second dim of output
  AffineExpr secondIndexAE = rewriter.getAffineConstantExpr(0);
  AffineExpr secondAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
  for (int64_t i = inputRank - 1; i >= axisValue; i--) {
    int64_t idx = i - axisValue;
    AffineExpr dimIndexAE = rewriter.getAffineDimExpr(idx);
    secondIndexAE = secondIndexAE + dimIndexAE * secondAccumulatedDimSizeAE;
    AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(idx);
    secondAccumulatedDimSizeAE = dimSizeAE * secondAccumulatedDimSizeAE;
  }
  AffineMap secondDimMap = AffineMap::get(
      inputRank - axisValue, inputRank - axisValue, secondIndexAE);

  SmallVector<Value, 4> secondMapArgList;
  for (size_t i = axisValue; i < inputRank; i++)
    secondMapArgList.emplace_back(iterationBlock.getArguments()[i]);
  for (size_t i = axisValue; i < inputRank; i++)
    secondMapArgList.emplace_back(create.mem.dim(input, i));

  auto secondDimVal = rewriter.create<affine::AffineApplyOp>(
      loc, secondDimMap, secondMapArgList);

  // Create the store
  SmallVector<Value, 2> outputMemRefVal = {firstDimVal, secondDimVal};
  create.krnl.store(inputVal, alloc, outputMemRefVal);

  return alloc;
}

//===----------------------------------------------------------------------===//
// Main conversion pattern
//
struct ONNXFlattenOpLowering : public OpConversionPattern<ONNXFlattenOp> {
  ONNXFlattenOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXFlattenOp flattenOp,
      ONNXFlattenOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // Gather info.
    Operation *op = flattenOp.getOperation();
    Location loc = ONNXLoc<ONNXFlattenOp>(op);

    Value input = adaptor.getInput();
    auto inputTy = mlir::cast<MemRefType>(input.getType());
    auto inputShape = inputTy.getShape();
    size_t inputRank = inputShape.size();
    int64_t axisValue = flattenOp.getAxis();
    if (axisValue < 0)
      axisValue = inputRank + axisValue;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    Value result = nullptr;

    // Strategy 1: Try memref.reinterpret_cast if input layout is contiguous
    if (canUseReinterpretCast(inputTy)) {
      result = createReinterpretCast(rewriter, loc, input, outputMemRefType, axisValue);
      if (result) {
        rewriter.replaceOp(op, result);
        onnxToKrnlSimdReport(op);
        return success();
      }
    }

    // Strategy 2: Try memref.reshape for static shapes
    result = createReshape(rewriter, loc, input, outputMemRefType, axisValue);
    if (result) {
      rewriter.replaceOp(op, result);
      onnxToKrnlSimdReport(op);
      return success();
    }

    // Strategy 3: Fallback to krnl loop implementation
    result = createKrnlLoopFlatten(rewriter, loc, input, outputMemRefType, axisValue);
    
    rewriter.replaceOp(op, result);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXFlattenOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir