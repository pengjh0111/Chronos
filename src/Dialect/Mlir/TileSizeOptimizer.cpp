// // #include "TileSizeOptimizer.h"
// // #include "llvm/ADT/STLExtras.h"
// // #include "llvm/Support/Debug.h"
// // #include "llvm/ADT/DenseSet.h"
// // #include "llvm/ADT/DenseMap.h"

// // #define DEBUG_TYPE "tile-size-optimizer"

// // using namespace mlir;
// // using namespace mlir::scf;
// // using namespace llvm;

// // //===----------------------------------------------------------------------===//
// // // TileSizeOptimizer实现
// // //===----------------------------------------------------------------------===//

// // CompleteTileConfig TileSizeOptimizer::optimizeTileSize(scf::ParallelOp parallelOp) {
// //   // Step 1: Extract loop information
// //   std::vector<LoopInfo> loopInfos = extractLoopInfo(parallelOp);
  
// //   // Step 2: Analyze memory access patterns
// //   std::vector<MemoryAccessInfo> memAccesses = analyzeMemoryAccesses(parallelOp);
  
// //   // Step 3: Analyze computation operations
// //   std::vector<ComputationInfo> computations = analyzeComputations(parallelOp);
  
// //   // Step 4: Detect computational pattern
// //   ComputationalPattern pattern = detectComputationalPattern(parallelOp, memAccesses);
  
// //   // Step 5: Generate candidate tile sizes with pattern awareness
// //   std::vector<std::vector<int64_t>> tileSizeCandidates = 
// //       generateTileSizeCandidates(loopInfos, pattern);
  
// //   // Step 6: Find optimal tile configuration using dynamic programming
// //   CompleteTileConfig optimalConfig = findOptimalTileConfig(
// //       loopInfos, memAccesses, computations, tileSizeCandidates);
  
// //   LLVM_DEBUG(llvm::dbgs() << "Optimal tile config: ";
// //            for (const auto &dim : optimalConfig.perDimConfig) {
// //              llvm::dbgs() << dim.tileSize << " ";
// //            }
// //            llvm::dbgs() << "Score: " << optimalConfig.overallPerformanceScore << "\n");
  
// //   return optimalConfig;
// // }

// // std::vector<LoopInfo> TileSizeOptimizer::extractLoopInfo(scf::ParallelOp parallelOp) {
// //   std::vector<LoopInfo> loopInfos;
  
// //   // 提取每个循环维度的信息
// //   for (unsigned i = 0; i < parallelOp.getNumLoops(); ++i) {
// //     LoopInfo info;
// //     info.dimension = i;
// //     info.lowerBound = parallelOp.getLowerBound()[i];
// //     info.upperBound = parallelOp.getUpperBound()[i];
// //     info.step = parallelOp.getStep()[i];
    
// //     // 尝试提取常量值（如果可用）
// //     if (auto constLB = dyn_cast_or_null<arith::ConstantIndexOp>(info.lowerBound.getDefiningOp())) {
// //       info.constantLowerBound = constLB.value();
      
// //       if (auto constUB = dyn_cast_or_null<arith::ConstantIndexOp>(info.upperBound.getDefiningOp())) {
// //         info.constantUpperBound = constUB.value();
        
// //         if (auto constStep = dyn_cast_or_null<arith::ConstantIndexOp>(info.step.getDefiningOp())) {
// //           info.constantStep = constStep.value();
// //           info.hasConstantBounds = true;
          
// //           // 计算总迭代次数
// //           info.tripCount = (info.constantUpperBound - info.constantLowerBound + info.constantStep - 1) / 
// //                            info.constantStep;
// //         }
// //       }
// //     }
    
// //     loopInfos.push_back(info);
// //   }
  
// //   return loopInfos;
// // }

// // std::vector<MemoryAccessInfo> TileSizeOptimizer::analyzeMemoryAccesses(scf::ParallelOp parallelOp) {
// //   std::vector<MemoryAccessInfo> memAccesses;
  
// //   // 收集循环中的所有内存操作
// //   parallelOp.getBody()->walk([&](Operation *op) {
// //     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
// //       MemoryAccessInfo info;
// //       info.memref = loadOp.getMemref();
// //       info.indices = SmallVector<Value, 4>(loadOp.getIndices().begin(), loadOp.getIndices().end());
// //       info.isLoad = true;
// //       info.isStore = false;
      
// //       // 确定数据类型大小
// //       if (auto memrefType = loadOp.getMemref().getType().dyn_cast<MemRefType>()) {
// //         Type elementType = memrefType.getElementType();
// //         if (elementType.isF32()) {
// //           info.dataTypeSizeInBytes = 4;
// //         } else if (elementType.isF64()) {
// //           info.dataTypeSizeInBytes = 8;
// //         } else if (elementType.isInteger(32)) {
// //           info.dataTypeSizeInBytes = 4;
// //         } else if (elementType.isInteger(64)) {
// //           info.dataTypeSizeInBytes = 8;
// //         } else {
// //           // 其他类型的默认值
// //           info.dataTypeSizeInBytes = 4;
// //         }
// //       } else {
// //         info.dataTypeSizeInBytes = 4;  // 默认值
// //       }
      
// //       // 分析访问模式（简化版）
// //       info.pattern = MemoryAccessPattern::SEQUENTIAL;
      
// //       // 寻找合并访问模式
// //       // 检查是否最内层循环索引用于最右侧维度
// //       if (!info.indices.empty() && 
// //           info.indices.back() == parallelOp.getInductionVars().back()) {
// //         info.pattern = MemoryAccessPattern::COALESCED;
// //       }
      
// //       memAccesses.push_back(info);
// //     } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
// //       MemoryAccessInfo info;
// //       info.memref = storeOp.getMemref();
// //       info.indices = SmallVector<Value, 4>(storeOp.getIndices().begin(), storeOp.getIndices().end());
// //       info.isLoad = false;
// //       info.isStore = true;
      
// //       // 确定数据类型大小（类似于loadOp）
// //       if (auto memrefType = storeOp.getMemref().getType().dyn_cast<MemRefType>()) {
// //         Type elementType = memrefType.getElementType();
// //         if (elementType.isF32()) {
// //           info.dataTypeSizeInBytes = 4;
// //         } else if (elementType.isF64()) {
// //           info.dataTypeSizeInBytes = 8;
// //         } else if (elementType.isInteger(32)) {
// //           info.dataTypeSizeInBytes = 4;
// //         } else if (elementType.isInteger(64)) {
// //           info.dataTypeSizeInBytes = 8;
// //         } else {
// //           info.dataTypeSizeInBytes = 4;  // 默认值
// //         }
// //       } else {
// //         info.dataTypeSizeInBytes = 4;  // 默认值
// //       }
      
// //       // 分析访问模式（简化版）
// //       info.pattern = MemoryAccessPattern::SEQUENTIAL;
      
// //       // 检查合并访问模式
// //       if (!info.indices.empty() && 
// //           info.indices.back() == parallelOp.getInductionVars().back()) {
// //         info.pattern = MemoryAccessPattern::COALESCED;
// //       }
      
// //       memAccesses.push_back(info);
// //     }
// //   });
  
// //   return memAccesses;
// // }

// // std::vector<ComputationInfo> TileSizeOptimizer::analyzeComputations(scf::ParallelOp parallelOp) {
// //   std::vector<ComputationInfo> computations;
  
// //   // 收集循环中的计算操作
// //   parallelOp.getBody()->walk([&](Operation *op) {
// //     ComputationInfo info;
// //     info.op = op;
// //     info.opCount = 1;  // 每个操作的默认计数
    
// //     // 识别浮点操作
// //     if (isa<arith::AddFOp>(op) || isa<arith::SubFOp>(op) ||
// //         isa<arith::MulFOp>(op) || isa<arith::DivFOp>(op)) {
// //       info.isFloatingPoint = true;
// //       computations.push_back(info);
// //     } 
// //     // 识别整数操作
// //     else if (isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
// //              isa<arith::MulIOp>(op) || isa<arith::DivSIOp>(op) ||
// //              isa<arith::DivUIOp>(op)) {
// //       info.isFloatingPoint = false;
// //       computations.push_back(info);
// //     }
// //   });
  
// //   return computations;
// // }

// // TileSizeOptimizer::ComputationalPattern TileSizeOptimizer::detectComputationalPattern(
// //     scf::ParallelOp parallelOp,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // Default pattern
// //   ComputationalPattern pattern = ComputationalPattern::GENERIC;
  
// //   // Get loop dimensions
// //   size_t numDims = parallelOp.getNumLoops();
  
// //   // Count memory operations
// //   int loadCount = 0;
// //   int storeCount = 0;
// //   llvm::DenseSet<Value> uniqueLoadMemrefs;
// //   llvm::DenseSet<Value> uniqueStoreMemrefs;
  
// //   for (const auto &access : memAccesses) {
// //     if (access.isLoad) {
// //       loadCount++;
// //       uniqueLoadMemrefs.insert(access.memref);
// //     }
// //     if (access.isStore) {
// //       storeCount++;
// //       uniqueStoreMemrefs.insert(access.memref);
// //     }
// //   }
  
// //   // Count arithmetic operations
// //   int addCount = 0, mulCount = 0, maxCount = 0;
  
// //   parallelOp.getBody()->walk([&](Operation *op) {
// //     if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
// //       addCount++;
// //     } else if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
// //       mulCount++;
// //     } else if (isa<arith::MaximumFOp>(op) || isa<arith::MaxSIOp>(op) || 
// //         isa<arith::MaxNumFOp>(op)) {
// //       maxCount++;
// //     }
// //   });
  
// //   // Matrix multiplication pattern detection
// //   // - Usually 3 dimensions (M, N, K)
// //   // - High ratio of multiply-adds
// //   // - Specific memory access patterns
// //   if (numDims >= 2 && 
// //       mulCount > 0 && 
// //       addCount > 0 && 
// //       mulCount + addCount > loadCount) {
// //     pattern = ComputationalPattern::MATMUL;
// //   }
  
// //   // Convolution pattern detection
// //   // - Usually have window/kernel dimensions
// //   // - Often use max operations (for pooling)
// //   // - High spatial locality
// //   if (numDims >= 3 && 
// //       (mulCount > 0 || maxCount > 0) && 
// //       uniqueLoadMemrefs.size() >= 1 && 
// //       uniqueStoreMemrefs.size() >= 1) {
// //     pattern = ComputationalPattern::CONV;
// //   }
  
// //   // Stencil pattern detection
// //   // - Multiple loads from adjacent locations
// //   // - Usually few arithmetic ops per load
// //   if (loadCount > 3 * (addCount + mulCount) && 
// //       uniqueLoadMemrefs.size() <= 2 &&
// //       uniqueStoreMemrefs.size() <= 2) {
// //     pattern = ComputationalPattern::STENCIL;
// //   }
  
// //   // Reduction pattern detection
// //   // - Usually has scf.reduce operation
// //   // - High ratio of adds or maxes
// //   if (numDims >= 1 && 
// //       (addCount > loadCount / 2 || maxCount > loadCount / 2)) {
    
// //     // Check for reduce operations
// //     bool hasReduce = false;
// //     parallelOp.getBody()->walk([&](Operation *op) {
// //       if (isa<scf::ReduceOp>(op)) {
// //         hasReduce = true;
// //       }
// //     });
    
// //     if (hasReduce) {
// //       pattern = ComputationalPattern::REDUCTION;
// //     }
// //   }
  
// //   // Element-wise pattern detection
// //   // - One load, one store per iteration
// //   // - Simple operations (add, mul, etc.)
// //   if (loadCount <= 2 * numDims && 
// //       storeCount <= numDims && 
// //       addCount + mulCount <= 3 * numDims) {
// //     pattern = ComputationalPattern::ELEMENTWISE;
// //   }
  
// //   LLVM_DEBUG(llvm::dbgs() << "Detected computational pattern: " 
// //             << static_cast<int>(pattern) << "\n");
  
// //   return pattern;
// // }

// // // std::vector<std::vector<int64_t>> TileSizeOptimizer::generateTileSizeCandidates(
// // //     const std::vector<LoopInfo> &loopInfos,
// // //     ComputationalPattern pattern) {
  
// // //   std::vector<std::vector<int64_t>> candidatesPerDim;
  
// // //   // Analyze memory access patterns to determine which dimensions are most important
// // //   // This would ideally be done by analyzing memAccesses, but for now we'll use
// // //   // heuristics based on loop position
  
// // //   // For CUDA/GPU optimization, innermost dimensions should be prioritized for coalescing
  
// // //   for (int dimIdx = 0; dimIdx < loopInfos.size(); ++dimIdx) {
// // //     const auto &loopInfo = loopInfos[dimIdx];
// // //     bool isInnermostDim = (dimIdx == loopInfos.size() - 1);
// // //     bool isOutermostDim = (dimIdx == 0);
// // //     std::vector<int64_t> candidatesForDim;
    
// // //     // Basic set to include in all cases - powers of two are generally good
// // //     for (int64_t size = 1; size <= 1024; size *= 2) {
// // //       candidatesForDim.push_back(size);
// // //     }
    
// // //     // Special handling for different dimensions
// // //     if (loopInfo.hasConstantBounds) {
// // //       int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
      
// // //       // If dimension size is small, include it as a candidate
// // //       if (loopRange <= 64) {
// // //         candidatesForDim.push_back(loopRange);
// // //       }
      
// // //       // For small dimensions, consider all divisors
// // //       if (loopRange <= 32) {
// // //         for (int64_t i = 1; i <= loopRange; ++i) {
// // //           if (loopRange % i == 0) {
// // //             candidatesForDim.push_back(i);
// // //           }
// // //         }
// // //       }
      
// // //       // For larger dimensions, include key divisors
// // //       else {
// // //         for (int64_t i = 1; i * i <= loopRange; ++i) {
// // //           if (loopRange % i == 0) {
// // //             candidatesForDim.push_back(i);
// // //             if (i != loopRange / i) {
// // //               candidatesForDim.push_back(loopRange / i);
// // //             }
// // //           }
// // //         }
// // //       }
      
// // //       // Hardware-specific tile size candidates
      
// // //       // For innermost dimension, prioritize coalescing - multiples of warp size
// // //       if (isInnermostDim) {
// // //         for (int i = 1; i <= 4; ++i) {
// // //           int64_t size = hwParams.warpSize * i;
// // //           if (size <= loopRange && size <= hwParams.maxBlockDimX) {
// // //             candidatesForDim.push_back(size);
// // //           }
// // //         }
        
// // //         // For innermost dimension, also try half-warp sizes
// // //         candidatesForDim.push_back(hwParams.warpSize / 2);
        
// // //         // Special case: if innermost dimension is small, include values 
// // //         // that are near multiples of warp size
// // //         if (loopRange < hwParams.warpSize) {
// // //           candidatesForDim.push_back(loopRange);
// // //         } else if (loopRange < hwParams.warpSize * 2) {
// // //           // For sizes between 32-64, try non-standard sizes like 48
// // //           candidatesForDim.push_back(48);
// // //         }
// // //       }
      
// // //       // For middle dimensions, balance between parallelism and data reuse
// // //       if (!isInnermostDim && !isOutermostDim) {
// // //         // Common values that often work well for middle dimensions
// // //         std::vector<int64_t> midDimCandidates = {4, 8, 12, 16, 24, 32};
// // //         for (auto size : midDimCandidates) {
// // //           if (size <= loopRange) {
// // //             candidatesForDim.push_back(size);
// // //           }
// // //         }
        
// // //         // Special handling for convolutional patterns - add kernel-size related values
// // //         std::vector<int64_t> convKernelSizes = {3, 5, 7, 9, 11};
// // //         for (auto kernelSize : convKernelSizes) {
// // //           if (kernelSize <= loopRange) {
// // //             candidatesForDim.push_back(kernelSize);
// // //             // Also add kernel size with halo regions for stencil/conv ops
// // //             if (kernelSize + 2 <= loopRange) {
// // //               candidatesForDim.push_back(kernelSize + 2);
// // //             }
// // //           }
// // //         }
// // //       }
      
// // //       // For outermost dimension, prioritize load balancing across blocks
// // //       if (isOutermostDim) {
// // //         // Try to find values that evenly divide the dimension for load balancing
// // //         for (int div = 2; div <= 16; ++div) {
// // //           if (loopRange % div == 0) {
// // //             int64_t size = loopRange / div;
// // //             if (size <= hwParams.maxBlockDimX) {
// // //               candidatesForDim.push_back(size);
// // //             }
// // //           }
// // //         }
        
// // //         // For small batch dimensions (often the outermost in ML workloads)
// // //         // include typical batch sizes
// // //         std::vector<int64_t> batchSizes = {1, 2, 4, 8, 16, 32, 64, 128};
// // //         for (auto size : batchSizes) {
// // //           if (size <= loopRange && size <= hwParams.maxBlockDimX) {
// // //             candidatesForDim.push_back(size);
// // //           }
// // //         }
// // //       }
      
// // //       // For all dimensions, include values close to sqrt of the range
// // //       // (often a good starting point for balanced tiling)
// // //       int64_t sqrtRange = static_cast<int64_t>(std::sqrt(loopRange));
// // //       candidatesForDim.push_back(sqrtRange);
// // //       if (sqrtRange > 1) candidatesForDim.push_back(sqrtRange - 1);
// // //       if (sqrtRange < loopRange) candidatesForDim.push_back(sqrtRange + 1);
      
// // //       // For dimensions that could be part of matrix multiplication patterns,
// // //       // include sizes that are good for matrix multiply (multiples of 8, 16, 32)
// // //       std::vector<int64_t> matmulSizes = {8, 16, 32, 64};
// // //       for (auto size : matmulSizes) {
// // //         if (size <= loopRange) {
// // //           candidatesForDim.push_back(size);
// // //         }
// // //       }
      
// // //       // For very large dimensions, include some larger tile sizes
// // //       if (loopRange > 1024) {
// // //         std::vector<int64_t> largeSizes = {128, 256, 384, 512, 768, 1024};
// // //         for (auto size : largeSizes) {
// // //           if (size <= loopRange && size <= hwParams.maxBlockDimX) {
// // //             candidatesForDim.push_back(size);
// // //           }
// // //         }
// // //       }
// // //     } else {
// // //       // For non-constant loop bounds, use a comprehensive set of candidates
// // //       candidatesForDim = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 
// // //                          256, 384, 512, 768, 1024};
      
// // //       // Special handling for different dimensions position
// // //       if (isInnermostDim) {
// // //         // For innermost, prioritize multiples of warp size
// // //         candidatesForDim.push_back(hwParams.warpSize);
// // //         candidatesForDim.push_back(hwParams.warpSize * 2);
// // //         candidatesForDim.push_back(hwParams.warpSize * 4);
// // //       }
// // //     }
    
// // //     // Sort and remove duplicates
// // //     std::sort(candidatesForDim.begin(), candidatesForDim.end());
// // //     candidatesForDim.erase(
// // //         std::unique(candidatesForDim.begin(), candidatesForDim.end()),
// // //         candidatesForDim.end());
    
// // //     // Filter candidates based on hardware constraints
// // //     std::vector<int64_t> validCandidates;
// // //     for (int64_t size : candidatesForDim) {
// // //       bool isValid = true;
      
// // //       // Basic constraint: max threads per dimension
// // //       if (dimIdx == 0 && size > hwParams.maxBlockDimX) isValid = false;
// // //       else if (dimIdx == 1 && size > hwParams.maxBlockDimY) isValid = false;
// // //       else if (dimIdx == 2 && size > hwParams.maxBlockDimZ) isValid = false;
      
// // //       // If this is a constant-bound loop, enforce loop range constraint
// // //       if (loopInfo.hasConstantBounds && 
// // //           size > loopInfo.constantUpperBound - loopInfo.constantLowerBound) {
// // //         isValid = false;
// // //       }
      
// // //       if (isValid) {
// // //         validCandidates.push_back(size);
// // //       }
// // //     }
    
// // //     // Edge case: if no valid candidates, add basic defaults
// // //     if (validCandidates.empty()) {
// // //       if (loopInfo.hasConstantBounds) {
// // //         int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
// // //         // Use smallest of: loop range, warp size, or max dimension size
// // //         int64_t defaultSize = std::min({loopRange, 
// // //                                       static_cast<int64_t>(hwParams.warpSize),
// // //                                       isOutermostDim ? 
// // //                                         static_cast<int64_t>(hwParams.maxBlockDimX) : 
// // //                                         static_cast<int64_t>(hwParams.maxThreadsPerBlock)});
// // //         validCandidates.push_back(std::max(static_cast<int64_t>(1), defaultSize));
// // //       } else {
// // //         validCandidates.push_back(isInnermostDim ? hwParams.warpSize : 16);
// // //       }
// // //     }
    
// // //     // Validate combined thread counts when building multi-dimensional candidates
// // //     // This needs to be done in findOptimalTileConfig since we need to check
// // //     // the product across dimensions
    
// // //     candidatesPerDim.push_back(validCandidates);
    
// // //     LLVM_DEBUG(llvm::dbgs() << "Dimension " << dimIdx 
// // //               << " candidates (" << validCandidates.size() << "): ");
// // //     LLVM_DEBUG(for (auto size : validCandidates) {
// // //       llvm::dbgs() << size << " ";
// // //     });
// // //     LLVM_DEBUG(llvm::dbgs() << "\n");
// // //   }
  
// // //   // Pattern-specific adjustments
// // //   switch (pattern) {
// // //     case ComputationalPattern::MATMUL: {
// // //       // For matrix multiplication, optimize for register blocking and shared memory
// // //       if (loopInfos.size() >= 3) {
// // //         // Common GEMM tile sizes for dimensions M, N
// // //         std::vector<int64_t> matmulTileSizes = {16, 32, 64, 128};
        
// // //         // Add these candidates for the first two dimensions (usually M, N)
// // //         for (int i = 0; i < std::min(size_t(2), loopInfos.size()); ++i) {
// // //           for (auto size : matmulTileSizes) {
// // //             if (loopInfos[i].hasConstantBounds) {
// // //               int64_t loopRange = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
// // //               if (size <= loopRange && size <= (i == 0 ? hwParams.maxBlockDimX : hwParams.maxBlockDimY)) {
// // //                 candidatesPerDim[i].push_back(size);
// // //               }
// // //             } else {
// // //               candidatesPerDim[i].push_back(size);
// // //             }
// // //           }
          
// // //           // Sort and remove duplicates
// // //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// // //           candidatesPerDim[i].erase(
// // //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// // //               candidatesPerDim[i].end());
// // //         }
        
// // //         // For K dimension (reduction), use smaller values to control register pressure
// // //         if (loopInfos.size() > 2) {
// // //           int kDim = 2; // Typically the 3rd dimension in GEMM
// // //           std::vector<int64_t> kDimSizes = {4, 8, 16};
          
// // //           if (loopInfos[kDim].hasConstantBounds) {
// // //             int64_t loopRange = loopInfos[kDim].constantUpperBound - loopInfos[kDim].constantLowerBound;
// // //             for (auto size : kDimSizes) {
// // //               if (size <= loopRange) {
// // //                 candidatesPerDim[kDim].push_back(size);
// // //               }
// // //             }
            
// // //             // Sort and remove duplicates
// // //             std::sort(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end());
// // //             candidatesPerDim[kDim].erase(
// // //                 std::unique(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end()),
// // //                 candidatesPerDim[kDim].end());
// // //           }
// // //         }
// // //       }
// // //       break;
// // //     }
    
// // //     case ComputationalPattern::CONV: {
// // //       // For convolutions, optimize spatial dimensions differently
// // //       if (loopInfos.size() >= 4) {
// // //         // Batch dimension (typically outermost)
// // //         // This is often small and best tiled at size 1 or the exact batch size
// // //         int batchDim = 0;
// // //         if (loopInfos[batchDim].hasConstantBounds) {
// // //           int64_t batchSize = loopInfos[batchDim].constantUpperBound - loopInfos[batchDim].constantLowerBound;
// // //           if (batchSize <= 32) {
// // //             // If batch size is small, consider using exact batch size
// // //             candidatesPerDim[batchDim].push_back(batchSize);
// // //           }
// // //         }
        
// // //         // Channel dimensions
// // //         int channelDim = 1;
// // //         if (loopInfos[channelDim].hasConstantBounds) {
// // //           int64_t channels = loopInfos[channelDim].constantUpperBound - loopInfos[channelDim].constantLowerBound;
// // //           // Common channel grouping sizes for channel dimensions
// // //           std::vector<int64_t> channelSizes = {1, 3, 4, 8, 16};
// // //           for (auto size : channelSizes) {
// // //             if (size <= channels) {
// // //               candidatesPerDim[channelDim].push_back(size);
// // //             }
// // //           }
// // //         }
        
// // //         // Spatial dimensions (typically innermost 2 dimensions)
// // //         for (int i = std::max(2, static_cast<int>(loopInfos.size()) - 2); 
// // //              i < loopInfos.size(); ++i) {
// // //           if (loopInfos[i].hasConstantBounds) {
// // //             int64_t spatialDim = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
            
// // //             // For spatial dimensions, include values that match common tensor core sizes
// // //             std::vector<int64_t> spatialSizes = {4, 8, 16, 32};
// // //             for (auto size : spatialSizes) {
// // //               if (size <= spatialDim) {
// // //                 candidatesPerDim[i].push_back(size);
// // //               }
// // //             }
            
// // //             // Add sizes based on common kernel dimensions plus padding
// // //             std::vector<int64_t> kernelSizes = {3, 5, 7};
// // //             for (auto kSize : kernelSizes) {
// // //               for (int padding = 0; padding <= 2; padding++) {
// // //                 int64_t tileSize = kSize + 2 * padding;
// // //                 if (tileSize <= spatialDim) {
// // //                   candidatesPerDim[i].push_back(tileSize);
// // //                 }
// // //               }
// // //             }
// // //           }
          
// // //           // Sort and remove duplicates
// // //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// // //           candidatesPerDim[i].erase(
// // //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// // //               candidatesPerDim[i].end());
// // //         }
// // //       }
// // //       break;
// // //     }
    
// // //     case ComputationalPattern::REDUCTION: {
// // //       // For reductions, the reduction dimensions should use larger tiles
// // //       if (!loopInfos.empty()) {
// // //         // Assume last dimension is reduction dimension (common pattern)
// // //         int redDim = loopInfos.size() - 1;
        
// // //         if (loopInfos[redDim].hasConstantBounds) {
// // //           int64_t redRange = loopInfos[redDim].constantUpperBound - loopInfos[redDim].constantLowerBound;
          
// // //           // For reduction, try larger tiles to reduce synchronization overhead
// // //           std::vector<int64_t> redSizes = {32, 64, 128, 256, 512};
// // //           for (auto size : redSizes) {
// // //             if (size <= redRange) {
// // //               candidatesPerDim[redDim].push_back(size);
// // //             }
// // //           }
          
// // //           // Also try full reduction in one tile
// // //           candidatesPerDim[redDim].push_back(redRange);
// // //         }
        
// // //         // Sort and remove duplicates
// // //         std::sort(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end());
// // //         candidatesPerDim[redDim].erase(
// // //             std::unique(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end()),
// // //             candidatesPerDim[redDim].end());
// // //       }
// // //       break;
// // //     }
    
// // //     case ComputationalPattern::STENCIL: {
// // //       // For stencils, consider halo regions
// // //       for (int i = 0; i < loopInfos.size(); ++i) {
// // //         if (loopInfos[i].hasConstantBounds) {
// // //           // Add tile sizes that account for typical stencil radii (1, 2, 3)
// // //           for (int radius = 1; radius <= 3; radius++) {
// // //             // For a radius-R stencil, a good tile size is often a multiple of 16 or 32, plus 2*R
// // //             for (int base : {16, 32}) {
// // //               int64_t tileSize = base + 2 * radius;
// // //               if (tileSize <= loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound) {
// // //                 candidatesPerDim[i].push_back(tileSize);
// // //               }
// // //             }
// // //           }
          
// // //           // Sort and remove duplicates
// // //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// // //           candidatesPerDim[i].erase(
// // //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// // //               candidatesPerDim[i].end());
// // //         }
// // //       }
// // //       break;
// // //     }
    
// // //     case ComputationalPattern::ELEMENTWISE: {
// // //       // For element-wise, prioritize coalesced memory access for innermost dim
// // //       if (!loopInfos.empty()) {
// // //         int innerDim = loopInfos.size() - 1;
        
// // //         // For innermost dimension, focus on warp and multiple-warp sizes
// // //         for (int warps = 1; warps <= 8; warps *= 2) {
// // //           int64_t size = hwParams.warpSize * warps;
// // //           if (loopInfos[innerDim].hasConstantBounds) {
// // //             int64_t dimSize = loopInfos[innerDim].constantUpperBound - loopInfos[innerDim].constantLowerBound;
// // //             if (size <= dimSize) {
// // //               candidatesPerDim[innerDim].push_back(size);
// // //             }
// // //           } else {
// // //             candidatesPerDim[innerDim].push_back(size);
// // //           }
// // //         }
        
// // //         // Sort and remove duplicates
// // //         std::sort(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end());
// // //         candidatesPerDim[innerDim].erase(
// // //             std::unique(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end()),
// // //             candidatesPerDim[innerDim].end());
// // //       }
// // //       break;
// // //     }
    
// // //     case ComputationalPattern::GENERIC:
// // //     default:
// // //       // No additional adjustments for generic pattern
// // //       break;
// // //   }
  
// // //   return candidatesPerDim;
// // // }

// // std::vector<std::vector<int64_t>> TileSizeOptimizer::generateTileSizeCandidates(
// //     const std::vector<LoopInfo> &loopInfos,
// //     ComputationalPattern pattern) {
  
// //   std::vector<std::vector<int64_t>> candidatesPerDim;
  
// //   // Helper lambda to check if a tile size can evenly divide the iteration count
// //   auto canDivideEvenly = [](const LoopInfo &loopInfo, int64_t tileSize) -> bool {
// //     if (!loopInfo.hasConstantBounds) {
// //       return true; // Cannot check for non-constant bounds, allow all candidates
// //     }
    
// //     int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
// //     int64_t iterationCount = loopRange / loopInfo.constantStep;
    
// //     return (iterationCount % tileSize == 0);
// //   };
  
// //   for (int dimIdx = 0; dimIdx < loopInfos.size(); ++dimIdx) {
// //     const auto &loopInfo = loopInfos[dimIdx];
// //     bool isInnermostDim = (dimIdx == loopInfos.size() - 1);
// //     bool isOutermostDim = (dimIdx == 0);
// //     std::vector<int64_t> candidatesForDim;
    
// //     // Basic set - powers of two that can evenly divide
// //     for (int64_t size = 1; size <= 1024; size *= 2) {
// //       if (canDivideEvenly(loopInfo, size)) {
// //         candidatesForDim.push_back(size);
// //       }
// //     }
    
// //     // Special handling for different dimensions
// //     if (loopInfo.hasConstantBounds) {
// //       int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
// //       int64_t iterationCount = loopRange / loopInfo.constantStep;
      
// //       // If dimension size is small, include it as a candidate (it always divides itself)
// //       if (loopRange <= 64) {
// //         candidatesForDim.push_back(iterationCount);
// //       }
      
// //       // For small dimensions, consider all divisors (these are guaranteed to divide evenly)
// //       if (iterationCount <= 32) {
// //         for (int64_t i = 1; i <= iterationCount; ++i) {
// //           if (iterationCount % i == 0) {
// //             candidatesForDim.push_back(i);
// //           }
// //         }
// //       }
      
// //       // For larger dimensions, include key divisors (these are guaranteed to divide evenly)
// //       else {
// //         for (int64_t i = 1; i * i <= iterationCount; ++i) {
// //           if (iterationCount % i == 0) {
// //             candidatesForDim.push_back(i);
// //             if (i != iterationCount / i) {
// //               candidatesForDim.push_back(iterationCount / i);
// //             }
// //           }
// //         }
// //       }
      
// //       // Hardware-specific tile size candidates
      
// //       // For innermost dimension, prioritize coalescing - multiples of warp size
// //       if (isInnermostDim) {
// //         for (int i = 1; i <= 4; ++i) {
// //           int64_t size = hwParams.warpSize * i;
// //           if (size <= loopRange && size <= hwParams.maxBlockDimX && canDivideEvenly(loopInfo, size)) {
// //             candidatesForDim.push_back(size);
// //           }
// //         }
        
// //         // For innermost dimension, also try half-warp sizes
// //         int64_t halfWarpSize = hwParams.warpSize / 2;
// //         if (canDivideEvenly(loopInfo, halfWarpSize)) {
// //           candidatesForDim.push_back(halfWarpSize);
// //         }
        
// //         // Special case: if innermost dimension is small, include values 
// //         // that are near multiples of warp size but still divide evenly
// //         if (loopRange < hwParams.warpSize) {
// //           // iterationCount already added above
// //         } else if (loopRange < hwParams.warpSize * 2) {
// //           // For sizes between 32-64, try non-standard sizes like 48 if they divide evenly
// //           if (canDivideEvenly(loopInfo, 48)) {
// //             candidatesForDim.push_back(48);
// //           }
// //         }
// //       }
      
// //       // For middle dimensions, balance between parallelism and data reuse
// //       if (!isInnermostDim && !isOutermostDim) {
// //         // Common values that often work well for middle dimensions
// //         std::vector<int64_t> midDimCandidates = {4, 8, 12, 16, 24, 32};
// //         for (auto size : midDimCandidates) {
// //           if (size <= loopRange && canDivideEvenly(loopInfo, size)) {
// //             candidatesForDim.push_back(size);
// //           }
// //         }
        
// //         // Special handling for convolutional patterns - add kernel-size related values
// //         std::vector<int64_t> convKernelSizes = {3, 5, 7, 9, 11};
// //         for (auto kernelSize : convKernelSizes) {
// //           if (kernelSize <= loopRange && canDivideEvenly(loopInfo, kernelSize)) {
// //             candidatesForDim.push_back(kernelSize);
// //             // Also add kernel size with halo regions for stencil/conv ops
// //             int64_t sizeWithHalo = kernelSize + 2;
// //             if (sizeWithHalo <= loopRange && canDivideEvenly(loopInfo, sizeWithHalo)) {
// //               candidatesForDim.push_back(sizeWithHalo);
// //             }
// //           }
// //         }
// //       }
      
// //       // For outermost dimension, prioritize load balancing across blocks
// //       if (isOutermostDim) {
// //         // Try to find values that evenly divide the dimension for load balancing
// //         // (these are guaranteed to divide evenly since we're computing divisors)
// //         for (int div = 2; div <= 16; ++div) {
// //           if (iterationCount % div == 0) {
// //             int64_t size = iterationCount / div;
// //             if (size <= hwParams.maxBlockDimX) {
// //               candidatesForDim.push_back(size);
// //             }
// //           }
// //         }
        
// //         // For small batch dimensions (often the outermost in ML workloads)
// //         // include typical batch sizes that divide evenly
// //         std::vector<int64_t> batchSizes = {1, 2, 4, 8, 16, 32, 64, 128};
// //         for (auto size : batchSizes) {
// //           if (size <= loopRange && size <= hwParams.maxBlockDimX && canDivideEvenly(loopInfo, size)) {
// //             candidatesForDim.push_back(size);
// //           }
// //         }
// //       }
      
// //       // For all dimensions, include values close to sqrt of the iteration count
// //       // (often a good starting point for balanced tiling)
// //       int64_t sqrtIterCount = static_cast<int64_t>(std::sqrt(iterationCount));
      
// //       // Try sqrt and nearby values that divide evenly
// //       for (int64_t candidate = std::max(static_cast<int64_t>(1), sqrtIterCount - 2); 
// //            candidate <= sqrtIterCount + 2 && candidate <= iterationCount; 
// //            ++candidate) {
// //         if (canDivideEvenly(loopInfo, candidate)) {
// //           candidatesForDim.push_back(candidate);
// //         }
// //       }
      
// //       // For dimensions that could be part of matrix multiplication patterns,
// //       // include sizes that are good for matrix multiply (multiples of 8, 16, 32)
// //       std::vector<int64_t> matmulSizes = {8, 16, 32, 64};
// //       for (auto size : matmulSizes) {
// //         if (size <= loopRange && canDivideEvenly(loopInfo, size)) {
// //           candidatesForDim.push_back(size);
// //         }
// //       }
      
// //       // For very large dimensions, include some larger tile sizes
// //       if (loopRange > 1024) {
// //         std::vector<int64_t> largeSizes = {128, 256, 384, 512, 768, 1024};
// //         for (auto size : largeSizes) {
// //           if (size <= loopRange && size <= hwParams.maxBlockDimX && canDivideEvenly(loopInfo, size)) {
// //             candidatesForDim.push_back(size);
// //           }
// //         }
// //       }
// //     } else {
// //       // For non-constant loop bounds, use a comprehensive set of candidates
// //       // Cannot check divisibility, so include all reasonable candidates
// //       candidatesForDim = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 
// //                          256, 384, 512, 768, 1024};
      
// //       // Special handling for different dimensions position
// //       if (isInnermostDim) {
// //         // For innermost, prioritize multiples of warp size
// //         candidatesForDim.push_back(hwParams.warpSize);
// //         candidatesForDim.push_back(hwParams.warpSize * 2);
// //         candidatesForDim.push_back(hwParams.warpSize * 4);
// //       }
// //     }
    
// //     // Sort and remove duplicates
// //     std::sort(candidatesForDim.begin(), candidatesForDim.end());
// //     candidatesForDim.erase(
// //         std::unique(candidatesForDim.begin(), candidatesForDim.end()),
// //         candidatesForDim.end());
    
// //     // Filter candidates based on hardware constraints
// //     std::vector<int64_t> validCandidates;
// //     for (int64_t size : candidatesForDim) {
// //       bool isValid = true;
      
// //       // Basic constraint: max threads per dimension
// //       if (dimIdx == 0 && size > hwParams.maxBlockDimX) isValid = false;
// //       else if (dimIdx == 1 && size > hwParams.maxBlockDimY) isValid = false;
// //       else if (dimIdx == 2 && size > hwParams.maxBlockDimZ) isValid = false;
      
// //       // If this is a constant-bound loop, enforce loop range constraint
// //       if (loopInfo.hasConstantBounds && 
// //           size > loopInfo.constantUpperBound - loopInfo.constantLowerBound) {
// //         isValid = false;
// //       }
      
// //       if (isValid) {
// //         validCandidates.push_back(size);
// //       }
// //     }
    
// //     // Edge case: if no valid candidates, add basic defaults that divide evenly
// //     if (validCandidates.empty()) {
// //       if (loopInfo.hasConstantBounds) {
// //         int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
// //         int64_t iterationCount = loopRange / loopInfo.constantStep;
        
// //         // Use 1 as fallback (always divides evenly)
// //         validCandidates.push_back(1);
        
// //         // Try to add a reasonable size that divides evenly
// //         for (int64_t candidate : {2, 4, 8, 16, 32}) {
// //           if (candidate <= iterationCount && canDivideEvenly(loopInfo, candidate)) {
// //             validCandidates.push_back(candidate);
// //             break;
// //           }
// //         }
// //       } else {
// //         validCandidates.push_back(isInnermostDim ? hwParams.warpSize : 16);
// //       }
// //     }
    
// //     candidatesPerDim.push_back(validCandidates);
    
// //     LLVM_DEBUG(llvm::dbgs() << "Dimension " << dimIdx 
// //               << " candidates (" << validCandidates.size() << "): ");
// //     LLVM_DEBUG(for (auto size : validCandidates) {
// //       llvm::dbgs() << size << " ";
// //     });
// //     LLVM_DEBUG(llvm::dbgs() << "\n");
// //   }
  
// //   // Pattern-specific adjustments - also need to check divisibility
// //   switch (pattern) {
// //     case ComputationalPattern::MATMUL: {
// //       // For matrix multiplication, optimize for register blocking and shared memory
// //       if (loopInfos.size() >= 3) {
// //         // Common GEMM tile sizes for dimensions M, N
// //         std::vector<int64_t> matmulTileSizes = {16, 32, 64, 128};
        
// //         // Add these candidates for the first two dimensions (usually M, N)
// //         for (int i = 0; i < std::min(size_t(2), loopInfos.size()); ++i) {
// //           for (auto size : matmulTileSizes) {
// //             if (loopInfos[i].hasConstantBounds) {
// //               int64_t loopRange = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
// //               if (size <= loopRange && 
// //                   size <= (i == 0 ? hwParams.maxBlockDimX : hwParams.maxBlockDimY) &&
// //                   canDivideEvenly(loopInfos[i], size)) {
// //                 candidatesPerDim[i].push_back(size);
// //               }
// //             } else {
// //               candidatesPerDim[i].push_back(size);
// //             }
// //           }
          
// //           // Sort and remove duplicates
// //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// //           candidatesPerDim[i].erase(
// //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// //               candidatesPerDim[i].end());
// //         }
        
// //         // For K dimension (reduction), use smaller values to control register pressure
// //         if (loopInfos.size() > 2) {
// //           int kDim = 2; // Typically the 3rd dimension in GEMM
// //           std::vector<int64_t> kDimSizes = {4, 8, 16};
          
// //           if (loopInfos[kDim].hasConstantBounds) {
// //             int64_t loopRange = loopInfos[kDim].constantUpperBound - loopInfos[kDim].constantLowerBound;
// //             for (auto size : kDimSizes) {
// //               if (size <= loopRange && canDivideEvenly(loopInfos[kDim], size)) {
// //                 candidatesPerDim[kDim].push_back(size);
// //               }
// //             }
            
// //             // Sort and remove duplicates
// //             std::sort(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end());
// //             candidatesPerDim[kDim].erase(
// //                 std::unique(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end()),
// //                 candidatesPerDim[kDim].end());
// //           }
// //         }
// //       }
// //       break;
// //     }
    
// //     case ComputationalPattern::CONV: {
// //       // For convolutions, optimize spatial dimensions differently
// //       if (loopInfos.size() >= 4) {
// //         // Batch dimension (typically outermost)
// //         int batchDim = 0;
// //         if (loopInfos[batchDim].hasConstantBounds) {
// //           int64_t batchRange = loopInfos[batchDim].constantUpperBound - loopInfos[batchDim].constantLowerBound;
// //           int64_t batchIterCount = batchRange / loopInfos[batchDim].constantStep;
// //           if (batchIterCount <= 32) {
// //             // If batch iteration count is small, consider using exact batch iteration count
// //             candidatesPerDim[batchDim].push_back(batchIterCount);
// //           }
// //         }
        
// //         // Channel dimensions
// //         int channelDim = 1;
// //         if (loopInfos[channelDim].hasConstantBounds) {
// //           int64_t channelRange = loopInfos[channelDim].constantUpperBound - loopInfos[channelDim].constantLowerBound;
// //           int64_t channelIterCount = channelRange / loopInfos[channelDim].constantStep;
// //           // Common channel grouping sizes for channel dimensions
// //           std::vector<int64_t> channelSizes = {1, 3, 4, 8, 16};
// //           for (auto size : channelSizes) {
// //             if (size <= channelIterCount && canDivideEvenly(loopInfos[channelDim], size)) {
// //               candidatesPerDim[channelDim].push_back(size);
// //             }
// //           }
// //         }
        
// //         // Spatial dimensions (typically innermost 2 dimensions)
// //         for (int i = std::max(2, static_cast<int>(loopInfos.size()) - 2); 
// //              i < loopInfos.size(); ++i) {
// //           if (loopInfos[i].hasConstantBounds) {
// //             int64_t spatialRange = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
// //             int64_t spatialIterCount = spatialRange / loopInfos[i].constantStep;
            
// //             // For spatial dimensions, include values that match common tensor core sizes
// //             std::vector<int64_t> spatialSizes = {4, 8, 16, 32};
// //             for (auto size : spatialSizes) {
// //               if (size <= spatialIterCount && canDivideEvenly(loopInfos[i], size)) {
// //                 candidatesPerDim[i].push_back(size);
// //               }
// //             }
            
// //             // Add sizes based on common kernel dimensions plus padding
// //             std::vector<int64_t> kernelSizes = {3, 5, 7};
// //             for (auto kSize : kernelSizes) {
// //               for (int padding = 0; padding <= 2; padding++) {
// //                 int64_t tileSize = kSize + 2 * padding;
// //                 if (tileSize <= spatialIterCount && canDivideEvenly(loopInfos[i], tileSize)) {
// //                   candidatesPerDim[i].push_back(tileSize);
// //                 }
// //               }
// //             }
// //           }
          
// //           // Sort and remove duplicates
// //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// //           candidatesPerDim[i].erase(
// //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// //               candidatesPerDim[i].end());
// //         }
// //       }
// //       break;
// //     }
    
// //     case ComputationalPattern::REDUCTION: {
// //       // For reductions, the reduction dimensions should use larger tiles
// //       if (!loopInfos.empty()) {
// //         // Assume last dimension is reduction dimension (common pattern)
// //         int redDim = loopInfos.size() - 1;
        
// //         if (loopInfos[redDim].hasConstantBounds) {
// //           int64_t redRange = loopInfos[redDim].constantUpperBound - loopInfos[redDim].constantLowerBound;
// //           int64_t redIterCount = redRange / loopInfos[redDim].constantStep;
          
// //           // For reduction, try larger tiles to reduce synchronization overhead
// //           std::vector<int64_t> redSizes = {32, 64, 128, 256, 512};
// //           for (auto size : redSizes) {
// //             if (size <= redIterCount && canDivideEvenly(loopInfos[redDim], size)) {
// //               candidatesPerDim[redDim].push_back(size);
// //             }
// //           }
          
// //           // Also try full reduction in one tile (always divides evenly)
// //           candidatesPerDim[redDim].push_back(redIterCount);
// //         }
        
// //         // Sort and remove duplicates
// //         std::sort(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end());
// //         candidatesPerDim[redDim].erase(
// //             std::unique(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end()),
// //             candidatesPerDim[redDim].end());
// //       }
// //       break;
// //     }
    
// //     case ComputationalPattern::STENCIL: {
// //       // For stencils, consider halo regions
// //       for (int i = 0; i < loopInfos.size(); ++i) {
// //         if (loopInfos[i].hasConstantBounds) {
// //           int64_t stencilRange = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
// //           int64_t stencilIterCount = stencilRange / loopInfos[i].constantStep;
          
// //           // Add tile sizes that account for typical stencil radii (1, 2, 3)
// //           for (int radius = 1; radius <= 3; radius++) {
// //             // For a radius-R stencil, a good tile size is often a multiple of 16 or 32, plus 2*R
// //             for (int base : {16, 32}) {
// //               int64_t tileSize = base + 2 * radius;
// //               if (tileSize <= stencilIterCount && canDivideEvenly(loopInfos[i], tileSize)) {
// //                 candidatesPerDim[i].push_back(tileSize);
// //               }
// //             }
// //           }
          
// //           // Sort and remove duplicates
// //           std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
// //           candidatesPerDim[i].erase(
// //               std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
// //               candidatesPerDim[i].end());
// //         }
// //       }
// //       break;
// //     }
    
// //     case ComputationalPattern::ELEMENTWISE: {
// //       // For element-wise, prioritize coalesced memory access for innermost dim
// //       if (!loopInfos.empty()) {
// //         int innerDim = loopInfos.size() - 1;
        
// //         // For innermost dimension, focus on warp and multiple-warp sizes
// //         for (int warps = 1; warps <= 8; warps *= 2) {
// //           int64_t size = hwParams.warpSize * warps;
// //           if (loopInfos[innerDim].hasConstantBounds) {
// //             int64_t dimRange = loopInfos[innerDim].constantUpperBound - loopInfos[innerDim].constantLowerBound;
// //             int64_t dimIterCount = dimRange / loopInfos[innerDim].constantStep;
// //             if (size <= dimIterCount && canDivideEvenly(loopInfos[innerDim], size)) {
// //               candidatesPerDim[innerDim].push_back(size);
// //             }
// //           } else {
// //             candidatesPerDim[innerDim].push_back(size);
// //           }
// //         }
        
// //         // Sort and remove duplicates
// //         std::sort(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end());
// //         candidatesPerDim[innerDim].erase(
// //             std::unique(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end()),
// //             candidatesPerDim[innerDim].end());
// //       }
// //       break;
// //     }
    
// //     case ComputationalPattern::GENERIC:
// //     default:
// //       // No additional adjustments for generic pattern
// //       break;
// //   }
  
// //   return candidatesPerDim;
// // }

// // bool TileSizeOptimizer::isValidTileConfig(const CompleteTileConfig &config) {
// //   // 检查每个维度的tile大小是否有效
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     // 确保tile大小在合理范围内
// //     if (tileConfig.tileSize < 1 || tileConfig.tileSize > hwParams.maxThreadsPerBlock) {
// //       return false;
// //     }
// //   }
  
// //   return true;
// // }

// // CompleteTileConfig TileSizeOptimizer::findOptimalTileConfig(
// //     const std::vector<LoopInfo> &loopInfos,
// //     const std::vector<MemoryAccessInfo> &memAccesses,
// //     const std::vector<ComputationInfo> &computations,
// //     const std::vector<std::vector<int64_t>> &tileSizeCandidates) {
  
// //   const int numDimensions = loopInfos.size();
  
// //   // 初始化DP表
// //   // dp[d][i] = 当第d维使用索引i对应的配置，且前d-1维使用最优配置时的最佳得分
// //   std::vector<std::vector<float>> dp(numDimensions);
// //   std::vector<std::vector<int>> prevConfig(numDimensions);
  
// //   // 保存最佳配置
// //   std::vector<std::vector<CompleteTileConfig>> bestConfigs(numDimensions);
  
// //   // 初始化第一维
// //   dp[0].resize(tileSizeCandidates[0].size(), -std::numeric_limits<float>::infinity());
// //   prevConfig[0].resize(tileSizeCandidates[0].size(), -1);
// //   bestConfigs[0].resize(tileSizeCandidates[0].size());
  
// //   for (size_t i = 0; i < tileSizeCandidates[0].size(); ++i) {
// //     CompleteTileConfig config;
    
// //     // 创建第一维配置
// //     TileConfig dimConfig;
// //     dimConfig.tileSize = tileSizeCandidates[0][i];
    
// //     config.perDimConfig.push_back(dimConfig);
    
// //     // 检查有效性并评估
// //     if (isValidTileConfig(config)) {
// //       // 评估单维度性能
// //       float score = evaluateConfig(config, loopInfos, memAccesses, computations);
// //       dp[0][i] = score;
// //       bestConfigs[0][i] = config;
// //     }
// //   }
  
// //   // 填充DP表，处理后续维度
// //   for (int dim = 1; dim < numDimensions; ++dim) {
// //     dp[dim].resize(tileSizeCandidates[dim].size(), -std::numeric_limits<float>::infinity());
// //     prevConfig[dim].resize(tileSizeCandidates[dim].size(), -1);
// //     bestConfigs[dim].resize(tileSizeCandidates[dim].size());
    
// //     for (size_t i = 0; i < tileSizeCandidates[dim].size(); ++i) {
// //       float bestScore = -std::numeric_limits<float>::infinity();
// //       int bestPrevIdx = -1;
      
// //       // 尝试与前一维度的每种配置组合
// //       for (size_t j = 0; j < dp[dim-1].size(); ++j) {
// //         // 跳过无效的前一配置
// //         if (dp[dim-1][j] == -std::numeric_limits<float>::infinity()) {
// //           continue;
// //         }
        
// //         // 创建新配置，扩展前一维度的最佳配置
// //         CompleteTileConfig newConfig = bestConfigs[dim-1][j];
        
// //         // 添加当前维度
// //         TileConfig dimConfig;
// //         dimConfig.tileSize = tileSizeCandidates[dim][i];
        
// //         newConfig.perDimConfig.push_back(dimConfig);
        
// //         // 检查有效性
// //         if (isValidTileConfig(newConfig)) {
// //           // 评估组合性能
// //           float score = evaluateConfig(newConfig, loopInfos, memAccesses, computations);
          
// //           if (score > bestScore) {
// //             bestScore = score;
// //             bestPrevIdx = j;
// //             bestConfigs[dim][i] = newConfig;
// //           }
// //         }
// //       }
      
// //       dp[dim][i] = bestScore;
// //       prevConfig[dim][i] = bestPrevIdx;
// //     }
// //   }
  
// //   // 在最终维度找到最佳配置
// //   float bestFinalScore = -std::numeric_limits<float>::infinity();
// //   int bestFinalIdx = -1;
  
// //   for (size_t i = 0; i < dp[numDimensions-1].size(); ++i) {
// //     if (dp[numDimensions-1][i] > bestFinalScore) {
// //       bestFinalScore = dp[numDimensions-1][i];
// //       bestFinalIdx = i;
// //     }
// //   }
  
// //   // 返回最佳配置
// //   CompleteTileConfig optimalConfig;
  
// //   if (bestFinalIdx != -1) {
// //     optimalConfig = bestConfigs[numDimensions-1][bestFinalIdx];
// //     optimalConfig.overallPerformanceScore = bestFinalScore;
// //   } else {
// //     // 如果没有找到有效配置，使用默认配置
// //     optimalConfig.perDimConfig.resize(numDimensions);
// //     for (int i = 0; i < numDimensions; ++i) {
// //       optimalConfig.perDimConfig[i].tileSize = (i == 0 || i == 1) ? 32 : 1;
// //     }
// //     optimalConfig.overallPerformanceScore = 0.0f;
// //   }
  
// //   return optimalConfig;
// // }

// // float TileSizeOptimizer::evaluateConfig(
// //     const CompleteTileConfig &config,
// //     const std::vector<LoopInfo> &loopInfos,
// //     const std::vector<MemoryAccessInfo> &memAccesses,
// //     const std::vector<ComputationInfo> &computations) {
  
// //   // Calculate individual performance factors with improved models
// //   float arithmeticIntensityScore = evaluateArithmeticIntensity(config, computations, memAccesses);
// //   float occupancyScore = evaluateOccupancy(config, computations);
// //   float memoryEfficiencyScore = evaluateMemoryEfficiency(config, loopInfos, memAccesses);
// //   float loadBalancingScore = evaluateLoadBalancing(config, loopInfos);
// //   float dataReuseScore = evaluateDataReuse(config, loopInfos, memAccesses);
  
// //   // Weights for different architectures (could be adjusted based on target)
// //   const float w_roofline = 0.35f;   // Arithmetic intensity importance
// //   const float w_occupancy = 0.25f;  // Resource utilization importance
// //   const float w_memory = 0.25f;     // Memory access pattern importance
// //   const float w_balance = 0.05f;    // Load balancing importance
// //   const float w_reuse = 0.10f;      // Data reuse importance
  
// //   // Combined score with all factors
// //   float totalScore = (
// //       w_roofline * arithmeticIntensityScore +
// //       w_occupancy * occupancyScore +
// //       w_memory * memoryEfficiencyScore +
// //       w_balance * loadBalancingScore +
// //       w_reuse * dataReuseScore
// //   ) / (w_roofline + w_occupancy + w_memory + w_balance + w_reuse);
  
// //   LLVM_DEBUG(llvm::dbgs() << "Tile Config: ";
// //              for (const auto &dim : config.perDimConfig) {
// //                llvm::dbgs() << dim.tileSize << " ";
// //              }
// //              llvm::dbgs() << "\n  AI Score: " << arithmeticIntensityScore
// //                          << ", Occ: " << occupancyScore
// //                          << ", Mem: " << memoryEfficiencyScore
// //                          << ", Bal: " << loadBalancingScore
// //                          << ", Reuse: " << dataReuseScore
// //                          << ", Total: " << totalScore << "\n");
  
// //   return totalScore;
// // }

// // float TileSizeOptimizer::evaluateArithmeticIntensity(
// //     const CompleteTileConfig &config,
// //     const std::vector<ComputationInfo> &computations,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // Calculate total tile size
// //   int64_t totalTileSize = 1;
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     totalTileSize *= tileConfig.tileSize;
// //   }
  
// //   // Calculate total operations with operation-specific weights
// //   int64_t totalOps = 0;
// //   for (const auto &comp : computations) {
// //     float opWeight = 1.0;
    
// //     // Assign different weights to different operations
// //     if (isa<arith::MulFOp>(comp.op) || isa<arith::MulIOp>(comp.op)) {
// //       opWeight = 1.5;  // Multiplications typically more expensive
// //     } else if (isa<arith::DivFOp>(comp.op) || isa<arith::DivSIOp>(comp.op) || 
// //                isa<arith::DivUIOp>(comp.op)) {
// //       opWeight = 3.0;  // Divisions are very expensive
// //     }
    
// //     totalOps += comp.opCount * opWeight;
// //   }
  
// //   // Estimate memory transactions with cache effects
// //   int64_t totalMemoryTransactions = 0;
// //   llvm::DenseMap<Value, float> memrefAccessCount;
  
// //   for (const auto &access : memAccesses) {
// //     // Count accesses per memref to model cache effects
// //     memrefAccessCount[access.memref] += 1.0;
// //   }
  
// //   // Calculate memory transactions with cache effects
// //   for (const auto &pair : memrefAccessCount) {
// //     Value memref = pair.first;
// //     float accessCount = pair.second;
    
// //     // Find the element size for this memref
// //     int elementSize = 4;  // Default to 4 bytes
// //     for (const auto &access : memAccesses) {
// //       if (access.memref == memref) {
// //         elementSize = access.dataTypeSizeInBytes;
// //         break;
// //       }
// //     }
    
// //     // Model cache behavior: first access is full cost, subsequent accesses benefit from cache
// //     float cacheFactor = std::min(1.0f, 1.0f / std::sqrt(accessCount));
// //     float effectiveBytes = totalTileSize * elementSize * cacheFactor;
    
// //     totalMemoryTransactions += static_cast<int64_t>(effectiveBytes);
// //   }
  
// //   // Avoid division by zero
// //   if (totalMemoryTransactions == 0) {
// //     return 0.5f;  // Default medium score
// //   }
  
// //   // Calculate arithmetic intensity (operations per byte)
// //   float arithmeticIntensity = static_cast<float>(totalOps) / totalMemoryTransactions;
  
// //   // Apply Roofline model with realistic performance curve
// //   float peakComputeThroughput = hwParams.peakComputePerformance;
// //   float peakMemoryBandwidth = hwParams.memoryBandwidth;
  
// //   // Calculate ridge point (arithmetic intensity where compute = memory bandwidth)
// //   float ridgePoint = peakComputeThroughput / peakMemoryBandwidth;
  
// //   // Calculate attainable performance as percentage of peak
// //   float attainablePerformance;
// //   if (arithmeticIntensity < ridgePoint) {
// //     // Memory-bound region
// //     attainablePerformance = arithmeticIntensity * peakMemoryBandwidth / peakComputeThroughput;
    
// //     // Add penalty for very low arithmetic intensity (heavy memory bottleneck)
// //     if (arithmeticIntensity < 0.1f * ridgePoint) {
// //       attainablePerformance *= 0.8f;
// //     }
// //   } else {
// //     // Compute-bound region
// //     attainablePerformance = 1.0f;
    
// //     // Add bonus for balanced computation (near ridge point)
// //     if (arithmeticIntensity < 2.0f * ridgePoint) {
// //       attainablePerformance *= 1.1f;
// //     }
// //   }
  
// //   // Normalize to [0,1] range
// //   return std::min(1.0f, attainablePerformance);
// // }

// // float TileSizeOptimizer::evaluateOccupancy(
// //     const CompleteTileConfig &config,
// //     const std::vector<ComputationInfo> &computations) {
  
// //   // Calculate total threads per block
// //   int64_t totalThreadsPerBlock = 1;
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     totalThreadsPerBlock *= tileConfig.tileSize;
// //   }
  
// //   // Check if thread count exceeds maximum
// //   if (totalThreadsPerBlock > hwParams.maxThreadsPerBlock) {
// //     return 0.0f;  // Invalid configuration
// //   }
  
// //   // Better register usage estimation based on operation types
// //   int registersPerThread = estimateRegistersPerThread(computations);
  
// //   // Better shared memory usage estimation
// //   int sharedMemoryPerBlock = 0;
// //   // Could be enhanced with more detailed analysis
  
// //   // Calculate theoretical occupancy limited by different factors
  
// //   // Threads per SM limit
// //   int maxBlocksPerSM_byThreadCount = hwParams.maxThreadsPerSM / totalThreadsPerBlock;
  
// //   // Registers per SM limit
// //   int maxBlocksPerSM_byRegisters = hwParams.maxRegistersPerSM / 
// //                                  (registersPerThread * totalThreadsPerBlock);
  
// //   // Shared memory per SM limit
// //   int maxBlocksPerSM_bySharedMem = (sharedMemoryPerBlock > 0) ?
// //                                  hwParams.maxSharedMemoryPerSM / sharedMemoryPerBlock :
// //                                  hwParams.maxBlocksPerSM;
  
// //   // Hardware blocks per SM limit
// //   int maxBlocksPerSM = std::min({
// //     hwParams.maxBlocksPerSM,
// //     maxBlocksPerSM_byThreadCount,
// //     maxBlocksPerSM_byRegisters,
// //     maxBlocksPerSM_bySharedMem
// //   });
  
// //   // Calculate occupancy as fraction of maximum possible warps
// //   int warpsPerBlock = (totalThreadsPerBlock + hwParams.warpSize - 1) / hwParams.warpSize;
// //   int activeWarps = maxBlocksPerSM * warpsPerBlock;
// //   int maxWarpsPerSM = hwParams.maxThreadsPerSM / hwParams.warpSize;
  
// //   float occupancy = static_cast<float>(activeWarps) / maxWarpsPerSM;
  
// //   // Apply occupancy-performance curve
// //   // Research shows that ~70% occupancy is often sufficient for good performance
// //   float normalizedOccupancy;
// //   if (occupancy < 0.2f) {
// //     // Very low occupancy is bad
// //     normalizedOccupancy = occupancy * 2.5f;
// //   } else if (occupancy < 0.7f) {
// //     // Medium occupancy - linear improvement
// //     normalizedOccupancy = 0.5f + (occupancy - 0.2f) * 0.7f;
// //   } else {
// //     // Diminishing returns after 70% occupancy
// //     normalizedOccupancy = 0.85f + (occupancy - 0.7f) * 0.5f;
// //   }
  
// //   // Warp utilization factor - bonus for sizes that are multiples of warp size
// //   float warpUtilization = 1.0f;
// //   if (totalThreadsPerBlock % hwParams.warpSize != 0) {
// //     // Partial warps reduce efficiency
// //     int lastWarpSize = totalThreadsPerBlock % hwParams.warpSize;
// //     warpUtilization = 1.0f - 0.2f * (1.0f - static_cast<float>(lastWarpSize) / hwParams.warpSize);
// //   }
  
// //   // Combine factors
// //   return normalizedOccupancy * warpUtilization;
// // }

// // float TileSizeOptimizer::evaluateMemoryEfficiency(
// //     const CompleteTileConfig &config,
// //     const std::vector<LoopInfo> &loopInfos,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // Evaluate coalesced memory access (weight: 0.6)
// //   float coalescedScore = estimateCoalescedAccess(config, memAccesses);
  
// //   // Evaluate cache utilization (weight: 0.3)
// //   float cacheScore = estimateCacheUtilization(config, memAccesses);
  
// //   // Evaluate memory bank conflicts (weight: 0.1)
// //   float bankConflictScore = estimateMemoryBankConflicts(config, memAccesses);
  
// //   // Combined memory efficiency score
// //   float memoryEfficiency = 0.6f * coalescedScore + 
// //                          0.3f * cacheScore + 
// //                          0.1f * bankConflictScore;
  
// //   return memoryEfficiency;
// // }

// // float TileSizeOptimizer::estimateCoalescedAccess(
// //     const CompleteTileConfig &config,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   float totalScore = 0.0f;
// //   int totalAccesses = 0;
  
// //   // If no memory accesses, return neutral score
// //   if (memAccesses.empty()) {
// //     return 0.5f;
// //   }
  
// //   // Process each memory access
// //   for (const auto &access : memAccesses) {
// //     totalAccesses++;
// //     float accessScore = 0.1f;  // Default low score
    
// //     // Evaluate based on access pattern
// //     switch (access.pattern) {
// //       case MemoryAccessPattern::COALESCED:
// //         // Check if innermost dimension tile size is warp-friendly
// //         if (!config.perDimConfig.empty()) {
// //           int64_t innermostTileSize = config.perDimConfig.back().tileSize;
          
// //           if (innermostTileSize == hwParams.warpSize) {
// //             // Perfect coalescing with warp size
// //             accessScore = 1.0f;
// //           } else if (innermostTileSize % hwParams.warpSize == 0) {
// //             // Multiple of warp size - still good
// //             accessScore = 0.95f;
// //           } else if (innermostTileSize % 16 == 0) {
// //             // Multiple of half-warp - reasonable
// //             accessScore = 0.85f;
// //           } else if (innermostTileSize % 8 == 0) {
// //             // Multiple of quarter-warp - acceptable
// //             accessScore = 0.75f;
// //           } else if (innermostTileSize >= hwParams.warpSize) {
// //             // Larger than warp size but not aligned - some inefficiency
// //             accessScore = 0.7f;
// //           } else {
// //             // Smaller than warp size - sub-optimal
// //             accessScore = 0.6f * static_cast<float>(innermostTileSize) / hwParams.warpSize;
// //           }
// //         }
// //         break;
        
// //       case MemoryAccessPattern::SEQUENTIAL:
// //         // Sequential but not coalesced - medium score
// //         accessScore = 0.5f;
// //         break;
        
// //       case MemoryAccessPattern::STRIDED:
// //         // Strided access - low score with some consideration for stride
// //         accessScore = 0.3f;
// //         break;
        
// //       case MemoryAccessPattern::RANDOM:
// //         // Random access - very low score
// //         accessScore = 0.1f;
// //         break;
// //     }
    
// //     totalScore += accessScore;
// //   }
  
// //   return totalScore / totalAccesses;
// // }

// // float TileSizeOptimizer::estimateCacheUtilization(
// //     const CompleteTileConfig &config,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // Calculate total tile size in elements
// //   int64_t totalTileSize = 1;
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     totalTileSize *= tileConfig.tileSize;
// //   }
  
// //   // Estimate working set size
// //   int64_t workingSetBytes = 0;
// //   llvm::DenseSet<Value> uniqueMemrefs;
  
// //   for (const auto &access : memAccesses) {
// //     if (uniqueMemrefs.insert(access.memref).second) {
// //       workingSetBytes += totalTileSize * access.dataTypeSizeInBytes;
// //     }
// //   }
  
// //   // Cache sizes - typical L1 and L2 sizes
// //   const int L1CacheSize = hwParams.l1CacheSize;
// //   const int L2CacheSize = hwParams.l2CacheSize;
  
// //   // Score based on working set vs cache size
// //   float cacheScore = 0.0f;
  
// //   if (workingSetBytes <= L1CacheSize) {
// //     // Working set fits in L1 cache - excellent
// //     cacheScore = 1.0f;
// //   } else if (workingSetBytes <= L2CacheSize) {
// //     // Working set fits in L2 cache - good
// //     float l2Ratio = static_cast<float>(L2CacheSize - workingSetBytes) / 
// //                    (L2CacheSize - L1CacheSize);
// //     cacheScore = 0.7f + 0.3f * l2Ratio;
// //   } else {
// //     // Working set exceeds L2 cache - poor
// //     float excessRatio = std::min(1.0f, static_cast<float>(L2CacheSize) / workingSetBytes);
// //     cacheScore = 0.3f * excessRatio;
// //   }
  
// //   // Spatial locality bonus - prefer larger tile sizes for better spatial locality
// //   float spatialLocalityFactor = std::min(1.0f, std::log2f(totalTileSize) / 8.0f);
  
// //   return cacheScore * (0.7f + 0.3f * spatialLocalityFactor);
// // }

// // float TileSizeOptimizer::estimateMemoryBankConflicts(
// //     const CompleteTileConfig &config,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // Simple model: Penalize tile sizes that are likely to cause bank conflicts
  
// //   // In NVIDIA GPUs, shared memory is divided into 32 banks
// //   const int numBanks = 32;
  
// //   // Check if any dimension size is likely to cause conflicts
// //   float conflictScore = 1.0f;
  
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     int64_t tileSize = tileConfig.tileSize;
    
// //     // Check for patterns known to cause bank conflicts
// //     if (tileSize % numBanks == 0) {
// //       // Perfect - no conflicts
// //       continue;
// //     } else if (tileSize % 16 == 0) {
// //       // Reasonably good
// //       conflictScore *= 0.95f;
// //     } else if (tileSize % 8 == 0) {
// //       // Some conflicts likely
// //       conflictScore *= 0.9f;
// //     } else if (tileSize % 2 == 0) {
// //       // More conflicts likely
// //       conflictScore *= 0.8f;
// //     } else {
// //       // Odd sizes can be better than even sizes for bank conflicts
// //       conflictScore *= 0.85f;
// //     }
// //   }
  
// //   return conflictScore;
// // }

// // float TileSizeOptimizer::evaluateLoadBalancing(
// //     const CompleteTileConfig &config,
// //     const std::vector<LoopInfo> &loopInfos) {
  
// //   float balanceScore = 1.0f;
  
// //   // Check each dimension for load balancing issues
// //   for (size_t i = 0; i < config.perDimConfig.size() && i < loopInfos.size(); ++i) {
// //     const auto &tileConfig = config.perDimConfig[i];
// //     const auto &loopInfo = loopInfos[i];
    
// //     // Skip dimensions without constant bounds
// //     if (!loopInfo.hasConstantBounds) {
// //       continue;
// //     }
    
// //     // Calculate loop range and number of blocks
// //     int64_t range = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
// //     int64_t tileSize = tileConfig.tileSize;
// //     int64_t numBlocks = (range + tileSize - 1) / tileSize;
    
// //     // Check for uneven division (remainder)
// //     if (range % tileSize != 0) {
// //       int64_t lastBlockSize = range % tileSize;
      
// //       // Calculate load imbalance factor
// //       float dimensionBalanceScore;
      
// //       if (numBlocks == 1) {
// //         // Only one block - no imbalance
// //         dimensionBalanceScore = 1.0f;
// //       } else {
// //         // Multiple blocks with one smaller block
// //         float fullBlocks = numBlocks - 1;
// //         float totalWork = fullBlocks * tileSize + lastBlockSize;
// //         float idealWork = totalWork / numBlocks;
// //         float maxDeviation = std::max(
// //             std::abs(tileSize - idealWork),
// //             std::abs(lastBlockSize - idealWork)
// //         ) / idealWork;
        
// //         // Score based on deviation from ideal
// //         dimensionBalanceScore = 1.0f - 0.5f * maxDeviation;
// //       }
      
// //       // Update overall balance score (use minimum across dimensions)
// //       balanceScore = std::min(balanceScore, dimensionBalanceScore);
// //     }
// //   }
  
// //   return balanceScore;
// // }

// // float TileSizeOptimizer::evaluateDataReuse(
// //     const CompleteTileConfig &config,
// //     const std::vector<LoopInfo> &loopInfos,
// //     const std::vector<MemoryAccessInfo> &memAccesses) {
  
// //   // This new function evaluates data reuse potential
  
// //   // Calculate tile sizes and total tile elements
// //   int64_t totalTileElements = 1;
// //   for (const auto &tileConfig : config.perDimConfig) {
// //     totalTileElements *= tileConfig.tileSize;
// //   }
  
// //   // Count memory references and accesses
// //   llvm::DenseMap<Value, int> memrefAccessCounts;
// //   int totalAccesses = 0;
  
// //   for (const auto &access : memAccesses) {
// //     memrefAccessCounts[access.memref]++;
// //     totalAccesses++;
// //   }
  
// //   // No accesses means no reuse
// //   if (totalAccesses == 0) {
// //     return 0.5f;
// //   }
  
// //   // Calculate average accesses per memref
// //   float avgAccessesPerMemref = static_cast<float>(totalAccesses) / memrefAccessCounts.size();
  
// //   // Higher average accesses means more potential data reuse
// //   float reuseScore = std::min(1.0f, avgAccessesPerMemref / 10.0f);
  
// //   // Adjust based on tile size - larger tiles can increase reuse
// //   float tileSizeFactor = std::min(1.0f, std::log2f(totalTileElements) / 10.0f);
  
// //   // Adjust based on access patterns - sequential/coalesced patterns often have better reuse
// //   float patternFactor = 0.5f;
// //   int coalescedCount = 0;
  
// //   for (const auto &access : memAccesses) {
// //     if (access.pattern == MemoryAccessPattern::COALESCED ||
// //         access.pattern == MemoryAccessPattern::SEQUENTIAL) {
// //       coalescedCount++;
// //     }
// //   }
  
// //   if (totalAccesses > 0) {
// //     patternFactor = 0.5f + 0.5f * static_cast<float>(coalescedCount) / totalAccesses;
// //   }
  
// //   // Combine factors
// //   return reuseScore * 0.6f + tileSizeFactor * 0.2f + patternFactor * 0.2f;
// // }

// // int TileSizeOptimizer::estimateRegistersPerThread(
// //     const std::vector<ComputationInfo> &computations) {
  
// //   // Base registers for loop overhead
// //   int baseRegisters = 16;
  
// //   // Maps to track unique operations and operands
// //   llvm::DenseSet<Operation*> uniqueOps;
// //   llvm::DenseSet<Value> uniqueOperands;
// //   int intermediateResults = 0;
  
// //   for (const auto &comp : computations) {
// //     // Track unique operations
// //     uniqueOps.insert(comp.op);
    
// //     // Track operands
// //     for (Value operand : comp.op->getOperands()) {
// //       uniqueOperands.insert(operand);
// //     }
    
// //     // Track results that need registers
// //     intermediateResults += comp.op->getNumResults();
    
// //     // Operation-specific register estimates
// //     if (isa<arith::MulFOp>(comp.op) || isa<arith::DivFOp>(comp.op)) {
// //       // Complex operations may need temporary registers
// //       baseRegisters += 1;
// //     }
// //   }
  
// //   // Calculate total register estimate - conservative approach
// //   int totalRegisters = baseRegisters + 
// //                      uniqueOperands.size() +
// //                      intermediateResults;
  
// //   // Apply upper bound limit
// //   return std::min(totalRegisters, hwParams.maxRegistersPerThread);
// // }

// #include "TileSizeOptimizer.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/DenseSet.h"
// #include "llvm/ADT/DenseMap.h"

// #define DEBUG_TYPE "tile-size-optimizer"

// using namespace mlir;
// using namespace mlir::scf;
// using namespace llvm;

// //===----------------------------------------------------------------------===//
// // 重构的TileSizeOptimizer实现 - 专注A100性能
// //===----------------------------------------------------------------------===//

// CompleteTileConfig TileSizeOptimizer::optimizeTileSize(scf::ParallelOp parallelOp) {
//   LLVM_DEBUG(llvm::dbgs() << "Starting tile size optimization for A100\n");
  
//   // Step 1: 分析阶段
//   std::vector<LoopInfo> loopInfos = extractLoopInfo(parallelOp);
//   std::vector<MemoryAccessInfo> memAccesses = analyzeMemoryAccesses(parallelOp);
//   std::vector<ComputationInfo> computations = analyzeComputations(parallelOp);
  
//   // Step 2: 模式检测
//   ComputationalPattern pattern = detectComputationalPattern(parallelOp, memAccesses, computations);
  
//   // Step 3: 智能候选生成
//   std::vector<std::vector<int64_t>> candidates = generateSmartTileCandidates(loopInfos, pattern);
  
//   // Step 4: 寻找最优配置
//   CompleteTileConfig optimalConfig = findOptimalConfiguration(
//       loopInfos, memAccesses, computations, candidates, pattern);
  
//   LLVM_DEBUG(llvm::dbgs() << "Optimal config found - Score: " << optimalConfig.overallPerformanceScore
//                          << ", Threads: " << optimalConfig.totalThreadsPerBlock << "\n");
  
//   return optimalConfig;
// }

// std::vector<LoopInfo> TileSizeOptimizer::extractLoopInfo(scf::ParallelOp parallelOp) {
//   std::vector<LoopInfo> loopInfos;
  
//   for (unsigned i = 0; i < parallelOp.getNumLoops(); ++i) {
//     LoopInfo info;
//     info.dimension = i;
//     info.lowerBound = parallelOp.getLowerBound()[i];
//     info.upperBound = parallelOp.getUpperBound()[i];
//     info.step = parallelOp.getStep()[i];
    
//     // 提取常量值
//     if (auto constLB = dyn_cast_or_null<arith::ConstantIndexOp>(info.lowerBound.getDefiningOp())) {
//       info.constantLowerBound = constLB.value();
      
//       if (auto constUB = dyn_cast_or_null<arith::ConstantIndexOp>(info.upperBound.getDefiningOp())) {
//         info.constantUpperBound = constUB.value();
        
//         if (auto constStep = dyn_cast_or_null<arith::ConstantIndexOp>(info.step.getDefiningOp())) {
//           info.constantStep = constStep.value();
//           info.hasConstantBounds = true;
          
//           // 计算实际迭代次数
//           int64_t range = info.constantUpperBound - info.constantLowerBound;
//           info.iterationCount = (range + info.constantStep - 1) / info.constantStep;
//         }
//       }
//     }
    
//     loopInfos.push_back(info);
//   }
  
//   return loopInfos;
// }

// std::vector<MemoryAccessInfo> TileSizeOptimizer::analyzeMemoryAccesses(scf::ParallelOp parallelOp) {
//   std::vector<MemoryAccessInfo> memAccesses;
//   llvm::DenseMap<Value, int> accessCounts; // 跟踪访问频率
  
//   parallelOp.getBody()->walk([&](Operation *op) {
//     if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
//       MemoryAccessInfo info;
//       info.memref = loadOp.getMemref();
//       info.indices = SmallVector<Value, 4>(loadOp.getIndices().begin(), loadOp.getIndices().end());
//       info.isLoad = true;
//       info.isStore = false;
      
//       // 确定数据类型大小
//       if (auto memrefType = mlir::dyn_cast<MemRefType>(loadOp.getMemref().getType())) {
//         Type elementType = memrefType.getElementType();
//         if (elementType.isF32()) info.dataTypeSizeInBytes = 4;
//         else if (elementType.isF64()) info.dataTypeSizeInBytes = 8;
//         else if (elementType.isF16()) info.dataTypeSizeInBytes = 2;
//         else if (elementType.isBF16()) info.dataTypeSizeInBytes = 2;
//         else if (elementType.isInteger(32)) info.dataTypeSizeInBytes = 4;
//         else if (elementType.isInteger(64)) info.dataTypeSizeInBytes = 8;
//         else info.dataTypeSizeInBytes = 4; // 默认
//       } else {
//         info.dataTypeSizeInBytes = 4;
//       }
      
//       // 改进的访问模式分析
//       info.pattern = analyzeAccessPattern(info.indices, parallelOp.getInductionVars());
      
//       // 计算访问频率
//       accessCounts[info.memref]++;
//       info.accessFrequency = 1.0f; // 基础频率，稍后调整
      
//       memAccesses.push_back(info);
//     } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
//       MemoryAccessInfo info;
//       info.memref = storeOp.getMemref();
//       info.indices = SmallVector<Value, 4>(storeOp.getIndices().begin(), storeOp.getIndices().end());
//       info.isLoad = false;
//       info.isStore = true;
      
//       // 类似的数据类型分析
//       if (auto memrefType = mlir::dyn_cast<MemRefType>(storeOp.getMemref().getType())) {
//         Type elementType = memrefType.getElementType();
//         if (elementType.isF32()) info.dataTypeSizeInBytes = 4;
//         else if (elementType.isF64()) info.dataTypeSizeInBytes = 8;
//         else if (elementType.isF16()) info.dataTypeSizeInBytes = 2;
//         else if (elementType.isBF16()) info.dataTypeSizeInBytes = 2;
//         else if (elementType.isInteger(32)) info.dataTypeSizeInBytes = 4;
//         else if (elementType.isInteger(64)) info.dataTypeSizeInBytes = 8;
//         else info.dataTypeSizeInBytes = 4;
//       } else {
//         info.dataTypeSizeInBytes = 4;
//       }
      
//       info.pattern = analyzeAccessPattern(info.indices, parallelOp.getInductionVars());
//       accessCounts[info.memref]++;
//       info.accessFrequency = 1.0f;
      
//       memAccesses.push_back(info);
//     }
//   });
  
//   // 调整访问频率
//   for (auto &access : memAccesses) {
//     access.accessFrequency = static_cast<float>(accessCounts[access.memref]);
//   }
  
//   return memAccesses;
// }

// MemoryAccessPattern TileSizeOptimizer::analyzeAccessPattern(
//     const SmallVector<Value, 4> &indices, ValueRange inductionVars) {
//   if (indices.empty() || inductionVars.empty()) {
//     return MemoryAccessPattern::SCATTERED;
//   }
  
//   // 检查最内层维度是否使用最内层循环变量（完全合并）
//   if (indices.back() == inductionVars.back()) {
//     return MemoryAccessPattern::COALESCED;
//   }
  
//   // 检查是否存在循环变量在索引中（部分合并或跨步）
//   bool hasInductionVar = false;
//   for (Value idx : indices) {
//     for (Value iv : inductionVars) {
//       if (idx == iv) {
//         hasInductionVar = true;
//         break;
//       }
//     }
//     if (hasInductionVar) break;
//   }
  
//   if (hasInductionVar) {
//     // 进一步分析是否为规律跨步
//     return MemoryAccessPattern::STRIDED;
//   }
  
//   return MemoryAccessPattern::SCATTERED;
// }

// std::vector<ComputationInfo> TileSizeOptimizer::analyzeComputations(scf::ParallelOp parallelOp) {
//   std::vector<ComputationInfo> computations;
  
//   parallelOp.getBody()->walk([&](Operation *op) {
//     ComputationInfo info;
//     info.op = op;
//     info.opCount = 1;
//     info.computeIntensity = 1.0f; // 默认强度
    
//     // 浮点操作
//     if (isa<arith::AddFOp>(op) || isa<arith::SubFOp>(op)) {
//       info.isFloatingPoint = true;
//       info.computeIntensity = 1.0f;
//       computations.push_back(info);
//     } else if (isa<arith::MulFOp>(op)) {
//       info.isFloatingPoint = true;
//       info.computeIntensity = 1.2f; // 乘法稍微昂贵
//       computations.push_back(info);
//     } else if (isa<arith::DivFOp>(op)) {
//       info.isFloatingPoint = true;
//       info.computeIntensity = 4.0f; // 除法很昂贵
//       computations.push_back(info);
//     }
//     // 整数操作
//     else if (isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
//              isa<arith::MulIOp>(op)) {
//       info.isFloatingPoint = false;
//       info.computeIntensity = 0.8f; // 整数运算相对便宜
//       computations.push_back(info);
//     } else if (isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op)) {
//       info.isFloatingPoint = false;
//       info.computeIntensity = 3.0f;
//       computations.push_back(info);
//     }
//     // 特殊操作
//     else if (isa<arith::MaximumFOp>(op) || isa<arith::MinimumFOp>(op) ||
//              isa<arith::MaxSIOp>(op) || isa<arith::MinSIOp>(op)) {
//       info.isFloatingPoint = isa<arith::MaximumFOp>(op) || isa<arith::MinimumFOp>(op);
//       info.computeIntensity = 1.1f;
//       computations.push_back(info);
//     }
//   });
  
//   return computations;
// }

// ComputationalPattern TileSizeOptimizer::detectComputationalPattern(
//     scf::ParallelOp parallelOp,
//     const std::vector<MemoryAccessInfo> &memAccesses,
//     const std::vector<ComputationInfo> &computations) {
  
//   size_t numDims = parallelOp.getNumLoops();
  
//   // 统计操作类型
//   int mulCount = 0, addCount = 0, maxCount = 0, totalOps = 0;
//   float totalComputeIntensity = 0.0f;
  
//   for (const auto &comp : computations) {
//     totalOps++;
//     totalComputeIntensity += comp.computeIntensity;
    
//     if (isa<arith::MulFOp>(comp.op) || isa<arith::MulIOp>(comp.op)) {
//       mulCount++;
//     } else if (isa<arith::AddFOp>(comp.op) || isa<arith::AddIOp>(comp.op)) {
//       addCount++;
//     } else if (isa<arith::MaximumFOp>(comp.op) || isa<arith::MaxSIOp>(comp.op)) {
//       maxCount++;
//     }
//   }
  
//   // 统计内存访问
//   int loadCount = 0, storeCount = 0;
//   int coalescedCount = 0;
//   llvm::DenseSet<Value> uniqueMemrefs;
  
//   for (const auto &access : memAccesses) {
//     if (access.isLoad) loadCount++;
//     if (access.isStore) storeCount++;
//     if (access.pattern == MemoryAccessPattern::COALESCED) coalescedCount++;
//     uniqueMemrefs.insert(access.memref);
//   }
  
//   float avgComputeIntensity = totalOps > 0 ? totalComputeIntensity / totalOps : 0.0f;
//   float computeToMemoryRatio = (loadCount + storeCount) > 0 ? 
//                               static_cast<float>(totalOps) / (loadCount + storeCount) : 0.0f;
  
//   // 模式检测逻辑 - 更精确
//   if (numDims >= 3 && mulCount > 0 && addCount > 0 && 
//       computeToMemoryRatio > 1.5f && avgComputeIntensity > 1.0f) {
//     LLVM_DEBUG(llvm::dbgs() << "Detected MATMUL pattern\n");
//     return ComputationalPattern::MATMUL;
//   }
  
//   if (numDims >= 3 && (mulCount > 0 || maxCount > 0) && 
//       uniqueMemrefs.size() >= 2 && computeToMemoryRatio > 0.8f) {
//     LLVM_DEBUG(llvm::dbgs() << "Detected CONV pattern\n");
//     return ComputationalPattern::CONV;
//   }
  
//   if (loadCount > 3 * totalOps && uniqueMemrefs.size() <= 3) {
//     LLVM_DEBUG(llvm::dbgs() << "Detected STENCIL pattern\n");
//     return ComputationalPattern::STENCIL;
//   }
  
//   // 检查归约操作
//   bool hasReduce = false;
//   parallelOp.getBody()->walk([&](Operation *op) {
//     if (isa<scf::ReduceOp>(op)) hasReduce = true;
//   });
  
//   if (hasReduce || (addCount > loadCount/2) || (maxCount > loadCount/2)) {
//     LLVM_DEBUG(llvm::dbgs() << "Detected REDUCTION pattern\n");
//     return ComputationalPattern::REDUCTION;
//   }
  
//   if (loadCount <= 2 && storeCount <= 1 && totalOps <= 3 && coalescedCount > 0) {
//     LLVM_DEBUG(llvm::dbgs() << "Detected ELEMENTWISE pattern\n");
//     return ComputationalPattern::ELEMENTWISE;
//   }
  
//   LLVM_DEBUG(llvm::dbgs() << "Using GENERIC pattern\n");
//   return ComputationalPattern::GENERIC;
// }

// std::vector<std::vector<int64_t>> TileSizeOptimizer::generateSmartTileCandidates(
//     const std::vector<LoopInfo> &loopInfos,
//     ComputationalPattern pattern) {
  
//   std::vector<std::vector<int64_t>> candidatesPerDim;
  
//   for (size_t i = 0; i < loopInfos.size(); ++i) {
//     bool isInnermostDim = (i == loopInfos.size() - 1);
//     bool isOutermostDim = (i == 0);
    
//     std::vector<int64_t> candidates = generateDimensionCandidates(
//         loopInfos[i], static_cast<int>(i), pattern, isInnermostDim, isOutermostDim);
    
//     candidatesPerDim.push_back(candidates);
//   }
  
//   return candidatesPerDim;
// }

// std::vector<int64_t> TileSizeOptimizer::generateDimensionCandidates(
//     const LoopInfo &loopInfo, 
//     int dimIndex,
//     ComputationalPattern pattern,
//     bool isInnermostDim,
//     bool isOutermostDim) {
  
//   std::vector<int64_t> candidates;
  
//   // A100友好的基础候选 - 优先warp对齐
//   std::vector<int64_t> baseCandidates = {
//     1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024
//   };
  
//   // 获取维度大小限制
//   int64_t maxSize = hwParams.maxThreadsPerBlock;
//   if (dimIndex == 0) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimX));
//   else if (dimIndex == 1) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimY));
//   else if (dimIndex == 2) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimZ));
  
//   int64_t iterCount = loopInfo.hasConstantBounds ? loopInfo.iterationCount : 1024;
//   int64_t effectiveMax = std::min(maxSize, iterCount);
  
//   // 添加基础候选
//   for (int64_t size : baseCandidates) {
//     if (size <= effectiveMax) {
//       candidates.push_back(size);
//     }
//   }
  
//   // 最内层维度 - 专注warp效率
//   if (isInnermostDim) {
//     // 确保包含warp相关大小
//     for (int mult = 1; mult <= 8; mult++) {
//       int64_t size = hwParams.warpSize * mult;
//       if (size <= effectiveMax) {
//         candidates.push_back(size);
//       }
//     }
    
//     // A100 tensor core友好大小
//     if (pattern == ComputationalPattern::MATMUL || pattern == ComputationalPattern::CONV) {
//       for (int mult = 1; mult <= 8; mult++) {
//         int64_t size = hwParams.tensorCoreAlignment * mult;
//         if (size <= effectiveMax) {
//           candidates.push_back(size);
//         }
//       }
//     }
//   }
  
//   // 排序并去重
//   std::sort(candidates.begin(), candidates.end());
//   candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  
//   // 移除0和无效值
//   candidates.erase(std::remove_if(candidates.begin(), candidates.end(), 
//                                  [](int64_t x) { return x <= 0; }), candidates.end());
  
//   return candidates;
// }

// CompleteTileConfig TileSizeOptimizer::findOptimalConfiguration(
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses,
//     const std::vector<ComputationInfo> &computations,
//     const std::vector<std::vector<int64_t>> &candidates,
//     ComputationalPattern pattern) {
  
//   CompleteTileConfig bestConfig;
//   float bestScore = -1.0f;
  
//   // 使用递归生成所有有效组合
//   std::function<void(std::vector<int64_t>&, int)> exploreConfigurations = 
//     [&](std::vector<int64_t> &currentConfig, int dimIndex) {
//       if (static_cast<size_t>(dimIndex) == loopInfos.size()) {
//         // 评估当前配置
//         if (isValidConfiguration(currentConfig)) {
//           float score = evaluateConfiguration(currentConfig, loopInfos, memAccesses, computations, pattern);
//           if (score > bestScore) {
//             bestScore = score;
            
//             // 构建最佳配置
//             bestConfig.perDimConfig.clear();
//             for (size_t i = 0; i < currentConfig.size(); ++i) {
//               TileConfig tileConfig;
//               tileConfig.tileSize = currentConfig[i];
//               tileConfig.efficiencyScore = score; // 简化，使用总分
//               bestConfig.perDimConfig.push_back(tileConfig);
//             }
//             bestConfig.overallPerformanceScore = score;
//             bestConfig.totalThreadsPerBlock = calculateTotalThreads(currentConfig);
//             bestConfig.occupancyScore = score; // 简化
//             bestConfig.memoryEfficiencyScore = score; // 简化
//             bestConfig.computeUtilizationScore = score; // 简化
//             bestConfig.estimatedActiveBlocks = calculateTotalThreads(currentConfig) / hwParams.warpSize;
//           }
//         }
//         return;
//       }
      
//       // 尝试当前维度的所有候选
//       for (int64_t candidate : candidates[dimIndex]) {
//         currentConfig[dimIndex] = candidate;
        
//         // 提前剪枝 - 检查线程数限制
//         int totalThreads = 1;
//         for (int i = 0; i <= dimIndex; i++) {
//           totalThreads *= currentConfig[i];
//         }
//         if (totalThreads > hwParams.maxThreadsPerBlock) {
//           continue; // 跳过这个候选
//         }
        
//         exploreConfigurations(currentConfig, dimIndex + 1);
//       }
//     };
  
//   std::vector<int64_t> currentConfig(loopInfos.size(), 1);
//   exploreConfigurations(currentConfig, 0);
  
//   // 如果没找到有效配置，使用保守默认值
//   if (bestScore < 0) {
//     bestConfig.perDimConfig.clear();
//     for (size_t i = 0; i < loopInfos.size(); ++i) {
//       TileConfig tileConfig;
//       tileConfig.tileSize = (i == loopInfos.size() - 1) ? hwParams.warpSize : 
//                            std::min(32L, loopInfos[i].hasConstantBounds ? 
//                                    loopInfos[i].iterationCount : 32L);
//       tileConfig.efficiencyScore = 0.5f;
//       bestConfig.perDimConfig.push_back(tileConfig);
//     }
//     bestConfig.overallPerformanceScore = 0.5f;
//     bestConfig.totalThreadsPerBlock = calculateTotalThreads(
//         [&bestConfig]() {
//           std::vector<int64_t> sizes;
//           for (const auto &dimConfig : bestConfig.perDimConfig) {
//             sizes.push_back(dimConfig.tileSize);
//           }
//           return sizes;
//         }());
//     bestConfig.occupancyScore = 0.5f;
//     bestConfig.memoryEfficiencyScore = 0.5f;
//     bestConfig.computeUtilizationScore = 0.5f;
//     bestConfig.estimatedActiveBlocks = bestConfig.totalThreadsPerBlock / hwParams.warpSize;
//   }
  
//   return bestConfig;
// }

// float TileSizeOptimizer::evaluateConfiguration(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses,
//     const std::vector<ComputationInfo> &computations,
//     ComputationalPattern pattern) {
  
//   // 新的权重分配 - 专注关键因素
//   const float OCCUPANCY_WEIGHT = 0.50f;    // 占用率最重要
//   const float MEMORY_WEIGHT = 0.30f;       // 内存效率次重要  
//   const float COMPUTE_WEIGHT = 0.20f;      // 计算利用率
  
//   float occupancyScore = evaluateOccupancy(tileSizes);
//   float memoryScore = evaluateMemoryEfficiency(tileSizes, loopInfos, memAccesses);
//   float computeScore = evaluateComputeUtilization(tileSizes, computations, pattern);
  
//   // A100特定奖励
//   float a100Bonus = isA100Friendly(tileSizes, pattern) ? 1.1f : 1.0f;
  
//   float totalScore = (OCCUPANCY_WEIGHT * occupancyScore +
//                      MEMORY_WEIGHT * memoryScore +
//                      COMPUTE_WEIGHT * computeScore) * a100Bonus;
  
//   return totalScore;
// }

// float TileSizeOptimizer::evaluateOccupancy(const std::vector<int64_t> &tileSizes) {
//   int totalThreads = calculateTotalThreads(tileSizes);
  
//   if (totalThreads > hwParams.maxThreadsPerBlock || totalThreads <= 0) {
//     return 0.0f; // 无效配置
//   }
  
//   // 估算寄存器使用 - 简化模型
//   int estimatedRegisters = 32; // 基础寄存器使用量
  
//   // 估算共享内存使用 - 简化模型  
//   int estimatedSharedMem = totalThreads * 16; // 每线程16字节
  
//   float occupancyScore = calculateA100OccupancyScore(totalThreads, estimatedRegisters, estimatedSharedMem);
  
//   // Warp利用率奖励
//   float warpUtilization = 1.0f;
//   if (totalThreads % hwParams.warpSize != 0) {
//     int activeThreadsInLastWarp = totalThreads % hwParams.warpSize;
//     warpUtilization = 0.8f + 0.2f * activeThreadsInLastWarp / hwParams.warpSize;
//   }
  
//   return occupancyScore * warpUtilization;
// }

// float TileSizeOptimizer::calculateA100OccupancyScore(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock) {
//   // 计算每个限制因素允许的blocks数量
//   int maxBlocksByThreads = hwParams.maxThreadsPerSM / threadsPerBlock;
//   int maxBlocksByRegisters = (registersPerThread > 0) ? 
//                             hwParams.maxRegistersPerSM / (registersPerThread * threadsPerBlock) :
//                             hwParams.maxBlocksPerSM;
//   int maxBlocksBySharedMem = (sharedMemoryPerBlock > 0) ?
//                             hwParams.maxSharedMemoryPerSM / sharedMemoryPerBlock :
//                             hwParams.maxBlocksPerSM;
  
//   int activeBlocks = std::min({hwParams.maxBlocksPerSM, maxBlocksByThreads, 
//                               maxBlocksByRegisters, maxBlocksBySharedMem});
  
//   int maxWarps = hwParams.maxThreadsPerSM / hwParams.warpSize;
//   int warpsPerBlock = (threadsPerBlock + hwParams.warpSize - 1) / hwParams.warpSize;
//   int activeWarps = activeBlocks * warpsPerBlock;
  
//   float occupancy = static_cast<float>(activeWarps) / maxWarps;
  
//   // A100特定的占用率性能曲线
//   if (occupancy < 0.25f) {
//     return occupancy * 2.0f; // 低占用率性能线性下降
//   } else if (occupancy < 0.75f) {
//     return 0.5f + (occupancy - 0.25f) * 1.0f; // 中等占用率区域
//   } else {
//     return 1.0f - (occupancy - 0.75f) * 0.2f; // 高占用率收益递减
//   }
// }

// float TileSizeOptimizer::evaluateMemoryEfficiency(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   if (memAccesses.empty()) return 0.5f;
  
//   float coalescingScore = 0.0f;
//   float totalWeight = 0.0f;
  
//   // 评估合并访问
//   for (const auto &access : memAccesses) {
//     float weight = access.accessFrequency;
//     totalWeight += weight;
    
//     float accessScore = 0.0f;
//     if (access.pattern == MemoryAccessPattern::COALESCED) {
//       // 检查最内层tile大小是否warp友好
//       if (!tileSizes.empty()) {
//         int64_t innermostSize = tileSizes.back();
//         if (innermostSize % hwParams.warpSize == 0) {
//           accessScore = 1.0f;
//         } else if (innermostSize >= hwParams.warpSize) {
//           accessScore = 0.8f;
//         } else {
//           accessScore = static_cast<float>(innermostSize) / hwParams.warpSize * 0.7f;
//         }
//       }
//     } else if (access.pattern == MemoryAccessPattern::PARTIAL_COALESCED) {
//       accessScore = 0.6f;
//     } else if (access.pattern == MemoryAccessPattern::STRIDED) {
//       accessScore = 0.4f;
//     } else {
//       accessScore = 0.2f;
//     }
    
//     coalescingScore += accessScore * weight;
//   }
  
//   if (totalWeight > 0) {
//     coalescingScore /= totalWeight;
//   }
  
//   return coalescingScore;
// }

// float TileSizeOptimizer::evaluateDivisibility(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<LoopInfo> &loopInfos) {
  
//   float totalDivisibilityScore = 1.0f;
//   int validDimensions = 0;
  
//   for (size_t i = 0; i < tileSizes.size() && i < loopInfos.size(); ++i) {
//     if (!loopInfos[i].hasConstantBounds) {
//       continue; // 跳过非常量边界
//     }
    
//     validDimensions++;
//     int64_t iterCount = loopInfos[i].iterationCount;
//     int64_t tileSize = tileSizes[i];
    
//     if (iterCount % tileSize == 0) {
//       // 完美整除 - 最佳情况
//       // 保持得分为1.0，不做任何处理
//     } else {
//       // 不能整除 - 计算惩罚
//       int64_t remainder = iterCount % tileSize;
//       float utilization = static_cast<float>(remainder) / tileSize;
      
//       // 根据剩余线程的利用率计算惩罚
//       if (utilization >= 0.75f) {
//         // 剩余部分 >= 75%，影响相对较小
//         totalDivisibilityScore *= 0.9f;
//       } else if (utilization >= 0.5f) {
//         // 剩余部分 >= 50%，中等影响
//         totalDivisibilityScore *= 0.7f;
//       } else if (utilization >= 0.25f) {
//         // 剩余部分 >= 25%，较大影响
//         totalDivisibilityScore *= 0.4f;
//       } else {
//         // 剩余部分 < 25%，严重影响（这种情况下大量线程空闲）
//         totalDivisibilityScore *= 0.1f;
//       }
      
//       // 额外惩罚：如果是最内层维度且不能整除，加重惩罚
//       // 因为最内层维度的分支发散对性能影响最大
//       if (i == tileSizes.size() - 1) {
//         totalDivisibilityScore *= 0.8f; // 最内层不整除额外20%惩罚
//       }
//     }
//   }
  
//   // 如果没有常量边界维度，返回中性得分
//   if (validDimensions == 0) {
//     return 0.8f; // 对未知情况给予谨慎的得分
//   }
  
//   return totalDivisibilityScore;
// }

// float TileSizeOptimizer::evaluateComputeUtilization(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<ComputationInfo> &computations,
//     ComputationalPattern pattern) {
  
//   if (computations.empty()) return 0.5f;
  
//   int64_t totalTileSize = calculateTotalThreads(tileSizes);
  
//   // 计算总的计算强度
//   float totalComputeOps = 0.0f;
//   for (const auto &comp : computations) {
//     totalComputeOps += comp.opCount * comp.computeIntensity;
//   }
  
//   // 每线程的计算量
//   float computePerThread = totalComputeOps / totalTileSize;
  
//   // 根据模式调整期望的计算量
//   float expectedComputePerThread = 1.0f;
//   switch (pattern) {
//     case ComputationalPattern::MATMUL:
//       expectedComputePerThread = 4.0f;
//       break;
//     case ComputationalPattern::CONV:
//       expectedComputePerThread = 2.0f;
//       break;
//     case ComputationalPattern::ELEMENTWISE:
//       expectedComputePerThread = 0.5f;
//       break;
//     case ComputationalPattern::REDUCTION:
//       expectedComputePerThread = 1.5f;
//       break;
//     default:
//       expectedComputePerThread = 1.0f;
//       break;
//   }
  
//   // 计算利用率得分
//   float utilizationRatio = computePerThread / expectedComputePerThread;
  
//   if (utilizationRatio < 0.5f) {
//     return utilizationRatio * 0.8f; // 计算量过少
//   } else if (utilizationRatio <= 2.0f) {
//     return 0.4f + utilizationRatio * 0.3f; // 理想区间
//   } else {
//     return 1.0f - (utilizationRatio - 2.0f) * 0.1f; // 计算量过多可能导致寄存器压力
//   }
// }

// bool TileSizeOptimizer::isValidConfiguration(const std::vector<int64_t> &tileSizes) {
//   int totalThreads = calculateTotalThreads(tileSizes);
  
//   if (totalThreads <= 0 || totalThreads > hwParams.maxThreadsPerBlock) {
//     return false;
//   }
  
//   // 检查维度限制
//   for (size_t i = 0; i < tileSizes.size() && i < 3; ++i) {
//     int maxDim = (i == 0) ? hwParams.maxBlockDimX :
//                 (i == 1) ? hwParams.maxBlockDimY : hwParams.maxBlockDimZ;
//     if (tileSizes[i] > maxDim) {
//       return false;
//     }
//   }
  
//   return true;
// }

// int TileSizeOptimizer::calculateTotalThreads(const std::vector<int64_t> &tileSizes) {
//   int64_t total = 1;
//   for (int64_t size : tileSizes) {
//     total *= size;
//     if (total > hwParams.maxThreadsPerBlock) {
//       return hwParams.maxThreadsPerBlock + 1; // 返回无效值
//     }
//   }
//   return static_cast<int>(total);
// }

// bool TileSizeOptimizer::isA100Friendly(const std::vector<int64_t> &tileSizes, ComputationalPattern pattern) {
//   if (tileSizes.empty()) return false;
  
//   int totalThreads = calculateTotalThreads(tileSizes);
  
//   // 检查warp对齐
//   bool warpAligned = (totalThreads % hwParams.warpSize == 0);
  
//   // 检查是否使用了合理的线程数
//   bool goodThreadCount = (totalThreads >= 128 && totalThreads <= 512);
  
//   // 检查最内层维度是否合理
//   bool goodInnermostDim = (tileSizes.back() % 32 == 0) || (tileSizes.back() >= 16);
  
//   return warpAligned && goodThreadCount && goodInnermostDim;
// }

// int TileSizeOptimizer::estimateRegistersPerThread(const std::vector<ComputationInfo> &computations) {
//   // 基础寄存器使用 + 每个操作的估算
//   return 32 + static_cast<int>(computations.size() * 2);
// }

// int TileSizeOptimizer::estimateSharedMemoryUsage(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
//   // 简化估算：假设每个线程需要少量共享内存
//   int totalThreads = calculateTotalThreads(tileSizes);
//   return totalThreads * 16; // 每线程16字节
// }


#include "TileSizeOptimizer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "tile-size-optimizer"

using namespace mlir;
using namespace mlir::scf;
using namespace llvm;

//===----------------------------------------------------------------------===//
// 重构的TileSizeOptimizer实现 - A100优化 + 强制整除性
//===----------------------------------------------------------------------===//

CompleteTileConfig TileSizeOptimizer::optimizeTileSize(scf::ParallelOp parallelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Starting enhanced tile size optimization for A100\n");
  
  // Step 1: 增强的分析阶段
  std::vector<LoopInfo> loopInfos = extractEnhancedLoopInfo(parallelOp);
  std::vector<MemoryAccessInfo> memAccesses = analyzeEnhancedMemoryAccesses(parallelOp);
  std::vector<ComputationInfo> computations = analyzeEnhancedComputations(parallelOp);
  
  // Step 2: 计算整除性信息
  computeDivisibilityInfo(loopInfos);
  
  // Step 3: 分析数据复用模式
  analyzeDataReuse(memAccesses, loopInfos);
  
  // Step 4: 模式检测
  ComputationalPattern pattern = detectComputationalPattern(parallelOp, memAccesses, computations);
  
  // Step 5: 生成严格整除的候选
  std::vector<std::vector<int64_t>> candidates = generateDivisibleTileCandidates(loopInfos, pattern);
  
  // Step 6: 智能搜索最优配置
  CompleteTileConfig optimalConfig = findOptimalConfigurationSmart(
      loopInfos, memAccesses, computations, candidates, pattern);
  
  LLVM_DEBUG(llvm::dbgs() << "Optimal config found:\n");
  LLVM_DEBUG(llvm::dbgs() << "  Score: " << optimalConfig.overallPerformanceScore << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Threads: " << optimalConfig.totalThreadsPerBlock << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Divisibility: " << optimalConfig.divisibilityScore << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Working Set Fit: " << optimalConfig.workingSetScore << "\n");
  
  return optimalConfig;
}

std::vector<LoopInfo> TileSizeOptimizer::extractEnhancedLoopInfo(scf::ParallelOp parallelOp) {
  std::vector<LoopInfo> loopInfos;
  
  for (unsigned i = 0; i < parallelOp.getNumLoops(); ++i) {
    LoopInfo info;
    info.dimension = i;
    info.lowerBound = parallelOp.getLowerBound()[i];
    info.upperBound = parallelOp.getUpperBound()[i];
    info.step = parallelOp.getStep()[i];
    info.isOuterMostLoop = (i == 0);
    info.isInnerMostLoop = (i == parallelOp.getNumLoops() - 1);
    
    // 提取常量值
    if (auto constLB = dyn_cast_or_null<arith::ConstantIndexOp>(info.lowerBound.getDefiningOp())) {
      info.constantLowerBound = constLB.value();
      
      if (auto constUB = dyn_cast_or_null<arith::ConstantIndexOp>(info.upperBound.getDefiningOp())) {
        info.constantUpperBound = constUB.value();
        
        if (auto constStep = dyn_cast_or_null<arith::ConstantIndexOp>(info.step.getDefiningOp())) {
          info.constantStep = constStep.value();
          info.hasConstantBounds = true;
          
          // 计算实际迭代次数
          int64_t range = info.constantUpperBound - info.constantLowerBound;
          info.iterationCount = (range + info.constantStep - 1) / info.constantStep;
          
          LLVM_DEBUG(llvm::dbgs() << "Dimension " << i << " iteration count: " << info.iterationCount << "\n");
        }
      }
    }
    
    loopInfos.push_back(info);
  }
  
  return loopInfos;
}

void TileSizeOptimizer::computeDivisibilityInfo(std::vector<LoopInfo> &loopInfos) {
  for (auto &info : loopInfos) {
    if (info.hasConstantBounds && info.iterationCount > 0) {
      info.possibleDivisors = getFactors(info.iterationCount);
      LLVM_DEBUG(llvm::dbgs() << "Dimension " << info.dimension << " divisors: ");
      for (auto div : info.possibleDivisors) {
        LLVM_DEBUG(llvm::dbgs() << div << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }
  }
}

std::vector<int64_t> TileSizeOptimizer::getFactors(int64_t number) {
  std::vector<int64_t> factors;
  if (number <= 0) return factors;
  
  for (int64_t i = 1; i * i <= number; ++i) {
    if (number % i == 0) {
      factors.push_back(i);
      if (i != number / i) {
        factors.push_back(number / i);
      }
    }
  }
  
  std::sort(factors.begin(), factors.end());
  return factors;
}

std::vector<MemoryAccessInfo> TileSizeOptimizer::analyzeEnhancedMemoryAccesses(scf::ParallelOp parallelOp) {
  std::vector<MemoryAccessInfo> memAccesses;
  llvm::DenseMap<Value, int> accessCounts;
  
  parallelOp.getBody()->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      MemoryAccessInfo info;
      info.memref = loadOp.getMemref();
      info.indices = SmallVector<Value, 4>(loadOp.getIndices().begin(), loadOp.getIndices().end());
      info.isLoad = true;
      info.isStore = false;
      
      // 确定数据类型大小
      if (auto memrefType = mlir::dyn_cast<MemRefType>(loadOp.getMemref().getType())) {
        Type elementType = memrefType.getElementType();
        if (elementType.isF32()) info.dataTypeSizeInBytes = 4;
        else if (elementType.isF64()) info.dataTypeSizeInBytes = 8;
        else if (elementType.isF16()) info.dataTypeSizeInBytes = 2;
        else if (elementType.isBF16()) info.dataTypeSizeInBytes = 2;
        else if (elementType.isInteger(32)) info.dataTypeSizeInBytes = 4;
        else if (elementType.isInteger(64)) info.dataTypeSizeInBytes = 8;
        else info.dataTypeSizeInBytes = 4;
      } else {
        info.dataTypeSizeInBytes = 4;
      }
      
      // 增强的访问模式分析
      info.pattern = analyzeAccessPatternEnhanced(info.indices, parallelOp.getInductionVars());
      
      accessCounts[info.memref]++;
      info.accessFrequency = 1.0f;
      
      memAccesses.push_back(info);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      MemoryAccessInfo info;
      info.memref = storeOp.getMemref();
      info.indices = SmallVector<Value, 4>(storeOp.getIndices().begin(), storeOp.getIndices().end());
      info.isLoad = false;
      info.isStore = true;
      
      if (auto memrefType = mlir::dyn_cast<MemRefType>(storeOp.getMemref().getType())) {
        Type elementType = memrefType.getElementType();
        if (elementType.isF32()) info.dataTypeSizeInBytes = 4;
        else if (elementType.isF64()) info.dataTypeSizeInBytes = 8;
        else if (elementType.isF16()) info.dataTypeSizeInBytes = 2;
        else if (elementType.isBF16()) info.dataTypeSizeInBytes = 2;
        else if (elementType.isInteger(32)) info.dataTypeSizeInBytes = 4;
        else if (elementType.isInteger(64)) info.dataTypeSizeInBytes = 8;
        else info.dataTypeSizeInBytes = 4;
      } else {
        info.dataTypeSizeInBytes = 4;
      }
      
      info.pattern = analyzeAccessPatternEnhanced(info.indices, parallelOp.getInductionVars());
      accessCounts[info.memref]++;
      info.accessFrequency = 1.0f;
      
      memAccesses.push_back(info);
    }
  });
  
  // 调整访问频率
  for (auto &access : memAccesses) {
    access.accessFrequency = static_cast<float>(accessCounts[access.memref]);
  }
  
  return memAccesses;
}

MemoryAccessPattern TileSizeOptimizer::analyzeAccessPatternEnhanced(
    const SmallVector<Value, 4> &indices, ValueRange inductionVars) {
  if (indices.empty() || inductionVars.empty()) {
    return MemoryAccessPattern::SCATTERED;
  }
  
  // 检查是否为广播访问（所有线程访问相同位置）
  bool isConstantAccess = true;
  for (Value idx : indices) {
    bool isConstant = false;
    if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(idx.getDefiningOp())) {
      isConstant = true;
    }
    if (!isConstant) {
      isConstantAccess = false;
      break;
    }
  }
  
  if (isConstantAccess) {
    return MemoryAccessPattern::BROADCAST;
  }
  
  // 检查最内层维度是否使用最内层循环变量（完全合并）
  if (indices.back() == inductionVars.back()) {
    return MemoryAccessPattern::COALESCED;
  }
  
  // 检查是否存在循环变量在索引中
  bool hasInductionVar = false;
  for (Value idx : indices) {
    for (Value iv : inductionVars) {
      if (idx == iv) {
        hasInductionVar = true;
        break;
      }
    }
    if (hasInductionVar) break;
  }
  
  if (hasInductionVar) {
    // 进一步分析是否为规律跨步
    return MemoryAccessPattern::STRIDED;
  }
  
  return MemoryAccessPattern::SCATTERED;
}

void TileSizeOptimizer::analyzeDataReuse(std::vector<MemoryAccessInfo> &memAccesses,
                                        const std::vector<LoopInfo> &loopInfos) {
  for (auto &access : memAccesses) {
    // 简化的复用分析
    if (access.pattern == MemoryAccessPattern::COALESCED || 
        access.pattern == MemoryAccessPattern::STRIDED) {
      access.isSpatialReuse = true;
    }
    
    if (access.accessFrequency > 1.0f) {
      access.isTemporalReuse = true;
    }
    
    // 判断是否需要共享内存优化
    if ((access.pattern == MemoryAccessPattern::STRIDED || 
         access.pattern == MemoryAccessPattern::SCATTERED) && 
        access.accessFrequency > 2.0f) {
      access.requiresSharedMemory = true;
    }
  }
}

std::vector<ComputationInfo> TileSizeOptimizer::analyzeEnhancedComputations(scf::ParallelOp parallelOp) {
  std::vector<ComputationInfo> computations;
  
  parallelOp.getBody()->walk([&](Operation *op) {
    ComputationInfo info;
    info.op = op;
    info.opCount = 1;
    info.computeIntensity = 1.0f;
    info.estimatedRegisters = 2; // 默认寄存器使用
    
    // 浮点操作
    if (isa<arith::AddFOp>(op) || isa<arith::SubFOp>(op)) {
      info.isFloatingPoint = true;
      info.computeIntensity = 1.0f;
      info.estimatedRegisters = 3; // src1, src2, dst
      info.isVectorizable = true;
      computations.push_back(info);
    } else if (isa<arith::MulFOp>(op)) {
      info.isFloatingPoint = true;
      info.computeIntensity = 1.2f;
      info.estimatedRegisters = 3;
      info.isVectorizable = true;
      info.isTensorCoreCapable = true; // FMA操作可以用Tensor Core
      computations.push_back(info);
    } else if (isa<arith::DivFOp>(op)) {
      info.isFloatingPoint = true;
      info.computeIntensity = 8.0f; // 除法很昂贵
      info.estimatedRegisters = 4;
      computations.push_back(info);
    }
    // 整数操作
    else if (isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
             isa<arith::MulIOp>(op)) {
      info.isFloatingPoint = false;
      info.computeIntensity = 0.8f;
      info.estimatedRegisters = 3;
      info.isVectorizable = true;
      computations.push_back(info);
    }
    // 特殊操作
    else if (isa<arith::MaximumFOp>(op) || isa<arith::MinimumFOp>(op)) {
      info.isFloatingPoint = true;
      info.computeIntensity = 1.1f;
      info.estimatedRegisters = 3;
      computations.push_back(info);
    }
  });
  
  return computations;
}

ComputationalPattern TileSizeOptimizer::detectComputationalPattern(
    scf::ParallelOp parallelOp,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<ComputationInfo> &computations) {
  
  size_t numDims = parallelOp.getNumLoops();
  
  // 统计操作类型
  int mulCount = 0, addCount = 0, maxCount = 0, totalOps = 0;
  int tensorCoreOps = 0;
  float totalComputeIntensity = 0.0f;
  
  for (const auto &comp : computations) {
    totalOps++;
    totalComputeIntensity += comp.computeIntensity;
    
    if (isa<arith::MulFOp>(comp.op) || isa<arith::MulIOp>(comp.op)) {
      mulCount++;
      if (comp.isTensorCoreCapable) tensorCoreOps++;
    } else if (isa<arith::AddFOp>(comp.op) || isa<arith::AddIOp>(comp.op)) {
      addCount++;
    } else if (isa<arith::MaximumFOp>(comp.op) || isa<arith::MaxSIOp>(comp.op)) {
      maxCount++;
    }
  }
  
  // 统计内存访问
  int loadCount = 0, storeCount = 0, coalescedCount = 0, reusableAccesses = 0;
  llvm::DenseSet<Value> uniqueMemrefs;
  
  for (const auto &access : memAccesses) {
    if (access.isLoad) loadCount++;
    if (access.isStore) storeCount++;
    if (access.pattern == MemoryAccessPattern::COALESCED) coalescedCount++;
    if (access.isTemporalReuse || access.isSpatialReuse) reusableAccesses++;
    uniqueMemrefs.insert(access.memref);
  }
  
  float avgComputeIntensity = totalOps > 0 ? totalComputeIntensity / totalOps : 0.0f;
  float computeToMemoryRatio = (loadCount + storeCount) > 0 ? 
                              static_cast<float>(totalOps) / (loadCount + storeCount) : 0.0f;
  float reuseRatio = (loadCount + storeCount) > 0 ? 
                     static_cast<float>(reusableAccesses) / (loadCount + storeCount) : 0.0f;
  
  // 增强的模式检测逻辑
  if (numDims >= 2 && tensorCoreOps > 0 && mulCount > 0 && addCount > 0 && 
      computeToMemoryRatio > 2.0f && avgComputeIntensity > 1.0f) {
    LLVM_DEBUG(llvm::dbgs() << "Detected MATMUL pattern (tensor core capable)\n");
    return ComputationalPattern::MATMUL;
  }
  
  if (numDims >= 3 && (mulCount > 0 || maxCount > 0) && 
      uniqueMemrefs.size() >= 2 && computeToMemoryRatio > 1.2f && reuseRatio > 0.3f) {
    LLVM_DEBUG(llvm::dbgs() << "Detected CONV pattern\n");
    return ComputationalPattern::CONV;
  }
  
  if (reuseRatio > 0.6f && uniqueMemrefs.size() <= 4 && numDims >= 2) {
    LLVM_DEBUG(llvm::dbgs() << "Detected STENCIL pattern\n");
    return ComputationalPattern::STENCIL;
  }
  
  // 检查归约操作
  bool hasReduce = false;
  parallelOp.getBody()->walk([&](Operation *op) {
    if (isa<scf::ReduceOp>(op)) hasReduce = true;
  });
  
  if (hasReduce || (addCount > loadCount/2) || (maxCount > loadCount/2)) {
    LLVM_DEBUG(llvm::dbgs() << "Detected REDUCTION pattern\n");
    return ComputationalPattern::REDUCTION;
  }
  
  // 检查转置模式
  int strideAccesses = 0;
  for (const auto &access : memAccesses) {
    if (access.pattern == MemoryAccessPattern::STRIDED) strideAccesses++;
  }
  if (strideAccesses > coalescedCount && numDims >= 2) {
    LLVM_DEBUG(llvm::dbgs() << "Detected TRANSPOSE pattern\n");
    return ComputationalPattern::TRANSPOSE;
  }
  
  if (loadCount <= 2 && storeCount <= 1 && totalOps <= 3 && coalescedCount > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Detected ELEMENTWISE pattern\n");
    return ComputationalPattern::ELEMENTWISE;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Using GENERIC pattern\n");
  return ComputationalPattern::GENERIC;
}

std::vector<std::vector<int64_t>> TileSizeOptimizer::generateDivisibleTileCandidates(
    const std::vector<LoopInfo> &loopInfos,
    ComputationalPattern pattern) {
  
  std::vector<std::vector<int64_t>> candidatesPerDim;
  
  for (size_t i = 0; i < loopInfos.size(); ++i) {
    bool isInnermostDim = (i == loopInfos.size() - 1);
    bool isOutermostDim = (i == 0);
    
    std::vector<int64_t> candidates = generateDivisibleCandidatesForDimension(
        loopInfos[i], static_cast<int>(i), pattern, isInnermostDim, isOutermostDim);
    
    candidatesPerDim.push_back(candidates);
  }
  
  return candidatesPerDim;
}

std::vector<int64_t> TileSizeOptimizer::generateDivisibleCandidatesForDimension(
    const LoopInfo &loopInfo, 
    int dimIndex,
    ComputationalPattern pattern,
    bool isInnermostDim,
    bool isOutermostDim) {
  
  std::vector<int64_t> candidates;
  
  // 如果有常量边界，优先使用除数
  if (loopInfo.hasConstantBounds && !loopInfo.possibleDivisors.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Using divisible candidates for dimension " << dimIndex << "\n");
    
    // 添加所有合适的除数
    for (int64_t divisor : loopInfo.possibleDivisors) {
      // 检查硬件限制
      int64_t maxSize = hwParams.maxThreadsPerBlock;
      if (dimIndex == 0) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimX));
      else if (dimIndex == 1) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimY));
      else if (dimIndex == 2) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimZ));
      
      if (divisor <= maxSize && divisor > 0) {
        candidates.push_back(divisor);
      }
    }
  } else {
    // 回退到A100友好的基础候选
    LLVM_DEBUG(llvm::dbgs() << "Using generic A100-friendly candidates for dimension " << dimIndex << "\n");
    
    std::vector<int64_t> baseCandidates = {
      1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512
    };
    
    int64_t maxSize = hwParams.maxThreadsPerBlock;
    if (dimIndex == 0) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimX));
    else if (dimIndex == 1) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimY));
    else if (dimIndex == 2) maxSize = std::min(maxSize, static_cast<int64_t>(hwParams.maxBlockDimZ));
    
    for (int64_t size : baseCandidates) {
      if (size <= maxSize) {
        candidates.push_back(size);
      }
    }
  }
  
  // 新增：强制添加一些性能友好的候选，即使不能完美整除
  std::vector<int64_t> performanceCandidates;
  if (isInnermostDim) {
    performanceCandidates = {16, 32, 64, 128, 256};  // 最内层维度偏好较大值
  } else if (dimIndex == static_cast<int>(loopInfo.dimension) - 2) {  // 次内层
    performanceCandidates = {8, 16, 32, 64};
  } else {  // 外层维度
    performanceCandidates = {1, 2, 4, 8, 16};
  }
  
  for (int64_t candidate : performanceCandidates) {
    if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
      // 检查是否在合理范围内
      if (loopInfo.hasConstantBounds) {
        if (candidate <= loopInfo.iterationCount) {  // 只要不超过总数就加入
          candidates.push_back(candidate);
        }
      } else {
        candidates.push_back(candidate);
      }
    }
  }

  // 最内层维度 - 确保warp对齐
  if (isInnermostDim) {
    // 添加warp的倍数
    for (int mult = 1; mult <= 16; mult++) {
      int64_t size = hwParams.warpSize * mult;
      if (size <= hwParams.maxThreadsPerBlock && 
          std::find(candidates.begin(), candidates.end(), size) == candidates.end()) {
        
        // 如果有常量边界，检查是否能整除
        if (loopInfo.hasConstantBounds) {
          if (loopInfo.iterationCount % size == 0) {
            candidates.push_back(size);
          }
        } else {
          candidates.push_back(size);
        }
      }
    }
    
    // Tensor Core友好大小
    if (pattern == ComputationalPattern::MATMUL || pattern == ComputationalPattern::CONV) {
      for (int mult = 1; mult <= 8; mult++) {
        int64_t size = hwParams.tensorCoreAlignment * mult;
        if (size <= hwParams.maxThreadsPerBlock && 
            std::find(candidates.begin(), candidates.end(), size) == candidates.end()) {
          
          if (loopInfo.hasConstantBounds) {
            if (loopInfo.iterationCount % size == 0) {
              candidates.push_back(size);
            }
          } else {
            candidates.push_back(size);
          }
        }
      }
    }
  }
  
  // 应用启发式过滤
  candidates = filterCandidatesByHeuristics(candidates, loopInfo, pattern);
  
  // 排序并去重
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  
  // 移除0和无效值
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(), 
                                 [](int64_t x) { return x <= 0; }), candidates.end());
  
  // 如果没有候选，至少保留1
  if (candidates.empty()) {
    candidates.push_back(1);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Generated " << candidates.size() << " candidates for dimension " << dimIndex << "\n");
  
  return candidates;
}

std::vector<int64_t> TileSizeOptimizer::filterCandidatesByHeuristics(
    const std::vector<int64_t> &candidates,
    const LoopInfo &loopInfo,
    ComputationalPattern pattern) {
  
  std::vector<int64_t> filtered;
  
  for (int64_t candidate : candidates) {
    bool keep = true;
    
    // 根据模式应用过滤规则
    switch (pattern) {
      case ComputationalPattern::MATMUL:
        // 矩阵乘法倾向于使用较大的tile
        if (loopInfo.isInnerMostLoop && candidate < 16) {
          keep = false; // 最内层至少16
        }
        break;
      
      case ComputationalPattern::ELEMENTWISE:
        // 逐元素操作倾向于使用warp对齐的tile
        if (loopInfo.isInnerMostLoop && candidate % hwParams.warpSize != 0 && candidate > hwParams.warpSize) {
          keep = false;
        }
        break;
      
      case ComputationalPattern::REDUCTION:
        // 归约操作需要考虑warp内同步
        if (loopInfo.isInnerMostLoop && candidate > 256) {
          keep = false; // 避免过大的tile导致同步开销
        }
        break;
      
      default:
        // 通用情况不进行额外过滤
        break;
    }
    
    // 基本合理性检查
    if (loopInfo.hasConstantBounds && candidate > loopInfo.iterationCount) {
      keep = false; // tile不能大于总迭代数
    }
    
    if (keep) {
      filtered.push_back(candidate);
    }
  }
  
  return filtered.empty() ? candidates : filtered; // 如果过滤后为空，返回原始候选
}

CompleteTileConfig TileSizeOptimizer::findOptimalConfigurationSmart(
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<ComputationInfo> &computations,
    const std::vector<std::vector<int64_t>> &candidates,
    ComputationalPattern pattern) {
  
  CompleteTileConfig bestConfig;
  float bestScore = -1.0f;
  int evaluatedConfigs = 0;
  int totalPossibleConfigs = 1;
  
  for (const auto &dimCandidates : candidates) {
    totalPossibleConfigs *= dimCandidates.size();
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Smart search: " << totalPossibleConfigs << " total configurations\n");
  
  // 生成引导候选以减少搜索空间
  std::vector<std::vector<int64_t>> guidedCandidates = generateGuidedCandidates(loopInfos, candidates, pattern);
  
  // 智能搜索算法：优先评估有前途的配置
  std::function<void(std::vector<int64_t>&, int)> smartSearch = 
    [&](std::vector<int64_t> &currentConfig, int dimIndex) {
      if (static_cast<size_t>(dimIndex) == loopInfos.size()) {
        // 评估当前配置
        if (isStrictlyValidConfiguration(currentConfig, loopInfos)) {
          evaluatedConfigs++;
          
          float score = evaluateConfigurationPrecise(currentConfig, loopInfos, memAccesses, computations, pattern);
          if (score > bestScore) {
            bestScore = score;
            
            // 构建最佳配置
            bestConfig.perDimConfig.clear();
            for (size_t i = 0; i < currentConfig.size(); ++i) {
              TileConfig tileConfig;
              tileConfig.tileSize = currentConfig[i];
              tileConfig.efficiencyScore = score;
              
              // 检查是否完全整除
              if (loopInfos[i].hasConstantBounds) {
                tileConfig.isExactlyDivisible = (loopInfos[i].iterationCount % currentConfig[i] == 0);
              } else {
                tileConfig.isExactlyDivisible = true; // 未知情况假设可整除
              }
              
              bestConfig.perDimConfig.push_back(tileConfig);
            }
            
            bestConfig.overallPerformanceScore = score;
            bestConfig.totalThreadsPerBlock = calculateTotalThreads(currentConfig);
            bestConfig.estimatedRegistersPerThread = estimateAccurateRegistersPerThread(computations);
            bestConfig.sharedMemoryUsagePerBlock = estimateAccurateSharedMemoryUsage(currentConfig, memAccesses, pattern);
            bestConfig.workingSetInfo = analyzeWorkingSet(loopInfos, memAccesses, currentConfig);
            bestConfig.divisibilityScore = evaluateStrictDivisibility(currentConfig, loopInfos);
            bestConfig.workingSetScore = evaluateWorkingSetFit(currentConfig, loopInfos, memAccesses);
            bestConfig.warpUtilizationScore = evaluateWarpUtilization(currentConfig);
            bestConfig.isA100Optimized = isStrictlyA100Friendly(currentConfig, loopInfos, pattern);
          }
        }
        return;
      }
      
      // 选择当前维度的候选，优先选择引导候选
      std::vector<int64_t> currentDimCandidates;
      if (static_cast<size_t>(dimIndex) < guidedCandidates.size()) {
        currentDimCandidates = guidedCandidates[dimIndex];
      } else {
        currentDimCandidates = candidates[dimIndex];
      }
      
      // 按优先级排序候选
      std::sort(currentDimCandidates.begin(), currentDimCandidates.end(), 
        [&](int64_t a, int64_t b) {
          // 优先选择能整除的候选
          if (loopInfos[dimIndex].hasConstantBounds) {
            bool aDivisible = (loopInfos[dimIndex].iterationCount % a == 0);
            bool bDivisible = (loopInfos[dimIndex].iterationCount % b == 0);
            if (aDivisible != bDivisible) {
              return aDivisible > bDivisible;
            }
          }
          
          // 最内层维度优先选择warp对齐的
          if (loopInfos[dimIndex].isInnerMostLoop) {
            bool aWarpAligned = (a % hwParams.warpSize == 0);
            bool bWarpAligned = (b % hwParams.warpSize == 0);
            if (aWarpAligned != bWarpAligned) {
              return aWarpAligned > bWarpAligned;
            }
          }
          
          return a < b; // 默认选择较小的值
        });
      
      for (int64_t candidate : currentDimCandidates) {
        currentConfig[dimIndex] = candidate;
        
        // 提前剪枝：检查当前线程数是否合理
        int currentThreads = 1;
        for (int i = 0; i <= dimIndex; i++) {
          currentThreads *= currentConfig[i];
        }
        
        if (currentThreads > hwParams.maxThreadsPerBlock) {
          continue; // 线程数过多，剪枝
        }
        
        // 如果已经评估了足够多的配置，进行启发式剪枝
        if (evaluatedConfigs > 1000 && bestScore > 0.8f) {
          // 对于高分配置，只继续评估有潜力超越的候选
          continue;
        }
        
        smartSearch(currentConfig, dimIndex + 1);
      }
    };
  
  std::vector<int64_t> currentConfig(loopInfos.size(), 1);
  smartSearch(currentConfig, 0);
  
  LLVM_DEBUG(llvm::dbgs() << "Smart search completed: evaluated " << evaluatedConfigs 
                         << "/" << totalPossibleConfigs << " configurations\n");
  
  // 如果没找到有效配置，使用保守默认值
  if (bestScore < 0) {
    bestConfig = getConservativeDefault(loopInfos);
  }
  
  return bestConfig;
}

std::vector<std::vector<int64_t>> TileSizeOptimizer::generateGuidedCandidates(
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<std::vector<int64_t>> &baseCandidates,
    ComputationalPattern pattern) {
  
  std::vector<std::vector<int64_t>> guided = baseCandidates;
  
  // 根据模式调整候选优先级
  for (size_t i = 0; i < guided.size(); ++i) {
    auto &candidates = guided[i];
    
    switch (pattern) {
      case ComputationalPattern::MATMUL:
        // 矩阵乘法：最内层优先tensor core对齐
        if (loopInfos[i].isInnerMostLoop) {
          // 将tensor core对齐的候选移到前面
          std::stable_partition(candidates.begin(), candidates.end(), 
            [this](int64_t x) { return x % hwParams.tensorCoreAlignment == 0; });
        }
        break;
      
      case ComputationalPattern::ELEMENTWISE:
        // 逐元素：优先warp对齐
        if (loopInfos[i].isInnerMostLoop) {
          std::stable_partition(candidates.begin(), candidates.end(), 
            [this](int64_t x) { return x % hwParams.warpSize == 0; });
        }
        break;
      
      default:
        break;
    }
    
    // 限制候选数量以加速搜索
    if (candidates.size() > 10) {
      candidates.resize(10);
    }
  }
  
  return guided;
}

// float TileSizeOptimizer::evaluateConfigurationPrecise(
//     const std::vector<int64_t> &tileSizes,
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses,
//     const std::vector<ComputationInfo> &computations,
//     ComputationalPattern pattern) {
  
//   // 更加精确的权重分配
//   const float DIVISIBILITY_WEIGHT = 0.40f;    // 整除性最重要 - 避免分支
//   const float OCCUPANCY_WEIGHT = 0.30f;       // SM占用率次重要
//   const float MEMORY_WEIGHT = 0.20f;          // 内存效率
//   const float WORKING_SET_WEIGHT = 0.10f;     // 工作集适配性
  
//   float divisibilityScore = evaluateStrictDivisibility(tileSizes, loopInfos);
//   float occupancyScore = evaluatePreciseOccupancy(tileSizes, computations, memAccesses);
//   float memoryScore = evaluatePreciseMemoryEfficiency(tileSizes, loopInfos, memAccesses);
//   float workingSetScore = evaluateWorkingSetFit(tileSizes, loopInfos, memAccesses);
  
//   // 如果不能完全整除，给予严重惩罚
//   if (divisibilityScore < 0.99f) {
//     divisibilityScore *= 0.1f; // 90%的惩罚
//   }
  
//   // A100特定奖励
//   float a100Bonus = isStrictlyA100Friendly(tileSizes, loopInfos, pattern) ? 1.1f : 1.0f;
  
//   float totalScore = (DIVISIBILITY_WEIGHT * divisibilityScore +
//                      OCCUPANCY_WEIGHT * occupancyScore +
//                      MEMORY_WEIGHT * memoryScore +
//                      WORKING_SET_WEIGHT * workingSetScore) * a100Bonus;
  
//   return totalScore;
// }

float TileSizeOptimizer::evaluateConfigurationPrecise(
    const std::vector<int64_t> &tileSizes,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<ComputationInfo> &computations,
    ComputationalPattern pattern) {
  
  int totalThreads = calculateTotalThreads(tileSizes);
  
  // 添加线程数检查 - 新增
  if (totalThreads <= 0 || totalThreads > hwParams.maxThreadsPerBlock) {
    return 0.0f;
  }
  
  // 重新调整的权重
  const float OCCUPANCY_WEIGHT = 0.35f;        // 提高！
  const float DIVISIBILITY_WEIGHT = 0.20f;     // 降低！
  const float MEMORY_WEIGHT = 0.25f;
  const float THREAD_COUNT_WEIGHT = 0.20f;     // 新增！
  
  float divisibilityScore = evaluateFlexibleDivisibility(tileSizes, loopInfos);  // 使用灵活版本
  float occupancyScore = evaluatePreciseOccupancy(tileSizes, computations, memAccesses);
  float memoryScore = evaluatePreciseMemoryEfficiency(tileSizes, loopInfos, memAccesses);
  
  // 新增：线程数合理性评估
  float threadCountScore = evaluateThreadCountReasonableness(totalThreads);
  
  // 特殊模式奖励
  float patternBonus = 1.0f;
  if (pattern == ComputationalPattern::TRANSPOSE && totalThreads >= 256 && totalThreads <= 512) {
    patternBonus = 1.15f;  // 转置操作的线程数奖励
  }
  
  float totalScore = (OCCUPANCY_WEIGHT * occupancyScore +
                     DIVISIBILITY_WEIGHT * divisibilityScore +
                     MEMORY_WEIGHT * memoryScore +
                     THREAD_COUNT_WEIGHT * threadCountScore) * patternBonus;
  
  return std::min(1.0f, totalScore);
}

float TileSizeOptimizer::evaluateThreadCountReasonableness(int totalThreads) {
  const int IDEAL_MIN = 128;
  const int IDEAL_OPTIMAL = 256;
  const int IDEAL_MAX = 512;
  
  if (totalThreads < 64) {
    return 0.1f;  // 太少线程，严重影响性能
  } else if (totalThreads < IDEAL_MIN) {
    return static_cast<float>(totalThreads) / IDEAL_MIN * 0.6f;  // 线性增长到0.6
  } else if (totalThreads <= IDEAL_OPTIMAL) {
    return 0.6f + 0.4f * (totalThreads - IDEAL_MIN) / (IDEAL_OPTIMAL - IDEAL_MIN);  // 0.6到1.0
  } else if (totalThreads <= IDEAL_MAX) {
    return 1.0f - 0.2f * (totalThreads - IDEAL_OPTIMAL) / (IDEAL_MAX - IDEAL_OPTIMAL);  // 1.0到0.8
  } else if (totalThreads <= hwParams.maxThreadsPerBlock) {
    return 0.8f - 0.4f * (totalThreads - IDEAL_MAX) / (hwParams.maxThreadsPerBlock - IDEAL_MAX);  // 0.8到0.4
  } else {
    return 0.0f;  // 超出限制
  }
}

float TileSizeOptimizer::evaluateFlexibleDivisibility(
    const std::vector<int64_t> &tileSizes,
    const std::vector<LoopInfo> &loopInfos) {
  
  if (tileSizes.size() != loopInfos.size()) return 0.5f;
  
  float totalPenalty = 0.0f;
  int validDimensions = 0;
  
  for (size_t i = 0; i < tileSizes.size(); ++i) {
    if (!loopInfos[i].hasConstantBounds) continue;
    
    validDimensions++;
    int64_t iterCount = loopInfos[i].iterationCount;
    int64_t tileSize = tileSizes[i];
    
    if (iterCount % tileSize == 0) {
      // 完美整除，无惩罚
      continue;
    }
    
    // 计算浪费比例并应用渐进式惩罚
    int64_t remainder = iterCount % tileSize;
    float wasteRatio = static_cast<float>(remainder) / tileSize;
    
    if (wasteRatio <= 0.1f) {
      totalPenalty += 0.05f;       // 浪费10%以内：轻微惩罚
    } else if (wasteRatio <= 0.25f) {
      totalPenalty += 0.15f;       // 浪费25%以内：中等惩罚
    } else if (wasteRatio <= 0.5f) {
      totalPenalty += 0.35f;       // 浪费50%以内：较重惩罚
    } else {
      totalPenalty += 0.6f;        // 浪费超过50%：重惩罚
    }
  }
  
  if (validDimensions == 0) return 0.8f;  // 无常量边界时给予较高分
  
  float avgPenalty = totalPenalty / validDimensions;
  return std::max(0.0f, 1.0f - avgPenalty);
}

float TileSizeOptimizer::evaluateStrictDivisibility(
    const std::vector<int64_t> &tileSizes,
    const std::vector<LoopInfo> &loopInfos) {
  
  float totalScore = 1.0f;
  int validDimensions = 0;
  
  for (size_t i = 0; i < tileSizes.size() && i < loopInfos.size(); ++i) {
    if (!loopInfos[i].hasConstantBounds) {
      continue; // 跳过非常量边界，给予好处的怀疑
    }
    
    validDimensions++;
    int64_t iterCount = loopInfos[i].iterationCount;
    int64_t tileSize = tileSizes[i];
    
    if (iterCount % tileSize == 0) {
      // 完美整除 - 满分
      continue;
    } else {
      // 不能整除 - 根据MLIR的特点，这会引入条件分支，严重影响性能
      totalScore = 0.0f; // 严格模式：不整除直接为0分
      break;
    }
  }
  
  // 如果没有常量边界维度，返回中性得分
  if (validDimensions == 0) {
    return 0.7f; // 对未知情况给予中等得分
  }
  
  return totalScore;
}

bool TileSizeOptimizer::isStrictlyValidConfiguration(const std::vector<int64_t> &tileSizes, 
                                                     const std::vector<LoopInfo> &loopInfos) {
  int totalThreads = calculateTotalThreads(tileSizes);
  
  if (totalThreads <= 0 || totalThreads > hwParams.maxThreadsPerBlock) {
    return false;
  }
  
  // 检查维度限制
  for (size_t i = 0; i < tileSizes.size() && i < 3; ++i) {
    int maxDim = (i == 0) ? hwParams.maxBlockDimX :
                (i == 1) ? hwParams.maxBlockDimY : hwParams.maxBlockDimZ;
    if (tileSizes[i] > maxDim) {
      return false;
    }
  }
  
  // 严格检查整除性（如果有常量边界）
  for (size_t i = 0; i < tileSizes.size() && i < loopInfos.size(); ++i) {
    if (loopInfos[i].hasConstantBounds) {
      if (loopInfos[i].iterationCount % tileSizes[i] != 0) {
        return false; // 严格模式：必须整除
      }
    }
  }
  
  return true;
}

WorkingSetInfo TileSizeOptimizer::analyzeWorkingSet(
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<int64_t> &tileSizes) {
  
  WorkingSetInfo info;
  
  // 计算每个tile的总数据大小
  int64_t tileVolume = 1;
  for (int64_t size : tileSizes) {
    tileVolume *= size;
  }
  
  llvm::DenseSet<Value> uniqueMemrefs;
  for (const auto &access : memAccesses) {
    uniqueMemrefs.insert(access.memref);
  }
  
  // 估算工作集大小
  for (const auto &access : memAccesses) {
    int64_t accessSize = tileVolume * access.dataTypeSizeInBytes;
    info.totalDataSize += accessSize;
    
    if (access.isTemporalReuse || access.isSpatialReuse) {
      info.reusableDataSize += accessSize;
    }
    
    if (access.requiresSharedMemory) {
      info.sharedMemoryRequirement += accessSize;
      info.needsSharedMemoryTiling = true;
    }
  }
  
  // 计算缓存适配比例
  info.l1CacheFitRatio = std::min(1.0f, static_cast<float>(hwParams.l1CacheSize) / info.totalDataSize);
  info.l2CacheFitRatio = std::min(1.0f, static_cast<float>(hwParams.l2CacheSize) / info.totalDataSize);
  
  return info;
}

float TileSizeOptimizer::evaluateWorkingSetFit(
    const std::vector<int64_t> &tileSizes,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  WorkingSetInfo wsInfo = analyzeWorkingSet(loopInfos, memAccesses, tileSizes);
  
  float score = 0.0f;
  
  // L1缓存适配性
  if (wsInfo.l1CacheFitRatio >= 0.8f) {
    score += 0.5f; // L1完全适配
  } else if (wsInfo.l1CacheFitRatio >= 0.5f) {
    score += 0.3f; // L1部分适配
  }
  
  // L2缓存适配性
  if (wsInfo.l2CacheFitRatio >= 0.8f) {
    score += 0.3f; // L2完全适配
  } else if (wsInfo.l2CacheFitRatio >= 0.3f) {
    score += 0.2f; // L2部分适配
  }
  
  // 共享内存需求检查
  if (wsInfo.needsSharedMemoryTiling) {
    if (wsInfo.sharedMemoryRequirement <= hwParams.maxSharedMemoryPerBlock) {
      score += 0.2f; // 共享内存需求合理
    } else {
      score -= 0.3f; // 共享内存需求过多
    }
  }
  
  return std::max(0.0f, std::min(1.0f, score));
}

float TileSizeOptimizer::evaluatePreciseOccupancy(
    const std::vector<int64_t> &tileSizes,
    const std::vector<ComputationInfo> &computations,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  int totalThreads = calculateTotalThreads(tileSizes);
  
  if (totalThreads > hwParams.maxThreadsPerBlock || totalThreads <= 0) {
    return 0.0f;
  }
  
  // 精确的寄存器使用估算
  int estimatedRegisters = estimateAccurateRegistersPerThread(computations);
  
  // 精确的共享内存使用估算  
  int estimatedSharedMem = 0;
  for (const auto &access : memAccesses) {
    if (access.requiresSharedMemory) {
      estimatedSharedMem += totalThreads * access.dataTypeSizeInBytes;
    }
  }
  estimatedSharedMem = std::max(estimatedSharedMem, totalThreads * 16); // 最小共享内存
  
  float occupancyScore = calculateA100OccupancyScore(totalThreads, estimatedRegisters, estimatedSharedMem);
  
  // Warp利用率奖励
  float warpUtilization = evaluateWarpUtilization(tileSizes);
  
  return occupancyScore * warpUtilization;
}

float TileSizeOptimizer::evaluateWarpUtilization(const std::vector<int64_t> &tileSizes) {
  int totalThreads = calculateTotalThreads(tileSizes);
  
  if (totalThreads <= 0) return 0.0f;
  
  // 计算warp利用率
  int fullWarps = totalThreads / hwParams.warpSize;
  int remainingThreads = totalThreads % hwParams.warpSize;
  
  float utilization = static_cast<float>(fullWarps * hwParams.warpSize + remainingThreads) / totalThreads;
  
  // 如果有不完整的warp，计算其利用率
  if (remainingThreads > 0) {
    float partialWarpUtilization = static_cast<float>(remainingThreads) / hwParams.warpSize;
    utilization = (static_cast<float>(fullWarps) + partialWarpUtilization) / 
                  ((fullWarps > 0 ? fullWarps : 0) + (remainingThreads > 0 ? 1 : 0));
  }
  
  return utilization;
}

int TileSizeOptimizer::estimateAccurateRegistersPerThread(const std::vector<ComputationInfo> &computations) {
  int totalRegisters = 8; // 基础寄存器使用
  
  for (const auto &comp : computations) {
    totalRegisters += comp.estimatedRegisters;
    
    // 复杂操作需要更多寄存器
    if (isa<arith::DivFOp>(comp.op)) {
      totalRegisters += 4; // 除法需要额外寄存器
    }
    if (comp.isTensorCoreCapable) {
      totalRegisters += 2; // Tensor Core操作需要额外寄存器
    }
  }
  
  return std::min(totalRegisters, hwParams.maxRegistersPerThread);
}

float TileSizeOptimizer::calculateA100OccupancyScore(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock) {
  // 计算每个限制因素允许的blocks数量
  int maxBlocksByThreads = hwParams.maxThreadsPerSM / threadsPerBlock;
  int maxBlocksByRegisters = (registersPerThread > 0) ? 
                            hwParams.maxRegistersPerSM / (registersPerThread * threadsPerBlock) :
                            hwParams.maxBlocksPerSM;
  int maxBlocksBySharedMem = (sharedMemoryPerBlock > 0) ?
                            hwParams.maxSharedMemoryPerSM / sharedMemoryPerBlock :
                            hwParams.maxBlocksPerSM;
  
  int activeBlocks = std::min({hwParams.maxBlocksPerSM, maxBlocksByThreads, 
                              maxBlocksByRegisters, maxBlocksBySharedMem});
  
  int maxWarps = hwParams.maxThreadsPerSM / hwParams.warpSize;
  int warpsPerBlock = (threadsPerBlock + hwParams.warpSize - 1) / hwParams.warpSize;
  int activeWarps = activeBlocks * warpsPerBlock;
  
  float occupancy = static_cast<float>(activeWarps) / maxWarps;
  
  // A100特定的占用率性能曲线 - 更精确建模
  if (occupancy < 0.125f) {
    return occupancy * 4.0f; // 极低占用率性能急剧下降
  } else if (occupancy < 0.25f) {
    return 0.5f + (occupancy - 0.125f) * 2.0f; // 低占用率区域
  } else if (occupancy < 0.5f) {
    return 0.75f + (occupancy - 0.25f) * 1.0f; // 中低占用率区域
  } else if (occupancy < 0.75f) {
    return 1.0f + (occupancy - 0.5f) * 0.0f; // 理想占用率区域
  } else {
    return 1.0f - (occupancy - 0.75f) * 0.4f; // 高占用率收益递减
  }
}

bool TileSizeOptimizer::isStrictlyA100Friendly(const std::vector<int64_t> &tileSizes, 
                                               const std::vector<LoopInfo> &loopInfos,
                                               ComputationalPattern pattern) {
  if (tileSizes.empty()) return false;
  
  int totalThreads = calculateTotalThreads(tileSizes);
  
  // 检查线程数是否合理
  if (totalThreads < 32 || totalThreads > 512) {
    return false; // A100最佳线程数范围
  }
  
  // 检查warp对齐
  if (totalThreads % hwParams.warpSize != 0) {
    return false; // 必须warp对齐
  }
  
  // 检查最内层维度
  if (!tileSizes.empty()) {
    int64_t innermostSize = tileSizes.back();
    
    // 根据模式检查最内层维度
    switch (pattern) {
      case ComputationalPattern::MATMUL:
        if (innermostSize % hwParams.tensorCoreAlignment != 0) {
          return false; // 矩阵乘法需要tensor core对齐
        }
        break;
      
      case ComputationalPattern::ELEMENTWISE:
        if (innermostSize % hwParams.warpSize != 0 && innermostSize > hwParams.warpSize) {
          return false; // 逐元素操作需要warp对齐或小于warp
        }
        break;
        
      default:
        if (innermostSize < 16) {
          return false; // 最内层至少16个线程
        }
        break;
    }
  }
  
  // 检查是否完全整除
  for (size_t i = 0; i < tileSizes.size() && i < loopInfos.size(); ++i) {
    if (loopInfos[i].hasConstantBounds) {
      if (loopInfos[i].iterationCount % tileSizes[i] != 0) {
        return false; // A100友好配置必须完全整除
      }
    }
  }
  
  return true;
}

int TileSizeOptimizer::calculateTotalThreads(const std::vector<int64_t> &tileSizes) {
  int64_t total = 1;
  for (int64_t size : tileSizes) {
    total *= size;
    if (total > hwParams.maxThreadsPerBlock) {
      return hwParams.maxThreadsPerBlock + 1; // 返回无效值
    }
  }
  return static_cast<int>(total);
}

CompleteTileConfig TileSizeOptimizer::getConservativeDefault(const std::vector<LoopInfo> &loopInfos) {
  CompleteTileConfig defaultConfig;
  
  for (size_t i = 0; i < loopInfos.size(); ++i) {
    TileConfig tileConfig;
    
    if (loopInfos[i].hasConstantBounds) {
      // 选择最大的能整除的因子，但不超过合理范围
      std::vector<int64_t> factors = getFactors(loopInfos[i].iterationCount);
      tileConfig.tileSize = 1;
      
      for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        if (*it <= 256 && *it >= 1) { // 合理的范围
          tileConfig.tileSize = *it;
          break;
        }
      }
      
      tileConfig.isExactlyDivisible = true;
    } else {
      // 未知边界情况使用保守值
      if (i == loopInfos.size() - 1) {
        tileConfig.tileSize = hwParams.warpSize; // 最内层使用warp大小
      } else {
        tileConfig.tileSize = 8; // 其他维度使用小值
      }
      tileConfig.isExactlyDivisible = false;
    }
    
    tileConfig.efficiencyScore = 0.5f;
    defaultConfig.perDimConfig.push_back(tileConfig);
  }
  
  std::vector<int64_t> sizes;
  for (const auto &config : defaultConfig.perDimConfig) {
    sizes.push_back(config.tileSize);
  }
  
  defaultConfig.totalThreadsPerBlock = calculateTotalThreads(sizes);
  defaultConfig.overallPerformanceScore = 0.5f;
  defaultConfig.occupancyScore = 0.5f;
  defaultConfig.memoryEfficiencyScore = 0.5f;
  defaultConfig.computeUtilizationScore = 0.5f;
  defaultConfig.workingSetScore = 0.5f;
  defaultConfig.divisibilityScore = 1.0f; // 保守默认应该是整除的
  defaultConfig.warpUtilizationScore = 0.8f;
  defaultConfig.isA100Optimized = false;
  
  return defaultConfig;
}

// 其他辅助函数的实现...
float TileSizeOptimizer::evaluatePreciseMemoryEfficiency(
    const std::vector<int64_t> &tileSizes,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  if (memAccesses.empty()) return 0.5f;
  
  float totalScore = 0.0f;
  float totalWeight = 0.0f;
  
  for (const auto &access : memAccesses) {
    float weight = access.accessFrequency;
    totalWeight += weight;
    
    float accessScore = 0.0f;
    
    switch (access.pattern) {
      case MemoryAccessPattern::COALESCED:
        // 检查最内层tile大小的合并访问效率
        if (!tileSizes.empty()) {
          int64_t innermostSize = tileSizes.back();
          if (innermostSize % hwParams.warpSize == 0) {
            accessScore = 1.0f; // 完美合并
          } else if (innermostSize >= hwParams.warpSize) {
            accessScore = 0.8f; // 大部分合并
          } else {
            accessScore = static_cast<float>(innermostSize) / hwParams.warpSize * 0.7f;
          }
        }
        break;
        
      case MemoryAccessPattern::STRIDED:
        // 跨步访问的效率取决于跨步大小和缓存行
        accessScore = 0.4f;
        if (access.isSpatialReuse) {
          accessScore += 0.2f; // 有空间局部性的跨步访问
        }
        break;
        
      case MemoryAccessPattern::BROADCAST:
        accessScore = 0.9f; // 广播访问通常很高效
        break;
        
      case MemoryAccessPattern::PARTIAL_COALESCED:
        accessScore = 0.6f;
        break;
        
      default:
        accessScore = 0.2f; // 散乱访问
        break;
    }
    
    // 考虑数据复用
    if (access.isTemporalReuse) {
      accessScore *= 1.2f; // 时间复用奖励
    }
    if (access.isSpatialReuse) {
      accessScore *= 1.1f; // 空间复用奖励
    }
    
    totalScore += accessScore * weight;
  }
  
  return totalWeight > 0 ? (totalScore / totalWeight) : 0.5f;
}

int TileSizeOptimizer::estimateAccurateSharedMemoryUsage(
    const std::vector<int64_t> &tileSizes,
    const std::vector<MemoryAccessInfo> &memAccesses,
    ComputationalPattern pattern) {
  
  int totalSharedMem = 0;
  int64_t tileVolume = 1;
  
  for (int64_t size : tileSizes) {
    tileVolume *= size;
  }
  
  // 根据计算模式估算共享内存需求
  switch (pattern) {
    case ComputationalPattern::MATMUL:
      // 矩阵乘法通常需要输入矩阵的tile在共享内存中
      for (const auto &access : memAccesses) {
        if (access.requiresSharedMemory || access.isTemporalReuse) {
          totalSharedMem += tileVolume * access.dataTypeSizeInBytes / 4; // 部分数据在共享内存
        }
      }
      break;
      
    case ComputationalPattern::CONV:
      // 卷积操作需要输入和权重在共享内存中
      for (const auto &access : memAccesses) {
        if (access.isLoad && (access.isTemporalReuse || access.isSpatialReuse)) {
          totalSharedMem += tileVolume * access.dataTypeSizeInBytes / 8; // 估算
        }
      }
      break;
      
    case ComputationalPattern::STENCIL:
      // 模板操作需要邻域数据在共享内存中
      totalSharedMem = tileVolume * 4 * 1.2; // 假设float+20%的halo
      break;
      
    default:
      // 默认情况下的基本共享内存使用
      totalSharedMem = std::max(static_cast<int>(tileVolume * 16), 1024); // 至少1KB
      break;
  }
  
  return std::min(totalSharedMem, hwParams.maxSharedMemoryPerBlock);
}