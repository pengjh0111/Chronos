// // #ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_
// // #define MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

// // #include "mlir/Dialect/SCF/IR/SCF.h"
// // #include "mlir/Dialect/MemRef/IR/MemRef.h"
// // #include "mlir/Dialect/Arith/IR/Arith.h"
// // #include "mlir/IR/Builders.h"
// // #include "mlir/IR/BuiltinOps.h"
// // #include "mlir/IR/MLIRContext.h"
// // #include "mlir/IR/PatternMatch.h"
// // #include "mlir/Support/LogicalResult.h"
// // #include "llvm/ADT/DenseSet.h"

// // #include <vector>
// // #include <unordered_map>
// // #include <set>
// // #include <cmath>
// // #include <limits>
// // #include <algorithm>


// // using namespace mlir;
// // using namespace mlir::scf;
// // using namespace llvm;

// // /// GPU硬件参数结构体，用于性能模型计算
// // struct GpuHardwareParameters {
// //   // 常见NVIDIA GPU参数
// //   int maxThreadsPerBlock = 1024;
// //   int maxBlockDimX = 1024;
// //   int maxBlockDimY = 1024;
// //   int maxBlockDimZ = 64;
// //   int warpSize = 32;
// //   int maxRegistersPerThread = 255;
// //   int maxRegistersPerSM = 65536;
// //   int maxThreadsPerSM = 2048;
// //   int maxBlocksPerSM = 32;
// //   int maxSharedMemoryPerSM = 49152; // 字节
// //   float peakComputePerformance = 9.7e12;  // FLOPS
// //   float memoryBandwidth = 1.6e12;         // 字节/秒
// //   int l1CacheSize = 128 * 1024;    // 字节
// //   int l2CacheSize = 6 * 1024 * 1024; // 字节
// // };

// // /// 内存访问模式枚举
// // enum class MemoryAccessPattern {
// //   SEQUENTIAL,  // 相同线程的顺序访问
// //   COALESCED,   // 相邻线程访问相邻内存
// //   STRIDED,     // 跨步访问
// //   RANDOM       // 不规则访问模式
// // };

// // /// 从MLIR并行循环提取的循环信息
// // struct LoopInfo {
// //   int dimension;
// //   Value lowerBound;
// //   Value upperBound;
// //   Value step;
// //   int64_t constantLowerBound = -1;
// //   int64_t constantUpperBound = -1;
// //   int64_t constantStep = -1;
// //   bool hasConstantBounds = false;
// //   int64_t tripCount = -1;
// // };

// // /// 内存访问信息，用于性能建模
// // struct MemoryAccessInfo {
// //   Value memref;
// //   SmallVector<Value, 4> indices;
// //   bool isLoad;
// //   bool isStore;
// //   MemoryAccessPattern pattern;
// //   int dataTypeSizeInBytes;
// // };

// // /// 计算操作信息
// // struct ComputationInfo {
// //   Operation *op;
// //   int64_t opCount;
// //   bool isFloatingPoint;
// // };

// // /// 单维度tile配置
// // struct TileConfig {
// //   int64_t tileSize;
// //   float performanceScore;
// // };

// // /// 所有维度的完整tile配置
// // struct CompleteTileConfig {
// //   std::vector<TileConfig> perDimConfig;
// //   float overallPerformanceScore;
// // };

// // /// Tile大小优化器 - 使用启发式规则和动态规划
// // class TileSizeOptimizer {
// // public:
// //   TileSizeOptimizer(MLIRContext *context, const GpuHardwareParameters &params = GpuHardwareParameters())
// //       : context(context), hwParams(params) {}

// //   /// 分析并行循环并返回最优tile配置
// //   CompleteTileConfig optimizeTileSize(scf::ParallelOp parallelOp);

// //     /// Recognized computational patterns for specialized tuning
// //     enum class ComputationalPattern {
// //     GENERIC,     // General case
// //     MATMUL,      // Matrix multiplication patterns
// //     CONV,        // Convolution patterns
// //     REDUCTION,   // Reduction operations (sum, max, etc.)
// //     STENCIL,     // Stencil computations (e.g., 2D/3D neighbor operations)
// //     ELEMENTWISE  // Element-wise operations (map operations)
// //     };

// // private:
// //   MLIRContext *context;
// //   GpuHardwareParameters hwParams;

// //   /// 从并行循环中提取循环信息
// //   std::vector<LoopInfo> extractLoopInfo(scf::ParallelOp parallelOp);
  
// //   /// 分析循环体中的内存访问模式
// //   std::vector<MemoryAccessInfo> analyzeMemoryAccesses(scf::ParallelOp parallelOp);
  
// //   /// 分析循环体中的计算操作
// //   std::vector<ComputationInfo> analyzeComputations(scf::ParallelOp parallelOp);
  
// // /// Additional method declarations
// // ComputationalPattern detectComputationalPattern(
// //     scf::ParallelOp parallelOp,
// //     const std::vector<MemoryAccessInfo> &memAccesses);

// // /// Updated method signature for pattern-aware candidate generation
// // std::vector<std::vector<int64_t>> generateTileSizeCandidates(
// //     const std::vector<LoopInfo> &loopInfos,
// //     ComputationalPattern pattern = ComputationalPattern::GENERIC);

// // //   /// 使用启发式规则为每个维度生成候选tile大小
// // //   std::vector<std::vector<int64_t>> generateTileSizeCandidates(const std::vector<LoopInfo> &loopInfos);
  
// //   /// 使用动态规划查找最优tile配置
// //   CompleteTileConfig findOptimalTileConfig(
// //       const std::vector<LoopInfo> &loopInfos,
// //       const std::vector<MemoryAccessInfo> &memAccesses,
// //       const std::vector<ComputationInfo> &computations,
// //       const std::vector<std::vector<int64_t>> &tileSizeCandidates);
  
// //   /// 检查tile配置是否满足硬件约束
// //   bool isValidTileConfig(const CompleteTileConfig &config);

// //   // ===== 性能评估模型方法 =====
  
// //   /// 评估完整tile配置的性能
// //     float evaluateConfig(
// //         const CompleteTileConfig &config,
// //         const std::vector<LoopInfo> &loopInfos,
// //         const std::vector<MemoryAccessInfo> &memAccesses,
// //         const std::vector<ComputationInfo> &computations);

// //     /// 基于计算和内存访问评估算术强度 - 已修改
// //     float evaluateArithmeticIntensity(
// //         const CompleteTileConfig &config,
// //         const std::vector<ComputationInfo> &computations,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// 基于资源使用评估占用率 - 已修改
// //     float evaluateOccupancy(
// //         const CompleteTileConfig &config,
// //         const std::vector<ComputationInfo> &computations);

// //     /// 评估内存访问效率
// //     float evaluateMemoryEfficiency(
// //         const CompleteTileConfig &config,
// //         const std::vector<LoopInfo> &loopInfos,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// 评估负载均衡
// //     float evaluateLoadBalancing(
// //         const CompleteTileConfig &config,
// //         const std::vector<LoopInfo> &loopInfos);

// //     /// 估计合并内存访问效率
// //     float estimateCoalescedAccess(
// //         const CompleteTileConfig &config,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// 估计缓存利用率
// //     float estimateCacheUtilization(
// //         const CompleteTileConfig &config,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// 估计每线程寄存器使用量
// //     int estimateRegistersPerThread(const std::vector<ComputationInfo> &computations);

// //     /// 计算共享内存使用量
// //     int calculateSharedMemoryUsage(
// //         const CompleteTileConfig &config,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// Evaluate data reuse potential
// //     float evaluateDataReuse(
// //         const CompleteTileConfig &config,
// //         const std::vector<LoopInfo> &loopInfos,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     /// Estimate shared memory bank conflicts
// //     float estimateMemoryBankConflicts(
// //         const CompleteTileConfig &config,
// //         const std::vector<MemoryAccessInfo> &memAccesses);

// //     // Add any other new function declarations here
// // };



// // #endif // MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

// #ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_
// #define MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Support/LogicalResult.h"
// #include "llvm/ADT/DenseSet.h"

// #include <vector>
// #include <unordered_map>
// #include <set>
// #include <cmath>
// #include <limits>
// #include <algorithm>

// using namespace mlir;
// using namespace mlir::scf;
// using namespace llvm;

// /// A100 GPU硬件参数 - 针对实际硬件优化
// struct A100HardwareParameters {
//   // 基本限制
//   int maxThreadsPerBlock = 1024;
//   int maxBlockDimX = 1024;
//   int maxBlockDimY = 1024; 
//   int maxBlockDimZ = 64;
//   int warpSize = 32;
  
//   // 寄存器和内存
//   int maxRegistersPerThread = 255;
//   int maxRegistersPerSM = 65536;
//   int maxThreadsPerSM = 2048;
//   int maxBlocksPerSM = 32;
//   int maxSharedMemoryPerSM = 164 * 1024; // 164KB
  
//   // 性能参数
//   int numSMs = 108;
//   float peakComputePerformance = 19.5e12;  // 19.5 TFLOPS (FP32)
//   float memoryBandwidth = 1935e9;          // 1935 GB/s
//   int l1CacheSize = 192 * 1024;            // 192KB per SM
//   int l2CacheSize = 40 * 1024 * 1024;      // 40MB shared
  
//   // A100特定优化参数
//   int tensorCoreAlignment = 16;  // Tensor Core要求16x16对齐
//   int preferredWarpMultiple = 4; // 偏好4个warp的倍数
//   int minEffectiveOccupancy = 50; // 最小有效占用率百分比
// };

// /// 内存访问模式
// enum class MemoryAccessPattern {
//   COALESCED,   // 完全合并访问
//   PARTIAL_COALESCED, // 部分合并
//   STRIDED,     // 规律跨步
//   SCATTERED    // 不规律访问
// };

// /// 计算模式
// enum class ComputationalPattern {
//   MATMUL,      // 矩阵乘法 - 高计算强度
//   CONV,        // 卷积 - 中等计算强度，复杂内存模式
//   ELEMENTWISE, // 逐元素 - 内存带宽受限
//   REDUCTION,   // 归约 - 需要同步
//   STENCIL,     // 模板 - 高内存复用
//   GENERIC      // 通用
// };

// /// 循环信息
// struct LoopInfo {
//   int dimension;
//   Value lowerBound;
//   Value upperBound; 
//   Value step;
//   int64_t constantLowerBound = 0;
//   int64_t constantUpperBound = -1;
//   int64_t constantStep = 1;
//   bool hasConstantBounds = false;
//   int64_t iterationCount = -1;
// };

// /// 内存访问信息
// struct MemoryAccessInfo {
//   Value memref;
//   SmallVector<Value, 4> indices;
//   bool isLoad;
//   bool isStore;
//   MemoryAccessPattern pattern;
//   int dataTypeSizeInBytes;
//   float accessFrequency; // 在循环中的访问频率
// };

// /// 计算操作信息  
// struct ComputationInfo {
//   Operation *op;
//   int64_t opCount;
//   bool isFloatingPoint;
//   float computeIntensity; // 操作的计算强度权重
// };

// /// 单维度tile配置
// struct TileConfig {
//   int64_t tileSize;
//   float efficiencyScore; // 该维度的效率得分
// };

// /// 完整tile配置
// struct CompleteTileConfig {
//   std::vector<TileConfig> perDimConfig;
//   float overallPerformanceScore;
  
//   // 性能分析信息
//   float occupancyScore;
//   float memoryEfficiencyScore;
//   float computeUtilizationScore;
//   int totalThreadsPerBlock;
//   int estimatedActiveBlocks;
// };

// /// 重构的Tile大小优化器 - 专注于A100性能
// class TileSizeOptimizer {
// public:
//   TileSizeOptimizer(MLIRContext *context, const A100HardwareParameters &params = A100HardwareParameters())
//       : context(context), hwParams(params) {}

//   /// 主优化接口
//   CompleteTileConfig optimizeTileSize(scf::ParallelOp parallelOp);

// private:
//   MLIRContext *context;
//   A100HardwareParameters hwParams;

//   // ==== 分析阶段 ====
//   std::vector<LoopInfo> extractLoopInfo(scf::ParallelOp parallelOp);
//   std::vector<MemoryAccessInfo> analyzeMemoryAccesses(scf::ParallelOp parallelOp);
//   std::vector<ComputationInfo> analyzeComputations(scf::ParallelOp parallelOp);
//   ComputationalPattern detectComputationalPattern(
//       scf::ParallelOp parallelOp,
//       const std::vector<MemoryAccessInfo> &memAccesses,
//       const std::vector<ComputationInfo> &computations);

//   // ==== 候选生成 - 重新设计 ====
//   std::vector<std::vector<int64_t>> generateSmartTileCandidates(
//       const std::vector<LoopInfo> &loopInfos,
//       ComputationalPattern pattern);
  
//   std::vector<int64_t> generateDimensionCandidates(
//       const LoopInfo &loopInfo, 
//       int dimIndex,
//       ComputationalPattern pattern,
//       bool isInnermostDim,
//       bool isOutermostDim);

//   // ==== 优化搜索 - 改进算法 ====
//   CompleteTileConfig findOptimalConfiguration(
//       const std::vector<LoopInfo> &loopInfos,
//       const std::vector<MemoryAccessInfo> &memAccesses,
//       const std::vector<ComputationInfo> &computations,
//       const std::vector<std::vector<int64_t>> &candidates,
//       ComputationalPattern pattern);

//   // ==== 性能评估 - 简化且聚焦关键因素 ====
//   float evaluateConfiguration(
//       const std::vector<int64_t> &tileSizes,
//       const std::vector<LoopInfo> &loopInfos,
//       const std::vector<MemoryAccessInfo> &memAccesses,
//       const std::vector<ComputationInfo> &computations,
//       ComputationalPattern pattern);

//   // 核心性能因素
//   float evaluateOccupancy(const std::vector<int64_t> &tileSizes);
//   float evaluateMemoryEfficiency(
//       const std::vector<int64_t> &tileSizes,
//       const std::vector<LoopInfo> &loopInfos,
//       const std::vector<MemoryAccessInfo> &memAccesses);
//   float evaluateComputeUtilization(
//       const std::vector<int64_t> &tileSizes,
//       const std::vector<ComputationInfo> &computations,
//       ComputationalPattern pattern);

//   // 整除性评估 - 避免分支发散
//   float evaluateDivisibility(
//       const std::vector<int64_t> &tileSizes,
//       const std::vector<LoopInfo> &loopInfos);

//   // 辅助函数
//   bool isValidConfiguration(const std::vector<int64_t> &tileSizes);
//   int calculateTotalThreads(const std::vector<int64_t> &tileSizes);
//   int estimateRegistersPerThread(const std::vector<ComputationInfo> &computations);
//   int estimateSharedMemoryUsage(
//       const std::vector<int64_t> &tileSizes,
//       const std::vector<MemoryAccessInfo> &memAccesses);
  
//   // 内存访问模式分析
//   MemoryAccessPattern analyzeAccessPattern(
//       const SmallVector<Value, 4> &indices, 
//       ValueRange inductionVars);
  
//   // A100特定优化
//   float calculateA100OccupancyScore(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock);
//   bool isA100Friendly(const std::vector<int64_t> &tileSizes, ComputationalPattern pattern);
// };

// #endif // MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace mlir;
using namespace mlir::scf;
using namespace llvm;

/// A100 GPU硬件参数 - 精确的硬件建模
struct A100HardwareParameters {
  // 基本限制
  int maxThreadsPerBlock = 1024;
  int maxBlockDimX = 1024;
  int maxBlockDimY = 1024; 
  int maxBlockDimZ = 64;
  int warpSize = 32;
  
  // 寄存器和内存 - 更精确的参数
  int maxRegistersPerThread = 255;
  int maxRegistersPerSM = 65536;
  int maxThreadsPerSM = 2048;
  int maxBlocksPerSM = 32;
  int maxSharedMemoryPerSM = 164 * 1024; // 164KB
  int maxSharedMemoryPerBlock = 48 * 1024; // 48KB per block
  
  // 性能参数
  int numSMs = 108;
  float peakComputePerformance = 19.5e12;  // 19.5 TFLOPS (FP32)
  float memoryBandwidth = 1935e9;          // 1935 GB/s
  int l1CacheSize = 192 * 1024;            // 192KB per SM
  int l2CacheSize = 40 * 1024 * 1024;      // 40MB shared
  
  // A100特定优化参数
  int tensorCoreAlignment = 16;  // Tensor Core要求16x16对齐
  int preferredWarpMultiple = 4; // 偏好4个warp的倍数
  int minEffectiveOccupancy = 50; // 最小有效占用率百分比
  
  // 新增：缓存和内存层次结构参数
  int l1CacheLineSize = 128;     // L1缓存行大小
  int l2CacheLineSize = 128;     // L2缓存行大小
  float l1CacheHitLatency = 4.0f;   // L1命中延迟（周期）
  float l2CacheHitLatency = 200.0f; // L2命中延迟（周期）
  float globalMemoryLatency = 400.0f; // 全局内存延迟（周期）
};

/// 内存访问模式 - 扩展分类
enum class MemoryAccessPattern {
  COALESCED,           // 完全合并访问
  PARTIAL_COALESCED,   // 部分合并
  STRIDED,             // 规律跨步
  SCATTERED,           // 不规律访问
  BROADCAST,           // 广播访问（所有线程访问相同位置）
  REDUCTION_FRIENDLY   // 适合归约操作的访问模式
};

/// 计算模式 - 扩展分类
enum class ComputationalPattern {
  MATMUL,              // 矩阵乘法 - 高计算强度
  CONV,                // 卷积 - 中等计算强度，复杂内存模式
  ELEMENTWISE,         // 逐元素 - 内存带宽受限
  REDUCTION,           // 归约 - 需要同步
  STENCIL,             // 模板 - 高内存复用
  TRANSPOSE,           // 转置 - 内存访问模式复杂
  GENERIC              // 通用
};

/// 循环信息 - 增强版
struct LoopInfo {
  int dimension;
  Value lowerBound;
  Value upperBound; 
  Value step;
  int64_t constantLowerBound = 0;
  int64_t constantUpperBound = -1;
  int64_t constantStep = 1;
  bool hasConstantBounds = false;
  int64_t iterationCount = -1;
  
  // 新增：循环特性分析
  bool isOuterMostLoop = false;
  bool isInnerMostLoop = false;
  float accessDensity = 1.0f;      // 访问密度：该维度上的内存访问频率
  float computeDensity = 1.0f;     // 计算密度：该维度上的计算操作频率
  std::vector<int64_t> possibleDivisors; // 可能的除数列表
};

/// 内存访问信息 - 增强版
struct MemoryAccessInfo {
  Value memref;
  SmallVector<Value, 4> indices;
  bool isLoad;
  bool isStore;
  MemoryAccessPattern pattern;
  int dataTypeSizeInBytes;
  float accessFrequency; // 在循环中的访问频率
  
  // 新增：内存访问特性
  int64_t strideInBytes = 1;       // 跨步大小（字节）
  bool isTemporalReuse = false;    // 是否有时间局部性
  bool isSpatialReuse = false;     // 是否有空间局部性
  int reuseDistance = -1;          // 复用距离
  bool requiresSharedMemory = false; // 是否需要共享内存优化
};

/// 计算操作信息 - 增强版
struct ComputationInfo {
  Operation *op;
  int64_t opCount;
  bool isFloatingPoint;
  float computeIntensity; // 操作的计算强度权重
  
  // 新增：寄存器使用分析
  int estimatedRegisters = 2;      // 估算的寄存器使用量
  bool isVectorizable = false;     // 是否可向量化
  bool isTensorCoreCapable = false; // 是否可使用Tensor Core
};

/// 工作集分析结果
struct WorkingSetInfo {
  int64_t totalDataSize = 0;           // 总工作集大小（字节）
  int64_t reusableDataSize = 0;        // 可复用数据大小（字节）
  float l1CacheFitRatio = 0.0f;        // L1缓存适配比例
  float l2CacheFitRatio = 0.0f;        // L2缓存适配比例
  bool needsSharedMemoryTiling = false; // 是否需要共享内存tiling
  int64_t sharedMemoryRequirement = 0;  // 共享内存需求（字节）
};

/// 单维度tile配置 - 增强版
struct TileConfig {
  int64_t tileSize;
  float efficiencyScore; // 该维度的效率得分
  bool isExactlyDivisible = false; // 是否完全整除
  float divisibilityPenalty = 0.0f; // 不整除的惩罚
};

/// 完整tile配置 - 增强版
struct CompleteTileConfig {
  std::vector<TileConfig> perDimConfig;
  float overallPerformanceScore;
  
  // 性能分析信息 - 更详细
  float occupancyScore;
  float memoryEfficiencyScore;
  float computeUtilizationScore;
  float workingSetScore;        // 工作集适配得分
  float divisibilityScore;      // 整除性得分
  
  int totalThreadsPerBlock;
  int estimatedActiveBlocks;
  int estimatedRegistersPerThread;
  int sharedMemoryUsagePerBlock;
  WorkingSetInfo workingSetInfo;
  
  // A100特定指标
  float warpUtilizationScore;   // warp利用率得分
  float tensorCoreUtilization = 0.0f; // Tensor Core利用率
  bool isA100Optimized = false; // 是否A100优化
};

/// 重构的Tile大小优化器 - 专注于A100性能和强制整除性
class TileSizeOptimizer {
public:
  TileSizeOptimizer(MLIRContext *context, const A100HardwareParameters &params = A100HardwareParameters())
      : context(context), hwParams(params) {}

  /// 主优化接口
  CompleteTileConfig optimizeTileSize(scf::ParallelOp parallelOp);

private:
  MLIRContext *context;
  A100HardwareParameters hwParams;

  // ==== 分析阶段 - 增强版 ====
  std::vector<LoopInfo> extractEnhancedLoopInfo(scf::ParallelOp parallelOp);
  std::vector<MemoryAccessInfo> analyzeEnhancedMemoryAccesses(scf::ParallelOp parallelOp);
  std::vector<ComputationInfo> analyzeEnhancedComputations(scf::ParallelOp parallelOp);
  ComputationalPattern detectComputationalPattern(
      scf::ParallelOp parallelOp,
      const std::vector<MemoryAccessInfo> &memAccesses,
      const std::vector<ComputationInfo> &computations);
  
  // 工作集分析
  WorkingSetInfo analyzeWorkingSet(
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses,
      const std::vector<int64_t> &tileSizes);
  
  // 可除性分析
  void computeDivisibilityInfo(std::vector<LoopInfo> &loopInfos);

  // ==== 候选生成 - 完全重新设计强制整除 ====
  std::vector<std::vector<int64_t>> generateDivisibleTileCandidates(
      const std::vector<LoopInfo> &loopInfos,
      ComputationalPattern pattern);
  
  std::vector<int64_t> generateDivisibleCandidatesForDimension(
      const LoopInfo &loopInfo, 
      int dimIndex,
      ComputationalPattern pattern,
      bool isInnermostDim,
      bool isOutermostDim);
  
  // 智能候选过滤
  std::vector<int64_t> filterCandidatesByHeuristics(
      const std::vector<int64_t> &candidates,
      const LoopInfo &loopInfo,
      ComputationalPattern pattern);

  // ==== 智能搜索算法 - 替代暴力搜索 ====
  CompleteTileConfig findOptimalConfigurationSmart(
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses,
      const std::vector<ComputationInfo> &computations,
      const std::vector<std::vector<int64_t>> &candidates,
      ComputationalPattern pattern);
  
  // 支配关系剪枝
  bool isDominatedConfiguration(
      const std::vector<int64_t> &config1,
      const std::vector<int64_t> &config2,
      const std::vector<LoopInfo> &loopInfos);
  
  // 启发式引导搜索
  std::vector<std::vector<int64_t>> generateGuidedCandidates(
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<std::vector<int64_t>> &baseCandidates,
      ComputationalPattern pattern);

  // ==== 精确性能评估 ====
  float evaluateConfigurationPrecise(
      const std::vector<int64_t> &tileSizes,
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses,
      const std::vector<ComputationInfo> &computations,
      ComputationalPattern pattern);

  // 精确的SM占用率评估
  float evaluatePreciseOccupancy(
      const std::vector<int64_t> &tileSizes,
      const std::vector<ComputationInfo> &computations,
      const std::vector<MemoryAccessInfo> &memAccesses);
  
  // 精确的内存效率评估
  float evaluatePreciseMemoryEfficiency(
      const std::vector<int64_t> &tileSizes,
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses);
  
  // 工作集适配性评估
  float evaluateWorkingSetFit(
      const std::vector<int64_t> &tileSizes,
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses);
  
  // 强制整除性评估 - 0容忍
  float evaluateStrictDivisibility(
      const std::vector<int64_t> &tileSizes,
      const std::vector<LoopInfo> &loopInfos);
  
  // Warp利用率评估
  float evaluateWarpUtilization(const std::vector<int64_t> &tileSizes);
  
  // 共享内存优化评估
  float evaluateSharedMemoryOptimization(
      const std::vector<int64_t> &tileSizes,
      const std::vector<MemoryAccessInfo> &memAccesses,
      ComputationalPattern pattern);

  // ==== 辅助函数 ====
  bool isStrictlyValidConfiguration(const std::vector<int64_t> &tileSizes, 
                                   const std::vector<LoopInfo> &loopInfos);
  int calculateTotalThreads(const std::vector<int64_t> &tileSizes);
  int estimateAccurateRegistersPerThread(const std::vector<ComputationInfo> &computations);
  int estimateAccurateSharedMemoryUsage(
      const std::vector<int64_t> &tileSizes,
      const std::vector<MemoryAccessInfo> &memAccesses,
      ComputationalPattern pattern);
  
  // 内存访问模式分析 - 增强版
  MemoryAccessPattern analyzeAccessPatternEnhanced(
      const SmallVector<Value, 4> &indices, 
      ValueRange inductionVars);
  
  // 复用分析
  void analyzeDataReuse(std::vector<MemoryAccessInfo> &memAccesses,
                       const std::vector<LoopInfo> &loopInfos);
  
  // A100特定优化
  float calculateA100OccupancyScore(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock);
  bool isStrictlyA100Friendly(const std::vector<int64_t> &tileSizes, 
                              const std::vector<LoopInfo> &loopInfos,
                              ComputationalPattern pattern);
  
  float evaluateThreadCountReasonableness(int totalThreads);
  float evaluateFlexibleDivisibility(
        const std::vector<int64_t> &tileSizes,
        const std::vector<LoopInfo> &loopInfos);

  // 工具函数：计算最大公因数和因子分解
  std::vector<int64_t> getFactors(int64_t number);
  int64_t gcd(int64_t a, int64_t b);
  std::vector<int64_t> getCommonDivisors(const std::vector<int64_t> &numbers);
  
  // 保守默认配置生成
  CompleteTileConfig getConservativeDefault(const std::vector<LoopInfo> &loopInfos);
};

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_