// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/DenseMap.h"
// #include <vector>
// #include <set>

// using namespace mlir;
// using namespace mlir::affine;

// #define DEBUG_TYPE "convert-memcpy-to-affine"

// namespace {

// // 存储单个 memcpy 操作的信息
// struct MemcpyInfo {
//   Operation* memcpyOp;
//   Value source;
//   Value target;
//   bool sourceIsSubview;
//   bool targetIsSubview;
  
//   // Subview 信息
//   Value baseMemref;  // 如果是subview，记录base memref
//   SmallVector<int64_t, 4> offsets;
//   SmallVector<int64_t, 4> sizes;
//   SmallVector<int64_t, 4> strides;
  
//   Location loc;
  
//   MemcpyInfo(Location location) : memcpyOp(nullptr), sourceIsSubview(false), 
//                                    targetIsSubview(false), loc(location) {}
// };

// // 存储一组相关的 memcpy 操作（concat 或 split）
// struct MemcpyGroup {
//   bool isConcatPattern;  // true: concat, false: split
//   Value largeMemref;     // 大的 memref (e.g., 64x256x256)
//   std::vector<Value> smallMemrefs;  // 小的 memrefs (e.g., 8x256x256)
//   std::vector<MemcpyInfo> memcpyOps;
//   std::vector<int64_t> axisOffsets;  // 每个分片在axis上的偏移
//   int64_t axis;  // 拼接/拆分的轴
//   SmallVector<int64_t, 4> largeShape;
//   SmallVector<int64_t, 4> smallShape;
//   Location loc;
  
//   MemcpyGroup(Location location) : isConcatPattern(false), axis(0), loc(location) {}
// };

// class ConvertMemcpyToAffinePass
//     : public PassWrapper<ConvertMemcpyToAffinePass, OperationPass<func::FuncOp>> {

// public:
//   StringRef getArgument() const final { 
//     return "convert-memcpy-to-affine"; 
//   }
  
//   StringRef getDescription() const final {
//     return "Convert gpu.memcpy operations with subviews to affine loop implementations";
//   }

//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
    
//     LLVM_DEBUG(llvm::dbgs() << "=== Convert Memcpy to Affine Pass ===\n");
    
//     // 收集所有的 gpu.memcpy 操作
//     std::vector<MemcpyInfo> allMemcpyOps;
//     if (failed(collectMemcpyOps(funcOp, allMemcpyOps))) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to collect memcpy operations\n");
//       signalPassFailure();
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Found " << allMemcpyOps.size() << " memcpy ops\n");
    
//     // 将 memcpy 操作分组
//     std::vector<MemcpyGroup> groups;
//     if (failed(groupMemcpyOps(allMemcpyOps, groups))) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to group memcpy operations\n");
//       signalPassFailure();
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Created " << groups.size() << " memcpy groups\n");
    
//     // 转换每个组
//     for (auto& group : groups) {
//       if (failed(convertMemcpyGroupToAffineLoop(group))) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to convert a memcpy group\n");
//         signalPassFailure();
//         return;
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully converted all memcpy groups\n");
//   }

// private:
//   // 收集所有 gpu.memcpy 操作
//   LogicalResult collectMemcpyOps(func::FuncOp funcOp, std::vector<MemcpyInfo>& memcpyOps) {
//     funcOp.walk([&](gpu::MemcpyOp op) {
//       MemcpyInfo info(op.getLoc());
//       if (succeeded(extractMemcpyInfo(op, info))) {
//         memcpyOps.push_back(info);
//       }
//     });
    
//     return success();
//   }

//   // 提取 memcpy 操作信息
//   LogicalResult extractMemcpyInfo(gpu::MemcpyOp op, MemcpyInfo& info) {
//     info.memcpyOp = op.getOperation();
//     info.target = op.getDst();
//     info.source = op.getSrc();
    
//     // 检查 source 是否是 subview
//     if (auto subviewOp = info.source.getDefiningOp<memref::SubViewOp>()) {
//       info.sourceIsSubview = true;
//       info.baseMemref = subviewOp.getSource();
      
//       // 提取 offsets, sizes, strides
//       extractSubviewParams(subviewOp, info.offsets, info.sizes, info.strides);
      
//       LLVM_DEBUG(llvm::dbgs() << "  Source is subview with offsets: ";
//                  for (auto off : info.offsets) llvm::dbgs() << off << " ";
//                  llvm::dbgs() << "\n");
//     }
    
//     // 检查 target 是否是 subview
//     if (auto subviewOp = info.target.getDefiningOp<memref::SubViewOp>()) {
//       info.targetIsSubview = true;
//       info.baseMemref = subviewOp.getSource();
      
//       // 提取 offsets, sizes, strides
//       extractSubviewParams(subviewOp, info.offsets, info.sizes, info.strides);
      
//       LLVM_DEBUG(llvm::dbgs() << "  Target is subview with offsets: ";
//                  for (auto off : info.offsets) llvm::dbgs() << off << " ";
//                  llvm::dbgs() << "\n");
//     }
    
//     // 至少有一个是 subview
//     if (!info.sourceIsSubview && !info.targetIsSubview) {
//       return failure();
//     }
    
//     return success();
//   }

//   // 从 SubViewOp 提取参数
//   void extractSubviewParams(memref::SubViewOp op, 
//                            SmallVector<int64_t, 4>& offsets,
//                            SmallVector<int64_t, 4>& sizes,
//                            SmallVector<int64_t, 4>& strides) {
//     offsets.clear();
//     sizes.clear();
//     strides.clear();
    
//     // 获取静态 offsets
//     auto staticOffsets = op.getStaticOffsets();
//     for (auto offset : staticOffsets) {
//       offsets.push_back(offset);
//     }
    
//     // 获取静态 sizes
//     auto staticSizes = op.getStaticSizes();
//     for (auto size : staticSizes) {
//       sizes.push_back(size);
//     }
    
//     // 获取静态 strides
//     auto staticStrides = op.getStaticStrides();
//     for (auto stride : staticStrides) {
//       strides.push_back(stride);
//     }
//   }

//   // 将 memcpy 操作分组
//   LogicalResult groupMemcpyOps(const std::vector<MemcpyInfo>& memcpyOps,
//                                std::vector<MemcpyGroup>& groups) {
//     // 按照 base memref 分组
//     llvm::DenseMap<Value, std::vector<MemcpyInfo>> groupsByBase;
    
//     for (const auto& info : memcpyOps) {
//       if (info.sourceIsSubview || info.targetIsSubview) {
//         groupsByBase[info.baseMemref].push_back(info);
//       }
//     }
    
//     // 为每个 base memref 创建一个 group
//     for (auto& [baseMemref, infos] : groupsByBase) {
//       if (infos.empty()) continue;
      
//       MemcpyGroup group(infos[0].loc);
      
//       if (failed(analyzeMemcpyGroup(baseMemref, infos, group))) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to analyze memcpy group\n");
//         continue;
//       }
      
//       groups.push_back(group);
//     }
    
//     return success();
//   }

//   // 分析一组 memcpy 操作
//   LogicalResult analyzeMemcpyGroup(Value baseMemref,
//                                    const std::vector<MemcpyInfo>& infos,
//                                    MemcpyGroup& group) {
//     if (infos.empty()) return failure();
    
//     group.largeMemref = baseMemref;
//     group.memcpyOps = infos;
//     group.loc = infos[0].loc;
    
//     // 获取大 memref 的 shape
//     auto largeType = baseMemref.getType().cast<MemRefType>();
//     for (auto dim : largeType.getShape()) {
//       group.largeShape.push_back(dim);
//     }
    
//     // 判断是 concat 还是 split
//     const auto& firstInfo = infos[0];
//     group.isConcatPattern = firstInfo.targetIsSubview;
    
//     // 找出拼接/拆分的轴（第一个非零 offset 的维度）
//     group.axis = 0;
//     for (size_t i = 0; i < firstInfo.offsets.size(); ++i) {
//       bool allDifferent = false;
//       std::set<int64_t> offsetSet;
      
//       for (const auto& info : infos) {
//         if (i < info.offsets.size()) {
//           offsetSet.insert(info.offsets[i]);
//         }
//       }
      
//       if (offsetSet.size() > 1) {
//         group.axis = i;
//         break;
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Group pattern: " 
//                << (group.isConcatPattern ? "concat" : "split")
//                << ", axis=" << group.axis << "\n");
    
//     // 收集小 memrefs 和 offsets
//     for (const auto& info : infos) {
//       if (group.isConcatPattern) {
//         group.smallMemrefs.push_back(info.source);
//         group.axisOffsets.push_back(info.offsets[group.axis]);
        
//         if (group.smallShape.empty()) {
//           auto type = info.source.getType().cast<MemRefType>();
//           for (auto dim : type.getShape()) {
//             group.smallShape.push_back(dim);
//           }
//         }
//       } else {
//         group.smallMemrefs.push_back(info.target);
//         group.axisOffsets.push_back(info.offsets[group.axis]);
        
//         if (group.smallShape.empty()) {
//           auto type = info.target.getType().cast<MemRefType>();
//           for (auto dim : type.getShape()) {
//             group.smallShape.push_back(dim);
//           }
//         }
//       }
//     }
    
//     return success();
//   }

//   // 将 memcpy group 转换为 affine loop
//   LogicalResult convertMemcpyGroupToAffineLoop(MemcpyGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "\nConverting memcpy group to affine loop\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Pattern: " << (group.isConcatPattern ? "concat" : "split") << "\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Axis: " << group.axis << "\n");
//     LLVM_DEBUG(llvm::dbgs() << "  Num operations: " << group.memcpyOps.size() << "\n");
    
//     // 找到第一个 memcpy 操作的位置
//     Operation* firstOp = group.memcpyOps[0].memcpyOp;
//     OpBuilder builder(firstOp);
    
//     // 创建嵌套的 affine.for 循环
//     size_t rank = group.smallShape.size();
    
//     std::function<void(OpBuilder&, size_t, SmallVector<Value, 4>&)> buildLoops;
//     buildLoops = [&](OpBuilder& b, size_t dim, SmallVector<Value, 4>& ivs) {
//       if (dim == rank) {
//         // 在最内层，为每个小 memref 生成 load/store
//         for (size_t i = 0; i < group.smallMemrefs.size(); ++i) {
//           Value smallMemref = group.smallMemrefs[i];
//           int64_t offset = group.axisOffsets[i];
          
//           // 构建索引
//           SmallVector<Value, 4> smallIndices = ivs;
//           SmallVector<Value, 4> largeIndices = ivs;
          
//           // 在 axis 维度上应用偏移
//           if (offset > 0) {
//             AffineExpr d0 = b.getAffineDimExpr(0);
//             AffineExpr offsetExpr = b.getAffineConstantExpr(offset);
//             AffineMap map = AffineMap::get(1, 0, d0 + offsetExpr, b.getContext());
//             Value axisIV = ivs[group.axis];
//             Value newAxisIdx = b.create<AffineApplyOp>(group.loc, map, ValueRange{axisIV});
//             largeIndices[group.axis] = newAxisIdx;
//           }
          
//           // 根据 concat/split 决定 load/store 的方向
//           if (group.isConcatPattern) {
//             // Concat: 从小 memref load，存到大 memref
//             Value val = b.create<AffineLoadOp>(group.loc, smallMemref, smallIndices);
//             b.create<AffineStoreOp>(group.loc, val, group.largeMemref, largeIndices);
//           } else {
//             // Split: 从大 memref load，存到小 memref
//             Value val = b.create<AffineLoadOp>(group.loc, group.largeMemref, largeIndices);
//             b.create<AffineStoreOp>(group.loc, val, smallMemref, smallIndices);
//           }
//         }
//         return;
//       }
      
//       // 创建当前维度的循环
//       int64_t upperBound = group.smallShape[dim];
//       auto forOp = b.create<AffineForOp>(group.loc, 0, upperBound);
//       ivs.push_back(forOp.getInductionVar());
      
//       // 在循环体中递归
//       b.setInsertionPointToStart(forOp.getBody());
//       buildLoops(b, dim + 1, ivs);
      
//       ivs.pop_back();
      
//       // 恢复插入点到循环之后
//       b.setInsertionPointAfter(forOp);
//     };
    
//     SmallVector<Value, 4> ivs;
//     buildLoops(builder, 0, ivs);
    
//     // 删除相关的 GPU 操作
//     if (failed(removeGpuMemcpyOps(group))) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to remove GPU operations\n");
//       return failure();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "  Successfully converted memcpy group\n");
//     return success();
//   }

//   // 删除 GPU memcpy 相关操作
//   LogicalResult removeGpuMemcpyOps(MemcpyGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "  Removing GPU memcpy operations\n");
    
//     // 收集所有需要删除的操作
//     SmallVector<Operation*, 16> opsToErase;
//     SmallVector<gpu::WaitOp, 8> waitOpsToErase;
    
//     // 第一步：收集所有 memcpy 操作和它们的前置 wait async
//     for (const auto& info : group.memcpyOps) {
//       auto memcpyOp = dyn_cast<gpu::MemcpyOp>(info.memcpyOp);
//       if (!memcpyOp) continue;
      
//       // 收集前置的 gpu.wait async 操作
//       for (auto asyncDep : memcpyOp.getAsyncDependencies()) {
//         if (auto waitOp = asyncDep.getDefiningOp<gpu::WaitOp>()) {
//           // 检查这个 wait 是否只被当前 memcpy 使用
//           if (waitOp.getAsyncToken() && waitOp.getAsyncToken().hasOneUse()) {
//             opsToErase.push_back(waitOp);
//             LLVM_DEBUG(llvm::dbgs() << "    Found gpu.wait async to remove\n");
//           }
//         }
//       }
      
//       opsToErase.push_back(info.memcpyOp);
//     }
    
//     // 第二步：找到后续的 gpu.wait 操作（同步等待所有 memcpy 完成）
//     Operation* lastMemcpy = group.memcpyOps.back().memcpyOp;
    
//     // 收集所有 memcpy 的 async token
//     SmallVector<Value, 8> memcpyTokens;
//     for (const auto& info : group.memcpyOps) {
//       if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(info.memcpyOp)) {
//         if (memcpyOp.getAsyncToken()) {
//           memcpyTokens.push_back(memcpyOp.getAsyncToken());
//         }
//       }
//     }
    
//     // 在 memcpy 后面查找 gpu.wait 操作
//     Operation* nextOp = lastMemcpy->getNextNode();
//     while (nextOp) {
//       if (auto waitOp = dyn_cast<gpu::WaitOp>(nextOp)) {
//         // 检查这个 wait 是否等待我们的 memcpy tokens
//         bool waitsForOurMemcpy = false;
        
//         // 如果 wait 没有 async token（同步 wait），检查它的依赖
//         if (!waitOp.getAsyncToken()) {
//           for (auto dep : waitOp.getAsyncDependencies()) {
//             for (auto memcpyToken : memcpyTokens) {
//               if (dep == memcpyToken) {
//                 waitsForOurMemcpy = true;
//                 break;
//               }
//             }
//             if (waitsForOurMemcpy) break;
//           }
//         }
        
//         if (waitsForOurMemcpy) {
//           waitOpsToErase.push_back(waitOp);
//           LLVM_DEBUG(llvm::dbgs() << "    Found gpu.wait (sync) to remove\n");
//           break;  // 找到一个就够了
//         }
//       }
//       nextOp = nextOp->getNextNode();
      
//       // 不要搜索太远
//       if (opsToErase.size() + waitOpsToErase.size() > 100) {
//         break;
//       }
//     }
    
//     // 第三步：替换 async token 的使用
//     // 对于每个 memcpy，如果它的 async token 被后续的 wait 使用，
//     // 我们需要移除这些使用关系
//     for (auto waitOp : waitOpsToErase) {
//       // 如果 wait 有 async token 输出，替换它的所有使用
//       if (waitOp.getAsyncToken()) {
//         // 尝试用它的输入 token 替换（如果有的话）
//         if (!waitOp.getAsyncDependencies().empty()) {
//           waitOp.getAsyncToken().replaceAllUsesWith(
//               waitOp.getAsyncDependencies().front());
//         } else {
//           // 如果没有输入，确保没有人使用输出
//           if (!waitOp.getAsyncToken().use_empty()) {
//             LLVM_DEBUG(llvm::dbgs() << "    Warning: wait op has uses but no dependencies\n");
//           }
//         }
//       }
//       opsToErase.push_back(waitOp);
//     }
    
//     // 第四步：删除操作（按照倒序删除，先删除使用者）
//     for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
//       Operation* op = *it;
      
//       // 在删除前，确保所有结果都没有被使用
//       for (auto result : op->getResults()) {
//         if (!result.use_empty()) {
//           LLVM_DEBUG(llvm::dbgs() << "    Warning: operation still has uses, replacing with null\n");
//           // 对于 async token，可以删除使用
//           result.dropAllUses();
//         }
//       }
      
//       LLVM_DEBUG(llvm::dbgs() << "    Erasing operation: " << op->getName() << "\n");
//       op->erase();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "  Removed " << opsToErase.size() << " GPU operations\n");
//     return success();
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {

// std::unique_ptr<Pass> createConvertMemcpyToAffinePass() {
//   return std::make_unique<ConvertMemcpyToAffinePass>();
// }

// } // namespace onnx_mlir

// // Pass 注册
// static mlir::PassRegistration<ConvertMemcpyToAffinePass> pass;


#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>
#include <set>

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "convert-memcpy-to-affine"

namespace {

// 包装函数信息
struct WrapperFuncInfo {
  func::CallOp callOp;
  SmallVector<Value, 4> targetMemrefs;  // 需要处理的memref参数
  bool isMultiHeadAttention;  // true: MultiHeadAttention, false: ReduceMean
  
  WrapperFuncInfo(func::CallOp op, bool isMHA) 
    : callOp(op), isMultiHeadAttention(isMHA) {}
};

// 存储单个 memcpy 操作的信息
struct MemcpyInfo {
  Operation* memcpyOp;
  Value source;
  Value target;
  bool sourceIsSubview;
  bool targetIsSubview;
  
  // Subview 信息
  Value baseMemref;
  SmallVector<int64_t, 4> offsets;
  SmallVector<int64_t, 4> sizes;
  SmallVector<int64_t, 4> strides;
  
  Location loc;
  
  MemcpyInfo(Location location) : memcpyOp(nullptr), sourceIsSubview(false), 
                                   targetIsSubview(false), loc(location) {}
};

// 存储一组相关的 memcpy 操作（concat 或 split）
struct MemcpyGroup {
  bool isConcatPattern;
  Value largeMemref;
  std::vector<Value> smallMemrefs;
  std::vector<MemcpyInfo> memcpyOps;
  std::vector<int64_t> axisOffsets;
  int64_t axis;
  SmallVector<int64_t, 4> largeShape;
  SmallVector<int64_t, 4> smallShape;
  Location loc;
  
  // 对于输出split，记录中间memcpy
  Operation* intermediateMemcpyOp = nullptr;
  Value wrapperOutputMemref;  // 包装函数直接输出的memref
  
  MemcpyGroup(Location location) : isConcatPattern(false), axis(0), loc(location) {}
};

class ConvertMemcpyToAffinePass
    : public PassWrapper<ConvertMemcpyToAffinePass, OperationPass<func::FuncOp>> {

public:
  StringRef getArgument() const final { 
    return "convert-memcpy-to-affine"; 
  }
  
  StringRef getDescription() const final {
    return "Convert gpu.memcpy operations for specific wrapper functions to affine loops";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== Convert Memcpy to Affine Pass ===\n");
    
    // 1. 识别目标包装函数
    std::vector<WrapperFuncInfo> wrapperFuncs;
    if (failed(collectWrapperFunctions(funcOp, wrapperFuncs))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect wrapper functions\n");
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << wrapperFuncs.size() << " wrapper function calls\n");
    
    // 2. 提取目标memref集合并追踪输出的中间buffer
    llvm::DenseSet<Value> targetMemrefs;
    llvm::DenseMap<Value, WrapperFuncInfo*> outputMemrefToWrapper;
    llvm::DenseMap<Value, Value> outputIntermediateBuffers; // wrapper output -> intermediate buffer
    
    for (auto& wrapper : wrapperFuncs) {
      for (auto memref : wrapper.targetMemrefs) {
        targetMemrefs.insert(memref);
      }
      
      // 对于输出memref（最后一个），追踪其使用
      Value outputMemref = wrapper.targetMemrefs.back();
      outputMemrefToWrapper[outputMemref] = &wrapper;
      
      // 查找wrapper调用后的中间memcpy: outputMemref -> intermediateBuffer
      Value intermediateBuffer = findIntermediateOutputBuffer(wrapper.callOp, outputMemref);
      if (intermediateBuffer) {
        LLVM_DEBUG(llvm::dbgs() << "  Found intermediate output buffer for wrapper\n");
        targetMemrefs.insert(intermediateBuffer);
        outputIntermediateBuffers[outputMemref] = intermediateBuffer;
        outputMemrefToWrapper[intermediateBuffer] = &wrapper;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Collected " << targetMemrefs.size() << " target memrefs\n");
    
    // 3. 收集与目标memref相关的memcpy操作
    std::vector<MemcpyInfo> relevantMemcpyOps;
    if (failed(collectRelevantMemcpyOps(funcOp, targetMemrefs, relevantMemcpyOps))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to collect relevant memcpy operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << relevantMemcpyOps.size() << " relevant memcpy ops\n");
    
    // 4. 将memcpy操作分组
    std::vector<MemcpyGroup> groups;
    if (failed(groupMemcpyOps(relevantMemcpyOps, groups, outputMemrefToWrapper, outputIntermediateBuffers))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to group memcpy operations\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Created " << groups.size() << " memcpy groups\n");
    
    // 5. 转换每个组
    for (auto& group : groups) {
      if (failed(convertMemcpyGroupToAffineLoop(group))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert a memcpy group\n");
        signalPassFailure();
        return;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted all memcpy groups\n");
  }

private:
  // 收集目标包装函数
  LogicalResult collectWrapperFunctions(func::FuncOp funcOp, 
                                        std::vector<WrapperFuncInfo>& wrappers) {
    funcOp.walk([&](func::CallOp callOp) {
      StringRef callee = callOp.getCallee();
      
      if (callee == "mgpuCudnnMultiHeadAttention") {
        // MultiHeadAttention: 参数索引 12,13,14 是 q,k,v，参数索引 16 是输出
        WrapperFuncInfo info(callOp, true);
        
        auto operands = callOp.getOperands();
        if (operands.size() > 16) {
          // 提取q,k,v对应的memref（通过llvm.inttoptr追溯）
          Value qPtr = operands[12];
          Value kPtr = operands[13];
          Value vPtr = operands[14];
          Value outPtr = operands[16];
          
          if (auto qMemref = traceBackToMemref(qPtr)) {
            info.targetMemrefs.push_back(qMemref);
          }
          if (auto kMemref = traceBackToMemref(kPtr)) {
            info.targetMemrefs.push_back(kMemref);
          }
          if (auto vMemref = traceBackToMemref(vPtr)) {
            info.targetMemrefs.push_back(vMemref);
          }
          if (auto outMemref = traceBackToMemref(outPtr)) {
            info.targetMemrefs.push_back(outMemref);
          }
          
          wrappers.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "Found mgpuCudnnMultiHeadAttention with " 
                     << info.targetMemrefs.size() << " target memrefs\n");
        }
      } else if (callee == "mgpuCudnnReduceMean") {
        // ReduceMean: 参数索引 12 是输入，参数索引 13 是输出
        WrapperFuncInfo info(callOp, false);
        
        auto operands = callOp.getOperands();
        if (operands.size() > 13) {
          Value inPtr = operands[12];
          Value outPtr = operands[13];
          
          if (auto inMemref = traceBackToMemref(inPtr)) {
            info.targetMemrefs.push_back(inMemref);
          }
          if (auto outMemref = traceBackToMemref(outPtr)) {
            info.targetMemrefs.push_back(outMemref);
          }
          
          wrappers.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "Found mgpuCudnnReduceMean with " 
                     << info.targetMemrefs.size() << " target memrefs\n");
        }
      } else if (callee == "mgpuCudnnConv2dForward") {
        // Conv2dForward: 参数索引 13 是输入，参数索引 16 是输出
        WrapperFuncInfo info(callOp, false);
        
        auto operands = callOp.getOperands();
        if (operands.size() > 16) {
          Value inPtr = operands[13];
          Value outPtr = operands[16];
          
          if (auto inMemref = traceBackToMemref(inPtr)) {
            info.targetMemrefs.push_back(inMemref);
          }
          if (auto outMemref = traceBackToMemref(outPtr)) {
            info.targetMemrefs.push_back(outMemref);
          }
          
          wrappers.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "Found mgpuCudnnConv2dForward with " 
                    << info.targetMemrefs.size() << " target memrefs\n");
        }
      } else if (callee == "mgpuCulibsFullyConnectedForward") {
        // FullyConnectedForward: 参数索引 4 是输入，参数索引 7 是输出
        // 参数索引 5 是权重，不需要处理
        WrapperFuncInfo info(callOp, false);
        
        auto operands = callOp.getOperands();
        if (operands.size() > 7) {
          Value inPtr = operands[4];   // %1177 - 输入
          // Value weightPtr = operands[5]; // %1179 - 权重，不处理
          Value outPtr = operands[7];  // %1181 - 输出
          
          if (auto inMemref = traceBackToMemref(inPtr)) {
            info.targetMemrefs.push_back(inMemref);
          }
          if (auto outMemref = traceBackToMemref(outPtr)) {
            info.targetMemrefs.push_back(outMemref);
          }
          
          wrappers.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "Found mgpuCulibsFullyConnectedForward with " 
                    << info.targetMemrefs.size() << " target memrefs\n");
        }
      } else if (callee == "mgpuCudnnMaxPoolForward") {
        // MaxPoolForward: 参数索引 15 是输入，参数索引 16 是输出
        WrapperFuncInfo info(callOp, false);
        
        auto operands = callOp.getOperands();
        if (operands.size() > 16) {
          Value inPtr = operands[15];  // 输入
          Value outPtr = operands[16]; // 输出
          // operands[17] 可能是workspace，不处理
          
          if (auto inMemref = traceBackToMemref(inPtr)) {
            info.targetMemrefs.push_back(inMemref);
          }
          if (auto outMemref = traceBackToMemref(outPtr)) {
            info.targetMemrefs.push_back(outMemref);
          }
          
          wrappers.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "Found mgpuCudnnMaxPoolForward with " 
                    << info.targetMemrefs.size() << " target memrefs\n");
        }
      }
    });
    
    return success();
  }
  
  // 从指针追溯到memref
  Value traceBackToMemref(Value ptr) {
    // ptr 是 !llvm.ptr 类型，追溯其定义
    Operation* defOp = ptr.getDefiningOp();
    if (!defOp) return nullptr;
    
    // llvm.inttoptr %i64 : i64 to !llvm.ptr
    if (defOp->getName().getStringRef() == "llvm.inttoptr") {
      if (defOp->getNumOperands() > 0) {
        Value i64Val = defOp->getOperand(0);
        Operation* castOp = i64Val.getDefiningOp();
        
        // arith.index_cast %index : index to i64
        if (castOp && castOp->getName().getStringRef() == "arith.index_cast") {
          if (castOp->getNumOperands() > 0) {
            Value indexVal = castOp->getOperand(0);
            Operation* extractOp = indexVal.getDefiningOp();
            
            // memref.extract_aligned_pointer_as_index %memref
            if (extractOp && extractOp->getName().getStringRef() == "memref.extract_aligned_pointer_as_index") {
              if (extractOp->getNumOperands() > 0) {
                return extractOp->getOperand(0);
              }
            }
          }
        }
      }
    }
    return nullptr;
  }
  
  // 查找包装函数输出后的中间buffer
  // 在wrapper调用后查找: gpu.memcpy %intermediate_buffer, %wrapper_output
  Value findIntermediateOutputBuffer(func::CallOp wrapperCall, Value wrapperOutput) {
    Operation* op = wrapperCall.getOperation();
    
    // 跳过stream同步操作
    while (op) {
      op = op->getNextNode();
      if (!op) break;
      
      // 跳过stream相关的调用
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        StringRef callee = callOp.getCallee();
        if (callee.contains("Stream") || callee.contains("mgpu")) {
          continue;
        }
      }
      
      // 查找gpu.wait
      if (isa<gpu::WaitOp>(op)) {
        continue;
      }
      
      // 查找gpu.memcpy
      if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(op)) {
        // 检查是否是 dst <- src，其中src是wrapper的输出
        if (memcpyOp.getSrc() == wrapperOutput) {
          return memcpyOp.getDst();
        }
        // 或者dst是wrapper的输出
        if (memcpyOp.getDst() == wrapperOutput) {
          return memcpyOp.getSrc();
        }
      }
      
      // 如果遇到其他类型的操作，可能已经过了相关区域
      if (!isa<gpu::WaitOp>(op) && !isa<gpu::MemcpyOp>(op) && 
          !isa<func::CallOp>(op)) {
        break;
      }
    }
    
    return nullptr;
  }

  // 收集与目标memref相关的memcpy操作
  LogicalResult collectRelevantMemcpyOps(func::FuncOp funcOp,
                                         const llvm::DenseSet<Value>& targetMemrefs,
                                         std::vector<MemcpyInfo>& memcpyOps) {
    funcOp.walk([&](gpu::MemcpyOp op) {
      MemcpyInfo info(op.getLoc());
      if (succeeded(extractMemcpyInfo(op, info))) {
        // 检查此memcpy是否与目标memref相关
        bool isRelevant = false;
        
        // 检查source
        if (info.sourceIsSubview && targetMemrefs.count(info.baseMemref)) {
          isRelevant = true;
        } else if (!info.sourceIsSubview && targetMemrefs.count(info.source)) {
          isRelevant = true;
        }
        
        // 检查target
        if (info.targetIsSubview && targetMemrefs.count(info.baseMemref)) {
          isRelevant = true;
        } else if (!info.targetIsSubview && targetMemrefs.count(info.target)) {
          isRelevant = true;
        }
        
        if (isRelevant) {
          memcpyOps.push_back(info);
          LLVM_DEBUG(llvm::dbgs() << "  Collected relevant memcpy\n");
        }
      }
    });
    
    return success();
  }

  // 提取memcpy操作信息
  LogicalResult extractMemcpyInfo(gpu::MemcpyOp op, MemcpyInfo& info) {
    info.memcpyOp = op.getOperation();
    info.target = op.getDst();
    info.source = op.getSrc();
    
    // 检查source是否是subview
    if (auto subviewOp = info.source.getDefiningOp<memref::SubViewOp>()) {
      info.sourceIsSubview = true;
      info.baseMemref = subviewOp.getSource();
      extractSubviewParams(subviewOp, info.offsets, info.sizes, info.strides);
    }
    
    // 检查target是否是subview
    if (auto subviewOp = info.target.getDefiningOp<memref::SubViewOp>()) {
      info.targetIsSubview = true;
      info.baseMemref = subviewOp.getSource();
      extractSubviewParams(subviewOp, info.offsets, info.sizes, info.strides);
    }
    
    // 至少有一个是subview
    if (!info.sourceIsSubview && !info.targetIsSubview) {
      return failure();
    }
    
    return success();
  }

  // 从SubViewOp提取参数
  void extractSubviewParams(memref::SubViewOp op, 
                           SmallVector<int64_t, 4>& offsets,
                           SmallVector<int64_t, 4>& sizes,
                           SmallVector<int64_t, 4>& strides) {
    offsets.clear();
    sizes.clear();
    strides.clear();
    
    auto staticOffsets = op.getStaticOffsets();
    for (auto offset : staticOffsets) {
      offsets.push_back(offset);
    }
    
    auto staticSizes = op.getStaticSizes();
    for (auto size : staticSizes) {
      sizes.push_back(size);
    }
    
    auto staticStrides = op.getStaticStrides();
    for (auto stride : staticStrides) {
      strides.push_back(stride);
    }
  }

  // 将memcpy操作分组
  LogicalResult groupMemcpyOps(const std::vector<MemcpyInfo>& memcpyOps,
                               std::vector<MemcpyGroup>& groups,
                               const llvm::DenseMap<Value, WrapperFuncInfo*>& outputMemrefToWrapper,
                               const llvm::DenseMap<Value, Value>& outputIntermediateBuffers) {
    // 按照base memref分组
    llvm::DenseMap<Value, std::vector<MemcpyInfo>> groupsByBase;
    
    for (const auto& info : memcpyOps) {
      if (info.sourceIsSubview || info.targetIsSubview) {
        groupsByBase[info.baseMemref].push_back(info);
      }
    }
    
    // 为每个base memref创建一个group
    for (auto& [baseMemref, infos] : groupsByBase) {
      if (infos.empty()) continue;
      
      MemcpyGroup group(infos[0].loc);
      
      if (failed(analyzeMemcpyGroup(baseMemref, infos, group, outputMemrefToWrapper, outputIntermediateBuffers))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to analyze memcpy group\n");
        continue;
      }
      
      groups.push_back(group);
    }
    
    return success();
  }

  // 分析一组memcpy操作
  LogicalResult analyzeMemcpyGroup(Value baseMemref,
                                   const std::vector<MemcpyInfo>& infos,
                                   MemcpyGroup& group,
                                   const llvm::DenseMap<Value, WrapperFuncInfo*>& outputMemrefToWrapper,
                                   const llvm::DenseMap<Value, Value>& outputIntermediateBuffers) {
    if (infos.empty()) return failure();
    
    group.largeMemref = baseMemref;
    group.memcpyOps = infos;
    group.loc = infos[0].loc;
    
    // 获取大memref的shape
    auto largeType = baseMemref.getType().cast<MemRefType>();
    for (auto dim : largeType.getShape()) {
      group.largeShape.push_back(dim);
    }
    
    // 判断是concat还是split
    const auto& firstInfo = infos[0];
    group.isConcatPattern = firstInfo.targetIsSubview;
    
    // 找出拼接/拆分的轴
    group.axis = 0;
    for (size_t i = 0; i < firstInfo.offsets.size(); ++i) {
      std::set<int64_t> offsetSet;
      for (const auto& info : infos) {
        if (i < info.offsets.size()) {
          offsetSet.insert(info.offsets[i]);
        }
      }
      if (offsetSet.size() > 1) {
        group.axis = i;
        break;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Group pattern: " 
               << (group.isConcatPattern ? "concat" : "split")
               << ", axis=" << group.axis << "\n");
    
    // 收集小memrefs和offsets
    for (const auto& info : infos) {
      if (group.isConcatPattern) {
        group.smallMemrefs.push_back(info.source);
        group.axisOffsets.push_back(info.offsets[group.axis]);
        
        if (group.smallShape.empty()) {
          auto type = info.source.getType().cast<MemRefType>();
          for (auto dim : type.getShape()) {
            group.smallShape.push_back(dim);
          }
        }
      } else {
        group.smallMemrefs.push_back(info.target);
        group.axisOffsets.push_back(info.offsets[group.axis]);
        
        if (group.smallShape.empty()) {
          auto type = info.target.getType().cast<MemRefType>();
          for (auto dim : type.getShape()) {
            group.smallShape.push_back(dim);
          }
        }
      }
    }
    
    // 对于split pattern，检查是否是输出的拆分
    if (!group.isConcatPattern) {
      // 检查baseMemref是否在outputMemrefToWrapper中
      // 或者是某个wrapper output的intermediate buffer
      if (outputMemrefToWrapper.count(baseMemref)) {
        LLVM_DEBUG(llvm::dbgs() << "  This is an output split\n");
        
        auto* wrapper = outputMemrefToWrapper.lookup(baseMemref);
        Value wrapperDirectOutput = wrapper->targetMemrefs.back();
        
        // 如果baseMemref是intermediate buffer，找到真正的wrapper output
        for (auto& [wrapperOut, intermediateBuffer] : outputIntermediateBuffers) {
          if (intermediateBuffer == baseMemref) {
            wrapperDirectOutput = wrapperOut;
            LLVM_DEBUG(llvm::dbgs() << "  Found wrapper direct output\n");
            break;
          }
        }
        
        // 查找中间memcpy (wrapperDirectOutput -> baseMemref)
        if (wrapperDirectOutput != baseMemref) {
          LLVM_DEBUG(llvm::dbgs() << "  Looking for intermediate memcpy\n");
          Operation* wrapperCall = wrapper->callOp.getOperation();
          group.intermediateMemcpyOp = findIntermediateMemcpy(wrapperCall, wrapperDirectOutput, baseMemref);
          group.wrapperOutputMemref = wrapperDirectOutput;
          
          if (group.intermediateMemcpyOp) {
            LLVM_DEBUG(llvm::dbgs() << "  Found intermediate memcpy to remove\n");
          }
        }
      }
    }
    
    return success();
  }
  
  // 查找中间memcpy操作
  Operation* findIntermediateMemcpy(Operation* wrapperCall, Value source, Value dest) {
    Operation* op = wrapperCall;
    
    while (op) {
      op = op->getNextNode();
      if (!op) break;
      
      // 跳过wait和stream操作
      if (isa<gpu::WaitOp>(op)) continue;
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (callOp.getCallee().contains("Stream") || 
            callOp.getCallee().contains("mgpu")) {
          continue;
        }
      }
      
      // 查找目标memcpy
      if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(op)) {
        if ((memcpyOp.getSrc() == source && memcpyOp.getDst() == dest) ||
            (memcpyOp.getSrc() == dest && memcpyOp.getDst() == source)) {
          return memcpyOp;
        }
      }
      
      // 限制搜索范围
      if (!isa<gpu::WaitOp>(op) && !isa<gpu::MemcpyOp>(op) && 
          !isa<func::CallOp>(op)) {
        break;
      }
    }
    
    return nullptr;
  }

  // 将memcpy group转换为affine loop
  LogicalResult convertMemcpyGroupToAffineLoop(MemcpyGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "\nConverting memcpy group to affine loop\n");
    LLVM_DEBUG(llvm::dbgs() << "  Pattern: " << (group.isConcatPattern ? "concat" : "split") << "\n");
    
    // 对于split pattern的输出，使用wrapper的直接输出memref
    Value sourceMemref = group.largeMemref;
    if (!group.isConcatPattern && group.wrapperOutputMemref) {
      sourceMemref = group.wrapperOutputMemref;
      LLVM_DEBUG(llvm::dbgs() << "  Using wrapper output memref directly\n");
    }
    
    // 找到第一个memcpy操作的位置
    Operation* firstOp = group.memcpyOps[0].memcpyOp;
    OpBuilder builder(firstOp);
    
    // 创建嵌套的affine.for循环
    size_t rank = group.smallShape.size();
    
    std::function<void(OpBuilder&, size_t, SmallVector<Value, 4>&)> buildLoops;
    buildLoops = [&](OpBuilder& b, size_t dim, SmallVector<Value, 4>& ivs) {
      if (dim == rank) {
        // 在最内层，为每个小memref生成load/store
        for (size_t i = 0; i < group.smallMemrefs.size(); ++i) {
          Value smallMemref = group.smallMemrefs[i];
          int64_t offset = group.axisOffsets[i];
          
          SmallVector<Value, 4> smallIndices = ivs;
          SmallVector<Value, 4> largeIndices = ivs;
          
          // 在axis维度上应用偏移
          if (offset > 0) {
            AffineExpr d0 = b.getAffineDimExpr(0);
            AffineExpr offsetExpr = b.getAffineConstantExpr(offset);
            AffineMap map = AffineMap::get(1, 0, d0 + offsetExpr, b.getContext());
            Value axisIV = ivs[group.axis];
            Value newAxisIdx = b.create<AffineApplyOp>(group.loc, map, ValueRange{axisIV});
            largeIndices[group.axis] = newAxisIdx;
          }
          
          // 根据concat/split决定load/store的方向
          if (group.isConcatPattern) {
            Value val = b.create<AffineLoadOp>(group.loc, smallMemref, smallIndices);
            b.create<AffineStoreOp>(group.loc, val, group.largeMemref, largeIndices);
          } else {
            Value val = b.create<AffineLoadOp>(group.loc, sourceMemref, largeIndices);
            b.create<AffineStoreOp>(group.loc, val, smallMemref, smallIndices);
          }
        }
        return;
      }
      
      int64_t upperBound = group.smallShape[dim];
      auto forOp = b.create<AffineForOp>(group.loc, 0, upperBound);
      ivs.push_back(forOp.getInductionVar());
      
      b.setInsertionPointToStart(forOp.getBody());
      buildLoops(b, dim + 1, ivs);
      
      ivs.pop_back();
      b.setInsertionPointAfter(forOp);
    };
    
    SmallVector<Value, 4> ivs;
    buildLoops(builder, 0, ivs);
    
    // 删除相关的GPU操作
    if (failed(removeGpuMemcpyOps(group))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to remove GPU operations\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted memcpy group\n");
    return success();
  }

  // 删除GPU memcpy相关操作
  LogicalResult removeGpuMemcpyOps(MemcpyGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "  Removing GPU memcpy operations\n");
    
    SmallVector<Operation*, 16> opsToErase;
    SmallVector<gpu::WaitOp, 8> waitOpsToErase;
    
    // 收集所有memcpy操作和它们的前置wait async
    for (const auto& info : group.memcpyOps) {
      auto memcpyOp = dyn_cast<gpu::MemcpyOp>(info.memcpyOp);
      if (!memcpyOp) continue;
      
      for (auto asyncDep : memcpyOp.getAsyncDependencies()) {
        if (auto waitOp = asyncDep.getDefiningOp<gpu::WaitOp>()) {
          if (waitOp.getAsyncToken() && waitOp.getAsyncToken().hasOneUse()) {
            opsToErase.push_back(waitOp);
          }
        }
      }
      
      opsToErase.push_back(info.memcpyOp);
    }
    
    // 如果有中间memcpy，也要删除它及其相关的wait
    if (group.intermediateMemcpyOp) {
      LLVM_DEBUG(llvm::dbgs() << "  Removing intermediate memcpy\n");
      
      auto intermediateMemcpy = dyn_cast<gpu::MemcpyOp>(group.intermediateMemcpyOp);
      if (intermediateMemcpy) {
        // 收集前置wait
        for (auto asyncDep : intermediateMemcpy.getAsyncDependencies()) {
          if (auto waitOp = asyncDep.getDefiningOp<gpu::WaitOp>()) {
            if (waitOp.getAsyncToken() && waitOp.getAsyncToken().hasOneUse()) {
              opsToErase.push_back(waitOp);
            }
          }
        }
        opsToErase.push_back(group.intermediateMemcpyOp);
        
        // 查找后续的同步wait
        Operation* nextOp = group.intermediateMemcpyOp->getNextNode();
        while (nextOp) {
          if (auto waitOp = dyn_cast<gpu::WaitOp>(nextOp)) {
            if (!waitOp.getAsyncToken()) {
              for (auto dep : waitOp.getAsyncDependencies()) {
                if (intermediateMemcpy.getAsyncToken() == dep) {
                  opsToErase.push_back(waitOp);
                  break;
                }
              }
            }
            break;
          }
          nextOp = nextOp->getNextNode();
        }
      }
    }
    
    // 找到后续的gpu.wait操作
    Operation* lastMemcpy = group.memcpyOps.back().memcpyOp;
    SmallVector<Value, 8> memcpyTokens;
    for (const auto& info : group.memcpyOps) {
      if (auto memcpyOp = dyn_cast<gpu::MemcpyOp>(info.memcpyOp)) {
        if (memcpyOp.getAsyncToken()) {
          memcpyTokens.push_back(memcpyOp.getAsyncToken());
        }
      }
    }
    
    Operation* nextOp = lastMemcpy->getNextNode();
    while (nextOp) {
      if (auto waitOp = dyn_cast<gpu::WaitOp>(nextOp)) {
        if (!waitOp.getAsyncToken()) {
          bool waitsForOurMemcpy = false;
          for (auto dep : waitOp.getAsyncDependencies()) {
            for (auto memcpyToken : memcpyTokens) {
              if (dep == memcpyToken) {
                waitsForOurMemcpy = true;
                break;
              }
            }
            if (waitsForOurMemcpy) break;
          }
          
          if (waitsForOurMemcpy) {
            waitOpsToErase.push_back(waitOp);
            break;
          }
        }
      }
      nextOp = nextOp->getNextNode();
      if (opsToErase.size() + waitOpsToErase.size() > 100) break;
    }
    
    // 替换async token的使用
    for (auto waitOp : waitOpsToErase) {
      if (waitOp.getAsyncToken()) {
        if (!waitOp.getAsyncDependencies().empty()) {
          waitOp.getAsyncToken().replaceAllUsesWith(
              waitOp.getAsyncDependencies().front());
        } else {
          if (!waitOp.getAsyncToken().use_empty()) {
            waitOp.getAsyncToken().dropAllUses();
          }
        }
      }
      opsToErase.push_back(waitOp);
    }
    
    // 删除操作（倒序）
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
      Operation* op = *it;
      for (auto result : op->getResults()) {
        if (!result.use_empty()) {
          result.dropAllUses();
        }
      }
      op->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Removed " << opsToErase.size() << " GPU operations\n");
    return success();
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createConvertMemcpyToAffinePass() {
  return std::make_unique<ConvertMemcpyToAffinePass>();
}

} // namespace onnx_mlir

static mlir::PassRegistration<ConvertMemcpyToAffinePass> pass;