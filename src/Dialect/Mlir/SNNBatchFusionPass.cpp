#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/DenseSet.h"

// 添加ONNX操作的头文件
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

#define DEBUG_TYPE "snn-batch-fusion"

namespace {

struct LayerInfo {
  std::string layerName;  // 实际的层名，如layer.0, layer.1
  int timestep;           // 时间步信息
  
  LayerInfo(std::string name, int ts) : layerName(std::move(name)), timestep(ts) {}
};

// 简化的layer信息结构
struct SimpleLayerInfo {
  int layerIndex;    // 层索引，如layer.1 -> 1
  int timestep;      // 时间步，如layer.1_5 -> 5，没有_则为0
  std::string opName; // 操作名
  
  SimpleLayerInfo(int layer, int ts, std::string op) 
      : layerIndex(layer), timestep(ts), opName(std::move(op)) {}
};

// 操作信息结构
struct OperationInfo {
  Operation* op;
  SimpleLayerInfo layerInfo;
  bool isLayerOp;        // 是否是layer操作
  bool isTimestepBoundary; // 是否是时间步边界
  
  OperationInfo(Operation* operation, SimpleLayerInfo info, bool isLayer, bool isBoundary)
      : op(operation), layerInfo(std::move(info)), isLayerOp(isLayer), isTimestepBoundary(isBoundary) {}
};

// 可融合操作组结构
struct FusibleGroup {
  int layerIndex;
  std::string opType;  // Conv, MaxPoolSingleOut, Gemm, MatMul
  SmallVector<Operation*> operations;
  SmallVector<int> timesteps;
  
  FusibleGroup() : layerIndex(0), opType("") {}  // 添加默认构造函数
  FusibleGroup(int layer, std::string type) : layerIndex(layer), opType(std::move(type)) {}
};

struct SNNBatchFusionPass
    : public PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>> {
  
  StringRef getArgument() const final { return "snn-batch-fusion"; }
  StringRef getDescription() const final {
    return "Fuse SNN operations across timesteps for batch processing with IR reordering";
  }
  
  Option<int> batchSize{*this, "batch-size", 
                        llvm::cl::desc("Batch size for fusion"), 
                        llvm::cl::init(2)};
  
  SNNBatchFusionPass() = default;
  SNNBatchFusionPass(const SNNBatchFusionPass& other) 
      : PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>>(),
        batchSize(*this, "batch-size", 
                  llvm::cl::desc("Batch size for fusion"), 
                  llvm::cl::init(other.batchSize)) {}
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect, arith::ArithDialect>();
  }
  
  // 修改后的主要执行流程
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "Running SNNBatchFusionPass with batch size: " 
               << batchSize << "\n");
    
    // 1. 预处理：给时序性算子添加layer字段
    preprocessTemporalOpsWithLayerInfo(funcOp);
    
    // 2. 直接执行基于layer信息的重排
    SmallVector<OperationInfo> operationInfos;
    collectOperationInfo(funcOp, operationInfos);
    
    if (operationInfos.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No operations with layer info found\n");
      return;
    }
    
    // 3. 确定时间步范围
    int maxTimestep = 0;
    for (const auto& opInfo : operationInfos) {
      maxTimestep = std::max(maxTimestep, opInfo.layerInfo.timestep);
    }
    int totalTimesteps = maxTimestep + 1;
    
    LLVM_DEBUG(llvm::dbgs() << "Total timesteps: " << totalTimesteps << "\n");
    
    if (totalTimesteps < batchSize) {
      LLVM_DEBUG(llvm::dbgs() << "Not enough timesteps for batching: " 
                 << totalTimesteps << " < " << batchSize << "\n");
      return;
    }
    
    // 4. 按batch进行重排（与fusion保持一致的分批逻辑）
    SmallVector<Operation*> reorderedOps;
    
    for (int batchStart = 0; batchStart + batchSize <= totalTimesteps; batchStart += batchSize) {
      LLVM_DEBUG(llvm::dbgs() << "\n--- Processing reorder batch starting at timestep " 
                 << batchStart << " (size: " << batchSize << ") ---\n");
      
      if (!reorderSingleBatch(operationInfos, batchStart, batchSize, reorderedOps)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to reorder batch starting at " << batchStart << "\n");
        return;
      }
    }
    
    // 5. 处理剩余的时间步（不足一个batch的）
    int remainingStart = (totalTimesteps / batchSize) * batchSize;
    if (remainingStart < totalTimesteps) {
      LLVM_DEBUG(llvm::dbgs() << "\n--- Processing remaining timesteps from " << remainingStart << " ---\n");
      
      SmallVector<Operation*> remainingOps;
      for (const auto& opInfo : operationInfos) {
        if (opInfo.layerInfo.timestep >= remainingStart) {
          remainingOps.push_back(opInfo.op);
        }
      }
      
      // 对剩余操作按原有顺序排序
      llvm::sort(remainingOps, [](Operation* a, Operation* b) {
        return a->isBeforeInBlock(b);
      });
      
      for (Operation* op : remainingOps) {
        reorderedOps.push_back(op);
      }
    }
    
    // 6. 执行最终重排
    if (!executeReordering(reorderedOps)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to execute reordering\n");
      return;
    }
    
    // 7. 在重排完成后，执行batch fusion（现在会按相同的batch size进行）
    LLVM_DEBUG(llvm::dbgs() << "\n=== Starting Batch Fusion (Batch Size: " << batchSize << ") ===\n");
    if (!performBatchFusion(funcOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to perform batch fusion\n");
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed SNNBatchFusionPass\n");
  }

private:
  
  // 预处理：给时序性算子添加layer字段
  void preprocessTemporalOpsWithLayerInfo(func::FuncOp funcOp) {
    Block& block = funcOp.getBody().front();
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Preprocessing: Adding Layer Info to Temporal Ops ===\n");
    
    // 首先确定最大的层索引，/Add操作将被标记为最后一层
    int maxLayerIndex = findMaxLayerIndex(funcOp);
    int addLayerIndex = maxLayerIndex + 1; // Add操作作为最后一层
    
    LLVM_DEBUG(llvm::dbgs() << "Max layer index found: " << maxLayerIndex 
               << ", Add operations will be layer " << addLayerIndex << "\n");
    
    std::string currentTemporalLayerName = "";
    int currentTimestep = 0;
    bool isAfterTimestepBoundary = false;
    bool hasSeenLayerOpInCurrentTimestep = false;
    
    for (Operation& op : block) {
      if (auto nodeNameAttr = op.getAttrOfType<StringAttr>("onnx_node_name")) {
        StringRef nodeName = nodeNameAttr.getValue();
        
        if (nodeName.contains("/layer/")) {
          // 解析层信息
          LayerInfo layerInfo = parseLayerNameCorrectly(nodeName);
          
          // 计算当前层后续时序性算子应该归属的层名
          currentTemporalLayerName = calculateNextLayerName(layerInfo.layerName, layerInfo.timestep);
          currentTimestep = layerInfo.timestep;
          hasSeenLayerOpInCurrentTimestep = true;
          isAfterTimestepBoundary = false;
          
          LLVM_DEBUG(llvm::dbgs() << "Found layer op: " << nodeName 
                     << " -> parsed as " << layerInfo.layerName 
                     << " (timestep: " << layerInfo.timestep 
                     << "), temporal ops will be tagged as: " << currentTemporalLayerName << "\n");
          
        } else if (nodeName.starts_with("/Add")) {
          // 处理时间步边界Add操作，给它们添加layer信息
          int timestepFromAdd = extractTimestepFromAdd(nodeName);
          
          std::string addLayerName;
          if (timestepFromAdd == 0) {
            addLayerName = "layer." + std::to_string(addLayerIndex);
          } else {
            addLayerName = "layer." + std::to_string(addLayerIndex) + "_" + std::to_string(timestepFromAdd);
          }
          
          std::string newNodeName = "/layer/" + addLayerName + "/" + nodeName.str();
          op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
          
          LLVM_DEBUG(llvm::dbgs() << "Updated Add operation: " << nodeName 
                     << " -> " << newNodeName << " (timestep: " << timestepFromAdd << ")\n");
          
          // 重置状态，进入下一个时间步
          currentTemporalLayerName = "";
          isAfterTimestepBoundary = true;
          hasSeenLayerOpInCurrentTimestep = false;
          currentTimestep = timestepFromAdd + 1;
          
        } else if (!isConstantOrUtilityOp(&op)) {
          // 时序性算子处理
          if (!currentTemporalLayerName.empty()) {
            // 正常情况：已经有了当前的时序层名，同一组时序性算子使用相同的层名
            std::string newNodeName = "/layer/" + currentTemporalLayerName + "/" + nodeName.str();
            op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
            
            LLVM_DEBUG(llvm::dbgs() << "Updated temporal op: " << nodeName 
                       << " -> " << newNodeName << "\n");
            
          } else if (isAfterTimestepBoundary && !hasSeenLayerOpInCurrentTimestep) {
            // 特殊情况：时间步边界后第一次遇到时序性操作，但还没有看到layer操作
            // 说明第0层被优化掉了，从第1层开始，这一组所有时序性算子都属于第1层
            if (currentTemporalLayerName.empty()) {
              if (currentTimestep == 0) {
                currentTemporalLayerName = "layer.1";  // 时间步0，第1层
              } else {
                currentTemporalLayerName = "layer.1_" + std::to_string(currentTimestep);  // 时间步N，第1层
              }
              
              LLVM_DEBUG(llvm::dbgs() << "Special case detected - layer 0 optimized away, "
                         << "temporal ops will use: " << currentTemporalLayerName << "\n");
            }
            
            std::string newNodeName = "/layer/" + currentTemporalLayerName + "/" + nodeName.str();
            op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
            
            LLVM_DEBUG(llvm::dbgs() << "Special case - updated temporal op: " 
                       << nodeName << " -> " << newNodeName << "\n");
            
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Skipping temporal op (no layer context): " << nodeName << "\n");
          }
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed preprocessing\n");
  }
  
  // 计算下一层的名称
  std::string calculateNextLayerName(const std::string& currentLayerName, int timestep) {
    // 从当前层名中提取层索引
    int currentLayerIndex = extractLayerIndex(currentLayerName);
    int nextLayerIndex = currentLayerIndex + 1;
    
    if (timestep == 0) {
      // 时间步0：layer.[next_layer_index]
      return "layer." + std::to_string(nextLayerIndex);
    } else {
      // 时间步N：layer.[next_layer_index]_[timestep]
      return "layer." + std::to_string(nextLayerIndex) + "_" + std::to_string(timestep);
    }
  }
  
  // 正确解析层名
  LayerInfo parseLayerNameCorrectly(StringRef nodeName) {
    // 解析形如 "/layer/layer.4/Conv" 或 "/layer/layer.3_1/Conv" 的名称
    size_t lastSlash = nodeName.rfind('/');
    if (lastSlash == StringRef::npos) {
      return LayerInfo(nodeName.str(), 0);
    }
    
    StringRef layerPart = nodeName.substr(0, lastSlash);
    lastSlash = layerPart.rfind('/');
    if (lastSlash == StringRef::npos) {
      return LayerInfo(layerPart.str(), 0);
    }
    
    StringRef fullLayerName = layerPart.substr(lastSlash + 1);
    
    // 检查是否有时间步格式 (layer.3_1 -> layer.3 at timestep 1)
    size_t underscorePos = fullLayerName.rfind('_');
    if (underscorePos != StringRef::npos) {
      StringRef layerNamePart = fullLayerName.substr(0, underscorePos);  // "layer.3"
      StringRef timestepPart = fullLayerName.substr(underscorePos + 1);   // "1"
      
      int timestep = 0;
      if (timestepPart.getAsInteger(10, timestep)) {
        timestep = 0; // 默认为0如果解析失败
      }
      
      LLVM_DEBUG(llvm::dbgs() << "    Parsed " << fullLayerName << " -> layer: " << layerNamePart 
                << " (timestep: " << timestep << ")\n");
      
      return LayerInfo(layerNamePart.str(), timestep);
      
    } else {
      // 没有下划线，说明是时间步0的操作（layer.0, layer.3等）
      return LayerInfo(fullLayerName.str(), 0);
    }
  }
  
  // 从Add操作名中提取时间步信息
  int extractTimestepFromAdd(StringRef nodeName) {
    // 解析 "/Add" -> 0, "/Add_1" -> 1, "/Add_2" -> 2 等
    if (nodeName == "/Add") {
      return 0;
    }
    
    if (nodeName.starts_with("/Add_")) {
      StringRef timestepStr = nodeName.substr(5); // 跳过 "/Add_"
      int timestep = 0;
      if (timestepStr.getAsInteger(10, timestep)) {
        return 0; // 解析失败时默认为0
      }
      return timestep;
    }
    
    return 0;
  }
  
  // 找到现有layer操作中的最大层索引
  int findMaxLayerIndex(func::FuncOp funcOp) {
    int maxIndex = 0;
    
    funcOp.walk([&](Operation* op) {
      if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
        StringRef nodeName = nodeNameAttr.getValue();
        
        // 只处理原始的layer操作，不包括Add操作
        if (nodeName.contains("/layer/") && !nodeName.starts_with("/Add")) {
          LayerInfo layerInfo = parseLayerNameCorrectly(nodeName);
          int layerIndex = extractLayerIndex(layerInfo.layerName);
          maxIndex = std::max(maxIndex, layerIndex);
        }
      }
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Found max layer index: " << maxIndex << "\n");
    return maxIndex;
  }
  
  // 从层名中提取层索引
  int extractLayerIndex(const std::string& layerName) {
    // 处理两种格式：
    // 1. "layer.X" -> 返回 X
    // 2. "layer.X_T" -> 返回 X（X是层索引，T是时间步）
    
    size_t underscorePos = layerName.find('_');
    std::string layerPart;
    
    if (underscorePos != std::string::npos) {
      // 格式: "layer.X_T" -> 提取 layer.X 部分
      layerPart = layerName.substr(0, underscorePos);
    } else {
      // 格式: "layer.X"
      layerPart = layerName;
    }
    
    // 从 layer.X 中提取 X
    size_t dotPos = layerPart.find('.');
    if (dotPos != std::string::npos) {
      std::string indexStr = layerPart.substr(dotPos + 1);
      int index = 0;
      if (sscanf(indexStr.c_str(), "%d", &index) == 1) {
        return index;
      }
    }
    return 0;
  }

  SimpleLayerInfo parseSimpleLayerInfo(StringRef nodeName, Operation* op = nullptr) {
    // 解析格式: /layer/layer.X_Y/OpName 或 /layer/layer.X/OpName
    
    // 先找到最后一个/，获取操作名
    size_t lastSlash = nodeName.rfind('/');
    std::string opNameFromNodeName = (lastSlash != StringRef::npos) ? 
                        nodeName.substr(lastSlash + 1).str() : nodeName.str();
    
    // 查找layer.X_Y或layer.X部分
    size_t layerStart = nodeName.find("/layer/layer.");
    if (layerStart == StringRef::npos) {
      // 如果没有找到layer信息，尝试从实际操作类型推断
      std::string actualOpName = getActualOperationType(op);
      return SimpleLayerInfo(0, 0, actualOpName);
    }
    
    StringRef layerPart = nodeName.substr(layerStart + 13); // 跳过"/layer/layer."
    size_t nextSlash = layerPart.find('/');
    if (nextSlash != StringRef::npos) {
      layerPart = layerPart.substr(0, nextSlash);
    }
    
    // 解析layer.X_Y格式
    size_t underscorePos = layerPart.find('_');
    int layerIndex = 0;
    int timestep = 0;
    
    if (underscorePos != StringRef::npos) {
      // 有时间步信息: layer.X_Y
      StringRef layerIndexStr = layerPart.substr(0, underscorePos);
      StringRef timestepStr = layerPart.substr(underscorePos + 1);
      
      layerIndexStr.getAsInteger(10, layerIndex);
      timestepStr.getAsInteger(10, timestep);
    } else {
      // 没有时间步信息: layer.X (默认时间步0)
      layerPart.getAsInteger(10, layerIndex);
      timestep = 0;
    }
    
    // 获取实际的操作类型而不是依赖节点名
    std::string actualOpName = getActualOperationType(op);
    if (actualOpName.empty()) {
      // 如果无法获取实际操作类型，则尝试从节点名推断
      actualOpName = inferOpTypeFromNodeName(opNameFromNodeName);
    }
    
    return SimpleLayerInfo(layerIndex, timestep, actualOpName);
  }

  // 添加获取实际操作类型的辅助函数
  std::string getActualOperationType(Operation* op) {
    if (!op) return "";
    
    StringRef opName = op->getName().getStringRef();
    
    if (opName == "onnx.Conv") {
      return "Conv";
    } else if (opName == "onnx.MaxPoolSingleOut") {
      return "MaxPool";
    } else if (opName == "onnx.Gemm") {
      return "Gemm";
    } else if (opName == "onnx.MatMul") {
      return "MatMul";
    } else if (opName == "onnx.Flatten") {
      return "Flatten";
    }
    
    return "";
  }

  std::string inferOpTypeFromNodeName(const std::string& nodeName) {
    // 移除后缀数字和下划线
    std::string cleanName = nodeName;
    size_t underscorePos = cleanName.find('_');
    if (underscorePos != std::string::npos) {
      cleanName = cleanName.substr(0, underscorePos);
    }
    
    if (cleanName.find("Conv") != std::string::npos) {
      return "Conv";
    } else if (cleanName.find("MaxPool") != std::string::npos) {
      return "MaxPool";
    } else if (cleanName.find("MatMul") != std::string::npos || 
              cleanName.find("Gemm") != std::string::npos) {
      return "Gemm"; // 先假设是Gemm，后续会检查bias
    } else if (cleanName.find("Flatten") != std::string::npos) {
      return "Flatten";
    }
    
    return cleanName;
  }
  
  void collectOperationInfo(func::FuncOp funcOp, SmallVector<OperationInfo>& operationInfos) {
    LLVM_DEBUG(llvm::dbgs() << "\n=== Collecting Operation Information ===\n");
    
    funcOp.walk([&](Operation* op) {
      // 跳过常量和工具操作
      if (isConstantOrUtilityOp(op)) {
        return;
      }
      
      if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
        StringRef nodeName = nodeNameAttr.getValue();
        
        if (nodeName.contains("/layer/")) {
          SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName, op); // 传递op指针
          
          // 检查是否是时间步边界Add操作
          bool isTimestepBoundary = nodeName.contains("/Add");
          
          // 更全面的layer操作检查，使用实际操作类型
          bool isLayerOp = !isTimestepBoundary && isFusibleOperationType(layerInfo.opName);
          
          operationInfos.emplace_back(op, layerInfo, isLayerOp, isTimestepBoundary);
          
          // 特别标注操作类型
          std::string opTypeInfo = "TEMPORAL";
          if (isLayerOp) {
            opTypeInfo = "LAYER";
            if (layerInfo.opName == "Gemm") {
              auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
              if (gemmOp && isBiasZero(gemmOp.getC())) {
                opTypeInfo += " (Gemm->MatMul)";
              } else {
                opTypeInfo += " (Gemm)";
              }
            } else if (layerInfo.opName == "MatMul") {
              opTypeInfo += " (MatMul)";
            }
          } else if (isTimestepBoundary) {
            opTypeInfo = "BOUNDARY";
          }
          
          LLVM_DEBUG(llvm::dbgs() << "Collected: " << nodeName 
                    << " -> layer " << layerInfo.layerIndex 
                    << ", timestep " << layerInfo.timestep 
                    << ", actual op: " << op->getName().getStringRef()
                    << ", parsed type: " << layerInfo.opName
                    << ", category: " << opTypeInfo << "\n");
        }
      }
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Total collected operations: " << operationInfos.size() << "\n");
  }

  // 处理单个batch的重排
  bool reorderSingleBatch(ArrayRef<OperationInfo> allOps, int batchStart, int batchSize, 
                         SmallVector<Operation*>& reorderedOps) {
    
    // 1. 筛选出当前batch的操作
    SmallVector<const OperationInfo*> batchOps;
    for (const auto& opInfo : allOps) {
      if (opInfo.layerInfo.timestep >= batchStart && 
          opInfo.layerInfo.timestep < batchStart + batchSize) {
        batchOps.push_back(&opInfo);
      }
    }
    
    if (batchOps.empty()) {
      return true;
    }
    
    // 2. 按layer index分组
    std::map<int, SmallVector<const OperationInfo*>> layerGroups;
    for (const auto* opInfo : batchOps) {
      layerGroups[opInfo->layerInfo.layerIndex].push_back(opInfo);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Batch has " << layerGroups.size() << " layers\n");
    
    // 3. 按layer顺序处理
    for (const auto& [layerIndex, layerOps] : layerGroups) {
      LLVM_DEBUG(llvm::dbgs() << "Processing layer " << layerIndex << " with " << layerOps.size() << " ops\n");
      
      // 4. 在每一层内，按时间步分组
      std::map<int, SmallVector<const OperationInfo*>> timestepGroups;
      for (const auto* opInfo : layerOps) {
        timestepGroups[opInfo->layerInfo.timestep].push_back(opInfo);
      }
      
      // 5. 按时间步顺序添加操作
      for (int ts = batchStart; ts < batchStart + batchSize; ++ts) {
        auto it = timestepGroups.find(ts);
        if (it != timestepGroups.end()) {
          LLVM_DEBUG(llvm::dbgs() << "  Timestep " << ts << ": " << it->second.size() << " ops\n");
          
          // 在同一时间步内，保持原有顺序
          SmallVector<const OperationInfo*> timestepOps = it->second;
          llvm::sort(timestepOps, [](const OperationInfo* a, const OperationInfo* b) {
            return a->op->isBeforeInBlock(b->op);
          });
          
          for (const auto* opInfo : timestepOps) {
            reorderedOps.push_back(opInfo->op);
            
            if (auto nodeNameAttr = opInfo->op->getAttrOfType<StringAttr>("onnx_node_name")) {
              LLVM_DEBUG(llvm::dbgs() << "    Added: " << nodeNameAttr.getValue() << "\n");
            }
          }
        }
      }
    }
    
    return true;
  }
  
  // 执行最终的重排
  bool executeReordering(ArrayRef<Operation*> reorderedOps) {
    if (reorderedOps.empty()) {
      return true;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Executing Final Reordering ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Total operations to reorder: " << reorderedOps.size() << "\n");
    
    // 找到正确的插入点：所有常量操作之后
    Operation* insertionPoint = findInsertionPointAfterConstants(reorderedOps[0]->getParentOp());
    
    if (!insertionPoint) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to find valid insertion point\n");
      return false;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found insertion point after constants\n");
    
    // 按顺序移动操作
    for (size_t i = 0; i < reorderedOps.size(); ++i) {
      Operation* opToMove = reorderedOps[i];
      
      if (i == 0) {
        // 第一个操作移动到插入点之后
        opToMove->moveAfter(insertionPoint);
        LLVM_DEBUG(llvm::dbgs() << "Moved first operation after insertion point\n");
      } else {
        // 后续操作移动到前一个操作之后
        Operation* prevOp = reorderedOps[i-1];
        if (opToMove != prevOp->getNextNode()) {
          opToMove->moveAfter(prevOp);
        }
      }
      
      if (auto nodeNameAttr = opToMove->getAttrOfType<StringAttr>("onnx_node_name")) {
        LLVM_DEBUG(llvm::dbgs() << "Moved: " << nodeNameAttr.getValue() << "\n");
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Reordering completed successfully\n");
    return true;
  }
  
  // 找到所有常量操作之后的插入点
  Operation* findInsertionPointAfterConstants(Operation* parentOp) {
    Block* block = nullptr;
    
    // 获取包含操作的block
    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      block = &funcOp.getBody().front();
    } else {
      block = parentOp->getBlock();
    }
    
    if (!block) {
      return nullptr;
    }
    
    Operation* lastConstant = nullptr;
    
    // 遍历block，找到最后一个常量或工具操作
    for (Operation& op : *block) {
      if (isConstantOrUtilityOp(&op)) {
        lastConstant = &op;
        LLVM_DEBUG(llvm::dbgs() << "Found constant/utility op: " << op.getName() << "\n");
      } else {
        // 一旦遇到非常量操作，如果之前找到了常量，就停止搜索
        if (lastConstant) {
          break;
        }
      }
    }
    
    if (lastConstant) {
      LLVM_DEBUG(llvm::dbgs() << "Last constant operation found: " << lastConstant->getName() << "\n");
      return lastConstant;
    }
    
    // 如果没有找到常量操作，返回block的第一个操作（如果存在）
    if (!block->empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No constants found, using first operation in block\n");
      return &block->front();
    }
    
    return nullptr;
  }

  // 检查是否是常量或工具操作，这些操作不应该被重排
  bool isConstantOrUtilityOp(Operation* op) {
    StringRef opName = op->getName().getStringRef();
    
    // 常量操作
    if (opName == "onnx.Constant") {
      return true;
    }
    
    // 其他不应该重排的操作
    if (opName == "func.return" || 
        opName == "func.call" ||
        opName.starts_with("builtin.") ||
        opName.starts_with("arith.") ||
        opName.starts_with("memref.global")) {
      return true;
    }
    
    return false;
  }

  // === 修复后的 Batch Fusion 逻辑 ===
  
  // 执行batch fusion的主函数 - 修复版本，按batch size分组
  bool performBatchFusion(func::FuncOp funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "Starting batch fusion analysis with batch size: " << batchSize << "\n");
    
    // 1. 识别可融合的操作组（现在按batch size分组）
    SmallVector<FusibleGroup> fusibleGroups;
    identifyFusibleGroups(funcOp, fusibleGroups);
    
    // 2. 调试打印融合组信息
    debugPrintFusibleGroups(fusibleGroups);
    
    if (fusibleGroups.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No fusible groups found\n");
      return true;
    }
    
    // 3. 验证每个组的大小不超过batch size
    for (const auto& group : fusibleGroups) {
      if (group.operations.size() > static_cast<size_t>(batchSize)) {
        LLVM_DEBUG(llvm::dbgs() << "ERROR: Group size " << group.operations.size() 
                   << " exceeds batch size " << batchSize << "\n");
        return false;
      }
    }
    
    // 4. 按照第一个操作在IR中的位置对融合组进行排序
    llvm::sort(fusibleGroups, [](const FusibleGroup& a, const FusibleGroup& b) {
      if (a.operations.empty() || b.operations.empty()) return false;
      return a.operations[0]->isBeforeInBlock(b.operations[0]);
    });
    
    // 5. 对每个可融合组执行fusion
    for (auto& group : fusibleGroups) {
      LLVM_DEBUG(llvm::dbgs() << "Fusing group: layer " << group.layerIndex 
                 << ", op type: " << group.opType 
                 << ", " << group.operations.size() << " operations"
                 << " (batch size limited to " << batchSize << ")\n");
      
      if (!fuseOperationGroup(funcOp, group)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to fuse operation group\n");
        return false;
      }
    }
    
    return true;
  }
  
  // 修改identifyFusibleGroups函数，确保正确的分组和相邻性检查
  void identifyFusibleGroups(func::FuncOp funcOp, SmallVector<FusibleGroup>& fusibleGroups) {
    // 按 (layer_index, normalized_op_type) 收集所有可融合的操作
    std::map<std::pair<int, std::string>, SmallVector<std::pair<Operation*, int>>> tempGroupMap;
    
    funcOp.walk([&](Operation* op) {
      if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
        StringRef nodeName = nodeNameAttr.getValue();
        
        if (nodeName.contains("/layer/")) {
          SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName, op); // 传递op指针
          
          // 检查是否是可融合的操作类型
          if (isFusibleOperationType(layerInfo.opName)) {
            // 标准化操作类型：将Gemm(bias=0)统一为MatMul处理
            std::string normalizedOpType = layerInfo.opName;
            if (layerInfo.opName == "Gemm") {
              auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
              if (gemmOp && isBiasZero(gemmOp.getC())) {
                normalizedOpType = "MatMul";  // bias=0的Gemm当作MatMul处理
                LLVM_DEBUG(llvm::dbgs() << "Gemm with zero bias treated as MatMul: " << nodeName << "\n");
              }
            }
            
            auto key = std::make_pair(layerInfo.layerIndex, normalizedOpType);
            tempGroupMap[key].emplace_back(op, layerInfo.timestep);
            
            LLVM_DEBUG(llvm::dbgs() << "Added to temp group: " << nodeName 
                      << " (layer " << layerInfo.layerIndex 
                      << ", timestep " << layerInfo.timestep 
                      << ", actual op: " << op->getName().getStringRef()
                      << ", normalized type: " << normalizedOpType << ")\n");
          }
        }
      }
    });

    
    // 按batch size将大的组拆分成小的组，并验证相邻性
    for (auto& [key, opTimestepPairs] : tempGroupMap) {
      if (opTimestepPairs.size() <= 1) {
        continue; // 单个操作无需融合
      }
      
      // 按时间步排序
      llvm::sort(opTimestepPairs, [](const auto& a, const auto& b) {
        return a.second < b.second;
      });
      
      LLVM_DEBUG(llvm::dbgs() << "Processing layer " << key.first 
                 << ", type " << key.second 
                 << " with " << opTimestepPairs.size() << " operations\n");
      
      // 按batch size分组，同时确保分组内操作的相邻性
      for (size_t i = 0; i < opTimestepPairs.size(); i += batchSize) {
        size_t groupSize = std::min(static_cast<size_t>(batchSize), 
                                    opTimestepPairs.size() - i);
        
        if (groupSize > 1) { // 只有多于1个操作的组才需要融合
          SmallVector<std::pair<Operation*, int>> candidateGroup;
          for (size_t j = i; j < i + groupSize; j++) {
            candidateGroup.push_back(opTimestepPairs[j]);
          }
          
          // 验证候选组是否满足相邻性要求（重排后的特性）
          if (validateGroupAdjacency(candidateGroup)) {
            FusibleGroup group(key.first, key.second);
            
            for (const auto& pair : candidateGroup) {
              group.operations.push_back(pair.first);
              group.timesteps.push_back(pair.second);
            }
            
            fusibleGroups.push_back(std::move(group));
            
            LLVM_DEBUG(llvm::dbgs() << "Created valid fusible group: layer " << key.first 
                       << ", type " << key.second 
                       << ", " << candidateGroup.size() << " ops"
                       << " (timesteps " << candidateGroup[0].second 
                       << " to " << candidateGroup.back().second << ")\n");
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Rejected non-adjacent group: layer " << key.first 
                       << ", type " << key.second 
                       << " (timesteps " << candidateGroup[0].second 
                       << " to " << candidateGroup.back().second << ")\n");
            
            // 如果组不相邻，尝试逐个添加相邻的操作
            createAdjacentSubgroups(candidateGroup, key.first, key.second, fusibleGroups);
          }
        }
      }
    }
  }
  
  // 验证操作组是否在IR中相邻（利用重排后的特性）
  bool validateGroupAdjacency(const SmallVector<std::pair<Operation*, int>>& candidateGroup) {
    if (candidateGroup.size() <= 1) {
      return true;
    }
    
    // 按IR中的顺序排序
    SmallVector<Operation*> sortedOps;
    for (const auto& pair : candidateGroup) {
      sortedOps.push_back(pair.first);
    }
    llvm::sort(sortedOps, [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
    
    // 检查操作是否基本相邻（允许中间有少量非融合操作）
    Operation* prevOp = sortedOps[0];
    int maxGap = 5; // 允许的最大间隔操作数
    
    for (size_t i = 1; i < sortedOps.size(); ++i) {
      Operation* currentOp = sortedOps[i];
      int gap = 0;
      Operation* checkOp = prevOp->getNextNode();
      
      // 计算两个操作之间的距离
      while (checkOp && checkOp != currentOp && gap < maxGap) {
        // 跳过常量和时序操作
        if (!isConstantOrUtilityOp(checkOp) && !isTemporalOp(checkOp)) {
          gap++;
        }
        checkOp = checkOp->getNextNode();
      }
      
      if (gap >= maxGap || !checkOp) {
        LLVM_DEBUG(llvm::dbgs() << "Operations too far apart (gap: " << gap << ")\n");
        return false;
      }
      
      prevOp = currentOp;
    }
    
    return true;
  }
  
  // 检查是否是时序操作（非融合操作）
  bool isTemporalOp(Operation* op) {
    if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
      StringRef nodeName = nodeNameAttr.getValue();
      if (nodeName.contains("/layer/")) {
        SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName);
        // 时序操作通常包括激活函数、归一化等
        return !isFusibleOperationType(layerInfo.opName) && 
               (layerInfo.opName.find("Div") != std::string::npos ||
                layerInfo.opName.find("Add") != std::string::npos ||
                layerInfo.opName.find("Sub") != std::string::npos ||
                layerInfo.opName.find("Mul") != std::string::npos ||
                layerInfo.opName.find("Cast") != std::string::npos ||
                layerInfo.opName.find("Neg") != std::string::npos ||
                layerInfo.opName.find("GreaterOrEqual") != std::string::npos ||
                layerInfo.opName.find("Flatten") != std::string::npos);
      }
    }
    return false;
  }
  
  // 创建相邻的子组
  void createAdjacentSubgroups(const SmallVector<std::pair<Operation*, int>>& candidateGroup,
                               int layerIndex, const std::string& opType,
                               SmallVector<FusibleGroup>& fusibleGroups) {
    // 按IR顺序排序
    SmallVector<std::pair<Operation*, int>> sortedGroup = candidateGroup;
    llvm::sort(sortedGroup, [](const auto& a, const auto& b) {
      return a.first->isBeforeInBlock(b.first);
    });
    
    // 找到连续的相邻操作子序列
    SmallVector<std::pair<Operation*, int>> currentSubgroup;
    
    for (size_t i = 0; i < sortedGroup.size(); ++i) {
      if (currentSubgroup.empty()) {
        currentSubgroup.push_back(sortedGroup[i]);
      } else {
        // 检查与当前子组最后一个操作的相邻性
        SmallVector<std::pair<Operation*, int>> testGroup = currentSubgroup;
        testGroup.push_back(sortedGroup[i]);
        
        if (validateGroupAdjacency(testGroup)) {
          currentSubgroup.push_back(sortedGroup[i]);
        } else {
          // 当前子组结束，创建融合组（如果有多个操作）
          if (currentSubgroup.size() > 1) {
            FusibleGroup group(layerIndex, opType);
            for (const auto& pair : currentSubgroup) {
              group.operations.push_back(pair.first);
              group.timesteps.push_back(pair.second);
            }
            fusibleGroups.push_back(std::move(group));
            
            LLVM_DEBUG(llvm::dbgs() << "Created adjacent subgroup: layer " << layerIndex 
                       << ", type " << opType 
                       << ", " << currentSubgroup.size() << " ops\n");
          }
          
          // 开始新的子组
          currentSubgroup.clear();
          currentSubgroup.push_back(sortedGroup[i]);
        }
      }
    }
    
    // 处理最后一个子组
    if (currentSubgroup.size() > 1) {
      FusibleGroup group(layerIndex, opType);
      for (const auto& pair : currentSubgroup) {
        group.operations.push_back(pair.first);
        group.timesteps.push_back(pair.second);
      }
      fusibleGroups.push_back(std::move(group));
      
      LLVM_DEBUG(llvm::dbgs() << "Created final adjacent subgroup: layer " << layerIndex 
                 << ", type " << opType 
                 << ", " << currentSubgroup.size() << " ops\n");
    }
  }
  
  // 检查是否是可融合的操作类型 - 修复版本
  bool isFusibleOperationType(const std::string& opName) {
    return opName == "Conv" || 
           opName == "MaxPool" ||           // 修复：原来是MaxPoolSingleOut
           opName == "MaxPoolSingleOut" ||  // 保留原有的，以防有其他变体
           opName == "Gemm" || 
           opName == "MatMul";
  }
  

  // 修改操作组融合函数，确保正确的插入位置和支配关系
  bool fuseOperationGroup(func::FuncOp funcOp, FusibleGroup& group) {
    if (group.operations.empty()) {
      return true;
    }
    
    // 按照IR中的顺序对操作进行排序，确保正确的支配关系
    llvm::sort(group.operations, [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
    
    // 找到所有操作的最后一个操作，作为插入点
    Operation* lastOp = group.operations.back();
    
    // 同时要确保所有输入操作都在插入点之前定义
    Operation* insertionPoint = lastOp;
    for (Operation* op : group.operations) {
      for (Value operand : op->getOperands()) {
        if (Operation* defOp = operand.getDefiningOp()) {
          if (insertionPoint->isBeforeInBlock(defOp)) {
            insertionPoint = defOp;
          }
        }
      }
    }
    
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointAfter(insertionPoint);
    
    LLVM_DEBUG(llvm::dbgs() << "Fusing group with " << group.operations.size() 
               << " operations of type: " << group.opType 
               << " (batch size: " << group.operations.size() << ")\n");
    
    // 根据标准化后的操作类型执行不同的融合策略
    if (group.opType == "Conv") {
        bool hack = fuseConvOperations(builder, group);
      return true;
    } else if (group.opType == "MaxPool" || group.opType == "MaxPoolSingleOut") {
      return fuseMaxPoolOperations(builder, group);
    } else if (group.opType == "MatMul") {
      // 这里处理纯MatMul或者Gemm(bias=0)的融合
      // return fuseMixedMatMulOperations(builder, group); //gemm 与 matmul混合的情况
      return fuseMatMulOperations(builder, group);
    } else if (group.opType == "Gemm") {
      // 这里处理有非零bias的Gemm融合
      return fuseGemmOperations(builder, group);
    } 
    
    LLVM_DEBUG(llvm::dbgs() << "Unknown operation type for fusion: " << group.opType << "\n");
    return false;
  }
  

  // 融合Conv操作 - 并行复制版本
  bool fuseConvOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Conv operations with parallel copying\n");
    
    Operation* firstOp = group.operations[0];
    auto firstConv = dyn_cast<mlir::ONNXConvOp>(firstOp);
    if (!firstConv) {
      LLVM_DEBUG(llvm::dbgs() << "First operation is not a Conv op\n");
      return false;
    }
    
    // 获取输入类型信息
    auto inputType = firstConv.getX().getType().cast<RankedTensorType>();
    auto weightType = firstConv.getW().getType().cast<RankedTensorType>();
    auto outputType = firstConv.getY().getType().cast<RankedTensorType>();
    
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    
    // 计算融合后的batch大小
    int64_t originalBatchSize = inputShape[0];
    int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
    // 创建融合后的形状
    SmallVector<int64_t> fusedInputShape(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
    fusedInputShape[0] = fusedBatchSize;
    fusedOutputShape[0] = fusedBatchSize;
    
    auto fusedInputType = RankedTensorType::get(fusedInputShape, inputType.getElementType());
    auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
    // 创建memref类型用于GPU操作
    auto inputMemRefType = MemRefType::get(fusedInputShape, inputType.getElementType());
    auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
    auto originalInputMemRefType = MemRefType::get(inputShape, inputType.getElementType());
    auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    
    // 分配GPU内存  
      // 分配内存 - 修改为使用memref.alloc
      Value batchedInput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), inputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      Value batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
    
    // 为每个原始操作分配独立的输出内存
      SmallVector<Value> individualOutputs;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
    
    LLVM_DEBUG(llvm::dbgs() << "Allocated GPU memory for batched input/output\n");
    
    // 创建输入和输出子视图
    SmallVector<Value> inputViews;
    SmallVector<Value> outputViews;
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      // 计算当前操作在batch中的偏移
      int64_t offset = i * originalBatchSize;
      
      // 创建输入子视图
      SmallVector<OpFoldResult> inputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> inputSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(inputShape[1]),
          builder.getI64IntegerAttr(inputShape[2]),
          builder.getI64IntegerAttr(inputShape[3])
      };
      SmallVector<OpFoldResult> inputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value inputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedInput, inputOffsets, inputSizes, inputStrides);
      inputViews.push_back(inputView);
      
      // 创建输出子视图
      SmallVector<OpFoldResult> outputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> outputSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(outputShape[1]),
          builder.getI64IntegerAttr(outputShape[2]),
          builder.getI64IntegerAttr(outputShape[3])
      };
      SmallVector<OpFoldResult> outputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value outputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
      outputViews.push_back(outputView);
      
      LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
    }
    
    // ========== 修改：并行执行输入数据复制 ==========
    // 1. 首先一起创建所有需要的异步流
    SmallVector<Value> copyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      copyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有必要的类型转换
    SmallVector<Value> inputMemRefs;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* currentOp = group.operations[i];
      Value originalInput = currentOp->getOperand(0);
      
      Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), 
          originalInputMemRefType,
          originalInput).getResult(0);
      inputMemRefs.push_back(inputMemRef);
    }
    
    // 3. 执行所有并行复制操作
    SmallVector<Value> inputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(copyStreams[i]),
          inputViews[i], inputMemRefs[i]);  // 目标，源
      inputCopyTokens.push_back(copyToken.getAsyncToken());
      
      LLVM_DEBUG(llvm::dbgs() << "Initiated parallel input copy for operation " << i << "\n");
    }
    
    // 4. 等待所有输入复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
    // =====================================================
    
    // 转换batched input为tensor用于ONNX操作
    Value batchedInputTensor = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), fusedInputType, batchedInput).getResult(0);
    
    // 创建融合的Conv操作
    Value fusedResult = builder.create<mlir::ONNXConvOp>(
        builder.getUnknownLoc(), 
        fusedOutputType,
        batchedInputTensor,
        firstConv.getW(),  // 权重可以共享
        firstConv.getB(),  // bias可以共享
        firstConv.getAutoPadAttr(),
        firstConv.getDilationsAttr(),
        firstConv.getGroupAttr(),
        firstConv.getKernelShapeAttr(),
        firstConv.getPadsAttr(),
        firstConv.getStridesAttr());
    
    // 转换融合结果为memref
    Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
    LLVM_DEBUG(llvm::dbgs() << "Created fused Conv operation\n");
    
    // 将融合结果复制到预分配的batched output memory
    auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
    auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
        builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()),
        ValueRange(fusedCopyStream.getAsyncToken()),
        batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
    LLVM_DEBUG(llvm::dbgs() << "Copied fused Conv result to batched output memory\n");
    
    // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
    // 1. 首先一起创建所有需要的输出异步流
    SmallVector<Value> outputCopyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      outputCopyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有并行输出复制操作
    SmallVector<Value> outputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(outputCopyStreams[i]),
          individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
      outputCopyTokens.push_back(copyToken.getAsyncToken());
      
      LLVM_DEBUG(llvm::dbgs() << "Initiated parallel output copy for operation " << i << "\n");
    }
    
    // 3. 等待所有输出复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
    // 4. 转换回tensor
    SmallVector<Value> splitResults;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
      splitResults.push_back(splitResult);
    }
    // =================================================================
    
    // 先替换所有使用，然后再删除操作（避免dominance问题）
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      op->getResult(0).replaceAllUsesWith(splitResults[i]);
      LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
    }
    
    // 延迟删除操作，确保所有replacement完成后再删除
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << "\n");
      op->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused Conv operations with parallel copying\n");
    return true;
  }

  // 融合MaxPool操作 - 并行复制版本
  bool fuseMaxPoolOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MaxPool operations with parallel copying\n");
    
    Operation* firstOp = group.operations[0];
    auto firstMaxPool = dyn_cast<mlir::ONNXMaxPoolSingleOutOp>(firstOp);
    if (!firstMaxPool) {
      LLVM_DEBUG(llvm::dbgs() << "First operation is not a MaxPool op\n");
      return false;
    }
    
    auto inputType = firstMaxPool.getX().getType().cast<RankedTensorType>();
    auto outputType = firstMaxPool.getResult().getType().cast<RankedTensorType>();
    
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    
    int64_t originalBatchSize = inputShape[0];
    int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
    SmallVector<int64_t> fusedInputShape(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
    fusedInputShape[0] = fusedBatchSize;
    fusedOutputShape[0] = fusedBatchSize;
    
    auto fusedInputType = RankedTensorType::get(fusedInputShape, inputType.getElementType());
    auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
    auto inputMemRefType = MemRefType::get(fusedInputShape, inputType.getElementType());
    auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
    auto originalInputMemRefType = MemRefType::get(inputShape, inputType.getElementType());
    auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    
    // 分配GPU内存
      // 分配内存 - 修改为使用memref.alloc
      Value batchedInput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), inputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      Value batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      // 为每个原始操作分配独立的输出内存
      SmallVector<Value> individualOutputs;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
    
    // 创建输入和输出子视图
    SmallVector<Value> inputViews;
    SmallVector<Value> outputViews;
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      int64_t offset = i * originalBatchSize;
      
      // 输入子视图
      SmallVector<OpFoldResult> inputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> inputSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(inputShape[1]),
          builder.getI64IntegerAttr(inputShape[2]),
          builder.getI64IntegerAttr(inputShape[3])
      };
      SmallVector<OpFoldResult> inputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value inputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedInput, inputOffsets, inputSizes, inputStrides);
      inputViews.push_back(inputView);
      
      // 输出子视图
      SmallVector<OpFoldResult> outputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> outputSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(outputShape[1]),
          builder.getI64IntegerAttr(outputShape[2]),
          builder.getI64IntegerAttr(outputShape[3])
      };
      SmallVector<OpFoldResult> outputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value outputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
      outputViews.push_back(outputView);
    }
    
    // ========== 修改：并行执行输入数据复制 ==========
    // 1. 首先一起创建所有需要的异步流
    SmallVector<Value> copyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      copyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有必要的类型转换
    SmallVector<Value> inputMemRefs;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* currentOp = group.operations[i];
      Value originalInput = currentOp->getOperand(0);
      
      Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), 
          originalInputMemRefType,
          originalInput).getResult(0);
      inputMemRefs.push_back(inputMemRef);
    }
    
    // 3. 执行所有并行复制操作
    SmallVector<Value> inputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(copyStreams[i]),
          inputViews[i], inputMemRefs[i]);  // 目标，源
      inputCopyTokens.push_back(copyToken.getAsyncToken());
    }
    
    // 4. 等待所有输入复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
    // =====================================================
    
    // 创建融合的MaxPool操作
    Value batchedInputTensor = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), fusedInputType, batchedInput).getResult(0);
    
    Value fusedResult = builder.create<mlir::ONNXMaxPoolSingleOutOp>(
        builder.getUnknownLoc(),
        fusedOutputType,
        batchedInputTensor,
        firstMaxPool.getAutoPadAttr(),
        firstMaxPool.getCeilModeAttr(),
        firstMaxPool.getDilationsAttr(),
        firstMaxPool.getKernelShapeAttr(),
        firstMaxPool.getPadsAttr(),
        firstMaxPool.getStorageOrderAttr(),
        firstMaxPool.getStridesAttr());
    
    Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
    LLVM_DEBUG(llvm::dbgs() << "Created fused MaxPool operation\n");
    
    // 将融合结果复制到预分配的batched output memory
    auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
    auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
        builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()),
        ValueRange(fusedCopyStream.getAsyncToken()),
        batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
    LLVM_DEBUG(llvm::dbgs() << "Copied fused MaxPool result to batched output memory\n");
    
    // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
    // 1. 首先一起创建所有需要的输出异步流
    SmallVector<Value> outputCopyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      outputCopyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有并行输出复制操作
    SmallVector<Value> outputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(outputCopyStreams[i]),
          individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
      outputCopyTokens.push_back(copyToken.getAsyncToken());
    }
    
    // 3. 等待所有输出复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
    // 4. 转换回tensor
    SmallVector<Value> splitResults;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
      splitResults.push_back(splitResult);
    }
    // =================================================================
    
    // 先替换所有使用，然后再删除操作（避免dominance问题）
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      op->getResult(0).replaceAllUsesWith(splitResults[i]);
      LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
    }
    
    // 延迟删除操作，确保所有replacement完成后再删除
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << "\n");
      op->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused MaxPool operations with parallel copying\n");
    return true;
  }

  // 混合MatMul操作融合（包括Gemm(bias=0)和MatMul）- 并行复制版本
  bool fuseMixedMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " mixed MatMul operations with parallel copying\n");
    
    Operation* firstOp = group.operations[0];
    
    // 确定是否第一个操作是MatMul还是Gemm
    auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
    auto firstGemm = dyn_cast<mlir::ONNXGemmOp>(firstOp);
    
    if (!firstMatMul && !firstGemm) {
      LLVM_DEBUG(llvm::dbgs() << "First operation is neither MatMul nor Gemm\n");
      return false;
    }
    
    // 获取输入输出类型 - 统一从MatMul或Gemm中获取
    Value firstA, firstB;
    Type outputType;
    
    if (firstMatMul) {
      firstA = firstMatMul.getA();
      firstB = firstMatMul.getB();
      outputType = firstMatMul.getY().getType();
    } else {
      firstA = firstGemm.getA();
      firstB = firstGemm.getB();
      outputType = firstGemm.getY().getType();
    }
    
    auto aType = firstA.getType().cast<RankedTensorType>();
    auto bType = firstB.getType().cast<RankedTensorType>();
    auto outputTensorType = outputType.cast<RankedTensorType>();
    
    ArrayRef<int64_t> aShape = aType.getShape();
    ArrayRef<int64_t> outputShape = outputTensorType.getShape();
    
    int64_t originalBatchSize = aShape[0];
    int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
    SmallVector<int64_t> fusedAShape(aShape.begin(), aShape.end());
    SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
    fusedAShape[0] = fusedBatchSize;
    fusedOutputShape[0] = fusedBatchSize;
    
    auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
    auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputTensorType.getElementType());
    
    auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
    auto outputMemRefType = MemRefType::get(fusedOutputShape, outputTensorType.getElementType());
    auto originalAMemRefType = MemRefType::get(aShape, aType.getElementType());
    auto originalOutputMemRefType = MemRefType::get(outputShape, outputTensorType.getElementType());
    
    // 分配GPU内存
      // 分配内存 - 修改为使用memref.alloc
      Value batchedA = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), aMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      Value batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      // 为每个原始操作分配独立的输出内存
      SmallVector<Value> individualOutputs;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
    
    // 创建输入和输出子视图
    SmallVector<Value> aViews;
    SmallVector<Value> outputViews;
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      int64_t offset = i * originalBatchSize;
      
      // A矩阵子视图
      SmallVector<OpFoldResult> aOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> aSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(aShape[1])
      };
      SmallVector<OpFoldResult> aStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value aView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
      aViews.push_back(aView);
      
      // 输出子视图
      SmallVector<OpFoldResult> outputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> outputSizes = {
          builder.getI64IntegerAttr(originalBatchSize),
          builder.getI64IntegerAttr(outputShape[1])
      };
      SmallVector<OpFoldResult> outputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value outputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
      outputViews.push_back(outputView);
    }
    
    // ========== 修改：并行执行输入数据复制 - 处理混合操作 ==========
    // 1. 首先一起创建所有需要的异步流
    SmallVector<Value> copyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      copyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有必要的类型转换 - 处理混合操作
    SmallVector<Value> aMemRefs;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* currentOp = group.operations[i];
      Value originalA;
      
      // 从MatMul或Gemm中获取A矩阵
      if (auto matmulOp = dyn_cast<mlir::ONNXMatMulOp>(currentOp)) {
        originalA = matmulOp.getA();
      } else if (auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(currentOp)) {
        originalA = gemmOp.getA();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported operation type in mixed fusion\n");
        return false;
      }
      
      Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), 
          originalAMemRefType,
          originalA).getResult(0);
      aMemRefs.push_back(aMemRef);
    }
    
    // 3. 执行所有并行复制操作
    SmallVector<Value> inputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(copyStreams[i]),
          aViews[i], aMemRefs[i]);  // 目标，源
      inputCopyTokens.push_back(copyToken.getAsyncToken());
    }
    
    // 4. 等待所有输入复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed for mixed operations\n");
    // =================================================================
    
    // 创建融合的MatMul操作（统一使用MatMul，即使原来有Gemm）
    Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
    
    Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
        builder.getUnknownLoc(),
        fusedOutputType,
        batchedATensor,
        firstB);  // B矩阵可以共享
    
    Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
    LLVM_DEBUG(llvm::dbgs() << "Created fused MatMul operation for mixed group\n");
    
    // 将融合结果复制到预分配的batched output memory
    auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
    auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
        builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()),
        ValueRange(fusedCopyStream.getAsyncToken()),
        batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
    LLVM_DEBUG(llvm::dbgs() << "Copied fused mixed MatMul result to batched output memory\n");
    
    // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
    // 1. 首先一起创建所有需要的输出异步流
    SmallVector<Value> outputCopyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      outputCopyStreams.push_back(copyStream.getAsyncToken());
    }
    
    // 2. 执行所有并行输出复制操作
    SmallVector<Value> outputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(outputCopyStreams[i]),
          individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
      outputCopyTokens.push_back(copyToken.getAsyncToken());
    }
    
    // 3. 等待所有输出复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed for mixed operations\n");
    
    // 4. 转换回tensor
    SmallVector<Value> splitResults;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), outputTensorType, individualOutputs[i]).getResult(0);
      splitResults.push_back(splitResult);
    }
    // =================================================================
    
    // 先替换所有使用，然后再删除操作（避免dominance问题）
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      op->getResult(0).replaceAllUsesWith(splitResults[i]);
      LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
    }
    
    // 延迟删除操作，确保所有replacement完成后再删除
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << "\n");
      op->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused mixed MatMul operations with parallel copying\n");
    return true;
  }

  // // 调整生成Op的顺序
  // // 融合MatMul操作 - 增强版本，支持batch matmul广播
  // bool fuseMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
  //   LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MatMul operations with parallel copying and batch matmul support\n");
    
  //   Operation* firstOp = group.operations[0];
  //   auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
  //   if (!firstMatMul) {
  //     return false;
  //   }
    
  //   auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
  //   auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
  //   auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
    
  //   ArrayRef<int64_t> aShape = aType.getShape();
  //   ArrayRef<int64_t> bShape = bType.getShape();
  //   ArrayRef<int64_t> outputShape = outputType.getShape();
    
  //   LLVM_DEBUG(llvm::dbgs() << "MatMul shapes: A=" << aShape.size() << "D, B=" << bShape.size() << "D, Output=" << outputShape.size() << "D\n");
  //   LLVM_DEBUG(llvm::dbgs() << "A shape: [");
  //   for (size_t i = 0; i < aShape.size(); ++i) {
  //     LLVM_DEBUG(llvm::dbgs() << aShape[i]);
  //     if (i < aShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
  //   }
  //   LLVM_DEBUG(llvm::dbgs() << "]\n");
  //   LLVM_DEBUG(llvm::dbgs() << "B shape: [");
  //   for (size_t i = 0; i < bShape.size(); ++i) {
  //     LLVM_DEBUG(llvm::dbgs() << bShape[i]);
  //     if (i < bShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
  //   }
  //   LLVM_DEBUG(llvm::dbgs() << "]\n");
    
  //   // 检测是否为batch matmul情况
  //   bool isBatchMatMul = (aShape.size() == 2 && bShape.size() == 3) || (aShape.size() == 3 && bShape.size() == 2);
    
  //   if (isBatchMatMul) {
  //     LLVM_DEBUG(llvm::dbgs() << "Detected batch matmul case, using specialized fusion\n");
  //     return fuseBatchMatMulOperations(builder, group);
  //   } else {
  //     LLVM_DEBUG(llvm::dbgs() << "Standard matmul case, using regular fusion\n");
  //     return fuseStandardMatMulOperations(builder, group);
  //   }
  // }

  // // 专门处理batch matmul的融合函数
  // bool fuseBatchMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
  //   LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Batch MatMul operations\n");
    
  //   Operation* firstOp = group.operations[0];
  //   auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
  //   if (!firstMatMul) {
  //     return false;
  //   }
    
  //   auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
  //   auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
  //   auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
    
  //   ArrayRef<int64_t> aShape = aType.getShape();
  //   ArrayRef<int64_t> bShape = bType.getShape();
  //   ArrayRef<int64_t> outputShape = outputType.getShape();
    
  //   LLVM_DEBUG(llvm::dbgs() << "Original shapes: A=[" << aShape[0] << "," << aShape[1] << "], ");
  //   LLVM_DEBUG(llvm::dbgs() << "B=[" << bShape[0] << "," << bShape[1] << "," << bShape[2] << "], ");
  //   LLVM_DEBUG(llvm::dbgs() << "Output=[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "]\n");
    
  //   // 确定广播情况和参数
  //   bool needsBroadcastA = (aShape.size() == 2 && bShape.size() == 3);
  //   bool needsBroadcastB = (aShape.size() == 3 && bShape.size() == 2);
    
  //   if (!needsBroadcastA && !needsBroadcastB) {
  //     LLVM_DEBUG(llvm::dbgs() << "Error: Expected batch matmul case but shapes don't match\n");
  //     return false;
  //   }
    
  //   // 计算广播后的形状
  //   SmallVector<int64_t> broadcastedAShape;
  //   SmallVector<int64_t> broadcastedBShape; 
  //   int64_t batchDim = 0;
    
  //   if (needsBroadcastA) {
  //     // A: 2D -> 3D, B已经是3D
  //     batchDim = bShape[0];
  //     broadcastedAShape = {batchDim, aShape[0], aShape[1]};
  //     broadcastedBShape = {bShape[0], bShape[1], bShape[2]};
  //   } else {
  //     // B: 2D -> 3D, A已经是3D  
  //     batchDim = aShape[0];
  //     broadcastedAShape = {aShape[0], aShape[1], aShape[2]};
  //     broadcastedBShape = {batchDim, bShape[0], bShape[1]};
  //   }
    
  //   LLVM_DEBUG(llvm::dbgs() << "After broadcasting: A=[" << broadcastedAShape[0] << "," << broadcastedAShape[1] << "," << broadcastedAShape[2] << "], ");
  //   LLVM_DEBUG(llvm::dbgs() << "B=[" << broadcastedBShape[0] << "," << broadcastedBShape[1] << "," << broadcastedBShape[2] << "]\n");
    
  //   // 计算融合后的batch大小
  //   int64_t fusedBatchSize = batchDim * group.operations.size();
    
  //   // 创建融合后的形状
  //   SmallVector<int64_t> fusedAShape = {fusedBatchSize, broadcastedAShape[1], broadcastedAShape[2]};
  //   SmallVector<int64_t> fusedBShape = {fusedBatchSize, broadcastedBShape[1], broadcastedBShape[2]};
  //   SmallVector<int64_t> fusedOutputShape = {fusedBatchSize, outputShape[1], outputShape[2]};
    
  //   LLVM_DEBUG(llvm::dbgs() << "Fused shapes: A=[" << fusedAShape[0] << "," << fusedAShape[1] << "," << fusedAShape[2] << "], ");
  //   LLVM_DEBUG(llvm::dbgs() << "B=[" << fusedBShape[0] << "," << fusedBShape[1] << "," << fusedBShape[2] << "], ");
  //   LLVM_DEBUG(llvm::dbgs() << "Output=[" << fusedOutputShape[0] << "," << fusedOutputShape[1] << "," << fusedOutputShape[2] << "]\n");
    
  //   // 创建类型
  //   auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
  //   auto fusedBType = RankedTensorType::get(fusedBShape, bType.getElementType());
  //   auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
  //   auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
  //   auto bMemRefType = MemRefType::get(fusedBShape, bType.getElementType());
  //   auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
  //   auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
  //   auto broadcastedAMemRefType = MemRefType::get(broadcastedAShape, aType.getElementType());
  //   auto broadcastedBMemRefType = MemRefType::get(broadcastedBShape, bType.getElementType());
    
  //   // ========== 阶段1：预先生成所有的广播操作 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 1: Generating all broadcast operations\n");
  //   SmallVector<Value> processedAValues;
  //   SmallVector<Value> processedBValues;
    
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     Operation* currentOp = group.operations[i];
  //     Value originalA = currentOp->getOperand(0);
  //     Value originalB = currentOp->getOperand(1);
      
  //     // 处理A矩阵的广播
  //     Value processedA;
  //     if (needsBroadcastA) {
  //       processedA = performBroadcast2Dto3D(builder, originalA, batchDim, aType.getElementType());
  //       LLVM_DEBUG(llvm::dbgs() << "Generated broadcast for A matrix " << i << "\n");
  //     } else {
  //       processedA = originalA;
  //       LLVM_DEBUG(llvm::dbgs() << "Using original A matrix " << i << "\n");
  //     }
  //     processedAValues.push_back(processedA);
      
  //     // 处理B矩阵的广播
  //     Value processedB;
  //     if (needsBroadcastB) {
  //       processedB = performBroadcast2Dto3D(builder, originalB, batchDim, bType.getElementType());
  //       LLVM_DEBUG(llvm::dbgs() << "Generated broadcast for B matrix " << i << "\n");
  //     } else {
  //       processedB = originalB;
  //       LLVM_DEBUG(llvm::dbgs() << "Using original B matrix " << i << "\n");
  //     }
  //     processedBValues.push_back(processedB);
  //   }
    
  //   // ========== 阶段2：分配所有GPU内存 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 2: Allocating all GPU memory\n");
  //   Value batchedA = builder.create<memref::AllocOp>(
  //       builder.getUnknownLoc(), aMemRefType, 
  //       ValueRange{}, builder.getI64IntegerAttr(16));
    
  //   Value batchedB = builder.create<memref::AllocOp>(
  //       builder.getUnknownLoc(), bMemRefType, 
  //       ValueRange{}, builder.getI64IntegerAttr(16));
    
  //   Value batchedOutput = builder.create<memref::AllocOp>(
  //       builder.getUnknownLoc(), outputMemRefType, 
  //       ValueRange{}, builder.getI64IntegerAttr(16));
    
  //   // 为每个原始操作分配独立的输出内存
  //   SmallVector<Value> individualOutputs;
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     Value individualOutput = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), originalOutputMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
  //     individualOutputs.push_back(individualOutput);
  //   }
    
  //   LLVM_DEBUG(llvm::dbgs() << "Allocated all GPU memory\n");
    
  //   // ========== 阶段3：生成所有的subviews ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 3: Creating all subviews\n");
  //   SmallVector<Value> aViews;
  //   SmallVector<Value> bViews;
  //   SmallVector<Value> outputViews;
    
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     int64_t offset = i * batchDim;
      
  //     // 创建A矩阵子视图
  //     SmallVector<OpFoldResult> aOffsets = {
  //         builder.getI64IntegerAttr(offset),
  //         builder.getI64IntegerAttr(0),
  //         builder.getI64IntegerAttr(0)
  //     };
  //     SmallVector<OpFoldResult> aSizes = {
  //         builder.getI64IntegerAttr(batchDim),
  //         builder.getI64IntegerAttr(broadcastedAShape[1]),
  //         builder.getI64IntegerAttr(broadcastedAShape[2])
  //     };
  //     SmallVector<OpFoldResult> aStrides = {
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1)
  //     };
      
  //     Value aView = builder.create<memref::SubViewOp>(
  //         builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
  //     aViews.push_back(aView);
      
  //     // 创建B矩阵子视图
  //     SmallVector<OpFoldResult> bOffsets = {
  //         builder.getI64IntegerAttr(offset),
  //         builder.getI64IntegerAttr(0),
  //         builder.getI64IntegerAttr(0)
  //     };
  //     SmallVector<OpFoldResult> bSizes = {
  //         builder.getI64IntegerAttr(batchDim),
  //         builder.getI64IntegerAttr(broadcastedBShape[1]),
  //         builder.getI64IntegerAttr(broadcastedBShape[2])
  //     };
  //     SmallVector<OpFoldResult> bStrides = {
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1)
  //     };
      
  //     Value bView = builder.create<memref::SubViewOp>(
  //         builder.getUnknownLoc(), batchedB, bOffsets, bSizes, bStrides);
  //     bViews.push_back(bView);
      
  //     // 创建输出子视图（用于后续输出复制）
  //     SmallVector<OpFoldResult> outputOffsets = {
  //         builder.getI64IntegerAttr(offset),
  //         builder.getI64IntegerAttr(0),
  //         builder.getI64IntegerAttr(0)
  //     };
  //     SmallVector<OpFoldResult> outputSizes = {
  //         builder.getI64IntegerAttr(batchDim),
  //         builder.getI64IntegerAttr(outputShape[1]),
  //         builder.getI64IntegerAttr(outputShape[2])
  //     };
  //     SmallVector<OpFoldResult> outputStrides = {
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1),
  //         builder.getI64IntegerAttr(1)
  //     };
      
  //     Value outputView = builder.create<memref::SubViewOp>(
  //         builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
  //     outputViews.push_back(outputView);
      
  //     LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
  //   }
    
  //   // ========== 阶段4：预先生成所有的类型转换 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 4: Generating all type conversions\n");
  //   SmallVector<Value> aMemRefs;
  //   SmallVector<Value> bMemRefs;
    
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     // A矩阵类型转换
  //     Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), broadcastedAMemRefType, processedAValues[i]).getResult(0);
  //     aMemRefs.push_back(aMemRef);
      
  //     // B矩阵类型转换
  //     Value bMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), broadcastedBMemRefType, processedBValues[i]).getResult(0);
  //     bMemRefs.push_back(bMemRef);
      
  //     LLVM_DEBUG(llvm::dbgs() << "Generated type conversions for operation " << i << "\n");
  //   }
    
  //   // ========== 阶段5：创建所有异步流并执行复制 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 5: Creating async streams and performing copies\n");
  //   SmallVector<Value> copyStreams;
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //     copyStreams.push_back(copyStream.getAsyncToken());
  //   }
    
  //   SmallVector<Value> allCopyTokens;
    
  //   // 执行A矩阵复制
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     auto copyToken = builder.create<gpu::MemcpyOp>(
  //         builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()),
  //         ValueRange(copyStreams[i]),
  //         aViews[i], aMemRefs[i]);
  //     allCopyTokens.push_back(copyToken.getAsyncToken());
      
  //     LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << "\n");
  //   }
    
  //   // 执行B矩阵复制
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     auto copyToken = builder.create<gpu::MemcpyOp>(
  //         builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()),
  //         ValueRange(copyStreams[i]),
  //         bViews[i], bMemRefs[i]);
  //     allCopyTokens.push_back(copyToken.getAsyncToken());
      
  //     LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << "\n");
  //   }
    
  //   // 等待所有输入复制完成
  //   builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
  //   LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
    
  //   // ========== 阶段6：执行融合计算 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 6: Performing fused computation\n");
  //   // 转换为tensor用于ONNX操作
  //   Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //       builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
  //   Value batchedBTensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //       builder.getUnknownLoc(), fusedBType, batchedB).getResult(0);
    
  //   // 创建融合的3D MatMul操作
  //   Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
  //       builder.getUnknownLoc(),
  //       fusedOutputType,
  //       batchedATensor,
  //       batchedBTensor);
    
  //   LLVM_DEBUG(llvm::dbgs() << "Created fused 3D MatMul operation: (" 
  //             << fusedAShape[0] << "x" << fusedAShape[1] << "x" << fusedAShape[2] << ") × ("
  //             << fusedBShape[0] << "x" << fusedBShape[1] << "x" << fusedBShape[2] << ") → ("
  //             << fusedOutputShape[0] << "x" << fusedOutputShape[1] << "x" << fusedOutputShape[2] << ")\n");
    
  //   // 转换融合结果为memref
  //   Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //       builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
  //   // 将融合结果复制到预分配的batched output memory
  //   auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //       gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //   auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
  //       builder.getUnknownLoc(), 
  //       gpu::AsyncTokenType::get(builder.getContext()),
  //       ValueRange(fusedCopyStream.getAsyncToken()),
  //       batchedOutput, fusedResultMemRef);
  //   builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
  //   LLVM_DEBUG(llvm::dbgs() << "Copied fused batch MatMul result to batched output memory\n");
    
  //   // ========== 阶段7：输出复制 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 7: Performing output copies\n");
  //   SmallVector<Value> outputCopyStreams;
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //     outputCopyStreams.push_back(copyStream.getAsyncToken());
  //   }
    
  //   SmallVector<Value> outputCopyTokens;
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     auto copyToken = builder.create<gpu::MemcpyOp>(
  //         builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()),
  //         ValueRange(outputCopyStreams[i]),
  //         individualOutputs[i], outputViews[i]);
  //     outputCopyTokens.push_back(copyToken.getAsyncToken());
  //   }
    
  //   builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
  //   LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
  //   // ========== 阶段8：结果替换和清理 ==========
  //   LLVM_DEBUG(llvm::dbgs() << "Stage 8: Result replacement and cleanup\n");
  //   // 转换回tensor并替换原始操作
  //   SmallVector<Value> splitResults;
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
  //     splitResults.push_back(splitResult);
  //   }
    
  //   // 替换所有使用并删除操作
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     Operation* op = group.operations[i];
  //     op->getResult(0).replaceAllUsesWith(splitResults[i]);
  //     LLVM_DEBUG(llvm::dbgs() << "Replaced batch matmul operation " << i << " result\n");
  //   }
    
  //   for (size_t i = 0; i < group.operations.size(); ++i) {
  //     Operation* op = group.operations[i];
  //     LLVM_DEBUG(llvm::dbgs() << "Erasing batch matmul operation " << i << "\n");
  //     op->erase();
  //   }
    
  //   LLVM_DEBUG(llvm::dbgs() << "Successfully fused batch MatMul operations with ordered IR generation\n");
  //   return true;
  // }

  // // 执行2D到3D的广播
  // Value performBroadcast2Dto3D(OpBuilder& builder, Value input2D, int64_t batchDim, Type elementType) {
  //   auto input2DType = input2D.getType().cast<RankedTensorType>();
  //   ArrayRef<int64_t> input2DShape = input2DType.getShape();
    
  //   // 创建广播后的3D形状 [batchDim, dim1, dim2]
  //   SmallVector<int64_t> broadcast3DShape = {batchDim, input2DShape[0], input2DShape[1]};
  //   auto broadcast3DType = RankedTensorType::get(broadcast3DShape, elementType);
    
  //   LLVM_DEBUG(llvm::dbgs() << "Broadcasting from [" << input2DShape[0] << ", " << input2DShape[1] 
  //             << "] to [" << batchDim << ", " << input2DShape[0] << ", " << input2DShape[1] << "]\n");
    
  //   // 使用ONNX的Unsqueeze + Expand来实现广播
  //   // 1. 首先使用Unsqueeze在第0维添加维度
  //   auto unsqueeze3DType = RankedTensorType::get({1, input2DShape[0], input2DShape[1]}, elementType);
    
  //   // 创建axes常量 [0] - 按照文档要求，axes是tensor of 64-bit signless integer values
  //   auto axesType = RankedTensorType::get({1}, builder.getI64Type());
  //   SmallVector<int64_t> axesData = {0};
  //   auto axesAttr = DenseElementsAttr::get(axesType, ArrayRef<int64_t>(axesData));
  //   Value axesConstant = builder.create<mlir::ONNXConstantOp>(
  //       builder.getUnknownLoc(), 
  //       mlir::Attribute(),  // sparse_value (为空)
  //       axesAttr);          // value attribute - 会自动从这个推断类型
    
  //   // 根据文档：ONNXUnsqueezeOp takes operands (data, axes)
  //   Value unsqueezed = builder.create<mlir::ONNXUnsqueezeOp>(
  //       builder.getUnknownLoc(), 
  //       unsqueeze3DType,    // result type
  //       input2D,            // data operand
  //       axesConstant);      // axes operand
    
  //   // 2. 创建目标形状常量并使用Expand进行广播
  //   // 按照文档要求，shape是tensor of 64-bit signless integer values
  //   auto shapeType = RankedTensorType::get({3}, builder.getI64Type());
  //   SmallVector<int64_t> shapeData = {batchDim, input2DShape[0], input2DShape[1]};
  //   auto shapeAttr = DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(shapeData));
  //   Value shapeConstant = builder.create<mlir::ONNXConstantOp>(
  //       builder.getUnknownLoc(), 
  //       mlir::Attribute(),  // sparse_value (为空)
  //       shapeAttr);         // value attribute - 会自动从这个推断类型
    
  //   // 根据文档：ONNXExpandOp takes operands (input, shape)
  //   Value expanded = builder.create<mlir::ONNXExpandOp>(
  //       builder.getUnknownLoc(), 
  //       broadcast3DType,    // result type
  //       unsqueezed,         // input operand
  //       shapeConstant);     // shape operand
    
  //   LLVM_DEBUG(llvm::dbgs() << "Created broadcast operation using Unsqueeze + Expand\n");
    
  //   return expanded;
  // }

  // // 处理标准matmul的融合 - 优化版本，避免重复升维和操作穿插
  // bool fuseStandardMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
  //     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Standard 2D MatMul operations with optimized broadcast\n");
      
  //     Operation* firstOp = group.operations[0];
  //     auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
  //     if (!firstMatMul) {
  //       return false;
  //     }
      
  //     auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
  //     auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
  //     auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
      
  //     ArrayRef<int64_t> aShape = aType.getShape();
  //     ArrayRef<int64_t> bShape = bType.getShape();
  //     ArrayRef<int64_t> outputShape = outputType.getShape();
      
  //     LLVM_DEBUG(llvm::dbgs() << "Original 2D shapes: A=[" << aShape[0] << "," << aShape[1] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "B=[" << bShape[0] << "," << bShape[1] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "Output=[" << outputShape[0] << "," << outputShape[1] << "]\n");
      
  //     // ========== 阶段1：分析B矩阵共享情况并预先生成广播操作 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 1: Analyzing B matrix sharing and generating broadcast operations\n");
      
  //     // 分析B矩阵共享情况
  //     SmallVector<Value> uniqueBMatrices;
  //     SmallVector<size_t> bMatrixIndices; // 每个操作对应的B矩阵索引
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
  //       Value currentB = currentMatMul.getB();
        
  //       // 查找是否已经存在相同的B矩阵
  //       size_t bIndex = uniqueBMatrices.size();
  //       for (size_t j = 0; j < uniqueBMatrices.size(); ++j) {
  //         if (uniqueBMatrices[j] == currentB) {
  //           bIndex = j;
  //           break;
  //         }
  //       }
        
  //       if (bIndex == uniqueBMatrices.size()) {
  //         // 新的B矩阵
  //         uniqueBMatrices.push_back(currentB);
  //       }
  //       bMatrixIndices.push_back(bIndex);
  //     }
      
  //     LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueBMatrices.size() << " unique B matrices among " << group.operations.size() << " operations\n");
      
  //     // 广播后的3D形状（batch维度为1）
  //     SmallVector<int64_t> broadcast3DAShape = {1, aShape[0], aShape[1]};
  //     SmallVector<int64_t> broadcast3DBShape = {1, bShape[0], bShape[1]};
  //     SmallVector<int64_t> broadcast3DOutputShape = {1, outputShape[0], outputShape[1]};
      
  //     // 广播所有A矩阵
  //     SmallVector<Value> broadcastedAValues;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Operation* currentOp = group.operations[i];
  //       Value originalA = currentOp->getOperand(0);
        
  //       Value broadcastedA = performBroadcast2Dto3D(builder, originalA, 1, aType.getElementType());
  //       broadcastedAValues.push_back(broadcastedA);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Broadcasted A matrix " << i << " from 2D to 3D\n");
  //     }
      
  //     // 只广播唯一的B矩阵
  //     SmallVector<Value> uniqueBroadcastedBValues;
  //     for (size_t i = 0; i < uniqueBMatrices.size(); ++i) {
  //       Value broadcastedB = performBroadcast2Dto3D(builder, uniqueBMatrices[i], 1, bType.getElementType());
  //       uniqueBroadcastedBValues.push_back(broadcastedB);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Broadcasted unique B matrix " << i << " from 2D to 3D\n");
  //     }
      
  //     // ========== 阶段2：计算融合后的形状并分配GPU内存 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 2: Computing fused shapes and allocating GPU memory\n");
      
  //     int64_t fusedBatchSize = group.operations.size(); // 每个操作贡献batch=1
  //     SmallVector<int64_t> fusedAShape = {fusedBatchSize, aShape[0], aShape[1]};
  //     SmallVector<int64_t> fusedBShape = {fusedBatchSize, bShape[0], bShape[1]};
  //     SmallVector<int64_t> fusedOutputShape = {fusedBatchSize, outputShape[0], outputShape[1]};
      
  //     LLVM_DEBUG(llvm::dbgs() << "Fused shapes: A=[" << fusedAShape[0] << "," << fusedAShape[1] << "," << fusedAShape[2] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "B=[" << fusedBShape[0] << "," << fusedBShape[1] << "," << fusedBShape[2] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "Output=[" << fusedOutputShape[0] << "," << fusedOutputShape[1] << "," << fusedOutputShape[2] << "]\n");
      
  //     // 创建类型
  //     auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
  //     auto fusedBType = RankedTensorType::get(fusedBShape, bType.getElementType());
  //     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
      
  //     auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
  //     auto bMemRefType = MemRefType::get(fusedBShape, bType.getElementType());
  //     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
  //     auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
  //     auto broadcast3DAMemRefType = MemRefType::get(broadcast3DAShape, aType.getElementType());
  //     auto broadcast3DBMemRefType = MemRefType::get(broadcast3DBShape, bType.getElementType());
      
  //     // 分配GPU内存
  //     Value batchedA = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), aMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     Value batchedB = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), bMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     Value batchedOutput = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), outputMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     // 为每个原始操作分配独立的2D输出内存（最终需要2D输出）
  //     SmallVector<Value> individualOutputs;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Value individualOutput = builder.create<memref::AllocOp>(
  //           builder.getUnknownLoc(), originalOutputMemRefType, 
  //           ValueRange{}, builder.getI64IntegerAttr(16));
  //       individualOutputs.push_back(individualOutput);
  //     }
      
  //     LLVM_DEBUG(llvm::dbgs() << "Allocated all GPU memory\n");
      
  //     // ========== 阶段3：创建所有subviews ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 3: Creating all subviews\n");
  //     SmallVector<Value> aViews;
  //     SmallVector<Value> bViews;
  //     SmallVector<Value> outputViews;
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       int64_t offset = i; // 每个操作在batch维度的偏移量为i（因为每个贡献batch=1）
        
  //       // 创建A矩阵子视图
  //       SmallVector<OpFoldResult> aOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> aSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(aShape[0]),
  //           builder.getI64IntegerAttr(aShape[1])
  //       };
  //       SmallVector<OpFoldResult> aStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value aView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
  //       aViews.push_back(aView);
        
  //       // 创建B矩阵子视图
  //       SmallVector<OpFoldResult> bOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> bSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(bShape[0]),
  //           builder.getI64IntegerAttr(bShape[1])
  //       };
  //       SmallVector<OpFoldResult> bStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value bView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedB, bOffsets, bSizes, bStrides);
  //       bViews.push_back(bView);
        
  //       // 创建输出子视图（3D，用于后续复制和降维）
  //       SmallVector<OpFoldResult> outputOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> outputSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(outputShape[0]),
  //           builder.getI64IntegerAttr(outputShape[1])
  //       };
  //       SmallVector<OpFoldResult> outputStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value outputView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
  //       outputViews.push_back(outputView);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
  //     }
      
  //     // ========== 阶段4：预先生成所有类型转换 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 4: Generating all type conversions\n");
  //     SmallVector<Value> aMemRefs;
  //     SmallVector<Value> bMemRefs;
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // A矩阵类型转换（3D广播后的）
  //       Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), broadcast3DAMemRefType, broadcastedAValues[i]).getResult(0);
  //       aMemRefs.push_back(aMemRef);
        
  //       // B矩阵类型转换（根据索引获取对应的唯一B矩阵）
  //       size_t bIndex = bMatrixIndices[i];
  //       Value bMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), broadcast3DBMemRefType, uniqueBroadcastedBValues[bIndex]).getResult(0);
  //       bMemRefs.push_back(bMemRef);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Generated type conversions for operation " << i << " (B matrix index: " << bIndex << ")\n");
  //     }
      
  //     // ========== 阶段5：预先生成所有输出降维操作 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 5: Pre-generating all output dimension reduction operations using memref.reinterpret_cast\n");
      
  //     SmallVector<Value> reinterpretedOutputMemRefs;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // 直接使用memref.reinterpret_cast将3D subview重新解释为2D memref
  //       // outputViews[i]是形状为[1, M, N]的3D subview，我们要重新解释为[M, N]的2D memref
        
  //       // 创建静态参数数组
  //       SmallVector<int64_t> staticOffsets = {0};
  //       SmallVector<int64_t> staticSizes = {outputShape[0], outputShape[1]};
  //       SmallVector<int64_t> staticStrides = {outputShape[1], 1};
        
  //       Value reinterpretedOutput = builder.create<memref::ReinterpretCastOp>(
  //           builder.getUnknownLoc(),
  //           originalOutputMemRefType,
  //           outputViews[i],
  //           ValueRange{}, // dynamic offsets (empty)
  //           ValueRange{}, // dynamic sizes (empty)
  //           ValueRange{}, // dynamic strides (empty)
  //           staticOffsets,   // static offsets
  //           staticSizes,     // static sizes
  //           staticStrides);  // static strides
        
  //       reinterpretedOutputMemRefs.push_back(reinterpretedOutput);
  //       LLVM_DEBUG(llvm::dbgs() << "Pre-generated reinterpret_cast operation for output " << i << " from 3D to 2D\n");
  //     }
      
  //     // ========== 阶段6：创建异步流并执行输入复制 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 6: Creating async streams and performing input copies\n");
  //     SmallVector<Value> copyStreams;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //       copyStreams.push_back(copyStream.getAsyncToken());
  //     }
      
  //     SmallVector<Value> allCopyTokens;
      
  //     // 执行A矩阵复制
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(copyStreams[i]),
  //           aViews[i], aMemRefs[i]);
  //       allCopyTokens.push_back(copyToken.getAsyncToken());
        
  //       LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << "\n");
  //     }
      
  //     // 执行B矩阵复制
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(copyStreams[i]),
  //           bViews[i], bMemRefs[i]);
  //       allCopyTokens.push_back(copyToken.getAsyncToken());
        
  //       LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << "\n");
  //     }
      
  //     // 等待所有输入复制完成
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
  //     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
      
  //     // ========== 阶段7：执行融合计算 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 7: Performing fused 3D computation\n");
      
  //     // 转换为tensor用于ONNX操作
  //     Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
      
  //     Value batchedBTensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), fusedBType, batchedB).getResult(0);
      
  //     // 创建融合的3D MatMul操作
  //     Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
  //         builder.getUnknownLoc(),
  //         fusedOutputType,
  //         batchedATensor,
  //         batchedBTensor);
      
  //     LLVM_DEBUG(llvm::dbgs() << "Created fused 3D MatMul operation: (" 
  //               << fusedAShape[0] << "x" << fusedAShape[1] << "x" << fusedAShape[2] << ") × ("
  //               << fusedBShape[0] << "x" << fusedBShape[1] << "x" << fusedBShape[2] << ") → ("
  //               << fusedOutputShape[0] << "x" << fusedOutputShape[1] << "x" << fusedOutputShape[2] << ")\n");
      
  //     // 转换融合结果为memref
  //     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
      
  //     // 将融合结果复制到预分配的batched output memory
  //     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
  //         builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()),
  //         ValueRange(fusedCopyStream.getAsyncToken()),
  //         batchedOutput, fusedResultMemRef);
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
      
  //     LLVM_DEBUG(llvm::dbgs() << "Copied fused 3D MatMul result to batched output memory\n");
      
  //     // ========== 阶段8：执行输出复制 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 8: Performing output copies\n");
  //     SmallVector<Value> outputCopyStreams;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //       outputCopyStreams.push_back(copyStream.getAsyncToken());
  //     }
      
  //     SmallVector<Value> outputCopyTokens;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // 直接使用预先生成的reinterpret_cast输出memref
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(outputCopyStreams[i]),
  //           individualOutputs[i], reinterpretedOutputMemRefs[i]);
  //       outputCopyTokens.push_back(copyToken.getAsyncToken());
  //     }
      
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
  //     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
      
  //     // ========== 阶段9：结果替换和清理 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 9: Result replacement and cleanup\n");
  //     // 转换回tensor并替换原始操作
  //     SmallVector<Value> splitResults;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
  //       splitResults.push_back(splitResult);
  //     }
      
  //     // 替换所有使用并删除操作
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Operation* op = group.operations[i];
  //       op->getResult(0).replaceAllUsesWith(splitResults[i]);
  //       LLVM_DEBUG(llvm::dbgs() << "Replaced standard matmul operation " << i << " result\n");
  //     }
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Operation* op = group.operations[i];
  //       LLVM_DEBUG(llvm::dbgs() << "Erasing standard matmul operation " << i << "\n");
  //       op->erase();
  //     }
      
  //     LLVM_DEBUG(llvm::dbgs() << "Successfully fused standard 2D MatMul operations with optimized broadcast (avoided " 
  //               << (group.operations.size() - uniqueBMatrices.size()) << " redundant B matrix broadcasts)\n");
  //     return true;
  // }
  

  //在调整顺序的基础上，避免重复复制
// 融合MatMul操作 - 增强版本，支持batch matmul广播
  bool fuseMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MatMul operations with parallel copying and batch matmul support\n");
    
    Operation* firstOp = group.operations[0];
    auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
    if (!firstMatMul) {
      return false;
    }
    
    auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
    auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
    auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
    
    ArrayRef<int64_t> aShape = aType.getShape();
    ArrayRef<int64_t> bShape = bType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    
    LLVM_DEBUG(llvm::dbgs() << "MatMul shapes: A=" << aShape.size() << "D, B=" << bShape.size() << "D, Output=" << outputShape.size() << "D\n");
    LLVM_DEBUG(llvm::dbgs() << "A shape: [");
    for (size_t i = 0; i < aShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << aShape[i]);
      if (i < aShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "B shape: [");
    for (size_t i = 0; i < bShape.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << bShape[i]);
      if (i < bShape.size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
    
    // 检测是否为batch matmul情况
    bool isBatchMatMul = (aShape.size() == 2 && bShape.size() == 3) || (aShape.size() == 3 && bShape.size() == 2);
    
    if (isBatchMatMul) {
      LLVM_DEBUG(llvm::dbgs() << "Detected batch matmul case, using specialized fusion\n");
      return fuseBatchMatMulOperations(builder, group);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Standard matmul case, using regular fusion\n");
      return fuseStandardMatMulOperations(builder, group);
    }
  }

  // 专门处理batch matmul的融合函数 - 增加权重去重优化
  bool fuseBatchMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Batch MatMul operations with weight deduplication\n");
    
    Operation* firstOp = group.operations[0];
    auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
    if (!firstMatMul) {
      return false;
    }
    
    auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
    auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
    auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
    
    ArrayRef<int64_t> aShape = aType.getShape();
    ArrayRef<int64_t> bShape = bType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    
    LLVM_DEBUG(llvm::dbgs() << "Original shapes: A=[" << aShape[0] << "," << aShape[1] << "], ");
    LLVM_DEBUG(llvm::dbgs() << "B=[" << bShape[0] << "," << bShape[1] << "," << bShape[2] << "], ");
    LLVM_DEBUG(llvm::dbgs() << "Output=[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "]\n");
    
    // 确定广播情况和参数
    bool needsBroadcastA = (aShape.size() == 2 && bShape.size() == 3);
    bool needsBroadcastB = (aShape.size() == 3 && bShape.size() == 2);
    
    if (!needsBroadcastA && !needsBroadcastB) {
      LLVM_DEBUG(llvm::dbgs() << "Error: Expected batch matmul case but shapes don't match\n");
      return false;
    }
    
    // 计算广播后的形状
    SmallVector<int64_t> broadcastedAShape;
    SmallVector<int64_t> broadcastedBShape; 
    int64_t batchDim = 0;
    
    if (needsBroadcastA) {
      // A: 2D -> 3D, B已经是3D
      batchDim = bShape[0];
      broadcastedAShape = {batchDim, aShape[0], aShape[1]};
      broadcastedBShape = {bShape[0], bShape[1], bShape[2]};
    } else {
      // B: 2D -> 3D, A已经是3D  
      batchDim = aShape[0];
      broadcastedAShape = {aShape[0], aShape[1], aShape[2]};
      broadcastedBShape = {batchDim, bShape[0], bShape[1]};
    }
    
    LLVM_DEBUG(llvm::dbgs() << "After broadcasting: A=[" << broadcastedAShape[0] << "," << broadcastedAShape[1] << "," << broadcastedAShape[2] << "], ");
    LLVM_DEBUG(llvm::dbgs() << "B=[" << broadcastedBShape[0] << "," << broadcastedBShape[1] << "," << broadcastedBShape[2] << "]\n");
    
    // ========== 阶段1：分析A和B矩阵共享情况并预先生成广播操作 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 1: Analyzing A and B matrix sharing and generating broadcast operations\n");
    
    // 分析A矩阵共享情况（权重去重）
    SmallVector<Value> uniqueAMatrices;
    SmallVector<size_t> aMatrixIndices; // 每个操作对应的A矩阵索引
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
      Value currentA = currentMatMul.getA();
      
      // 查找是否已经存在相同的A矩阵
      size_t aIndex = uniqueAMatrices.size();
      for (size_t j = 0; j < uniqueAMatrices.size(); ++j) {
        if (uniqueAMatrices[j] == currentA) {
          aIndex = j;
          break;
        }
      }
      
      if (aIndex == uniqueAMatrices.size()) {
        // 新的A矩阵
        uniqueAMatrices.push_back(currentA);
      }
      aMatrixIndices.push_back(aIndex);
    }
    
    // 分析B矩阵共享情况
    SmallVector<Value> uniqueBMatrices;
    SmallVector<size_t> bMatrixIndices; // 每个操作对应的B矩阵索引
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
      Value currentB = currentMatMul.getB();
      
      // 查找是否已经存在相同的B矩阵
      size_t bIndex = uniqueBMatrices.size();
      for (size_t j = 0; j < uniqueBMatrices.size(); ++j) {
        if (uniqueBMatrices[j] == currentB) {
          bIndex = j;
          break;
        }
      }
      
      if (bIndex == uniqueBMatrices.size()) {
        // 新的B矩阵
        uniqueBMatrices.push_back(currentB);
      }
      bMatrixIndices.push_back(bIndex);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueAMatrices.size() << " unique A matrices among " << group.operations.size() << " operations\n");
    LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueBMatrices.size() << " unique B matrices among " << group.operations.size() << " operations\n");
    
    // 只广播唯一的A矩阵（需要广播的情况下）
    SmallVector<Value> uniqueBroadcastedAValues;
    if (needsBroadcastA) {
      for (size_t i = 0; i < uniqueAMatrices.size(); ++i) {
        Value broadcastedA = performBroadcast2Dto3D(builder, uniqueAMatrices[i], batchDim, aType.getElementType());
        uniqueBroadcastedAValues.push_back(broadcastedA);
        LLVM_DEBUG(llvm::dbgs() << "Broadcasted unique A matrix " << i << " from 2D to 3D\n");
      }
    } else {
      // A已经是3D，直接使用
      uniqueBroadcastedAValues = uniqueAMatrices;
    }
    
    // 只广播唯一的B矩阵（需要广播的情况下）
    SmallVector<Value> uniqueBroadcastedBValues;
    if (needsBroadcastB) {
      for (size_t i = 0; i < uniqueBMatrices.size(); ++i) {
        Value broadcastedB = performBroadcast2Dto3D(builder, uniqueBMatrices[i], batchDim, bType.getElementType());
        uniqueBroadcastedBValues.push_back(broadcastedB);
        LLVM_DEBUG(llvm::dbgs() << "Broadcasted unique B matrix " << i << " from 2D to 3D\n");
      }
    } else {
      // B已经是3D，直接使用
      uniqueBroadcastedBValues = uniqueBMatrices;
    }
    
    // 计算融合后的batch大小
    int64_t fusedBatchSize = batchDim * group.operations.size();
    
    // 创建融合后的形状
    SmallVector<int64_t> fusedAShape = {fusedBatchSize, broadcastedAShape[1], broadcastedAShape[2]};
    SmallVector<int64_t> fusedBShape = {fusedBatchSize, broadcastedBShape[1], broadcastedBShape[2]};
    SmallVector<int64_t> fusedOutputShape = {fusedBatchSize, outputShape[1], outputShape[2]};
    
    LLVM_DEBUG(llvm::dbgs() << "Fused shapes: A=[" << fusedAShape[0] << "," << fusedAShape[1] << "," << fusedAShape[2] << "], ");
    LLVM_DEBUG(llvm::dbgs() << "B=[" << fusedBShape[0] << "," << fusedBShape[1] << "," << fusedBShape[2] << "], ");
    LLVM_DEBUG(llvm::dbgs() << "Output=[" << fusedOutputShape[0] << "," << fusedOutputShape[1] << "," << fusedOutputShape[2] << "]\n");
    
    // 创建类型
    auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
    auto fusedBType = RankedTensorType::get(fusedBShape, bType.getElementType());
    auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
    auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
    auto bMemRefType = MemRefType::get(fusedBShape, bType.getElementType());
    auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
    auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    auto broadcastedAMemRefType = MemRefType::get(broadcastedAShape, aType.getElementType());
    auto broadcastedBMemRefType = MemRefType::get(broadcastedBShape, bType.getElementType());
    
    // ========== 阶段2：分配所有GPU内存 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 2: Allocating all GPU memory\n");
    Value batchedA = builder.create<memref::AllocOp>(
        builder.getUnknownLoc(), aMemRefType, 
        ValueRange{}, builder.getI64IntegerAttr(16));
    
    Value batchedB = builder.create<memref::AllocOp>(
        builder.getUnknownLoc(), bMemRefType, 
        ValueRange{}, builder.getI64IntegerAttr(16));
    
    Value batchedOutput = builder.create<memref::AllocOp>(
        builder.getUnknownLoc(), outputMemRefType, 
        ValueRange{}, builder.getI64IntegerAttr(16));
    
    // 为每个原始操作分配独立的输出内存
    SmallVector<Value> individualOutputs;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Value individualOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), originalOutputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      individualOutputs.push_back(individualOutput);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Allocated all GPU memory\n");
    
    // ========== 阶段3：生成所有的subviews ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 3: Creating all subviews\n");
    SmallVector<Value> aViews;
    SmallVector<Value> bViews;
    SmallVector<Value> outputViews;
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      int64_t offset = i * batchDim;
      
      // 创建A矩阵子视图
      SmallVector<OpFoldResult> aOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> aSizes = {
          builder.getI64IntegerAttr(batchDim),
          builder.getI64IntegerAttr(broadcastedAShape[1]),
          builder.getI64IntegerAttr(broadcastedAShape[2])
      };
      SmallVector<OpFoldResult> aStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value aView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
      aViews.push_back(aView);
      
      // 创建B矩阵子视图
      SmallVector<OpFoldResult> bOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> bSizes = {
          builder.getI64IntegerAttr(batchDim),
          builder.getI64IntegerAttr(broadcastedBShape[1]),
          builder.getI64IntegerAttr(broadcastedBShape[2])
      };
      SmallVector<OpFoldResult> bStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value bView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedB, bOffsets, bSizes, bStrides);
      bViews.push_back(bView);
      
      // 创建输出子视图（用于后续输出复制）
      SmallVector<OpFoldResult> outputOffsets = {
          builder.getI64IntegerAttr(offset),
          builder.getI64IntegerAttr(0),
          builder.getI64IntegerAttr(0)
      };
      SmallVector<OpFoldResult> outputSizes = {
          builder.getI64IntegerAttr(batchDim),
          builder.getI64IntegerAttr(outputShape[1]),
          builder.getI64IntegerAttr(outputShape[2])
      };
      SmallVector<OpFoldResult> outputStrides = {
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(1)
      };
      
      Value outputView = builder.create<memref::SubViewOp>(
          builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
      outputViews.push_back(outputView);
      
      LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
    }
    
    // ========== 阶段4：预先生成所有类型转换 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 4: Generating all type conversions\n");
    SmallVector<Value> aMemRefs;
    SmallVector<Value> bMemRefs;
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      // A矩阵类型转换（根据索引获取对应的唯一A矩阵）
      size_t aIndex = aMatrixIndices[i];
      Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), broadcastedAMemRefType, uniqueBroadcastedAValues[aIndex]).getResult(0);
      aMemRefs.push_back(aMemRef);
      
      // B矩阵类型转换（根据索引获取对应的唯一B矩阵）
      size_t bIndex = bMatrixIndices[i];
      Value bMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), broadcastedBMemRefType, uniqueBroadcastedBValues[bIndex]).getResult(0);
      bMemRefs.push_back(bMemRef);
      
      LLVM_DEBUG(llvm::dbgs() << "Generated type conversions for operation " << i << " (A matrix index: " << aIndex << ", B matrix index: " << bIndex << ")\n");
    }
    
    // // ========== 阶段5：创建所有异步流并执行复制 ==========
    // LLVM_DEBUG(llvm::dbgs() << "Stage 5: Creating async streams and performing copies\n");
    // SmallVector<Value> copyStreams;
    // for (size_t i = 0; i < group.operations.size(); ++i) {
    //   auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
    //       gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
    //   copyStreams.push_back(copyStream.getAsyncToken());
    // }
    
    // SmallVector<Value> allCopyTokens;
    
    // // 执行A矩阵复制
    // for (size_t i = 0; i < group.operations.size(); ++i) {
    //   auto copyToken = builder.create<gpu::MemcpyOp>(
    //       builder.getUnknownLoc(), 
    //       gpu::AsyncTokenType::get(builder.getContext()),
    //       ValueRange(copyStreams[i]),
    //       aViews[i], aMemRefs[i]);
    //   allCopyTokens.push_back(copyToken.getAsyncToken());
      
    //   LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << "\n");
    // }
    
    // // 执行B矩阵复制
    // for (size_t i = 0; i < group.operations.size(); ++i) {
    //   auto copyToken = builder.create<gpu::MemcpyOp>(
    //       builder.getUnknownLoc(), 
    //       gpu::AsyncTokenType::get(builder.getContext()),
    //       ValueRange(copyStreams[i]),
    //       bViews[i], bMemRefs[i]);
    //   allCopyTokens.push_back(copyToken.getAsyncToken());
      
    //   LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << "\n");
    // }
    
    // // 等待所有输入复制完成
    // builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
    // LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
    
    // 修复复制stream声明问题
    // ========== 阶段5：创建所有异步流并执行复制 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 5: Creating async streams and performing copies\n");

    // 为每个复制操作创建独立的stream（A矩阵复制 + B矩阵复制）
    SmallVector<Value> aCopyStreams;
    SmallVector<Value> bCopyStreams;

    for (size_t i = 0; i < group.operations.size(); ++i) {
      // 为A矩阵复制创建独立的stream
      auto aCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      aCopyStreams.push_back(aCopyStream.getAsyncToken());
      
      // 为B矩阵复制创建独立的stream
      auto bCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      bCopyStreams.push_back(bCopyStream.getAsyncToken());
    }

    SmallVector<Value> allCopyTokens;

    // 执行A矩阵复制 - 每个使用独立的stream
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(aCopyStreams[i]),  // 使用A矩阵专用的stream
          aViews[i], aMemRefs[i]);
      allCopyTokens.push_back(copyToken.getAsyncToken());
      
      LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << " on dedicated stream\n");
    }

    // 执行B矩阵复制 - 每个使用独立的stream
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(bCopyStreams[i]),  // 使用B矩阵专用的stream
          bViews[i], bMemRefs[i]);
      allCopyTokens.push_back(copyToken.getAsyncToken());
      
      LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << " on dedicated stream\n");
    }

    // 等待所有输入复制完成
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed with " << (group.operations.size() * 2) << " independent streams\n");


    // ========== 阶段6：执行融合计算 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 6: Performing fused computation\n");
    // 转换为tensor用于ONNX操作
    Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
    Value batchedBTensor = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), fusedBType, batchedB).getResult(0);
    
    // 创建融合的3D MatMul操作
    Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
        builder.getUnknownLoc(),
        fusedOutputType,
        batchedATensor,
        batchedBTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Created fused 3D MatMul operation: (" 
              << fusedAShape[0] << "x" << fusedAShape[1] << "x" << fusedAShape[2] << ") × ("
              << fusedBShape[0] << "x" << fusedBShape[1] << "x" << fusedBShape[2] << ") → ("
              << fusedOutputShape[0] << "x" << fusedOutputShape[1] << "x" << fusedOutputShape[2] << ")\n");
    
    // 转换融合结果为memref
    Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
    // 将融合结果复制到预分配的batched output memory
    auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
    auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
        builder.getUnknownLoc(), 
        gpu::AsyncTokenType::get(builder.getContext()),
        ValueRange(fusedCopyStream.getAsyncToken()),
        batchedOutput, fusedResultMemRef);
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
    LLVM_DEBUG(llvm::dbgs() << "Copied fused batch MatMul result to batched output memory\n");
    
    // ========== 阶段7：输出复制 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 7: Performing output copies\n");
    SmallVector<Value> outputCopyStreams;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      outputCopyStreams.push_back(copyStream.getAsyncToken());
    }
    
    SmallVector<Value> outputCopyTokens;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      auto copyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(outputCopyStreams[i]),
          individualOutputs[i], outputViews[i]);
      outputCopyTokens.push_back(copyToken.getAsyncToken());
    }
    
    builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
    LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
    // ========== 阶段8：结果替换和清理 ==========
    LLVM_DEBUG(llvm::dbgs() << "Stage 8: Result replacement and cleanup\n");
    // 转换回tensor并替换原始操作
    SmallVector<Value> splitResults;
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
      splitResults.push_back(splitResult);
    }
    
    // 替换所有使用并删除操作
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      op->getResult(0).replaceAllUsesWith(splitResults[i]);
      LLVM_DEBUG(llvm::dbgs() << "Replaced batch matmul operation " << i << " result\n");
    }
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      Operation* op = group.operations[i];
      LLVM_DEBUG(llvm::dbgs() << "Erasing batch matmul operation " << i << "\n");
      op->erase();
    }
    
    size_t aBroadcastsSaved = group.operations.size() - uniqueAMatrices.size();
    size_t bBroadcastsSaved = group.operations.size() - uniqueBMatrices.size();
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused batch MatMul operations with weight deduplication\n");
    LLVM_DEBUG(llvm::dbgs() << "Optimization: avoided " << aBroadcastsSaved << " redundant A matrix broadcasts\n");
    LLVM_DEBUG(llvm::dbgs() << "Optimization: avoided " << bBroadcastsSaved << " redundant B matrix broadcasts\n");
    return true;
  }

  // 执行2D到3D的广播
  Value performBroadcast2Dto3D(OpBuilder& builder, Value input2D, int64_t batchDim, Type elementType) {
    auto input2DType = input2D.getType().cast<RankedTensorType>();
    ArrayRef<int64_t> input2DShape = input2DType.getShape();
    
    // 创建广播后的3D形状 [batchDim, dim1, dim2]
    SmallVector<int64_t> broadcast3DShape = {batchDim, input2DShape[0], input2DShape[1]};
    auto broadcast3DType = RankedTensorType::get(broadcast3DShape, elementType);
    
    LLVM_DEBUG(llvm::dbgs() << "Broadcasting from [" << input2DShape[0] << ", " << input2DShape[1] 
              << "] to [" << batchDim << ", " << input2DShape[0] << ", " << input2DShape[1] << "]\n");
    
    // 使用ONNX的Unsqueeze + Expand来实现广播
    // 1. 首先使用Unsqueeze在第0维添加维度
    auto unsqueeze3DType = RankedTensorType::get({1, input2DShape[0], input2DShape[1]}, elementType);
    
    // 创建axes常量 [0] - 按照文档要求，axes是tensor of 64-bit signless integer values
    auto axesType = RankedTensorType::get({1}, builder.getI64Type());
    SmallVector<int64_t> axesData = {0};
    auto axesAttr = DenseElementsAttr::get(axesType, ArrayRef<int64_t>(axesData));
    Value axesConstant = builder.create<mlir::ONNXConstantOp>(
        builder.getUnknownLoc(), 
        mlir::Attribute(),  // sparse_value (为空)
        axesAttr);          // value attribute - 会自动从这个推断类型
    
    // 根据文档：ONNXUnsqueezeOp takes operands (data, axes)
    Value unsqueezed = builder.create<mlir::ONNXUnsqueezeOp>(
        builder.getUnknownLoc(), 
        unsqueeze3DType,    // result type
        input2D,            // data operand
        axesConstant);      // axes operand
    
    // 2. 创建目标形状常量并使用Expand进行广播
    // 按照文档要求，shape是tensor of 64-bit signless integer values
    auto shapeType = RankedTensorType::get({3}, builder.getI64Type());
    SmallVector<int64_t> shapeData = {batchDim, input2DShape[0], input2DShape[1]};
    auto shapeAttr = DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(shapeData));
    Value shapeConstant = builder.create<mlir::ONNXConstantOp>(
        builder.getUnknownLoc(), 
        mlir::Attribute(),  // sparse_value (为空)
        shapeAttr);         // value attribute - 会自动从这个推断类型
    
    // 根据文档：ONNXExpandOp takes operands (input, shape)
    Value expanded = builder.create<mlir::ONNXExpandOp>(
        builder.getUnknownLoc(), 
        broadcast3DType,    // result type
        unsqueezed,         // input operand
        shapeConstant);     // shape operand
    
    LLVM_DEBUG(llvm::dbgs() << "Created broadcast operation using Unsqueeze + Expand\n");
    
    return expanded;
  }

  // // 处理标准matmul的融合 - 优化版本，对A和B矩阵都进行去重
  // bool fuseStandardMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
  //     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Standard 2D MatMul operations with A/B matrix deduplication\n");
      
  //     Operation* firstOp = group.operations[0];
  //     auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
  //     if (!firstMatMul) {
  //       return false;
  //     }
      
  //     auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
  //     auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
  //     auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
      
  //     ArrayRef<int64_t> aShape = aType.getShape();
  //     ArrayRef<int64_t> bShape = bType.getShape();
  //     ArrayRef<int64_t> outputShape = outputType.getShape();
      
  //     LLVM_DEBUG(llvm::dbgs() << "Original 2D shapes: A=[" << aShape[0] << "," << aShape[1] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "B=[" << bShape[0] << "," << bShape[1] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "Output=[" << outputShape[0] << "," << outputShape[1] << "]\n");
      
  //     // ========== 阶段1：分析A和B矩阵共享情况并预先生成广播操作 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 1: Analyzing A and B matrix sharing and generating broadcast operations\n");
      
  //     // 分析A矩阵共享情况（权重去重）
  //     SmallVector<Value> uniqueAMatrices;
  //     SmallVector<size_t> aMatrixIndices; // 每个操作对应的A矩阵索引
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
  //       Value currentA = currentMatMul.getA();
        
  //       // 查找是否已经存在相同的A矩阵
  //       size_t aIndex = uniqueAMatrices.size();
  //       for (size_t j = 0; j < uniqueAMatrices.size(); ++j) {
  //         if (uniqueAMatrices[j] == currentA) {
  //           aIndex = j;
  //           break;
  //         }
  //       }
        
  //       if (aIndex == uniqueAMatrices.size()) {
  //         // 新的A矩阵
  //         uniqueAMatrices.push_back(currentA);
  //       }
  //       aMatrixIndices.push_back(aIndex);
  //     }
      
  //     // 分析B矩阵共享情况
  //     SmallVector<Value> uniqueBMatrices;
  //     SmallVector<size_t> bMatrixIndices; // 每个操作对应的B矩阵索引
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
  //       Value currentB = currentMatMul.getB();
        
  //       // 查找是否已经存在相同的B矩阵
  //       size_t bIndex = uniqueBMatrices.size();
  //       for (size_t j = 0; j < uniqueBMatrices.size(); ++j) {
  //         if (uniqueBMatrices[j] == currentB) {
  //           bIndex = j;
  //           break;
  //         }
  //       }
        
  //       if (bIndex == uniqueBMatrices.size()) {
  //         // 新的B矩阵
  //         uniqueBMatrices.push_back(currentB);
  //       }
  //       bMatrixIndices.push_back(bIndex);
  //     }
      
  //     LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueAMatrices.size() << " unique A matrices among " << group.operations.size() << " operations\n");
  //     LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueBMatrices.size() << " unique B matrices among " << group.operations.size() << " operations\n");
      
  //     // 广播后的3D形状（batch维度为1）
  //     SmallVector<int64_t> broadcast3DAShape = {1, aShape[0], aShape[1]};
  //     SmallVector<int64_t> broadcast3DBShape = {1, bShape[0], bShape[1]};
  //     SmallVector<int64_t> broadcast3DOutputShape = {1, outputShape[0], outputShape[1]};
      
  //     // 只广播唯一的A矩阵
  //     SmallVector<Value> uniqueBroadcastedAValues;
  //     for (size_t i = 0; i < uniqueAMatrices.size(); ++i) {
  //       Value broadcastedA = performBroadcast2Dto3D(builder, uniqueAMatrices[i], 1, aType.getElementType());
  //       uniqueBroadcastedAValues.push_back(broadcastedA);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Broadcasted unique A matrix " << i << " from 2D to 3D\n");
  //     }
      
  //     // 只广播唯一的B矩阵
  //     SmallVector<Value> uniqueBroadcastedBValues;
  //     for (size_t i = 0; i < uniqueBMatrices.size(); ++i) {
  //       Value broadcastedB = performBroadcast2Dto3D(builder, uniqueBMatrices[i], 1, bType.getElementType());
  //       uniqueBroadcastedBValues.push_back(broadcastedB);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Broadcasted unique B matrix " << i << " from 2D to 3D\n");
  //     }
      
  //     // ========== 阶段2：计算融合后的形状并分配GPU内存 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 2: Computing fused shapes and allocating GPU memory\n");
      
  //     int64_t fusedBatchSize = group.operations.size(); // 每个操作贡献batch=1
  //     SmallVector<int64_t> fusedAShape = {fusedBatchSize, aShape[0], aShape[1]};
  //     SmallVector<int64_t> fusedBShape = {fusedBatchSize, bShape[0], bShape[1]};
  //     SmallVector<int64_t> fusedOutputShape = {fusedBatchSize, outputShape[0], outputShape[1]};
      
  //     LLVM_DEBUG(llvm::dbgs() << "Fused shapes: A=[" << fusedAShape[0] << "," << fusedAShape[1] << "," << fusedAShape[2] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "B=[" << fusedBShape[0] << "," << fusedBShape[1] << "," << fusedBShape[2] << "], ");
  //     LLVM_DEBUG(llvm::dbgs() << "Output=[" << fusedOutputShape[0] << "," << fusedOutputShape[1] << "," << fusedOutputShape[2] << "]\n");
      
  //     // 创建类型
  //     auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
  //     auto fusedBType = RankedTensorType::get(fusedBShape, bType.getElementType());
  //     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
      
  //     auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
  //     auto bMemRefType = MemRefType::get(fusedBShape, bType.getElementType());
  //     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
  //     auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
  //     auto broadcast3DAMemRefType = MemRefType::get(broadcast3DAShape, aType.getElementType());
  //     auto broadcast3DBMemRefType = MemRefType::get(broadcast3DBShape, bType.getElementType());
      
  //     // 分配GPU内存
  //     Value batchedA = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), aMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     Value batchedB = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), bMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     Value batchedOutput = builder.create<memref::AllocOp>(
  //         builder.getUnknownLoc(), outputMemRefType, 
  //         ValueRange{}, builder.getI64IntegerAttr(16));
      
  //     // 为每个原始操作分配独立的2D输出内存（最终需要2D输出）
  //     SmallVector<Value> individualOutputs;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Value individualOutput = builder.create<memref::AllocOp>(
  //           builder.getUnknownLoc(), originalOutputMemRefType, 
  //           ValueRange{}, builder.getI64IntegerAttr(16));
  //       individualOutputs.push_back(individualOutput);
  //     }
      
  //     LLVM_DEBUG(llvm::dbgs() << "Allocated all GPU memory\n");
      
  //     // ========== 阶段3：创建所有subviews ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 3: Creating all subviews\n");
  //     SmallVector<Value> aViews;
  //     SmallVector<Value> bViews;
  //     SmallVector<Value> outputViews;
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       int64_t offset = i; // 每个操作在batch维度的偏移量为i（因为每个贡献batch=1）
        
  //       // 创建A矩阵子视图
  //       SmallVector<OpFoldResult> aOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> aSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(aShape[0]),
  //           builder.getI64IntegerAttr(aShape[1])
  //       };
  //       SmallVector<OpFoldResult> aStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value aView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
  //       aViews.push_back(aView);
        
  //       // 创建B矩阵子视图
  //       SmallVector<OpFoldResult> bOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> bSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(bShape[0]),
  //           builder.getI64IntegerAttr(bShape[1])
  //       };
  //       SmallVector<OpFoldResult> bStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value bView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedB, bOffsets, bSizes, bStrides);
  //       bViews.push_back(bView);
        
  //       // 创建输出子视图（3D，用于后续复制和降维）
  //       SmallVector<OpFoldResult> outputOffsets = {
  //           builder.getI64IntegerAttr(offset),
  //           builder.getI64IntegerAttr(0),
  //           builder.getI64IntegerAttr(0)
  //       };
  //       SmallVector<OpFoldResult> outputSizes = {
  //           builder.getI64IntegerAttr(1), // batch size = 1
  //           builder.getI64IntegerAttr(outputShape[0]),
  //           builder.getI64IntegerAttr(outputShape[1])
  //       };
  //       SmallVector<OpFoldResult> outputStrides = {
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1),
  //           builder.getI64IntegerAttr(1)
  //       };
        
  //       Value outputView = builder.create<memref::SubViewOp>(
  //           builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
  //       outputViews.push_back(outputView);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
  //     }
      
  //     // ========== 阶段4：预先生成所有类型转换 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 4: Generating all type conversions\n");
  //     SmallVector<Value> aMemRefs;
  //     SmallVector<Value> bMemRefs;
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // A矩阵类型转换（根据索引获取对应的唯一A矩阵）
  //       size_t aIndex = aMatrixIndices[i];
  //       Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), broadcast3DAMemRefType, uniqueBroadcastedAValues[aIndex]).getResult(0);
  //       aMemRefs.push_back(aMemRef);
        
  //       // B矩阵类型转换（根据索引获取对应的唯一B矩阵）
  //       size_t bIndex = bMatrixIndices[i];
  //       Value bMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), broadcast3DBMemRefType, uniqueBroadcastedBValues[bIndex]).getResult(0);
  //       bMemRefs.push_back(bMemRef);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Generated type conversions for operation " << i << " (A matrix index: " << aIndex << ", B matrix index: " << bIndex << ")\n");
  //     }
      
  //     // ========== 阶段5：预先生成所有输出降维操作 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 5: Pre-generating all output dimension reduction operations using memref.reinterpret_cast\n");
      
  //     SmallVector<Value> reinterpretedOutputMemRefs;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // 直接使用memref.reinterpret_cast将3D subview重新解释为2D memref
  //       // outputViews[i]是形状为[1, M, N]的3D subview，我们要重新解释为[M, N]的2D memref
        
  //       // 创建静态参数数组
  //       SmallVector<int64_t> staticOffsets = {0};
  //       SmallVector<int64_t> staticSizes = {outputShape[0], outputShape[1]};
  //       SmallVector<int64_t> staticStrides = {outputShape[1], 1};
        
  //       Value reinterpretedOutput = builder.create<memref::ReinterpretCastOp>(
  //           builder.getUnknownLoc(),
  //           originalOutputMemRefType,
  //           outputViews[i],
  //           ValueRange{}, // dynamic offsets (empty)
  //           ValueRange{}, // dynamic sizes (empty)
  //           ValueRange{}, // dynamic strides (empty)
  //           staticOffsets,   // static offsets
  //           staticSizes,     // static sizes
  //           staticStrides);  // static strides
        
  //       reinterpretedOutputMemRefs.push_back(reinterpretedOutput);
  //       LLVM_DEBUG(llvm::dbgs() << "Pre-generated reinterpret_cast operation for output " << i << " from 3D to 2D\n");
  //     }
      
  //     // // ========== 阶段6：创建异步流并执行输入复制 ==========
  //     // LLVM_DEBUG(llvm::dbgs() << "Stage 6: Creating async streams and performing input copies\n");
  //     // SmallVector<Value> copyStreams;
  //     // for (size_t i = 0; i < group.operations.size(); ++i) {
  //     //   auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //     //       gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //     //   copyStreams.push_back(copyStream.getAsyncToken());
  //     // }
      
  //     // SmallVector<Value> allCopyTokens;
      
  //     // // 执行A矩阵复制
  //     // for (size_t i = 0; i < group.operations.size(); ++i) {
  //     //   auto copyToken = builder.create<gpu::MemcpyOp>(
  //     //       builder.getUnknownLoc(), 
  //     //       gpu::AsyncTokenType::get(builder.getContext()),
  //     //       ValueRange(copyStreams[i]),
  //     //       aViews[i], aMemRefs[i]);
  //     //   allCopyTokens.push_back(copyToken.getAsyncToken());
        
  //     //   LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << "\n");
  //     // }
      
  //     // // 执行B矩阵复制
  //     // for (size_t i = 0; i < group.operations.size(); ++i) {
  //     //   auto copyToken = builder.create<gpu::MemcpyOp>(
  //     //       builder.getUnknownLoc(), 
  //     //       gpu::AsyncTokenType::get(builder.getContext()),
  //     //       ValueRange(copyStreams[i]),
  //     //       bViews[i], bMemRefs[i]);
  //     //   allCopyTokens.push_back(copyToken.getAsyncToken());
        
  //     //   LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << "\n");
  //     // }
      
  //     // // 等待所有输入复制完成
  //     // builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
  //     // LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
      
  //     //修复复制stream分配问题
  //     // ========== 阶段6：创建异步流并执行输入复制 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 6: Creating async streams and performing input copies\n");

  //     // 为每个复制操作创建独立的stream
  //     SmallVector<Value> aCopyStreams_std;
  //     SmallVector<Value> bCopyStreams_std;

  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // 为A矩阵复制创建独立的stream
  //       auto aCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //       aCopyStreams_std.push_back(aCopyStream.getAsyncToken());
        
  //       // 为B矩阵复制创建独立的stream
  //       auto bCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //       bCopyStreams_std.push_back(bCopyStream.getAsyncToken());
  //     }

  //     SmallVector<Value> allCopyTokens_std;

  //     // 执行A矩阵复制
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(aCopyStreams_std[i]),  // 使用A矩阵专用的stream
  //           aViews[i], aMemRefs[i]);
  //       allCopyTokens_std.push_back(copyToken.getAsyncToken());
        
  //       LLVM_DEBUG(llvm::dbgs() << "Initiated A matrix copy for operation " << i << " on dedicated stream\n");
  //     }

  //     // 执行B矩阵复制
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(bCopyStreams_std[i]),  // 使用B矩阵专用的stream
  //           bViews[i], bMemRefs[i]);
  //       allCopyTokens_std.push_back(copyToken.getAsyncToken());
        
  //       LLVM_DEBUG(llvm::dbgs() << "Initiated B matrix copy for operation " << i << " on dedicated stream\n");
  //     }

  //     // 等待所有输入复制完成
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens_std);
  //     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed with " << (group.operations.size() * 2) << " independent streams\n");


  //     // ========== 阶段7：执行融合计算 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 7: Performing fused 3D computation\n");
      
  //     // 转换为tensor用于ONNX操作
  //     Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
      
  //     Value batchedBTensor = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), fusedBType, batchedB).getResult(0);
      
  //     // 创建融合的3D MatMul操作
  //     Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
  //         builder.getUnknownLoc(),
  //         fusedOutputType,
  //         batchedATensor,
  //         batchedBTensor);
      
  //     LLVM_DEBUG(llvm::dbgs() << "Created fused 3D MatMul operation: (" 
  //               << fusedAShape[0] << "x" << fusedAShape[1] << "x" << fusedAShape[2] << ") × ("
  //               << fusedBShape[0] << "x" << fusedBShape[1] << "x" << fusedBShape[2] << ") → ("
  //               << fusedOutputShape[0] << "x" << fusedOutputShape[1] << "x" << fusedOutputShape[2] << ")\n");
      
  //     // 转换融合结果为memref
  //     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
  //         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
      
  //     // 将融合结果复制到预分配的batched output memory
  //     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
  //         builder.getUnknownLoc(), 
  //         gpu::AsyncTokenType::get(builder.getContext()),
  //         ValueRange(fusedCopyStream.getAsyncToken()),
  //         batchedOutput, fusedResultMemRef);
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
      
  //     LLVM_DEBUG(llvm::dbgs() << "Copied fused 3D MatMul result to batched output memory\n");
      
  //     // ========== 阶段8：执行输出复制 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 8: Performing output copies\n");
  //     SmallVector<Value> outputCopyStreams;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
  //       outputCopyStreams.push_back(copyStream.getAsyncToken());
  //     }
      
  //     SmallVector<Value> outputCopyTokens;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       // 直接使用预先生成的reinterpret_cast输出memref
  //       auto copyToken = builder.create<gpu::MemcpyOp>(
  //           builder.getUnknownLoc(), 
  //           gpu::AsyncTokenType::get(builder.getContext()),
  //           ValueRange(outputCopyStreams[i]),
  //           individualOutputs[i], reinterpretedOutputMemRefs[i]);
  //       outputCopyTokens.push_back(copyToken.getAsyncToken());
  //     }
      
  //     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
  //     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
      
  //     // ========== 阶段9：结果替换和清理 ==========
  //     LLVM_DEBUG(llvm::dbgs() << "Stage 9: Result replacement and cleanup\n");
  //     // 转换回tensor并替换原始操作
  //     SmallVector<Value> splitResults;
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
  //           builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
  //       splitResults.push_back(splitResult);
  //     }
      
  //     // 替换所有使用并删除操作
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Operation* op = group.operations[i];
  //       op->getResult(0).replaceAllUsesWith(splitResults[i]);
  //       LLVM_DEBUG(llvm::dbgs() << "Replaced standard matmul operation " << i << " result\n");
  //     }
      
  //     for (size_t i = 0; i < group.operations.size(); ++i) {
  //       Operation* op = group.operations[i];
  //       LLVM_DEBUG(llvm::dbgs() << "Erasing standard matmul operation " << i << "\n");
  //       op->erase();
  //     }
      
  //     size_t aBroadcastsSaved = group.operations.size() - uniqueAMatrices.size();
  //     size_t bBroadcastsSaved = group.operations.size() - uniqueBMatrices.size();
  //     LLVM_DEBUG(llvm::dbgs() << "Successfully fused standard 2D MatMul operations with A/B matrix deduplication\n");
  //     LLVM_DEBUG(llvm::dbgs() << "Optimization: avoided " << aBroadcastsSaved << " redundant A matrix broadcasts\n");
  //     LLVM_DEBUG(llvm::dbgs() << "Optimization: avoided " << bBroadcastsSaved << " redundant B matrix broadcasts\n");
  //     return true;
  // }

  // 在以上基础之上，只对输入参数进行batch维度拼接，不进行升维
  bool fuseStandardMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
      LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Standard 2D MatMul operations with input batch concatenation\n");
      
      Operation* firstOp = group.operations[0];
      auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
      if (!firstMatMul) {
        return false;
      }
      
      auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
      auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
      auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
      
      ArrayRef<int64_t> aShape = aType.getShape();
      ArrayRef<int64_t> bShape = bType.getShape();
      ArrayRef<int64_t> outputShape = outputType.getShape();
      
      LLVM_DEBUG(llvm::dbgs() << "Original 2D shapes: A=[" << aShape[0] << "," << aShape[1] << "], ");
      LLVM_DEBUG(llvm::dbgs() << "B=[" << bShape[0] << "," << bShape[1] << "], ");
      LLVM_DEBUG(llvm::dbgs() << "Output=[" << outputShape[0] << "," << outputShape[1] << "]\n");
      
      // ========== 阶段1：收集输入和权重矩阵 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 1: Collecting input and weight matrices\n");
      
      // 对于ONNX MatMul: A是输入参数(需要拼接), B是权重参数(共享)
      SmallVector<Value> inputMatrices;  // A矩阵（输入）
      Value sharedWeight;  // B矩阵（权重，所有操作共享）
      
      // 收集所有输入矩阵（A参数）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        auto currentMatMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
        Value currentInput = currentMatMul.getA();  // 第一个参数是输入
        inputMatrices.push_back(currentInput);
      }
      
      // 获取共享的权重矩阵（B参数）- 所有操作应该使用相同的权重
      sharedWeight = firstMatMul.getB();  // 第二个参数是权重
      
      LLVM_DEBUG(llvm::dbgs() << "Collected " << inputMatrices.size() << " input matrices (A parameters)\n");
      LLVM_DEBUG(llvm::dbgs() << "Using shared weight matrix (B parameter)\n");
      
      // 验证所有操作确实使用相同的权重（调试模式下的检查）
      #ifdef LLVM_ENABLE_ASSERTIONS
      for (size_t i = 1; i < group.operations.size(); ++i) {
        auto matMul = dyn_cast<mlir::ONNXMatMulOp>(group.operations[i]);
        assert(matMul.getB() == sharedWeight && "All operations should use the same weight matrix");
      }
      #endif
      
      // ========== 阶段2：计算融合后的形状 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 2: Computing fused shapes\n");
      
      int64_t fusedBatchSize = group.operations.size();
      // A矩阵（输入）在batch维度拼接：[64x147] * N -> [64*N x 147]
      SmallVector<int64_t> fusedAShape = {aShape[0] * fusedBatchSize, aShape[1]};
      // B矩阵（权重）保持不变：[147x48] -> [147x48]
      SmallVector<int64_t> fusedBShape = {bShape[0], bShape[1]};
      // 输出矩阵在batch维度拼接：[64x48] * N -> [64*N x 48]
      SmallVector<int64_t> fusedOutputShape = {outputShape[0] * fusedBatchSize, outputShape[1]};
      
      LLVM_DEBUG(llvm::dbgs() << "Fused shapes: A=[" << fusedAShape[0] << "," << fusedAShape[1] << "], ");
      LLVM_DEBUG(llvm::dbgs() << "B=[" << fusedBShape[0] << "," << fusedBShape[1] << "], ");
      LLVM_DEBUG(llvm::dbgs() << "Output=[" << fusedOutputShape[0] << "," << fusedOutputShape[1] << "]\n");
      
      // 创建类型
      auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
      auto fusedBType = RankedTensorType::get(fusedBShape, bType.getElementType());
      auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
      
      auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
      auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
      auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
      auto originalAMemRefType = MemRefType::get(aShape, aType.getElementType());
      
      // ========== 阶段3：分配GPU内存 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 3: Allocating GPU memory\n");
      
      // 只为输入矩阵分配拼接后的内存（权重直接使用原始tensor）
      Value batchedA = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), aMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      Value batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      
      // 为每个原始操作分配独立的2D输出内存
      SmallVector<Value> individualOutputs;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
      
      LLVM_DEBUG(llvm::dbgs() << "Allocated GPU memory (input concatenation buffer and individual outputs)\n");
      
      // ========== 阶段4：创建所有subviews ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 4: Creating all subviews\n");
      
      // 创建输入矩阵的subviews
      SmallVector<Value> inputViews;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        int64_t batchOffset = i * aShape[0]; // 每个batch在行维度的偏移
        
        SmallVector<OpFoldResult> aOffsets = {
            builder.getI64IntegerAttr(batchOffset),
            builder.getI64IntegerAttr(0)
        };
        SmallVector<OpFoldResult> aSizes = {
            builder.getI64IntegerAttr(aShape[0]),
            builder.getI64IntegerAttr(aShape[1])
        };
        SmallVector<OpFoldResult> aStrides = {
            builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(1)
        };
        
        Value aView = builder.create<memref::SubViewOp>(
            builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
        inputViews.push_back(aView);
        
        LLVM_DEBUG(llvm::dbgs() << "Created input subview " << i << " at offset " << batchOffset << "\n");
      }
      
      // 创建输出矩阵的subviews
      SmallVector<Value> outputViews;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        int64_t outputOffset = i * outputShape[0];
        
        SmallVector<OpFoldResult> outputOffsets = {
            builder.getI64IntegerAttr(outputOffset),
            builder.getI64IntegerAttr(0)
        };
        SmallVector<OpFoldResult> outputSizes = {
            builder.getI64IntegerAttr(outputShape[0]),
            builder.getI64IntegerAttr(outputShape[1])
        };
        SmallVector<OpFoldResult> outputStrides = {
            builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(1)
        };
        
        Value outputView = builder.create<memref::SubViewOp>(
            builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
        outputViews.push_back(outputView);
        
        LLVM_DEBUG(llvm::dbgs() << "Created output subview " << i << " at offset " << outputOffset << "\n");
      }
      
      LLVM_DEBUG(llvm::dbgs() << "All subviews created\n");
      
      // ========== 阶段5：执行输入数据复制 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 5: Performing input data copies\n");
      
      // 为输入复制创建异步流
      SmallVector<Value> inputCopyStreams;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
            gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
        inputCopyStreams.push_back(copyStream.getAsyncToken());
      }
      
      SmallVector<Value> allCopyTokens;
      
      // 复制输入矩阵（A参数）到对应的subview
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value currentInput = inputMatrices[i];
        
        // 转换输入为memref并复制到对应的subview
        Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
            builder.getUnknownLoc(), originalAMemRefType, currentInput).getResult(0);
        
        auto copyToken = builder.create<gpu::MemcpyOp>(
            builder.getUnknownLoc(), 
            gpu::AsyncTokenType::get(builder.getContext()),
            ValueRange(inputCopyStreams[i]),
            inputViews[i], aMemRef);
        allCopyTokens.push_back(copyToken.getAsyncToken());
        
        LLVM_DEBUG(llvm::dbgs() << "Initiated input matrix copy for operation " << i << "\n");
      }
      
      // 等待所有输入复制完成
      builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, allCopyTokens);
      LLVM_DEBUG(llvm::dbgs() << "All input copies completed\n");
      
      // ========== 阶段6：执行融合计算 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 6: Performing fused computation\n");
      
      // 转换为tensor用于ONNX操作
      Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
      
      // 直接使用原始的权重tensor，不需要额外的内存分配和复制
      
      // 创建融合的2D MatMul操作
      Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
          builder.getUnknownLoc(),
          fusedOutputType,
          batchedATensor,
          sharedWeight);  // 直接使用原始权重tensor
      
      LLVM_DEBUG(llvm::dbgs() << "Created fused 2D MatMul operation: (" 
                << fusedAShape[0] << "x" << fusedAShape[1] << ") × ("
                << fusedBShape[0] << "x" << fusedBShape[1] << ") → ("
                << fusedOutputShape[0] << "x" << fusedOutputShape[1] << ")\n");
      
      // 转换融合结果为memref
      Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
      
      // 将融合结果复制到预分配的batched output memory
      auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
      auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
          builder.getUnknownLoc(), 
          gpu::AsyncTokenType::get(builder.getContext()),
          ValueRange(fusedCopyStream.getAsyncToken()),
          batchedOutput, fusedResultMemRef);
      builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
      
      LLVM_DEBUG(llvm::dbgs() << "Copied fused MatMul result to batched output memory\n");
      
      // ========== 阶段7：执行输出复制 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 7: Performing output copies\n");
      
      SmallVector<Value> outputCopyStreams;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
            gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
        outputCopyStreams.push_back(copyStream.getAsyncToken());
      }
      
      SmallVector<Value> outputCopyTokens;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        // 使用预先创建的输出subview
        auto copyToken = builder.create<gpu::MemcpyOp>(
            builder.getUnknownLoc(), 
            gpu::AsyncTokenType::get(builder.getContext()),
            ValueRange(outputCopyStreams[i]),
            individualOutputs[i], outputViews[i]);
        outputCopyTokens.push_back(copyToken.getAsyncToken());
      }
      
      builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
      LLVM_DEBUG(llvm::dbgs() << "All output copies completed\n");
      
      // ========== 阶段8：结果替换和清理 ==========
      LLVM_DEBUG(llvm::dbgs() << "Stage 8: Result replacement and cleanup\n");
      
      // 转换回tensor并替换原始操作
      SmallVector<Value> splitResults;
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
            builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
        splitResults.push_back(splitResult);
      }
      
      // 替换所有使用并删除操作
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        op->getResult(0).replaceAllUsesWith(splitResults[i]);
        LLVM_DEBUG(llvm::dbgs() << "Replaced matmul operation " << i << " result\n");
      }
      
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        LLVM_DEBUG(llvm::dbgs() << "Erasing matmul operation " << i << "\n");
        op->erase();
      }
      
      size_t operationCount = group.operations.size();
      LLVM_DEBUG(llvm::dbgs() << "Successfully fused " << operationCount << " 2D MatMul operations with input batch concatenation\n");
      LLVM_DEBUG(llvm::dbgs() << "Optimization: reused 1 shared weight matrix across " << operationCount << " operations\n");
      return true;
  }

  // 辅助函数：将共享的B矩阵扩展到指定的batch大小
  Value performExpandSharedMatrix(OpBuilder& builder, Value sharedMatrix3D, int64_t targetBatchSize, Type elementType, ArrayRef<int64_t> originalShape) {
      auto sharedType = sharedMatrix3D.getType().cast<RankedTensorType>();
      ArrayRef<int64_t> sharedShape = sharedType.getShape(); // [1, rows, cols]
      
      // 创建目标形状 [targetBatchSize, rows, cols]
      SmallVector<int64_t> targetShape = {targetBatchSize, sharedShape[1], sharedShape[2]};
      auto targetType = RankedTensorType::get(targetShape, elementType);
      
      LLVM_DEBUG(llvm::dbgs() << "Expanding shared B matrix from [" << sharedShape[0] << ", " << sharedShape[1] 
                << ", " << sharedShape[2] << "] to [" << targetBatchSize << ", " << sharedShape[1] 
                << ", " << sharedShape[2] << "]\n");
      
      // 创建目标形状常量
      auto shapeType = RankedTensorType::get({3}, builder.getI64Type());
      SmallVector<int64_t> shapeData = {targetBatchSize, sharedShape[1], sharedShape[2]};
      auto shapeAttr = DenseElementsAttr::get(shapeType, ArrayRef<int64_t>(shapeData));
      Value shapeConstant = builder.create<mlir::ONNXConstantOp>(
          builder.getUnknownLoc(), 
          mlir::Attribute(),  // sparse_value (为空)
          shapeAttr);         // value attribute
      
      // 使用Expand操作进行广播
      Value expanded = builder.create<mlir::ONNXExpandOp>(
          builder.getUnknownLoc(), 
          targetType,         // result type
          sharedMatrix3D,     // input operand
          shapeConstant);     // shape operand
      
      LLVM_DEBUG(llvm::dbgs() << "Created expand operation for shared B matrix\n");
      return expanded;
  }

  // 辅助函数：将3D张量降维为2D（去掉batch维度1）
  Value performSqueeze3Dto2D(OpBuilder& builder, Value input3D, Type elementType) {
      // 这里input3D是一个3D memref subview (1, m, n)，需要转换为2D张量
      auto input3DType = input3D.getType().cast<MemRefType>();
      ArrayRef<int64_t> input3DShape = input3DType.getShape(); // [1, m, n]
      
      // 创建2D形状 [m, n]
      SmallVector<int64_t> output2DShape = {input3DShape[1], input3DShape[2]};
      auto output2DType = RankedTensorType::get(output2DShape, elementType);
      
      LLVM_DEBUG(llvm::dbgs() << "Squeezing from [" << input3DShape[0] << ", " << input3DShape[1] 
                << ", " << input3DShape[2] << "] to [" << output2DShape[0] << ", " << output2DShape[1] << "]\n");
      
      // 首先转换为tensor
      Value input3DTensor = builder.create<mlir::UnrealizedConversionCastOp>(
          builder.getUnknownLoc(), 
          RankedTensorType::get(input3DShape, elementType), 
          input3D).getResult(0);
      
      // 创建axes常量 [0] - squeeze掉第0维
      auto axesType = RankedTensorType::get({1}, builder.getI64Type());
      SmallVector<int64_t> axesData = {0};
      auto axesAttr = DenseElementsAttr::get(axesType, ArrayRef<int64_t>(axesData));
      Value axesConstant = builder.create<mlir::ONNXConstantOp>(
          builder.getUnknownLoc(), 
          mlir::Attribute(),  // sparse_value (为空)
          axesAttr);          // value attribute
      
      // 使用Squeeze操作移除batch维度
      Value squeezed = builder.create<mlir::ONNXSqueezeOp>(
          builder.getUnknownLoc(), 
          output2DType,       // result type
          input3DTensor,      // data operand
          axesConstant);      // axes operand
      
      LLVM_DEBUG(llvm::dbgs() << "Created squeeze operation from 3D to 2D\n");
      return squeezed;
  }


  // 检查bias是否为0 - 修复版本
  bool isBiasZero(Value bias) {
    if (!bias) {
      return true;  // 没有bias也算zero bias
    }
    
    if (auto constOp = bias.getDefiningOp<mlir::ONNXConstantOp>()) {
      auto valueAttr = constOp.getValue();
      if (valueAttr.has_value()) {
        if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr.value())) {
          if (denseAttr.isSplat()) {
            auto splatValue = denseAttr.getSplatValue<APFloat>();
            // 使用isPosZero()和isNegZero()来检查正零和负零
            return splatValue.isPosZero() || splatValue.isNegZero();
          }
          // 检查是否所有元素都是0（包括-0.0）
          for (auto value : denseAttr.getValues<APFloat>()) {
            if (!value.isPosZero() && !value.isNegZero()) {
              return false;
            }
          }
          return true;
        }
      }
    }
    
    // 检查是否是0常量
    if (auto constOp = bias.getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>()) {
        APFloat floatValue = floatAttr.getValue();
        return floatValue.isPosZero() || floatValue.isNegZero();
      }
    }
    
    return false;
  }

  // 融合Gemm操作
  bool fuseGemmOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Gemm operations\n");
    
    Operation* firstOp = group.operations[0];
    auto firstGemm = dyn_cast<mlir::ONNXGemmOp>(firstOp);
    if (!firstGemm) {
      LLVM_DEBUG(llvm::dbgs() << "First operation is not a Gemm op\n");
      return false;
    }
    
    // 检查是否所有Gemm操作的bias都是0，如果是则转换为MatMul
    bool allBiasZero = true;
    for (Operation* op : group.operations) {
      auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
      if (!gemmOp || !isBiasZero(gemmOp.getC())) {
        allBiasZero = false;
        break;
      }
    }
    
    if (allBiasZero) {
      LLVM_DEBUG(llvm::dbgs() << "All Gemm operations have zero bias, converting to MatMul fusion\n");
      return fuseGemmAsMatMul(builder, group);
    }
    
    // 如果有非零bias，执行正常的Gemm融合
    return fuseGemmNormal(builder, group);
  }

  // 将bias为0的Gemm融合为MatMul
  bool fuseGemmAsMatMul(OpBuilder& builder, FusibleGroup& group) {
    // 创建MatMul组并执行MatMul融合
    FusibleGroup matmulGroup(group.layerIndex, "MatMul");
    matmulGroup.operations = group.operations;
    matmulGroup.timesteps = group.timesteps;
    
    return fuseMatMulOperations(builder, matmulGroup);
  }
  
  // 正常的Gemm融合
  bool fuseGemmNormal(OpBuilder& builder, FusibleGroup& group) {
    // TODO
    // 实现正常的Gemm融合逻辑 
    // 这里简化实现，直接返回true
    LLVM_DEBUG(llvm::dbgs() << "Normal Gemm fusion not fully implemented\n");
    return true;
  }

  // 修改调试打印函数，显示更详细的融合信息
  void debugPrintFusibleGroups(const SmallVector<FusibleGroup>& fusibleGroups) {
    LLVM_DEBUG(llvm::dbgs() << "\n=== Debug: Fusible Groups Summary (Batch Size: " << batchSize << ") ===\n");
    
    // 首先打印所有收集到的操作信息
    LLVM_DEBUG(llvm::dbgs() << "=== All Fusible Operations ===\n");
    getOperation().walk([&](Operation* op) {
      if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
        StringRef nodeName = nodeNameAttr.getValue();
        if (nodeName.contains("/layer/")) {
          SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName);
          if (isFusibleOperationType(layerInfo.opName)) {
            std::string normalizedOpType = layerInfo.opName;
            if (layerInfo.opName == "Gemm") {
              auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
              if (gemmOp && isBiasZero(gemmOp.getC())) {
                normalizedOpType = "MatMul (from Gemm)";
              }
            }
            LLVM_DEBUG(llvm::dbgs() << "  " << nodeName 
                       << " -> layer " << layerInfo.layerIndex 
                       << ", timestep " << layerInfo.timestep 
                       << ", type: " << normalizedOpType << "\n");
          }
        }
      }
    });
    
    LLVM_DEBUG(llvm::dbgs() << "\n=== Formed Fusible Groups ===\n");
    for (size_t i = 0; i < fusibleGroups.size(); ++i) {
      const auto& group = fusibleGroups[i];
      LLVM_DEBUG(llvm::dbgs() << "Group " << i << ": Layer " << group.layerIndex 
                 << ", Type " << group.opType 
                 << ", " << group.operations.size() << " operations");
      
      if (!group.timesteps.empty()) {
        LLVM_DEBUG(llvm::dbgs() << " (timesteps " << group.timesteps[0] 
                   << " to " << group.timesteps.back() << ")");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      
      for (size_t j = 0; j < group.operations.size(); ++j) {
        if (auto nodeNameAttr = group.operations[j]->getAttrOfType<StringAttr>("onnx_node_name")) {
          StringRef nodeName = nodeNameAttr.getValue();
          std::string originalType = "unknown";
          
          // 检查原始操作类型
          if (nodeName.contains("Gemm")) {
            originalType = "Gemm";
          } else if (nodeName.contains("MatMul")) {
            originalType = "MatMul";
          } else if (nodeName.contains("Conv")) {
            originalType = "Conv";
          } else if (nodeName.contains("MaxPool")) {
            originalType = "MaxPool";
          }
          
          LLVM_DEBUG(llvm::dbgs() << "  Op " << j << ": " << nodeNameAttr.getValue() 
                     << " (timestep " << group.timesteps[j] 
                     << ", original: " << originalType << ")\n");
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "=== End Debug Summary ===\n\n");
  }
};

} // end anonymous namespace

namespace onnx_mlir {
    std::unique_ptr<Pass> createSNNBatchFusionPass() {
      return std::make_unique<SNNBatchFusionPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<SNNBatchFusionPass> pass;



// // 相邻op 避免复制
// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/Interfaces/InferTypeOpInterface.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/StringMap.h"
// #include "llvm/ADT/StringSet.h"
// #include "llvm/ADT/DenseSet.h"

// // 添加ONNX操作的头文件
// #include "src/Dialect/ONNX/ONNXOps.hpp"

// using namespace mlir;

// #define DEBUG_TYPE "snn-batch-fusion"

// namespace {

// struct LayerInfo {
//   std::string layerName;  // 实际的层名，如layer.0, layer.1
//   int timestep;           // 时间步信息
  
//   LayerInfo(std::string name, int ts) : layerName(std::move(name)), timestep(ts) {}
// };

// // 简化的layer信息结构
// struct SimpleLayerInfo {
//   int layerIndex;    // 层索引，如layer.1 -> 1
//   int timestep;      // 时间步，如layer.1_5 -> 5，没有_则为0
//   std::string opName; // 操作名
  
//   SimpleLayerInfo(int layer, int ts, std::string op) 
//       : layerIndex(layer), timestep(ts), opName(std::move(op)) {}
// };

// // 操作信息结构
// struct OperationInfo {
//   Operation* op;
//   SimpleLayerInfo layerInfo;
//   bool isLayerOp;        // 是否是layer操作
//   bool isTimestepBoundary; // 是否是时间步边界
  
//   OperationInfo(Operation* operation, SimpleLayerInfo info, bool isLayer, bool isBoundary)
//       : op(operation), layerInfo(std::move(info)), isLayerOp(isLayer), isTimestepBoundary(isBoundary) {}
// };

// // 可融合操作组结构 - 新增优化字段
// struct FusibleGroup {
//   int layerIndex;
//   std::string opType;  // Conv, MaxPoolSingleOut, Gemm, MatMul
//   SmallVector<Operation*> operations;
//   SmallVector<int> timesteps;
  
//   // 新增：相邻group优化字段
//   bool needInputCopy = true;    // 是否需要输入复制
//   bool needOutputCopy = true;   // 是否需要输出复制
//   Value sharedBatchedInput;     // 共享的batched输入内存
//   Value sharedBatchedOutput;    // 共享的batched输出内存
  
//   // *** 新增：依赖关系字段，用于延迟删除 ***
//   bool dependsOnPrevGroup = false;      // 是否依赖前一个组
//   bool hasNextGroupDependency = false;  // 是否有下一个组依赖自己
//   SmallVector<Operation*> pendingDeleteOps; // 待删除的操作（用于延迟删除）
  
//   FusibleGroup() : layerIndex(0), opType("") {}
//   FusibleGroup(int layer, std::string type) : layerIndex(layer), opType(std::move(type)) {}
// };

// struct SNNBatchFusionPass
//     : public PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>> {
  
//   StringRef getArgument() const final { return "snn-batch-fusion"; }
//   StringRef getDescription() const final {
//     return "Fuse SNN operations across timesteps for batch processing with IR reordering and adjacent group optimization";
//   }
  
//   Option<int> batchSize{*this, "batch-size", 
//                         llvm::cl::desc("Batch size for fusion"), 
//                         llvm::cl::init(2)};
  
//   SNNBatchFusionPass() = default;
//   SNNBatchFusionPass(const SNNBatchFusionPass& other) 
//       : PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>>(),
//         batchSize(*this, "batch-size", 
//                   llvm::cl::desc("Batch size for fusion"), 
//                   llvm::cl::init(other.batchSize)) {}
  
//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<gpu::GPUDialect, memref::MemRefDialect, arith::ArithDialect>();
//   }
  
//   // 修改后的主要执行流程
//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
    
//     LLVM_DEBUG(llvm::dbgs() << "Running SNNBatchFusionPass with batch size: " 
//                << batchSize << "\n");
    
//     // 1. 预处理：给时序性算子添加layer字段
//     preprocessTemporalOpsWithLayerInfo(funcOp);
    
//     // 2. 直接执行基于layer信息的重排
//     SmallVector<OperationInfo> operationInfos;
//     collectOperationInfo(funcOp, operationInfos);
    
//     if (operationInfos.empty()) {
//       LLVM_DEBUG(llvm::dbgs() << "No operations with layer info found\n");
//       return;
//     }
    
//     // 3. 确定时间步范围
//     int maxTimestep = 0;
//     for (const auto& opInfo : operationInfos) {
//       maxTimestep = std::max(maxTimestep, opInfo.layerInfo.timestep);
//     }
//     int totalTimesteps = maxTimestep + 1;
    
//     LLVM_DEBUG(llvm::dbgs() << "Total timesteps: " << totalTimesteps << "\n");
    
//     if (totalTimesteps < batchSize) {
//       LLVM_DEBUG(llvm::dbgs() << "Not enough timesteps for batching: " 
//                  << totalTimesteps << " < " << batchSize << "\n");
//       return;
//     }
    
//     // 4. 按batch进行重排（与fusion保持一致的分批逻辑）
//     SmallVector<Operation*> reorderedOps;
    
//     for (int batchStart = 0; batchStart + batchSize <= totalTimesteps; batchStart += batchSize) {
//       LLVM_DEBUG(llvm::dbgs() << "\n--- Processing reorder batch starting at timestep " 
//                  << batchStart << " (size: " << batchSize << ") ---\n");
      
//       if (!reorderSingleBatch(operationInfos, batchStart, batchSize, reorderedOps)) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to reorder batch starting at " << batchStart << "\n");
//         return;
//       }
//     }
    
//     // 5. 处理剩余的时间步（不足一个batch的）
//     int remainingStart = (totalTimesteps / batchSize) * batchSize;
//     if (remainingStart < totalTimesteps) {
//       LLVM_DEBUG(llvm::dbgs() << "\n--- Processing remaining timesteps from " << remainingStart << " ---\n");
      
//       SmallVector<Operation*> remainingOps;
//       for (const auto& opInfo : operationInfos) {
//         if (opInfo.layerInfo.timestep >= remainingStart) {
//           remainingOps.push_back(opInfo.op);
//         }
//       }
      
//       // 对剩余操作按原有顺序排序
//       llvm::sort(remainingOps, [](Operation* a, Operation* b) {
//         return a->isBeforeInBlock(b);
//       });
      
//       for (Operation* op : remainingOps) {
//         reorderedOps.push_back(op);
//       }
//     }
    
//     // 6. 执行最终重排
//     if (!executeReordering(reorderedOps)) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to execute reordering\n");
//       return;
//     }
    
//     // 7. 在重排完成后，执行batch fusion（现在会按相同的batch size进行）
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Starting Batch Fusion (Batch Size: " << batchSize << ") ===\n");
//     if (!performBatchFusionWithOptimizations(funcOp)) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to perform batch fusion\n");
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed SNNBatchFusionPass\n");
//   }

// private:
  
//   // 预处理：给时序性算子添加layer字段
//   void preprocessTemporalOpsWithLayerInfo(func::FuncOp funcOp) {
//     Block& block = funcOp.getBody().front();
    
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Preprocessing: Adding Layer Info to Temporal Ops ===\n");
    
//     // 首先确定最大的层索引，/Add操作将被标记为最后一层
//     int maxLayerIndex = findMaxLayerIndex(funcOp);
//     int addLayerIndex = maxLayerIndex + 1; // Add操作作为最后一层
    
//     LLVM_DEBUG(llvm::dbgs() << "Max layer index found: " << maxLayerIndex 
//                << ", Add operations will be layer " << addLayerIndex << "\n");
    
//     std::string currentTemporalLayerName = "";
//     int currentTimestep = 0;
//     bool isAfterTimestepBoundary = false;
//     bool hasSeenLayerOpInCurrentTimestep = false;
    
//     for (Operation& op : block) {
//       if (auto nodeNameAttr = op.getAttrOfType<StringAttr>("onnx_node_name")) {
//         StringRef nodeName = nodeNameAttr.getValue();
        
//         if (nodeName.contains("/layer/")) {
//           // 解析层信息
//           LayerInfo layerInfo = parseLayerNameCorrectly(nodeName);
          
//           // 计算当前层后续时序性算子应该归属的层名
//           currentTemporalLayerName = calculateNextLayerName(layerInfo.layerName, layerInfo.timestep);
//           currentTimestep = layerInfo.timestep;
//           hasSeenLayerOpInCurrentTimestep = true;
//           isAfterTimestepBoundary = false;
          
//           LLVM_DEBUG(llvm::dbgs() << "Found layer op: " << nodeName 
//                      << " -> parsed as " << layerInfo.layerName 
//                      << " (timestep: " << layerInfo.timestep 
//                      << "), temporal ops will be tagged as: " << currentTemporalLayerName << "\n");
          
//         } else if (nodeName.starts_with("/Add")) {
//           // 处理时间步边界Add操作，给它们添加layer信息
//           int timestepFromAdd = extractTimestepFromAdd(nodeName);
          
//           std::string addLayerName;
//           if (timestepFromAdd == 0) {
//             addLayerName = "layer." + std::to_string(addLayerIndex);
//           } else {
//             addLayerName = "layer." + std::to_string(addLayerIndex) + "_" + std::to_string(timestepFromAdd);
//           }
          
//           std::string newNodeName = "/layer/" + addLayerName + "/" + nodeName.str();
//           op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
          
//           LLVM_DEBUG(llvm::dbgs() << "Updated Add operation: " << nodeName 
//                      << " -> " << newNodeName << " (timestep: " << timestepFromAdd << ")\n");
          
//           // 重置状态，进入下一个时间步
//           currentTemporalLayerName = "";
//           isAfterTimestepBoundary = true;
//           hasSeenLayerOpInCurrentTimestep = false;
//           currentTimestep = timestepFromAdd + 1;
          
//         } else if (!isConstantOrUtilityOp(&op)) {
//           // 时序性算子处理
//           if (!currentTemporalLayerName.empty()) {
//             // 正常情况：已经有了当前的时序层名，同一组时序性算子使用相同的层名
//             std::string newNodeName = "/layer/" + currentTemporalLayerName + "/" + nodeName.str();
//             op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
            
//             LLVM_DEBUG(llvm::dbgs() << "Updated temporal op: " << nodeName 
//                        << " -> " << newNodeName << "\n");
            
//           } else if (isAfterTimestepBoundary && !hasSeenLayerOpInCurrentTimestep) {
//             // 特殊情况：时间步边界后第一次遇到时序性操作，但还没有看到layer操作
//             // 说明第0层被优化掉了，从第1层开始，这一组所有时序性算子都属于第1层
//             if (currentTemporalLayerName.empty()) {
//               if (currentTimestep == 0) {
//                 currentTemporalLayerName = "layer.1";  // 时间步0，第1层
//               } else {
//                 currentTemporalLayerName = "layer.1_" + std::to_string(currentTimestep);  // 时间步N，第1层
//               }
              
//               LLVM_DEBUG(llvm::dbgs() << "Special case detected - layer 0 optimized away, "
//                          << "temporal ops will use: " << currentTemporalLayerName << "\n");
//             }
            
//             std::string newNodeName = "/layer/" + currentTemporalLayerName + "/" + nodeName.str();
//             op.setAttr("onnx_node_name", StringAttr::get(op.getContext(), newNodeName));
            
//             LLVM_DEBUG(llvm::dbgs() << "Special case - updated temporal op: " 
//                        << nodeName << " -> " << newNodeName << "\n");
            
//           } else {
//             LLVM_DEBUG(llvm::dbgs() << "Skipping temporal op (no layer context): " << nodeName << "\n");
//           }
//         }
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Completed preprocessing\n");
//   }
  
//   // 计算下一层的名称
//   std::string calculateNextLayerName(const std::string& currentLayerName, int timestep) {
//     // 从当前层名中提取层索引
//     int currentLayerIndex = extractLayerIndex(currentLayerName);
//     int nextLayerIndex = currentLayerIndex + 1;
    
//     if (timestep == 0) {
//       // 时间步0：layer.[next_layer_index]
//       return "layer." + std::to_string(nextLayerIndex);
//     } else {
//       // 时间步N：layer.[next_layer_index]_[timestep]
//       return "layer." + std::to_string(nextLayerIndex) + "_" + std::to_string(timestep);
//     }
//   }
  
//   // 正确解析层名
//   LayerInfo parseLayerNameCorrectly(StringRef nodeName) {
//     // 解析形如 "/layer/layer.4/Conv" 或 "/layer/layer.3_1/Conv" 的名称
//     size_t lastSlash = nodeName.rfind('/');
//     if (lastSlash == StringRef::npos) {
//       return LayerInfo(nodeName.str(), 0);
//     }
    
//     StringRef layerPart = nodeName.substr(0, lastSlash);
//     lastSlash = layerPart.rfind('/');
//     if (lastSlash == StringRef::npos) {
//       return LayerInfo(layerPart.str(), 0);
//     }
    
//     StringRef fullLayerName = layerPart.substr(lastSlash + 1);
    
//     // 检查是否有时间步格式 (layer.3_1 -> layer.3 at timestep 1)
//     size_t underscorePos = fullLayerName.rfind('_');
//     if (underscorePos != StringRef::npos) {
//       StringRef layerNamePart = fullLayerName.substr(0, underscorePos);  // "layer.3"
//       StringRef timestepPart = fullLayerName.substr(underscorePos + 1);   // "1"
      
//       int timestep = 0;
//       if (timestepPart.getAsInteger(10, timestep)) {
//         timestep = 0; // 默认为0如果解析失败
//       }
      
//       LLVM_DEBUG(llvm::dbgs() << "    Parsed " << fullLayerName << " -> layer: " << layerNamePart 
//                 << " (timestep: " << timestep << ")\n");
      
//       return LayerInfo(layerNamePart.str(), timestep);
      
//     } else {
//       // 没有下划线，说明是时间步0的操作（layer.0, layer.3等）
//       return LayerInfo(fullLayerName.str(), 0);
//     }
//   }
  
//   // 从Add操作名中提取时间步信息
//   int extractTimestepFromAdd(StringRef nodeName) {
//     // 解析 "/Add" -> 0, "/Add_1" -> 1, "/Add_2" -> 2 等
//     if (nodeName == "/Add") {
//       return 0;
//     }
    
//     if (nodeName.starts_with("/Add_")) {
//       StringRef timestepStr = nodeName.substr(5); // 跳过 "/Add_"
//       int timestep = 0;
//       if (timestepStr.getAsInteger(10, timestep)) {
//         return 0; // 解析失败时默认为0
//       }
//       return timestep;
//     }
    
//     return 0;
//   }
  
//   // 找到现有layer操作中的最大层索引
//   int findMaxLayerIndex(func::FuncOp funcOp) {
//     int maxIndex = 0;
    
//     funcOp.walk([&](Operation* op) {
//       if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//         StringRef nodeName = nodeNameAttr.getValue();
        
//         // 只处理原始的layer操作，不包括Add操作
//         if (nodeName.contains("/layer/") && !nodeName.starts_with("/Add")) {
//           LayerInfo layerInfo = parseLayerNameCorrectly(nodeName);
//           int layerIndex = extractLayerIndex(layerInfo.layerName);
//           maxIndex = std::max(maxIndex, layerIndex);
//         }
//       }
//     });
    
//     LLVM_DEBUG(llvm::dbgs() << "Found max layer index: " << maxIndex << "\n");
//     return maxIndex;
//   }
  
//   // 从层名中提取层索引
//   int extractLayerIndex(const std::string& layerName) {
//     // 处理两种格式：
//     // 1. "layer.X" -> 返回 X
//     // 2. "layer.X_T" -> 返回 X（X是层索引，T是时间步）
    
//     size_t underscorePos = layerName.find('_');
//     std::string layerPart;
    
//     if (underscorePos != std::string::npos) {
//       // 格式: "layer.X_T" -> 提取 layer.X 部分
//       layerPart = layerName.substr(0, underscorePos);
//     } else {
//       // 格式: "layer.X"
//       layerPart = layerName;
//     }
    
//     // 从 layer.X 中提取 X
//     size_t dotPos = layerPart.find('.');
//     if (dotPos != std::string::npos) {
//       std::string indexStr = layerPart.substr(dotPos + 1);
//       int index = 0;
//       if (sscanf(indexStr.c_str(), "%d", &index) == 1) {
//         return index;
//       }
//     }
//     return 0;
//   }

//   SimpleLayerInfo parseSimpleLayerInfo(StringRef nodeName, Operation* op = nullptr) {
//     // 解析格式: /layer/layer.X_Y/OpName 或 /layer/layer.X/OpName
    
//     // 先找到最后一个/，获取操作名
//     size_t lastSlash = nodeName.rfind('/');
//     std::string opNameFromNodeName = (lastSlash != StringRef::npos) ? 
//                         nodeName.substr(lastSlash + 1).str() : nodeName.str();
    
//     // 查找layer.X_Y或layer.X部分
//     size_t layerStart = nodeName.find("/layer/layer.");
//     if (layerStart == StringRef::npos) {
//       // 如果没有找到layer信息，尝试从实际操作类型推断
//       std::string actualOpName = getActualOperationType(op);
//       return SimpleLayerInfo(0, 0, actualOpName);
//     }
    
//     StringRef layerPart = nodeName.substr(layerStart + 13); // 跳过"/layer/layer."
//     size_t nextSlash = layerPart.find('/');
//     if (nextSlash != StringRef::npos) {
//       layerPart = layerPart.substr(0, nextSlash);
//     }
    
//     // 解析layer.X_Y格式
//     size_t underscorePos = layerPart.find('_');
//     int layerIndex = 0;
//     int timestep = 0;
    
//     if (underscorePos != StringRef::npos) {
//       // 有时间步信息: layer.X_Y
//       StringRef layerIndexStr = layerPart.substr(0, underscorePos);
//       StringRef timestepStr = layerPart.substr(underscorePos + 1);
      
//       layerIndexStr.getAsInteger(10, layerIndex);
//       timestepStr.getAsInteger(10, timestep);
//     } else {
//       // 没有时间步信息: layer.X (默认时间步0)
//       layerPart.getAsInteger(10, layerIndex);
//       timestep = 0;
//     }
    
//     // 获取实际的操作类型而不是依赖节点名
//     std::string actualOpName = getActualOperationType(op);
//     if (actualOpName.empty()) {
//       // 如果无法获取实际操作类型，则尝试从节点名推断
//       actualOpName = inferOpTypeFromNodeName(opNameFromNodeName);
//     }
    
//     return SimpleLayerInfo(layerIndex, timestep, actualOpName);
//   }

//   // 添加获取实际操作类型的辅助函数
//   std::string getActualOperationType(Operation* op) {
//     if (!op) return "";
    
//     StringRef opName = op->getName().getStringRef();
    
//     if (opName == "onnx.Conv") {
//       return "Conv";
//     } else if (opName == "onnx.MaxPoolSingleOut") {
//       return "MaxPool";
//     } else if (opName == "onnx.Gemm") {
//       return "Gemm";
//     } else if (opName == "onnx.MatMul") {
//       return "MatMul";
//     } else if (opName == "onnx.Flatten") {
//       return "Flatten";
//     }
    
//     return "";
//   }

//   std::string inferOpTypeFromNodeName(const std::string& nodeName) {
//     // 移除后缀数字和下划线
//     std::string cleanName = nodeName;
//     size_t underscorePos = cleanName.find('_');
//     if (underscorePos != std::string::npos) {
//       cleanName = cleanName.substr(0, underscorePos);
//     }
    
//     if (cleanName.find("Conv") != std::string::npos) {
//       return "Conv";
//     } else if (cleanName.find("MaxPool") != std::string::npos) {
//       return "MaxPool";
//     } else if (cleanName.find("MatMul") != std::string::npos || 
//               cleanName.find("Gemm") != std::string::npos) {
//       return "Gemm"; // 先假设是Gemm，后续会检查bias
//     } else if (cleanName.find("Flatten") != std::string::npos) {
//       return "Flatten";
//     }
    
//     return cleanName;
//   }
  
//   void collectOperationInfo(func::FuncOp funcOp, SmallVector<OperationInfo>& operationInfos) {
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Collecting Operation Information ===\n");
    
//     funcOp.walk([&](Operation* op) {
//       // 跳过常量和工具操作
//       if (isConstantOrUtilityOp(op)) {
//         return;
//       }
      
//       if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//         StringRef nodeName = nodeNameAttr.getValue();
        
//         if (nodeName.contains("/layer/")) {
//           SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName, op); // 传递op指针
          
//           // 检查是否是时间步边界Add操作
//           bool isTimestepBoundary = nodeName.contains("/Add");
          
//           // 更全面的layer操作检查，使用实际操作类型
//           bool isLayerOp = !isTimestepBoundary && isFusibleOperationType(layerInfo.opName);
          
//           operationInfos.emplace_back(op, layerInfo, isLayerOp, isTimestepBoundary);
          
//           // 特别标注操作类型
//           std::string opTypeInfo = "TEMPORAL";
//           if (isLayerOp) {
//             opTypeInfo = "LAYER";
//             if (layerInfo.opName == "Gemm") {
//               auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
//               if (gemmOp && isBiasZero(gemmOp.getC())) {
//                 opTypeInfo += " (Gemm->MatMul)";
//               } else {
//                 opTypeInfo += " (Gemm)";
//               }
//             } else if (layerInfo.opName == "MatMul") {
//               opTypeInfo += " (MatMul)";
//             }
//           } else if (isTimestepBoundary) {
//             opTypeInfo = "BOUNDARY";
//           }
          
//           LLVM_DEBUG(llvm::dbgs() << "Collected: " << nodeName 
//                     << " -> layer " << layerInfo.layerIndex 
//                     << ", timestep " << layerInfo.timestep 
//                     << ", actual op: " << op->getName().getStringRef()
//                     << ", parsed type: " << layerInfo.opName
//                     << ", category: " << opTypeInfo << "\n");
//         }
//       }
//     });
    
//     LLVM_DEBUG(llvm::dbgs() << "Total collected operations: " << operationInfos.size() << "\n");
//   }

//   // 处理单个batch的重排
//   bool reorderSingleBatch(ArrayRef<OperationInfo> allOps, int batchStart, int batchSize, 
//                          SmallVector<Operation*>& reorderedOps) {
    
//     // 1. 筛选出当前batch的操作
//     SmallVector<const OperationInfo*> batchOps;
//     for (const auto& opInfo : allOps) {
//       if (opInfo.layerInfo.timestep >= batchStart && 
//           opInfo.layerInfo.timestep < batchStart + batchSize) {
//         batchOps.push_back(&opInfo);
//       }
//     }
    
//     if (batchOps.empty()) {
//       return true;
//     }
    
//     // 2. 按layer index分组
//     std::map<int, SmallVector<const OperationInfo*>> layerGroups;
//     for (const auto* opInfo : batchOps) {
//       layerGroups[opInfo->layerInfo.layerIndex].push_back(opInfo);
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Batch has " << layerGroups.size() << " layers\n");
    
//     // 3. 按layer顺序处理
//     for (const auto& [layerIndex, layerOps] : layerGroups) {
//       LLVM_DEBUG(llvm::dbgs() << "Processing layer " << layerIndex << " with " << layerOps.size() << " ops\n");
      
//       // 4. 在每一层内，按时间步分组
//       std::map<int, SmallVector<const OperationInfo*>> timestepGroups;
//       for (const auto* opInfo : layerOps) {
//         timestepGroups[opInfo->layerInfo.timestep].push_back(opInfo);
//       }
      
//       // 5. 按时间步顺序添加操作
//       for (int ts = batchStart; ts < batchStart + batchSize; ++ts) {
//         auto it = timestepGroups.find(ts);
//         if (it != timestepGroups.end()) {
//           LLVM_DEBUG(llvm::dbgs() << "  Timestep " << ts << ": " << it->second.size() << " ops\n");
          
//           // 在同一时间步内，保持原有顺序
//           SmallVector<const OperationInfo*> timestepOps = it->second;
//           llvm::sort(timestepOps, [](const OperationInfo* a, const OperationInfo* b) {
//             return a->op->isBeforeInBlock(b->op);
//           });
          
//           for (const auto* opInfo : timestepOps) {
//             reorderedOps.push_back(opInfo->op);
            
//             if (auto nodeNameAttr = opInfo->op->getAttrOfType<StringAttr>("onnx_node_name")) {
//               LLVM_DEBUG(llvm::dbgs() << "    Added: " << nodeNameAttr.getValue() << "\n");
//             }
//           }
//         }
//       }
//     }
    
//     return true;
//   }
  
//   // 执行最终的重排
//   bool executeReordering(ArrayRef<Operation*> reorderedOps) {
//     if (reorderedOps.empty()) {
//       return true;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Executing Final Reordering ===\n");
//     LLVM_DEBUG(llvm::dbgs() << "Total operations to reorder: " << reorderedOps.size() << "\n");
    
//     // 找到正确的插入点：所有常量操作之后
//     Operation* insertionPoint = findInsertionPointAfterConstants(reorderedOps[0]->getParentOp());
    
//     if (!insertionPoint) {
//       LLVM_DEBUG(llvm::dbgs() << "Failed to find valid insertion point\n");
//       return false;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Found insertion point after constants\n");
    
//     // 按顺序移动操作
//     for (size_t i = 0; i < reorderedOps.size(); ++i) {
//       Operation* opToMove = reorderedOps[i];
      
//       if (i == 0) {
//         // 第一个操作移动到插入点之后
//         opToMove->moveAfter(insertionPoint);
//         LLVM_DEBUG(llvm::dbgs() << "Moved first operation after insertion point\n");
//       } else {
//         // 后续操作移动到前一个操作之后
//         Operation* prevOp = reorderedOps[i-1];
//         if (opToMove != prevOp->getNextNode()) {
//           opToMove->moveAfter(prevOp);
//         }
//       }
      
//       if (auto nodeNameAttr = opToMove->getAttrOfType<StringAttr>("onnx_node_name")) {
//         LLVM_DEBUG(llvm::dbgs() << "Moved: " << nodeNameAttr.getValue() << "\n");
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Reordering completed successfully\n");
//     return true;
//   }
  
//   // 找到所有常量操作之后的插入点
//   Operation* findInsertionPointAfterConstants(Operation* parentOp) {
//     Block* block = nullptr;
    
//     // 获取包含操作的block
//     if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
//       block = &funcOp.getBody().front();
//     } else {
//       block = parentOp->getBlock();
//     }
    
//     if (!block) {
//       return nullptr;
//     }
    
//     Operation* lastConstant = nullptr;
    
//     // 遍历block，找到最后一个常量或工具操作
//     for (Operation& op : *block) {
//       if (isConstantOrUtilityOp(&op)) {
//         lastConstant = &op;
//         LLVM_DEBUG(llvm::dbgs() << "Found constant/utility op: " << op.getName() << "\n");
//       } else {
//         // 一旦遇到非常量操作，如果之前找到了常量，就停止搜索
//         if (lastConstant) {
//           break;
//         }
//       }
//     }
    
//     if (lastConstant) {
//       LLVM_DEBUG(llvm::dbgs() << "Last constant operation found: " << lastConstant->getName() << "\n");
//       return lastConstant;
//     }
    
//     // 如果没有找到常量操作，返回block的第一个操作（如果存在）
//     if (!block->empty()) {
//       LLVM_DEBUG(llvm::dbgs() << "No constants found, using first operation in block\n");
//       return &block->front();
//     }
    
//     return nullptr;
//   }

//   // 检查是否是常量或工具操作，这些操作不应该被重排
//   bool isConstantOrUtilityOp(Operation* op) {
//     StringRef opName = op->getName().getStringRef();
    
//     // 常量操作
//     if (opName == "onnx.Constant") {
//       return true;
//     }
    
//     // 其他不应该重排的操作
//     if (opName == "func.return" || 
//         opName == "func.call" ||
//         opName.starts_with("builtin.") ||
//         opName.starts_with("arith.") ||
//         opName.starts_with("memref.global")) {
//       return true;
//     }
    
//     return false;
//   }

//   // === 修复后的 Batch Fusion 逻辑 - 新增相邻组优化 ===
  
//   // 执行batch fusion的主函数 - 修复版本，增加相邻组优化
//   bool performBatchFusionWithOptimizations(func::FuncOp funcOp) {
//     LLVM_DEBUG(llvm::dbgs() << "Starting batch fusion analysis with batch size: " << batchSize << "\n");
    
//     // 1. 识别可融合的操作组（现在按batch size分组）
//     SmallVector<FusibleGroup, 4> fusibleGroups;  // 显式指定内联元素数量
//     identifyFusibleGroups(funcOp, fusibleGroups);
    
//     // 2. 调试打印融合组信息
//     debugPrintFusibleGroups(fusibleGroups);
    
//     if (fusibleGroups.empty()) {
//       LLVM_DEBUG(llvm::dbgs() << "No fusible groups found\n");
//       return true;
//     }
    
//     // 3. 验证每个组的大小不超过batch size
//     for (const auto& group : fusibleGroups) {
//       if (group.operations.size() > static_cast<size_t>(batchSize)) {
//         LLVM_DEBUG(llvm::dbgs() << "ERROR: Group size " << group.operations.size() 
//                    << " exceeds batch size " << batchSize << "\n");
//         return false;
//       }
//     }
    
//     // 4. 按照第一个操作在IR中的位置对融合组进行排序
//     llvm::sort(fusibleGroups, [](const FusibleGroup& a, const FusibleGroup& b) {
//       if (a.operations.empty() || b.operations.empty()) return false;
//       return a.operations[0]->isBeforeInBlock(b.operations[0]);
//     });
    
//     // *** 5. 分析相邻组关系并设置优化标记 ***
//     analyzeAdjacentGroups(fusibleGroups);
    
//     // *** 删除：不再预先设置共享内存，改为动态获取 ***
    
//     // 6. 对每个可融合组执行fusion
//     for (size_t i = 0; i < fusibleGroups.size(); ++i) {
//       auto& group = fusibleGroups[i];
      
//       // *** 新增：动态设置共享输入内存 ***
//       if (!group.needInputCopy && i > 0) {
//         // 从前一个组获取共享输出内存作为当前组的输入
//         auto& prevGroup = fusibleGroups[i-1];
//         group.sharedBatchedInput = prevGroup.sharedBatchedOutput;
        
//         LLVM_DEBUG(llvm::dbgs() << "Group " << i << " will use shared memory from Group " << (i-1) << "\n");
//       }
      
//       LLVM_DEBUG(llvm::dbgs() << "Fusing group: layer " << group.layerIndex 
//                  << ", op type: " << group.opType 
//                  << ", " << group.operations.size() << " operations"
//                  << " (batch size limited to " << batchSize << ")"
//                  << " [InputCopy: " << (group.needInputCopy ? "YES" : "NO") 
//                  << ", OutputCopy: " << (group.needOutputCopy ? "YES" : "NO") << "]\n");
      
//       if (!fuseOperationGroup(funcOp, group)) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to fuse operation group\n");
//         return false;
//       }
//     }
    
//     // *** 7. 执行延迟删除操作 ***
//     performDeferredDeletion(fusibleGroups);
    
//     return true;
//   }
  
//   // *** 新增：执行延迟删除操作 ***
//   void performDeferredDeletion(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Performing Deferred Deletion ===\n");
    
//     for (auto& group : fusibleGroups) {
//       if (!group.pendingDeleteOps.empty()) {
//         LLVM_DEBUG(llvm::dbgs() << "Deleting " << group.pendingDeleteOps.size() 
//                    << " pending operations from group (layer " << group.layerIndex 
//                    << ", type " << group.opType << ")\n");
        
//         for (Operation* op : group.pendingDeleteOps) {
//           if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//             LLVM_DEBUG(llvm::dbgs() << "  Deleting: " << nodeNameAttr.getValue() << "\n");
//           }
//           op->erase();
//         }
//         group.pendingDeleteOps.clear();
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "=== Deferred Deletion Complete ===\n\n");
//   }
  
//   // *** 新增：分析相邻组关系的函数 ***
//   void analyzeAdjacentGroups(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Analyzing Adjacent Groups for Optimization ===\n");
    
//     for (size_t i = 0; i < fusibleGroups.size(); ++i) {
//       auto& currentGroup = fusibleGroups[i];
      
//       // 检查是否与前一个组相邻
//       if (i > 0) {
//         auto& prevGroup = fusibleGroups[i-1];
        
//         if (areAdjacentGroups(prevGroup, currentGroup)) {
//           // 设置优化标记
//           currentGroup.needInputCopy = false;
//           prevGroup.needOutputCopy = false;
          
//           // *** 新增：建立相邻组的依赖关系，用于延迟删除 ***
//           currentGroup.dependsOnPrevGroup = true;
//           prevGroup.hasNextGroupDependency = true;
          
//           LLVM_DEBUG(llvm::dbgs() << "Found adjacent groups: "
//                      << "Group " << (i-1) << " (layer " << prevGroup.layerIndex << ", " << prevGroup.opType << ") "
//                      << "-> Group " << i << " (layer " << currentGroup.layerIndex << ", " << currentGroup.opType << ")\n");
//           LLVM_DEBUG(llvm::dbgs() << "  Optimization: Skip output copy for group " << (i-1) 
//                      << " and input copy for group " << i << "\n");
//           LLVM_DEBUG(llvm::dbgs() << "  Dependency: Group " << i << " depends on Group " << (i-1) << "\n");
//         }
//       }
      
//       LLVM_DEBUG(llvm::dbgs() << "Group " << i << " optimization flags: "
//                  << "needInputCopy=" << currentGroup.needInputCopy 
//                  << ", needOutputCopy=" << currentGroup.needOutputCopy 
//                  << ", dependsOnPrev=" << currentGroup.dependsOnPrevGroup
//                  << ", hasNextDep=" << currentGroup.hasNextGroupDependency << "\n");
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "=== Adjacent Group Analysis Complete ===\n\n");
//   }
  
//   // *** 新增：检查两个组是否相邻的函数 ***
//   bool areAdjacentGroups(const FusibleGroup& group1, const FusibleGroup& group2) {
//     // 检查层次上是否相邻：下一层的索引应该是当前层索引+1
//     bool layerAdjacent = (group2.layerIndex == group1.layerIndex + 1);
    
//     // 检查时间步是否一致（相同的batch）
//     bool sameTimesteps = (group1.timesteps == group2.timesteps);
    
//     // 检查IR中位置是否相邻（允许一些中间的temporal操作）
//     bool irAdjacent = areAdjacentInIR(group1, group2);
    
//     bool result = layerAdjacent && sameTimesteps && irAdjacent;
    
//     LLVM_DEBUG(llvm::dbgs() << "Checking adjacency between groups (layer " << group1.layerIndex 
//                << " -> " << group2.layerIndex << "): "
//                << "layerAdjacent=" << layerAdjacent 
//                << ", sameTimesteps=" << sameTimesteps 
//                << ", irAdjacent=" << irAdjacent 
//                << " -> " << (result ? "ADJACENT" : "NOT ADJACENT") << "\n");
    
//     return result;
//   }
  
//   // *** 新增：检查两个组在IR中是否相邻 ***
//   bool areAdjacentInIR(const FusibleGroup& group1, const FusibleGroup& group2) {
//     if (group1.operations.empty() || group2.operations.empty()) {
//       return false;
//     }
    
//     // 找到group1的最后一个操作和group2的第一个操作
//     Operation* lastOp1 = group1.operations.back();
//     Operation* firstOp2 = group2.operations.front();
    
//     // 检查它们之间的距离，允许有一些temporal操作
//     int maxAllowedGap = 10; // 允许的最大间隔操作数
//     int gap = 0;
//     Operation* current = lastOp1->getNextNode();
    
//     while (current && current != firstOp2 && gap < maxAllowedGap) {
//       // 跳过常量和工具操作
//       if (!isConstantOrUtilityOp(current) && !isTemporalOp(current)) {
//         gap++;
//       }
//       current = current->getNextNode();
//     }
    
//     bool adjacent = (current == firstOp2 && gap < maxAllowedGap);
    
//     LLVM_DEBUG(llvm::dbgs() << "  IR adjacency check: gap=" << gap 
//                << ", found=" << (current == firstOp2) 
//                << " -> " << (adjacent ? "ADJACENT" : "NOT ADJACENT") << "\n");
    
//     return adjacent;
//   }
  
//   // *** 新增：在fusion之前设置共享内存 ***
//   void setupSharedMemory(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Setting up Shared Memory for Adjacent Groups ===\n");
    
//     for (size_t i = 1; i < fusibleGroups.size(); ++i) {
//       auto& currentGroup = fusibleGroups[i];
//       auto& prevGroup = fusibleGroups[i-1];
      
//       // 如果当前组不需要输入复制，说明它可以复用前一个组的输出
//       if (!currentGroup.needInputCopy) {
//         // 前一个组的输出将作为当前组的输入
//         currentGroup.sharedBatchedInput = prevGroup.sharedBatchedOutput;
        
//         LLVM_DEBUG(llvm::dbgs() << "Group " << i << " will reuse output memory from Group " << (i-1) << "\n");
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "=== Shared Memory Setup Complete ===\n\n");
//   }
  
//   // 修改identifyFusibleGroups函数，确保正确的分组和相邻性检查
//   void identifyFusibleGroups(func::FuncOp funcOp, SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     // 按 (layer_index, normalized_op_type) 收集所有可融合的操作
//     std::map<std::pair<int, std::string>, SmallVector<std::pair<Operation*, int>>> tempGroupMap;
    
//     funcOp.walk([&](Operation* op) {
//       if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//         StringRef nodeName = nodeNameAttr.getValue();
        
//         if (nodeName.contains("/layer/")) {
//           SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName, op); // 传递op指针
          
//           // 检查是否是可融合的操作类型
//           if (isFusibleOperationType(layerInfo.opName)) {
//             // 标准化操作类型：将Gemm(bias=0)统一为MatMul处理
//             std::string normalizedOpType = layerInfo.opName;
//             if (layerInfo.opName == "Gemm") {
//               auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
//               if (gemmOp && isBiasZero(gemmOp.getC())) {
//                 normalizedOpType = "MatMul";  // bias=0的Gemm当作MatMul处理
//                 LLVM_DEBUG(llvm::dbgs() << "Gemm with zero bias treated as MatMul: " << nodeName << "\n");
//               }
//             }
            
//             auto key = std::make_pair(layerInfo.layerIndex, normalizedOpType);
//             tempGroupMap[key].emplace_back(op, layerInfo.timestep);
            
//             LLVM_DEBUG(llvm::dbgs() << "Added to temp group: " << nodeName 
//                       << " (layer " << layerInfo.layerIndex 
//                       << ", timestep " << layerInfo.timestep 
//                       << ", actual op: " << op->getName().getStringRef()
//                       << ", normalized type: " << normalizedOpType << ")\n");
//           }
//         }
//       }
//     });

    
//     // 按batch size将大的组拆分成小的组，并验证相邻性
//     for (auto& [key, opTimestepPairs] : tempGroupMap) {
//       if (opTimestepPairs.size() <= 1) {
//         continue; // 单个操作无需融合
//       }
      
//       // 按时间步排序
//       llvm::sort(opTimestepPairs, [](const auto& a, const auto& b) {
//         return a.second < b.second;
//       });
      
//       LLVM_DEBUG(llvm::dbgs() << "Processing layer " << key.first 
//                  << ", type " << key.second 
//                  << " with " << opTimestepPairs.size() << " operations\n");
      
//       // 按batch size分组，同时确保分组内操作的相邻性
//       for (size_t i = 0; i < opTimestepPairs.size(); i += batchSize) {
//         size_t groupSize = std::min(static_cast<size_t>(batchSize), 
//                                     opTimestepPairs.size() - i);
        
//         if (groupSize > 1) { // 只有多于1个操作的组才需要融合
//           SmallVector<std::pair<Operation*, int>> candidateGroup;
//           for (size_t j = i; j < i + groupSize; j++) {
//             candidateGroup.push_back(opTimestepPairs[j]);
//           }
          
//           // 验证候选组是否满足相邻性要求（重排后的特性）
//           if (validateGroupAdjacency(candidateGroup)) {
//             FusibleGroup group(key.first, key.second);
            
//             for (const auto& pair : candidateGroup) {
//               group.operations.push_back(pair.first);
//               group.timesteps.push_back(pair.second);
//             }
            
//             fusibleGroups.push_back(std::move(group));
            
//             LLVM_DEBUG(llvm::dbgs() << "Created valid fusible group: layer " << key.first 
//                        << ", type " << key.second 
//                        << ", " << candidateGroup.size() << " ops"
//                        << " (timesteps " << candidateGroup[0].second 
//                        << " to " << candidateGroup.back().second << ")\n");
//           } else {
//             LLVM_DEBUG(llvm::dbgs() << "Rejected non-adjacent group: layer " << key.first 
//                        << ", type " << key.second 
//                        << " (timesteps " << candidateGroup[0].second 
//                        << " to " << candidateGroup.back().second << ")\n");
            
//             // 如果组不相邻，尝试逐个添加相邻的操作
//             createAdjacentSubgroups(candidateGroup, key.first, key.second, fusibleGroups);
//           }
//         }
//       }
//     }
//   }
  
//   // 验证操作组是否在IR中相邻（利用重排后的特性）
//   bool validateGroupAdjacency(const SmallVector<std::pair<Operation*, int>>& candidateGroup) {
//     if (candidateGroup.size() <= 1) {
//       return true;
//     }
    
//     // 按IR中的顺序排序
//     SmallVector<Operation*> sortedOps;
//     for (const auto& pair : candidateGroup) {
//       sortedOps.push_back(pair.first);
//     }
//     llvm::sort(sortedOps, [](Operation* a, Operation* b) {
//       return a->isBeforeInBlock(b);
//     });
    
//     // 检查操作是否基本相邻（允许中间有少量非融合操作）
//     Operation* prevOp = sortedOps[0];
//     int maxGap = 5; // 允许的最大间隔操作数
    
//     for (size_t i = 1; i < sortedOps.size(); ++i) {
//       Operation* currentOp = sortedOps[i];
//       int gap = 0;
//       Operation* checkOp = prevOp->getNextNode();
      
//       // 计算两个操作之间的距离
//       while (checkOp && checkOp != currentOp && gap < maxGap) {
//         // 跳过常量和时序操作
//         if (!isConstantOrUtilityOp(checkOp) && !isTemporalOp(checkOp)) {
//           gap++;
//         }
//         checkOp = checkOp->getNextNode();
//       }
      
//       if (gap >= maxGap || !checkOp) {
//         LLVM_DEBUG(llvm::dbgs() << "Operations too far apart (gap: " << gap << ")\n");
//         return false;
//       }
      
//       prevOp = currentOp;
//     }
    
//     return true;
//   }
  
//   // 检查是否是时序操作（非融合操作）
//   bool isTemporalOp(Operation* op) {
//     if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//       StringRef nodeName = nodeNameAttr.getValue();
//       if (nodeName.contains("/layer/")) {
//         SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName);
//         // 时序操作通常包括激活函数、归一化等
//         return !isFusibleOperationType(layerInfo.opName) && 
//                (layerInfo.opName.find("Div") != std::string::npos ||
//                 layerInfo.opName.find("Add") != std::string::npos ||
//                 layerInfo.opName.find("Sub") != std::string::npos ||
//                 layerInfo.opName.find("Mul") != std::string::npos ||
//                 layerInfo.opName.find("Cast") != std::string::npos ||
//                 layerInfo.opName.find("Neg") != std::string::npos ||
//                 layerInfo.opName.find("GreaterOrEqual") != std::string::npos ||
//                 layerInfo.opName.find("Flatten") != std::string::npos);
//       }
//     }
//     return false;
//   }
  
//   // 创建相邻的子组
//   void createAdjacentSubgroups(const SmallVector<std::pair<Operation*, int>>& candidateGroup,
//                                int layerIndex, const std::string& opType,
//                                SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     // 按IR顺序排序
//     SmallVector<std::pair<Operation*, int>> sortedGroup = candidateGroup;
//     llvm::sort(sortedGroup, [](const auto& a, const auto& b) {
//       return a.first->isBeforeInBlock(b.first);
//     });
    
//     // 找到连续的相邻操作子序列
//     SmallVector<std::pair<Operation*, int>> currentSubgroup;
    
//     for (size_t i = 0; i < sortedGroup.size(); ++i) {
//       if (currentSubgroup.empty()) {
//         currentSubgroup.push_back(sortedGroup[i]);
//       } else {
//         // 检查与当前子组最后一个操作的相邻性
//         SmallVector<std::pair<Operation*, int>> testGroup = currentSubgroup;
//         testGroup.push_back(sortedGroup[i]);
        
//         if (validateGroupAdjacency(testGroup)) {
//           currentSubgroup.push_back(sortedGroup[i]);
//         } else {
//           // 当前子组结束，创建融合组（如果有多个操作）
//           if (currentSubgroup.size() > 1) {
//             FusibleGroup group(layerIndex, opType);
//             for (const auto& pair : currentSubgroup) {
//               group.operations.push_back(pair.first);
//               group.timesteps.push_back(pair.second);
//             }
//             fusibleGroups.push_back(std::move(group));
            
//             LLVM_DEBUG(llvm::dbgs() << "Created adjacent subgroup: layer " << layerIndex 
//                        << ", type " << opType 
//                        << ", " << currentSubgroup.size() << " ops\n");
//           }
          
//           // 开始新的子组
//           currentSubgroup.clear();
//           currentSubgroup.push_back(sortedGroup[i]);
//         }
//       }
//     }
    
//     // 处理最后一个子组
//     if (currentSubgroup.size() > 1) {
//       FusibleGroup group(layerIndex, opType);
//       for (const auto& pair : currentSubgroup) {
//         group.operations.push_back(pair.first);
//         group.timesteps.push_back(pair.second);
//       }
//       fusibleGroups.push_back(std::move(group));
      
//       LLVM_DEBUG(llvm::dbgs() << "Created final adjacent subgroup: layer " << layerIndex 
//                  << ", type " << opType 
//                  << ", " << currentSubgroup.size() << " ops\n");
//     }
//   }
  
//   // 检查是否是可融合的操作类型 - 修复版本
//   bool isFusibleOperationType(const std::string& opName) {
//     return opName == "Conv" || 
//            opName == "MaxPool" ||           // 修复：原来是MaxPoolSingleOut
//            opName == "MaxPoolSingleOut" ||  // 保留原有的，以防有其他变体
//            opName == "Gemm" || 
//            opName == "MatMul";
//   }
  
//   // 修改操作组融合函数，确保正确的插入位置和支配关系
//   bool fuseOperationGroup(func::FuncOp funcOp, FusibleGroup& group) {
//     if (group.operations.empty()) {
//       return true;
//     }
    
//     // 按照IR中的顺序对操作进行排序，确保正确的支配关系
//     llvm::sort(group.operations, [](Operation* a, Operation* b) {
//       return a->isBeforeInBlock(b);
//     });
    
//     // 找到所有操作的最后一个操作，作为插入点
//     Operation* lastOp = group.operations.back();
    
//     // 同时要确保所有输入操作都在插入点之前定义
//     Operation* insertionPoint = lastOp;
//     for (Operation* op : group.operations) {
//       for (Value operand : op->getOperands()) {
//         if (Operation* defOp = operand.getDefiningOp()) {
//           if (insertionPoint->isBeforeInBlock(defOp)) {
//             insertionPoint = defOp;
//           }
//         }
//       }
//     }
    
//     OpBuilder builder(funcOp.getContext());
//     builder.setInsertionPointAfter(insertionPoint);
    
//     LLVM_DEBUG(llvm::dbgs() << "Fusing group with " << group.operations.size() 
//                << " operations of type: " << group.opType 
//                << " (batch size: " << group.operations.size() << ")\n");
    
//     // 根据标准化后的操作类型执行不同的融合策略
//     if (group.opType == "Conv") {
//       return fuseConvOperations(builder, group);
//     } else if (group.opType == "MaxPool" || group.opType == "MaxPoolSingleOut") {
//       return fuseMaxPoolOperations(builder, group);
//     } else if (group.opType == "MatMul") {
//       // 这里处理纯MatMul或者Gemm(bias=0)的融合
//       return fuseMixedMatMulOperations(builder, group);
//     } else if (group.opType == "Gemm") {
//       // 这里处理有非零bias的Gemm融合
//       return fuseGemmOperations(builder, group);
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Unknown operation type for fusion: " << group.opType << "\n");
//     return false;
//   }
  

//   // *** 修改：融合Conv操作 - 支持相邻组优化 ***
//   bool fuseConvOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Conv operations with optimizations\n");
//     LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
//                << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
//     Operation* firstOp = group.operations[0];
//     auto firstConv = dyn_cast<mlir::ONNXConvOp>(firstOp);
//     if (!firstConv) {
//       LLVM_DEBUG(llvm::dbgs() << "First operation is not a Conv op\n");
//       return false;
//     }
    
//     // 获取输入类型信息
//     auto inputType = firstConv.getX().getType().cast<RankedTensorType>();
//     auto weightType = firstConv.getW().getType().cast<RankedTensorType>();
//     auto outputType = firstConv.getY().getType().cast<RankedTensorType>();
    
//     ArrayRef<int64_t> inputShape = inputType.getShape();
//     ArrayRef<int64_t> outputShape = outputType.getShape();
    
//     // 计算融合后的batch大小
//     int64_t originalBatchSize = inputShape[0];
//     int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
//     // 创建融合后的形状
//     SmallVector<int64_t> fusedInputShape(inputShape.begin(), inputShape.end());
//     SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
//     fusedInputShape[0] = fusedBatchSize;
//     fusedOutputShape[0] = fusedBatchSize;
    
//     auto fusedInputType = RankedTensorType::get(fusedInputShape, inputType.getElementType());
//     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
//     // 创建memref类型用于GPU操作
//     auto inputMemRefType = MemRefType::get(fusedInputShape, inputType.getElementType());
//     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
//     auto originalInputMemRefType = MemRefType::get(inputShape, inputType.getElementType());
//     auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    
//     // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
//     Value batchedInput;
//     if (group.needInputCopy) {
//       // 正常情况：分配新的输入内存
//       batchedInput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), inputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
//     } else {
//       // 优化情况：复用前一个组的输出内存作为输入
//       batchedInput = group.sharedBatchedInput;
      
//       // *** 新增：类型兼容性检查 ***
//       bool typeCompatible = false;
//       if (batchedInput) {
//         auto sharedType = batchedInput.getType().dyn_cast<MemRefType>();
//         if (sharedType && sharedType.getShape() == inputMemRefType.getShape() && 
//             sharedType.getElementType() == inputMemRefType.getElementType()) {
//           typeCompatible = true;
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
//         } else {
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed - shared type vs required type mismatch\n");
//           if (sharedType) {
//             LLVM_DEBUG(llvm::dbgs() << "  Shared shape: [");
//             for (size_t i = 0; i < sharedType.getShape().size(); ++i) {
//               LLVM_DEBUG(llvm::dbgs() << sharedType.getShape()[i]);
//               if (i < sharedType.getShape().size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//             }
//             LLVM_DEBUG(llvm::dbgs() << "]\n");
//             LLVM_DEBUG(llvm::dbgs() << "  Required shape: [");
//             for (size_t i = 0; i < inputMemRefType.getShape().size(); ++i) {
//               LLVM_DEBUG(llvm::dbgs() << inputMemRefType.getShape()[i]);
//               if (i < inputMemRefType.getShape().size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
//             }
//             LLVM_DEBUG(llvm::dbgs() << "]\n");
//           }
//         }
//       }
      
//       if (!batchedInput || !typeCompatible) {
//         LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
//         batchedInput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), inputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         // 重置为需要输入复制，因为我们现在需要分配新内存
//         group.needInputCopy = true;
//       }
//     }
    
//     // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
//     Value batchedOutput;
//     if (group.needOutputCopy) {
//       // 正常情况：分配新的输出内存
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
//     } else {
//       // 优化情况：分配输出内存，但会被下一个组复用
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       // 设置为下一个组的共享输入（这个在analyzeAdjacentGroups中处理）
//       group.sharedBatchedOutput = batchedOutput;
//       LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
//     }
    
//     // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
//     SmallVector<Value> individualOutputs;
//     if (group.needOutputCopy) {
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
//       LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
//     }
    
//     // 创建输入和输出子视图
//     SmallVector<Value> inputViews;
//     SmallVector<Value> outputViews;
    
//     // *** 新增：验证 batchedInput 的有效性 ***
//     if (!batchedInput) {
//       LLVM_DEBUG(llvm::dbgs() << "ERROR: batchedInput is invalid, cannot create subviews\n");
//       return false;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Creating subviews for " << group.operations.size() << " operations\n");
//     LLVM_DEBUG(llvm::dbgs() << "Input shape: [" << fusedInputShape[0] << ", " << fusedInputShape[1] 
//                << ", " << fusedInputShape[2] << ", " << fusedInputShape[3] << "]\n");
//     LLVM_DEBUG(llvm::dbgs() << "Original batch size: " << originalBatchSize << "\n");
    
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       // 计算当前操作在batch中的偏移
//       int64_t offset = i * originalBatchSize;
      
//       LLVM_DEBUG(llvm::dbgs() << "Creating subview " << i << " with offset " << offset << "\n");
      
//       // 创建输入子视图
//       SmallVector<OpFoldResult> inputOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> inputSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(inputShape[1]),
//           builder.getI64IntegerAttr(inputShape[2]),
//           builder.getI64IntegerAttr(inputShape[3])
//       };
//       SmallVector<OpFoldResult> inputStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value inputView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedInput, inputOffsets, inputSizes, inputStrides);
//       inputViews.push_back(inputView);
      
//       // 创建输出子视图
//       SmallVector<OpFoldResult> outputOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> outputSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(outputShape[1]),
//           builder.getI64IntegerAttr(outputShape[2]),
//           builder.getI64IntegerAttr(outputShape[3])
//       };
//       SmallVector<OpFoldResult> outputStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value outputView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
//       outputViews.push_back(outputView);
      
//       LLVM_DEBUG(llvm::dbgs() << "Created subviews for operation " << i << "\n");
//     }
    
//     // *** 修改：根据needInputCopy决定是否执行输入数据复制 ***
//     // 注意：needInputCopy 可能在类型兼容性检查后被动态修改
//     if (group.needInputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel)\n");
      
//       // 1. 首先一起创建所有需要的异步流
//       SmallVector<Value> copyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         copyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有必要的类型转换
//       SmallVector<Value> inputMemRefs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* currentOp = group.operations[i];
//         Value originalInput = currentOp->getOperand(0);
        
//         Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), 
//             originalInputMemRefType,
//             originalInput).getResult(0);
//         inputMemRefs.push_back(inputMemRef);
//       }
      
//       // 3. 执行所有并行复制操作
//       SmallVector<Value> inputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(copyStreams[i]),
//             inputViews[i], inputMemRefs[i]);  // 目标，源
//         inputCopyTokens.push_back(copyToken.getAsyncToken());
        
//         LLVM_DEBUG(llvm::dbgs() << "Initiated parallel input copy for operation " << i << "\n");
//       }
      
//       // 4. 等待所有输入复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
//     }
    
//     // 转换batched input为tensor用于ONNX操作
//     Value batchedInputTensor = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), fusedInputType, batchedInput).getResult(0);
    
//     // 创建融合的Conv操作
//     Value fusedResult = builder.create<mlir::ONNXConvOp>(
//         builder.getUnknownLoc(), 
//         fusedOutputType,
//         batchedInputTensor,
//         firstConv.getW(),  // 权重可以共享
//         firstConv.getB(),  // bias可以共享
//         firstConv.getAutoPadAttr(),
//         firstConv.getDilationsAttr(),
//         firstConv.getGroupAttr(),
//         firstConv.getKernelShapeAttr(),
//         firstConv.getPadsAttr(),
//         firstConv.getStridesAttr());
    
//     // 转换融合结果为memref
//     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
//     LLVM_DEBUG(llvm::dbgs() << "Created fused Conv operation\n");
    
//     // 将融合结果复制到预分配的batched output memory
//     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
//         builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()),
//         ValueRange(fusedCopyStream.getAsyncToken()),
//         batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
//     LLVM_DEBUG(llvm::dbgs() << "Copied fused Conv result to batched output memory\n");
    
//     // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
//     if (group.needOutputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
//       // 1. 首先一起创建所有需要的输出异步流
//       SmallVector<Value> outputCopyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         outputCopyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有并行输出复制操作
//       SmallVector<Value> outputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(outputCopyStreams[i]),
//             individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//         outputCopyTokens.push_back(copyToken.getAsyncToken());
//       }
      
//       // 3. 等待所有输出复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
      
//       // 4. 转换回tensor
//       SmallVector<Value> splitResults;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
//         splitResults.push_back(splitResult);
//       }
      
//       // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         op->getResult(0).replaceAllUsesWith(splitResults[i]);
//         LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//       }
      
//       // 6. 立即删除操作（因为已经完成替换）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << "\n");
//         op->erase();
//       }
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
//       // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
//       if (group.hasNextGroupDependency) {
//         LLVM_DEBUG(llvm::dbgs() << "Adding operations to pending deletion (next group depends on this)\n");
//         // 将操作添加到待删除列表，延迟到所有fusion完成后删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           group.pendingDeleteOps.push_back(op);
//           LLVM_DEBUG(llvm::dbgs() << "Added operation " << i << " to pending deletion\n");
//         }
//       } else {
//         LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting operations immediately\n");
//         // 没有下一个组依赖，可以立即删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << " (no dependency)\n");
//           op->erase();
//         }
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused Conv operations with optimizations\n");
//     return true;
//   }

//   // *** 修改：融合MaxPool操作 - 支持相邻组优化（类似Conv的逻辑）***
//   bool fuseMaxPoolOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MaxPool operations with optimizations\n");
//     LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
//                << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
//     Operation* firstOp = group.operations[0];
//     auto firstMaxPool = dyn_cast<mlir::ONNXMaxPoolSingleOutOp>(firstOp);
//     if (!firstMaxPool) {
//       LLVM_DEBUG(llvm::dbgs() << "First operation is not a MaxPool op\n");
//       return false;
//     }
    
//     auto inputType = firstMaxPool.getX().getType().cast<RankedTensorType>();
//     auto outputType = firstMaxPool.getResult().getType().cast<RankedTensorType>();
    
//     ArrayRef<int64_t> inputShape = inputType.getShape();
//     ArrayRef<int64_t> outputShape = outputType.getShape();
    
//     int64_t originalBatchSize = inputShape[0];
//     int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
//     SmallVector<int64_t> fusedInputShape(inputShape.begin(), inputShape.end());
//     SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
//     fusedInputShape[0] = fusedBatchSize;
//     fusedOutputShape[0] = fusedBatchSize;
    
//     auto fusedInputType = RankedTensorType::get(fusedInputShape, inputType.getElementType());
//     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
//     auto inputMemRefType = MemRefType::get(fusedInputShape, inputType.getElementType());
//     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
//     auto originalInputMemRefType = MemRefType::get(inputShape, inputType.getElementType());
//     auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    
//     // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
//     Value batchedInput;
//     if (group.needInputCopy) {
//       batchedInput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), inputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
//     } else {
//       batchedInput = group.sharedBatchedInput;
      
//       // *** 新增：类型兼容性检查 ***
//       bool typeCompatible = false;
//       if (batchedInput) {
//         auto sharedType = batchedInput.getType().dyn_cast<MemRefType>();
//         if (sharedType && sharedType.getShape() == inputMemRefType.getShape() && 
//             sharedType.getElementType() == inputMemRefType.getElementType()) {
//           typeCompatible = true;
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
//         } else {
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed for MaxPool\n");
//         }
//       }
      
//       if (!batchedInput || !typeCompatible) {
//         LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
//         batchedInput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), inputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         group.needInputCopy = true;
//       }
//     }
    
//     // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
//     Value batchedOutput;
//     if (group.needOutputCopy) {
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
//     } else {
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       group.sharedBatchedOutput = batchedOutput;
//       LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
//     }
    
//     // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
//     SmallVector<Value> individualOutputs;
//     if (group.needOutputCopy) {
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
//       LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
//     }
    
//     // 创建输入和输出子视图
//     SmallVector<Value> inputViews;
//     SmallVector<Value> outputViews;
    
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       int64_t offset = i * originalBatchSize;
      
//       // 输入子视图
//       SmallVector<OpFoldResult> inputOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> inputSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(inputShape[1]),
//           builder.getI64IntegerAttr(inputShape[2]),
//           builder.getI64IntegerAttr(inputShape[3])
//       };
//       SmallVector<OpFoldResult> inputStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value inputView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedInput, inputOffsets, inputSizes, inputStrides);
//       inputViews.push_back(inputView);
      
//       // 输出子视图
//       SmallVector<OpFoldResult> outputOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> outputSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(outputShape[1]),
//           builder.getI64IntegerAttr(outputShape[2]),
//           builder.getI64IntegerAttr(outputShape[3])
//       };
//       SmallVector<OpFoldResult> outputStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value outputView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
//       outputViews.push_back(outputView);
//     }
    
//     // *** 修改：根据needInputCopy决定是否执行输入数据复制 ***
//     // 注意：needInputCopy 可能在类型兼容性检查后被动态修改
//     if (group.needInputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel)\n");
      
//       // 1. 首先一起创建所有需要的异步流
//       SmallVector<Value> copyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         copyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有必要的类型转换
//       SmallVector<Value> inputMemRefs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* currentOp = group.operations[i];
//         Value originalInput = currentOp->getOperand(0);
        
//         Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), 
//             originalInputMemRefType,
//             originalInput).getResult(0);
//         inputMemRefs.push_back(inputMemRef);
//       }
      
//       // 3. 执行所有并行复制操作
//       SmallVector<Value> inputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(copyStreams[i]),
//             inputViews[i], inputMemRefs[i]);  // 目标，源
//         inputCopyTokens.push_back(copyToken.getAsyncToken());
//       }
      
//       // 4. 等待所有输入复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
//     }
    
//     // 创建融合的MaxPool操作
//     Value batchedInputTensor = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), fusedInputType, batchedInput).getResult(0);
    
//     Value fusedResult = builder.create<mlir::ONNXMaxPoolSingleOutOp>(
//         builder.getUnknownLoc(),
//         fusedOutputType,
//         batchedInputTensor,
//         firstMaxPool.getAutoPadAttr(),
//         firstMaxPool.getCeilModeAttr(),
//         firstMaxPool.getDilationsAttr(),
//         firstMaxPool.getKernelShapeAttr(),
//         firstMaxPool.getPadsAttr(),
//         firstMaxPool.getStorageOrderAttr(),
//         firstMaxPool.getStridesAttr());
    
//     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
//     LLVM_DEBUG(llvm::dbgs() << "Created fused MaxPool operation\n");
    
//     // 将融合结果复制到预分配的batched output memory
//     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
//         builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()),
//         ValueRange(fusedCopyStream.getAsyncToken()),
//         batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
//     LLVM_DEBUG(llvm::dbgs() << "Copied fused MaxPool result to batched output memory\n");
    
//     // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
//     if (group.needOutputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
//       // 1. 首先一起创建所有需要的输出异步流
//       SmallVector<Value> outputCopyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         outputCopyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有并行输出复制操作
//       SmallVector<Value> outputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(outputCopyStreams[i]),
//             individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//         outputCopyTokens.push_back(copyToken.getAsyncToken());
//       }
      
//       // 3. 等待所有输出复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
      
//       // 4. 转换回tensor
//       SmallVector<Value> splitResults;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
//         splitResults.push_back(splitResult);
//       }
      
//       // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         op->getResult(0).replaceAllUsesWith(splitResults[i]);
//         LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//       }
      
//       // 6. 立即删除操作（因为已经完成替换）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << "\n");
//         op->erase();
//       }
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
//       // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
//       if (group.hasNextGroupDependency) {
//         LLVM_DEBUG(llvm::dbgs() << "Adding MaxPool operations to pending deletion (next group depends on this)\n");
//         // 将操作添加到待删除列表，延迟到所有fusion完成后删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           group.pendingDeleteOps.push_back(op);
//           LLVM_DEBUG(llvm::dbgs() << "Added MaxPool operation " << i << " to pending deletion\n");
//         }
//       } else {
//         LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting MaxPool operations immediately\n");
//         // 没有下一个组依赖，可以立即删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << " (no dependency)\n");
//           op->erase();
//         }
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused MaxPool operations with optimizations\n");
//     return true;
//   }

//   // *** 修改：混合MatMul操作融合 - 支持相邻组优化（类似Conv的逻辑）***
//   bool fuseMixedMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " mixed MatMul operations with optimizations\n");
//     LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
//                << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
//     Operation* firstOp = group.operations[0];
    
//     // 确定是否第一个操作是MatMul还是Gemm
//     auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
//     auto firstGemm = dyn_cast<mlir::ONNXGemmOp>(firstOp);
    
//     if (!firstMatMul && !firstGemm) {
//       LLVM_DEBUG(llvm::dbgs() << "First operation is neither MatMul nor Gemm\n");
//       return false;
//     }
    
//     // 获取输入输出类型 - 统一从MatMul或Gemm中获取
//     Value firstA, firstB;
//     Type outputType;
    
//     if (firstMatMul) {
//       firstA = firstMatMul.getA();
//       firstB = firstMatMul.getB();
//       outputType = firstMatMul.getY().getType();
//     } else {
//       firstA = firstGemm.getA();
//       firstB = firstGemm.getB();
//       outputType = firstGemm.getY().getType();
//     }
    
//     auto aType = firstA.getType().cast<RankedTensorType>();
//     auto bType = firstB.getType().cast<RankedTensorType>();
//     auto outputTensorType = outputType.cast<RankedTensorType>();
    
//     ArrayRef<int64_t> aShape = aType.getShape();
//     ArrayRef<int64_t> outputShape = outputTensorType.getShape();
    
//     int64_t originalBatchSize = aShape[0];
//     int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
//     SmallVector<int64_t> fusedAShape(aShape.begin(), aShape.end());
//     SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
//     fusedAShape[0] = fusedBatchSize;
//     fusedOutputShape[0] = fusedBatchSize;
    
//     auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
//     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputTensorType.getElementType());
    
//     auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
//     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputTensorType.getElementType());
//     auto originalAMemRefType = MemRefType::get(aShape, aType.getElementType());
//     auto originalOutputMemRefType = MemRefType::get(outputShape, outputTensorType.getElementType());
    
//     // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
//     Value batchedA;
//     if (group.needInputCopy) {
//       batchedA = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), aMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
//     } else {
//       batchedA = group.sharedBatchedInput;
      
//       // *** 新增：类型兼容性检查 ***
//       bool typeCompatible = false;
//       if (batchedA) {
//         auto sharedType = batchedA.getType().dyn_cast<MemRefType>();
//         if (sharedType && sharedType.getShape() == aMemRefType.getShape() && 
//             sharedType.getElementType() == aMemRefType.getElementType()) {
//           typeCompatible = true;
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
//         } else {
//           LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed for MatMul\n");
//         }
//       }
      
//       if (!batchedA || !typeCompatible) {
//         LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
//         batchedA = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), aMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         group.needInputCopy = true;
//       }
//     }
    
//     // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
//     Value batchedOutput;
//     if (group.needOutputCopy) {
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
//     } else {
//       batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
//       group.sharedBatchedOutput = batchedOutput;
//       LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
//     }
    
//     // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
//     SmallVector<Value> individualOutputs;
//     if (group.needOutputCopy) {
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
//       LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
//     }
    
//     // 创建输入和输出子视图
//     SmallVector<Value> aViews;
//     SmallVector<Value> outputViews;
    
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       int64_t offset = i * originalBatchSize;
      
//       // A矩阵子视图
//       SmallVector<OpFoldResult> aOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> aSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(aShape[1])
//       };
//       SmallVector<OpFoldResult> aStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value aView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedA, aOffsets, aSizes, aStrides);
//       aViews.push_back(aView);
      
//       // 输出子视图
//       SmallVector<OpFoldResult> outputOffsets = {
//           builder.getI64IntegerAttr(offset),
//           builder.getI64IntegerAttr(0)
//       };
//       SmallVector<OpFoldResult> outputSizes = {
//           builder.getI64IntegerAttr(originalBatchSize),
//           builder.getI64IntegerAttr(outputShape[1])
//       };
//       SmallVector<OpFoldResult> outputStrides = {
//           builder.getI64IntegerAttr(1),
//           builder.getI64IntegerAttr(1)
//       };
      
//       Value outputView = builder.create<memref::SubViewOp>(
//           builder.getUnknownLoc(), batchedOutput, outputOffsets, outputSizes, outputStrides);
//       outputViews.push_back(outputView);
//     }
    
//     // *** 修改：根据needInputCopy决定是否执行输入数据复制 - 处理混合操作 ***
//     if (group.needInputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel, mixed operations)\n");
      
//       // 1. 首先一起创建所有需要的异步流
//       SmallVector<Value> copyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         copyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有必要的类型转换 - 处理混合操作
//       SmallVector<Value> aMemRefs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* currentOp = group.operations[i];
//         Value originalA;
        
//         // 从MatMul或Gemm中获取A矩阵
//         if (auto matmulOp = dyn_cast<mlir::ONNXMatMulOp>(currentOp)) {
//           originalA = matmulOp.getA();
//         } else if (auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(currentOp)) {
//           originalA = gemmOp.getA();
//         } else {
//           LLVM_DEBUG(llvm::dbgs() << "Unsupported operation type in mixed fusion\n");
//           return false;
//         }
        
//         Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), 
//             originalAMemRefType,
//             originalA).getResult(0);
//         aMemRefs.push_back(aMemRef);
//       }
      
//       // 3. 执行所有并行复制操作
//       SmallVector<Value> inputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(copyStreams[i]),
//             aViews[i], aMemRefs[i]);  // 目标，源
//         inputCopyTokens.push_back(copyToken.getAsyncToken());
//       }
      
//       // 4. 等待所有输入复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed for mixed operations\n");
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
//     }
    
//     // 创建融合的MatMul操作（统一使用MatMul，即使原来有Gemm）
//     Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
    
//     Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
//         builder.getUnknownLoc(),
//         fusedOutputType,
//         batchedATensor,
//         firstB);  // B矩阵可以共享
    
//     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
//     LLVM_DEBUG(llvm::dbgs() << "Created fused MatMul operation for mixed group\n");
    
//     // 将融合结果复制到预分配的batched output memory
//     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
//         builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()),
//         ValueRange(fusedCopyStream.getAsyncToken()),
//         batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
//     LLVM_DEBUG(llvm::dbgs() << "Copied fused mixed MatMul result to batched output memory\n");
    
//     // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
//     if (group.needOutputCopy) {
//       LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
//       // 1. 首先一起创建所有需要的输出异步流
//       SmallVector<Value> outputCopyStreams;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//         outputCopyStreams.push_back(copyStream.getAsyncToken());
//       }
      
//       // 2. 执行所有并行输出复制操作
//       SmallVector<Value> outputCopyTokens;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         auto copyToken = builder.create<gpu::MemcpyOp>(
//             builder.getUnknownLoc(), 
//             gpu::AsyncTokenType::get(builder.getContext()),
//             ValueRange(outputCopyStreams[i]),
//             individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//         outputCopyTokens.push_back(copyToken.getAsyncToken());
//       }
      
//       // 3. 等待所有输出复制完成
//       builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//       LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed for mixed operations\n");
      
//       // 4. 转换回tensor
//       SmallVector<Value> splitResults;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//             builder.getUnknownLoc(), outputTensorType, individualOutputs[i]).getResult(0);
//         splitResults.push_back(splitResult);
//       }
      
//       // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         op->getResult(0).replaceAllUsesWith(splitResults[i]);
//         LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//       }
      
//       // 6. 立即删除操作（因为已经完成替换）
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Operation* op = group.operations[i];
//         LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << "\n");
//         op->erase();
//       }
//     } else {
//       LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
//       // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
//       if (group.hasNextGroupDependency) {
//         LLVM_DEBUG(llvm::dbgs() << "Adding mixed MatMul operations to pending deletion (next group depends on this)\n");
//         // 将操作添加到待删除列表，延迟到所有fusion完成后删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           group.pendingDeleteOps.push_back(op);
//           LLVM_DEBUG(llvm::dbgs() << "Added mixed MatMul operation " << i << " to pending deletion\n");
//         }
//       } else {
//         LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting mixed MatMul operations immediately\n");
//         // 没有下一个组依赖，可以立即删除
//         for (size_t i = 0; i < group.operations.size(); ++i) {
//           Operation* op = group.operations[i];
//           LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << " (no dependency)\n");
//           op->erase();
//         }
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused mixed MatMul operations with optimizations\n");
//     return true;
//   }


//   // 检查bias是否为0 - 修复版本
//   bool isBiasZero(Value bias) {
//     if (!bias) {
//       return true;  // 没有bias也算zero bias
//     }
    
//     if (auto constOp = bias.getDefiningOp<mlir::ONNXConstantOp>()) {
//       auto valueAttr = constOp.getValue();
//       if (valueAttr.has_value()) {
//         if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr.value())) {
//           if (denseAttr.isSplat()) {
//             auto splatValue = denseAttr.getSplatValue<APFloat>();
//             // 使用isPosZero()和isNegZero()来检查正零和负零
//             return splatValue.isPosZero() || splatValue.isNegZero();
//           }
//           // 检查是否所有元素都是0（包括-0.0）
//           for (auto value : denseAttr.getValues<APFloat>()) {
//             if (!value.isPosZero() && !value.isNegZero()) {
//               return false;
//             }
//           }
//           return true;
//         }
//       }
//     }
    
//     // 检查是否是0常量
//     if (auto constOp = bias.getDefiningOp<arith::ConstantOp>()) {
//       if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>()) {
//         APFloat floatValue = floatAttr.getValue();
//         return floatValue.isPosZero() || floatValue.isNegZero();
//       }
//     }
    
//     return false;
//   }

//   // 融合Gemm操作
//   bool fuseGemmOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Gemm operations\n");
    
//     Operation* firstOp = group.operations[0];
//     auto firstGemm = dyn_cast<mlir::ONNXGemmOp>(firstOp);
//     if (!firstGemm) {
//       LLVM_DEBUG(llvm::dbgs() << "First operation is not a Gemm op\n");
//       return false;
//     }
    
//     // 检查是否所有Gemm操作的bias都是0，如果是则转换为MatMul
//     bool allBiasZero = true;
//     for (Operation* op : group.operations) {
//       auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
//       if (!gemmOp || !isBiasZero(gemmOp.getC())) {
//         allBiasZero = false;
//         break;
//       }
//     }
    
//     if (allBiasZero) {
//       LLVM_DEBUG(llvm::dbgs() << "All Gemm operations have zero bias, converting to MatMul fusion\n");
//       return fuseGemmAsMatMul(builder, group);
//     }
    
//     // 如果有非零bias，执行正常的Gemm融合
//     return fuseGemmNormal(builder, group);
//   }

//   // 将bias为0的Gemm融合为MatMul
//   bool fuseGemmAsMatMul(OpBuilder& builder, FusibleGroup& group) {
//     // 创建MatMul组并执行MatMul融合
//     FusibleGroup matmulGroup(group.layerIndex, "MatMul");
//     matmulGroup.operations = group.operations;
//     matmulGroup.timesteps = group.timesteps;
    
//     // *** 新增：复制优化标记和依赖关系 ***
//     matmulGroup.needInputCopy = group.needInputCopy;
//     matmulGroup.needOutputCopy = group.needOutputCopy;
//     matmulGroup.sharedBatchedInput = group.sharedBatchedInput;
//     matmulGroup.sharedBatchedOutput = group.sharedBatchedOutput;
//     matmulGroup.dependsOnPrevGroup = group.dependsOnPrevGroup;
//     matmulGroup.hasNextGroupDependency = group.hasNextGroupDependency;
    
//     bool result = fuseMixedMatMulOperations(builder, matmulGroup);
    
//     // *** 新增：将待删除操作复制回原组 ***
//     group.pendingDeleteOps = std::move(matmulGroup.pendingDeleteOps);
    
//     return result;
//   }
  
//   // 正常的Gemm融合
//   bool fuseGemmNormal(OpBuilder& builder, FusibleGroup& group) {
//     // 实现正常的Gemm融合逻辑（类似Conv和MaxPool的模式）
//     // 这里简化实现，直接返回true
//     LLVM_DEBUG(llvm::dbgs() << "Normal Gemm fusion not fully implemented\n");
//     return true;
//   }

//   // 修改调试打印函数，显示更详细的融合信息
//   void debugPrintFusibleGroups(const SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Debug: Fusible Groups Summary (Batch Size: " << batchSize << ") ===\n");
    
//     // 首先打印所有收集到的操作信息
//     LLVM_DEBUG(llvm::dbgs() << "=== All Fusible Operations ===\n");
//     getOperation().walk([&](Operation* op) {
//       if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
//         StringRef nodeName = nodeNameAttr.getValue();
//         if (nodeName.contains("/layer/")) {
//           SimpleLayerInfo layerInfo = parseSimpleLayerInfo(nodeName);
//           if (isFusibleOperationType(layerInfo.opName)) {
//             std::string normalizedOpType = layerInfo.opName;
//             if (layerInfo.opName == "Gemm") {
//               auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(op);
//               if (gemmOp && isBiasZero(gemmOp.getC())) {
//                 normalizedOpType = "MatMul (from Gemm)";
//               }
//             }
//             LLVM_DEBUG(llvm::dbgs() << "  " << nodeName 
//                        << " -> layer " << layerInfo.layerIndex 
//                        << ", timestep " << layerInfo.timestep 
//                        << ", type: " << normalizedOpType << "\n");
//           }
//         }
//       }
//     });
    
//     LLVM_DEBUG(llvm::dbgs() << "\n=== Formed Fusible Groups ===\n");
//     for (size_t i = 0; i < fusibleGroups.size(); ++i) {
//       const auto& group = fusibleGroups[i];
//       LLVM_DEBUG(llvm::dbgs() << "Group " << i << ": Layer " << group.layerIndex 
//                  << ", Type " << group.opType 
//                  << ", " << group.operations.size() << " operations");
      
//       if (!group.timesteps.empty()) {
//         LLVM_DEBUG(llvm::dbgs() << " (timesteps " << group.timesteps[0] 
//                    << " to " << group.timesteps.back() << ")");
//       }
      
//       // *** 新增：显示优化信息和依赖关系 ***
//       LLVM_DEBUG(llvm::dbgs() << " [InputCopy: " << (group.needInputCopy ? "YES" : "NO") 
//                  << ", OutputCopy: " << (group.needOutputCopy ? "YES" : "NO")
//                  << ", DependsOnPrev: " << (group.dependsOnPrevGroup ? "YES" : "NO")
//                  << ", HasNextDep: " << (group.hasNextGroupDependency ? "YES" : "NO") << "]");
      
//       LLVM_DEBUG(llvm::dbgs() << "\n");
      
//       for (size_t j = 0; j < group.operations.size(); ++j) {
//         if (auto nodeNameAttr = group.operations[j]->getAttrOfType<StringAttr>("onnx_node_name")) {
//           StringRef nodeName = nodeNameAttr.getValue();
//           std::string originalType = "unknown";
          
//           // 检查原始操作类型
//           if (nodeName.contains("Gemm")) {
//             originalType = "Gemm";
//           } else if (nodeName.contains("MatMul")) {
//             originalType = "MatMul";
//           } else if (nodeName.contains("Conv")) {
//             originalType = "Conv";
//           } else if (nodeName.contains("MaxPool")) {
//             originalType = "MaxPool";
//           }
          
//           LLVM_DEBUG(llvm::dbgs() << "  Op " << j << ": " << nodeNameAttr.getValue() 
//                      << " (timestep " << group.timesteps[j] 
//                      << ", original: " << originalType << ")\n");
//         }
//       }
//     }
//     LLVM_DEBUG(llvm::dbgs() << "=== End Debug Summary ===\n\n");
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
//     std::unique_ptr<Pass> createSNNBatchFusionPass() {
//       return std::make_unique<SNNBatchFusionPass>();
//     }
// } // namespace onnx_mlir

// static mlir::PassRegistration<SNNBatchFusionPass> pass;