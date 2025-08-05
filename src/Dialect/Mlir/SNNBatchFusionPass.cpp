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

// // 可融合操作组结构
// struct FusibleGroup {
//   int layerIndex;
//   std::string opType;  // Conv, MaxPoolSingleOut, Gemm, MatMul
//   SmallVector<Operation*> operations;
//   SmallVector<int> timesteps;
  
//   FusibleGroup() : layerIndex(0), opType("") {}  // 添加默认构造函数
//   FusibleGroup(int layer, std::string type) : layerIndex(layer), opType(std::move(type)) {}
// };

// struct SNNBatchFusionPass
//     : public PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>> {
  
//   StringRef getArgument() const final { return "snn-batch-fusion"; }
//   StringRef getDescription() const final {
//     return "Fuse SNN operations across timesteps for batch processing with IR reordering";
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
//     if (!performBatchFusion(funcOp)) {
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

//   // === 修复后的 Batch Fusion 逻辑 ===
  
//   // 执行batch fusion的主函数 - 修复版本，按batch size分组
//   bool performBatchFusion(func::FuncOp funcOp) {
//     LLVM_DEBUG(llvm::dbgs() << "Starting batch fusion analysis with batch size: " << batchSize << "\n");
    
//     // 1. 识别可融合的操作组（现在按batch size分组）
//     SmallVector<FusibleGroup> fusibleGroups;
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
    
//     // 5. 对每个可融合组执行fusion
//     for (auto& group : fusibleGroups) {
//       LLVM_DEBUG(llvm::dbgs() << "Fusing group: layer " << group.layerIndex 
//                  << ", op type: " << group.opType 
//                  << ", " << group.operations.size() << " operations"
//                  << " (batch size limited to " << batchSize << ")\n");
      
//       if (!fuseOperationGroup(funcOp, group)) {
//         LLVM_DEBUG(llvm::dbgs() << "Failed to fuse operation group\n");
//         return false;
//       }
//     }
    
//     return true;
//   }
  
//   // 修改identifyFusibleGroups函数，确保正确的分组和相邻性检查
//   void identifyFusibleGroups(func::FuncOp funcOp, SmallVector<FusibleGroup>& fusibleGroups) {
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
//                                SmallVector<FusibleGroup>& fusibleGroups) {
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
  

//   // 融合Conv操作 - 并行复制版本
//   bool fuseConvOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Conv operations with parallel copying\n");
    
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
    
//     // 分配GPU内存  
//       // 分配内存 - 修改为使用memref.alloc
//       Value batchedInput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), inputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       Value batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
    
//     // 为每个原始操作分配独立的输出内存
//       SmallVector<Value> individualOutputs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
    
//     LLVM_DEBUG(llvm::dbgs() << "Allocated GPU memory for batched input/output\n");
    
//     // 创建输入和输出子视图
//     SmallVector<Value> inputViews;
//     SmallVector<Value> outputViews;
    
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       // 计算当前操作在batch中的偏移
//       int64_t offset = i * originalBatchSize;
      
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
    
//     // ========== 修改：并行执行输入数据复制 ==========
//     // 1. 首先一起创建所有需要的异步流
//     SmallVector<Value> copyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       copyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有必要的类型转换
//     SmallVector<Value> inputMemRefs;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* currentOp = group.operations[i];
//       Value originalInput = currentOp->getOperand(0);
      
//       Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), 
//           originalInputMemRefType,
//           originalInput).getResult(0);
//       inputMemRefs.push_back(inputMemRef);
//     }
    
//     // 3. 执行所有并行复制操作
//     SmallVector<Value> inputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(copyStreams[i]),
//           inputViews[i], inputMemRefs[i]);  // 目标，源
//       inputCopyTokens.push_back(copyToken.getAsyncToken());
      
//       LLVM_DEBUG(llvm::dbgs() << "Initiated parallel input copy for operation " << i << "\n");
//     }
    
//     // 4. 等待所有输入复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
//     // =====================================================
    
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
    
//     // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
//     // 1. 首先一起创建所有需要的输出异步流
//     SmallVector<Value> outputCopyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       outputCopyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有并行输出复制操作
//     SmallVector<Value> outputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(outputCopyStreams[i]),
//           individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//       outputCopyTokens.push_back(copyToken.getAsyncToken());
      
//       LLVM_DEBUG(llvm::dbgs() << "Initiated parallel output copy for operation " << i << "\n");
//     }
    
//     // 3. 等待所有输出复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
//     // 4. 转换回tensor
//     SmallVector<Value> splitResults;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
//       splitResults.push_back(splitResult);
//     }
//     // =================================================================
    
//     // 先替换所有使用，然后再删除操作（避免dominance问题）
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       op->getResult(0).replaceAllUsesWith(splitResults[i]);
//       LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//     }
    
//     // 延迟删除操作，确保所有replacement完成后再删除
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << "\n");
//       op->erase();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused Conv operations with parallel copying\n");
//     return true;
//   }

//   // 融合MaxPool操作 - 并行复制版本
//   bool fuseMaxPoolOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MaxPool operations with parallel copying\n");
    
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
    
//     // 分配GPU内存
//       // 分配内存 - 修改为使用memref.alloc
//       Value batchedInput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), inputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       Value batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       // 为每个原始操作分配独立的输出内存
//       SmallVector<Value> individualOutputs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
    
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
    
//     // ========== 修改：并行执行输入数据复制 ==========
//     // 1. 首先一起创建所有需要的异步流
//     SmallVector<Value> copyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       copyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有必要的类型转换
//     SmallVector<Value> inputMemRefs;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* currentOp = group.operations[i];
//       Value originalInput = currentOp->getOperand(0);
      
//       Value inputMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), 
//           originalInputMemRefType,
//           originalInput).getResult(0);
//       inputMemRefs.push_back(inputMemRef);
//     }
    
//     // 3. 执行所有并行复制操作
//     SmallVector<Value> inputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(copyStreams[i]),
//           inputViews[i], inputMemRefs[i]);  // 目标，源
//       inputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 4. 等待所有输入复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
//     // =====================================================
    
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
    
//     // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
//     // 1. 首先一起创建所有需要的输出异步流
//     SmallVector<Value> outputCopyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       outputCopyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有并行输出复制操作
//     SmallVector<Value> outputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(outputCopyStreams[i]),
//           individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//       outputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 3. 等待所有输出复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
//     // 4. 转换回tensor
//     SmallVector<Value> splitResults;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
//       splitResults.push_back(splitResult);
//     }
//     // =================================================================
    
//     // 先替换所有使用，然后再删除操作（避免dominance问题）
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       op->getResult(0).replaceAllUsesWith(splitResults[i]);
//       LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//     }
    
//     // 延迟删除操作，确保所有replacement完成后再删除
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << "\n");
//       op->erase();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused MaxPool operations with parallel copying\n");
//     return true;
//   }

//   // 融合MatMul操作 - 并行复制版本
//   bool fuseMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MatMul operations with parallel copying\n");
    
//     Operation* firstOp = group.operations[0];
//     auto firstMatMul = dyn_cast<mlir::ONNXMatMulOp>(firstOp);
//     if (!firstMatMul) {
//       return false;
//     }
    
//     auto aType = firstMatMul.getA().getType().cast<RankedTensorType>();
//     auto bType = firstMatMul.getB().getType().cast<RankedTensorType>();
//     auto outputType = firstMatMul.getY().getType().cast<RankedTensorType>();
    
//     ArrayRef<int64_t> aShape = aType.getShape();
//     ArrayRef<int64_t> outputShape = outputType.getShape();
    
//     int64_t originalBatchSize = aShape[0];
//     int64_t fusedBatchSize = originalBatchSize * group.operations.size();
    
//     SmallVector<int64_t> fusedAShape(aShape.begin(), aShape.end());
//     SmallVector<int64_t> fusedOutputShape(outputShape.begin(), outputShape.end());
//     fusedAShape[0] = fusedBatchSize;
//     fusedOutputShape[0] = fusedBatchSize;
    
//     auto fusedAType = RankedTensorType::get(fusedAShape, aType.getElementType());
//     auto fusedOutputType = RankedTensorType::get(fusedOutputShape, outputType.getElementType());
    
//     auto aMemRefType = MemRefType::get(fusedAShape, aType.getElementType());
//     auto outputMemRefType = MemRefType::get(fusedOutputShape, outputType.getElementType());
//     auto originalAMemRefType = MemRefType::get(aShape, aType.getElementType());
//     auto originalOutputMemRefType = MemRefType::get(outputShape, outputType.getElementType());
    
//     // 分配GPU内存
//       // 分配内存 - 修改为使用memref.alloc
//       Value batchedA = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), aMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       Value batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       // 为每个原始操作分配独立的输出内存
//       SmallVector<Value> individualOutputs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
    
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
    
//     // ========== 修改：并行执行输入数据复制 ==========
//     // 1. 首先一起创建所有需要的异步流
//     SmallVector<Value> copyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       copyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有必要的类型转换
//     SmallVector<Value> aMemRefs;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* currentOp = group.operations[i];
//       Value originalA = currentOp->getOperand(0);
      
//       Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), 
//           originalAMemRefType,
//           originalA).getResult(0);
//       aMemRefs.push_back(aMemRef);
//     }
    
//     // 3. 执行所有并行复制操作
//     SmallVector<Value> inputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(copyStreams[i]),
//           aViews[i], aMemRefs[i]);  // 目标，源
//       inputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 4. 等待所有输入复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed\n");
//     // =====================================================
    
//     // 创建融合的MatMul操作
//     Value batchedATensor = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), fusedAType, batchedA).getResult(0);
    
//     Value fusedResult = builder.create<mlir::ONNXMatMulOp>(
//         builder.getUnknownLoc(),
//         fusedOutputType,
//         batchedATensor,
//         firstMatMul.getB());  // B矩阵可以共享
    
//     Value fusedResultMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//         builder.getUnknownLoc(), outputMemRefType, fusedResult).getResult(0);
    
//     LLVM_DEBUG(llvm::dbgs() << "Created fused MatMul operation\n");
    
//     // 将融合结果复制到预分配的batched output memory
//     auto fusedCopyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//     auto fusedCopyToken = builder.create<gpu::MemcpyOp>(
//         builder.getUnknownLoc(), 
//         gpu::AsyncTokenType::get(builder.getContext()),
//         ValueRange(fusedCopyStream.getAsyncToken()),
//         batchedOutput, fusedResultMemRef);  // 目标：预分配的batchedOutput，源：融合结果
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, ValueRange(fusedCopyToken.getAsyncToken()));
    
//     LLVM_DEBUG(llvm::dbgs() << "Copied fused MatMul result to batched output memory\n");
    
//     // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
//     // 1. 首先一起创建所有需要的输出异步流
//     SmallVector<Value> outputCopyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       outputCopyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有并行输出复制操作
//     SmallVector<Value> outputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(outputCopyStreams[i]),
//           individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//       outputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 3. 等待所有输出复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed\n");
    
//     // 4. 转换回tensor
//     SmallVector<Value> splitResults;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), outputType, individualOutputs[i]).getResult(0);
//       splitResults.push_back(splitResult);
//     }
//     // =================================================================
    
//     // 先替换所有使用，然后再删除操作（避免dominance问题）
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       op->getResult(0).replaceAllUsesWith(splitResults[i]);
//       LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//     }
    
//     // 延迟删除操作，确保所有replacement完成后再删除
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       LLVM_DEBUG(llvm::dbgs() << "Erasing MatMul operation " << i << "\n");
//       op->erase();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused MatMul operations with parallel copying\n");
//     return true;
//   }

//   // 混合MatMul操作融合（包括Gemm(bias=0)和MatMul）- 并行复制版本
//   bool fuseMixedMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
//     LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " mixed MatMul operations with parallel copying\n");
    
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
    
//     // 分配GPU内存
//       // 分配内存 - 修改为使用memref.alloc
//       Value batchedA = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), aMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       Value batchedOutput = builder.create<memref::AllocOp>(
//           builder.getUnknownLoc(), outputMemRefType, 
//           ValueRange{}, builder.getI64IntegerAttr(16));
      
//       // 为每个原始操作分配独立的输出内存
//       SmallVector<Value> individualOutputs;
//       for (size_t i = 0; i < group.operations.size(); ++i) {
//         Value individualOutput = builder.create<memref::AllocOp>(
//             builder.getUnknownLoc(), originalOutputMemRefType, 
//             ValueRange{}, builder.getI64IntegerAttr(16));
//         individualOutputs.push_back(individualOutput);
//       }
    
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
    
//     // ========== 修改：并行执行输入数据复制 - 处理混合操作 ==========
//     // 1. 首先一起创建所有需要的异步流
//     SmallVector<Value> copyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       copyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有必要的类型转换 - 处理混合操作
//     SmallVector<Value> aMemRefs;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* currentOp = group.operations[i];
//       Value originalA;
      
//       // 从MatMul或Gemm中获取A矩阵
//       if (auto matmulOp = dyn_cast<mlir::ONNXMatMulOp>(currentOp)) {
//         originalA = matmulOp.getA();
//       } else if (auto gemmOp = dyn_cast<mlir::ONNXGemmOp>(currentOp)) {
//         originalA = gemmOp.getA();
//       } else {
//         LLVM_DEBUG(llvm::dbgs() << "Unsupported operation type in mixed fusion\n");
//         return false;
//       }
      
//       Value aMemRef = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), 
//           originalAMemRefType,
//           originalA).getResult(0);
//       aMemRefs.push_back(aMemRef);
//     }
    
//     // 3. 执行所有并行复制操作
//     SmallVector<Value> inputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(copyStreams[i]),
//           aViews[i], aMemRefs[i]);  // 目标，源
//       inputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 4. 等待所有输入复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, inputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel input copies completed for mixed operations\n");
//     // =================================================================
    
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
    
//     // ========== 修改：并行复制输出数据并替换原始操作的使用 ==========
//     // 1. 首先一起创建所有需要的输出异步流
//     SmallVector<Value> outputCopyStreams;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyStream = builder.create<gpu::WaitOp>(builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()), ValueRange());
//       outputCopyStreams.push_back(copyStream.getAsyncToken());
//     }
    
//     // 2. 执行所有并行输出复制操作
//     SmallVector<Value> outputCopyTokens;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       auto copyToken = builder.create<gpu::MemcpyOp>(
//           builder.getUnknownLoc(), 
//           gpu::AsyncTokenType::get(builder.getContext()),
//           ValueRange(outputCopyStreams[i]),
//           individualOutputs[i], outputViews[i]);  // 目标：individual output，源：batched output subview
//       outputCopyTokens.push_back(copyToken.getAsyncToken());
//     }
    
//     // 3. 等待所有输出复制完成
//     builder.create<gpu::WaitOp>(builder.getUnknownLoc(), Type{}, outputCopyTokens);
//     LLVM_DEBUG(llvm::dbgs() << "All parallel output copies completed for mixed operations\n");
    
//     // 4. 转换回tensor
//     SmallVector<Value> splitResults;
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Value splitResult = builder.create<mlir::UnrealizedConversionCastOp>(
//           builder.getUnknownLoc(), outputTensorType, individualOutputs[i]).getResult(0);
//       splitResults.push_back(splitResult);
//     }
//     // =================================================================
    
//     // 先替换所有使用，然后再删除操作（避免dominance问题）
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       op->getResult(0).replaceAllUsesWith(splitResults[i]);
//       LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
//     }
    
//     // 延迟删除操作，确保所有replacement完成后再删除
//     for (size_t i = 0; i < group.operations.size(); ++i) {
//       Operation* op = group.operations[i];
//       LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << "\n");
//       op->erase();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Successfully fused mixed MatMul operations with parallel copying\n");
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
    
//     return fuseMatMulOperations(builder, matmulGroup);
//   }
  
//   // 正常的Gemm融合
//   bool fuseGemmNormal(OpBuilder& builder, FusibleGroup& group) {
//     // 实现正常的Gemm融合逻辑（类似Conv和MaxPool的模式）
//     // 这里简化实现，直接返回true
//     LLVM_DEBUG(llvm::dbgs() << "Normal Gemm fusion not fully implemented\n");
//     return true;
//   }

//   // 修改调试打印函数，显示更详细的融合信息
//   void debugPrintFusibleGroups(const SmallVector<FusibleGroup>& fusibleGroups) {
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

// 可融合操作组结构 - 新增优化字段
struct FusibleGroup {
  int layerIndex;
  std::string opType;  // Conv, MaxPoolSingleOut, Gemm, MatMul
  SmallVector<Operation*> operations;
  SmallVector<int> timesteps;
  
  // 新增：相邻group优化字段
  bool needInputCopy = true;    // 是否需要输入复制
  bool needOutputCopy = true;   // 是否需要输出复制
  Value sharedBatchedInput;     // 共享的batched输入内存
  Value sharedBatchedOutput;    // 共享的batched输出内存
  
  // *** 新增：依赖关系字段，用于延迟删除 ***
  bool dependsOnPrevGroup = false;      // 是否依赖前一个组
  bool hasNextGroupDependency = false;  // 是否有下一个组依赖自己
  SmallVector<Operation*> pendingDeleteOps; // 待删除的操作（用于延迟删除）
  
  FusibleGroup() : layerIndex(0), opType("") {}
  FusibleGroup(int layer, std::string type) : layerIndex(layer), opType(std::move(type)) {}
};

struct SNNBatchFusionPass
    : public PassWrapper<SNNBatchFusionPass, OperationPass<func::FuncOp>> {
  
  StringRef getArgument() const final { return "snn-batch-fusion"; }
  StringRef getDescription() const final {
    return "Fuse SNN operations across timesteps for batch processing with IR reordering and adjacent group optimization";
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
    if (!performBatchFusionWithOptimizations(funcOp)) {
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

  // === 修复后的 Batch Fusion 逻辑 - 新增相邻组优化 ===
  
  // 执行batch fusion的主函数 - 修复版本，增加相邻组优化
  bool performBatchFusionWithOptimizations(func::FuncOp funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "Starting batch fusion analysis with batch size: " << batchSize << "\n");
    
    // 1. 识别可融合的操作组（现在按batch size分组）
    SmallVector<FusibleGroup, 4> fusibleGroups;  // 显式指定内联元素数量
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
    
    // *** 5. 分析相邻组关系并设置优化标记 ***
    analyzeAdjacentGroups(fusibleGroups);
    
    // *** 删除：不再预先设置共享内存，改为动态获取 ***
    
    // 6. 对每个可融合组执行fusion
    for (size_t i = 0; i < fusibleGroups.size(); ++i) {
      auto& group = fusibleGroups[i];
      
      // *** 新增：动态设置共享输入内存 ***
      if (!group.needInputCopy && i > 0) {
        // 从前一个组获取共享输出内存作为当前组的输入
        auto& prevGroup = fusibleGroups[i-1];
        group.sharedBatchedInput = prevGroup.sharedBatchedOutput;
        
        LLVM_DEBUG(llvm::dbgs() << "Group " << i << " will use shared memory from Group " << (i-1) << "\n");
      }
      
      LLVM_DEBUG(llvm::dbgs() << "Fusing group: layer " << group.layerIndex 
                 << ", op type: " << group.opType 
                 << ", " << group.operations.size() << " operations"
                 << " (batch size limited to " << batchSize << ")"
                 << " [InputCopy: " << (group.needInputCopy ? "YES" : "NO") 
                 << ", OutputCopy: " << (group.needOutputCopy ? "YES" : "NO") << "]\n");
      
      if (!fuseOperationGroup(funcOp, group)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to fuse operation group\n");
        return false;
      }
    }
    
    // *** 7. 执行延迟删除操作 ***
    performDeferredDeletion(fusibleGroups);
    
    return true;
  }
  
  // *** 新增：执行延迟删除操作 ***
  void performDeferredDeletion(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
    LLVM_DEBUG(llvm::dbgs() << "\n=== Performing Deferred Deletion ===\n");
    
    for (auto& group : fusibleGroups) {
      if (!group.pendingDeleteOps.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Deleting " << group.pendingDeleteOps.size() 
                   << " pending operations from group (layer " << group.layerIndex 
                   << ", type " << group.opType << ")\n");
        
        for (Operation* op : group.pendingDeleteOps) {
          if (auto nodeNameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
            LLVM_DEBUG(llvm::dbgs() << "  Deleting: " << nodeNameAttr.getValue() << "\n");
          }
          op->erase();
        }
        group.pendingDeleteOps.clear();
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "=== Deferred Deletion Complete ===\n\n");
  }
  
  // *** 新增：分析相邻组关系的函数 ***
  void analyzeAdjacentGroups(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
    LLVM_DEBUG(llvm::dbgs() << "\n=== Analyzing Adjacent Groups for Optimization ===\n");
    
    for (size_t i = 0; i < fusibleGroups.size(); ++i) {
      auto& currentGroup = fusibleGroups[i];
      
      // 检查是否与前一个组相邻
      if (i > 0) {
        auto& prevGroup = fusibleGroups[i-1];
        
        if (areAdjacentGroups(prevGroup, currentGroup)) {
          // 设置优化标记
          currentGroup.needInputCopy = false;
          prevGroup.needOutputCopy = false;
          
          // *** 新增：建立相邻组的依赖关系，用于延迟删除 ***
          currentGroup.dependsOnPrevGroup = true;
          prevGroup.hasNextGroupDependency = true;
          
          LLVM_DEBUG(llvm::dbgs() << "Found adjacent groups: "
                     << "Group " << (i-1) << " (layer " << prevGroup.layerIndex << ", " << prevGroup.opType << ") "
                     << "-> Group " << i << " (layer " << currentGroup.layerIndex << ", " << currentGroup.opType << ")\n");
          LLVM_DEBUG(llvm::dbgs() << "  Optimization: Skip output copy for group " << (i-1) 
                     << " and input copy for group " << i << "\n");
          LLVM_DEBUG(llvm::dbgs() << "  Dependency: Group " << i << " depends on Group " << (i-1) << "\n");
        }
      }
      
      LLVM_DEBUG(llvm::dbgs() << "Group " << i << " optimization flags: "
                 << "needInputCopy=" << currentGroup.needInputCopy 
                 << ", needOutputCopy=" << currentGroup.needOutputCopy 
                 << ", dependsOnPrev=" << currentGroup.dependsOnPrevGroup
                 << ", hasNextDep=" << currentGroup.hasNextGroupDependency << "\n");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "=== Adjacent Group Analysis Complete ===\n\n");
  }
  
  // *** 新增：检查两个组是否相邻的函数 ***
  bool areAdjacentGroups(const FusibleGroup& group1, const FusibleGroup& group2) {
    // 检查层次上是否相邻：下一层的索引应该是当前层索引+1
    bool layerAdjacent = (group2.layerIndex == group1.layerIndex + 1);
    
    // 检查时间步是否一致（相同的batch）
    bool sameTimesteps = (group1.timesteps == group2.timesteps);
    
    // 检查IR中位置是否相邻（允许一些中间的temporal操作）
    bool irAdjacent = areAdjacentInIR(group1, group2);
    
    bool result = layerAdjacent && sameTimesteps && irAdjacent;
    
    LLVM_DEBUG(llvm::dbgs() << "Checking adjacency between groups (layer " << group1.layerIndex 
               << " -> " << group2.layerIndex << "): "
               << "layerAdjacent=" << layerAdjacent 
               << ", sameTimesteps=" << sameTimesteps 
               << ", irAdjacent=" << irAdjacent 
               << " -> " << (result ? "ADJACENT" : "NOT ADJACENT") << "\n");
    
    return result;
  }
  
  // *** 新增：检查两个组在IR中是否相邻 ***
  bool areAdjacentInIR(const FusibleGroup& group1, const FusibleGroup& group2) {
    if (group1.operations.empty() || group2.operations.empty()) {
      return false;
    }
    
    // 找到group1的最后一个操作和group2的第一个操作
    Operation* lastOp1 = group1.operations.back();
    Operation* firstOp2 = group2.operations.front();
    
    // 检查它们之间的距离，允许有一些temporal操作
    int maxAllowedGap = 10; // 允许的最大间隔操作数
    int gap = 0;
    Operation* current = lastOp1->getNextNode();
    
    while (current && current != firstOp2 && gap < maxAllowedGap) {
      // 跳过常量和工具操作
      if (!isConstantOrUtilityOp(current) && !isTemporalOp(current)) {
        gap++;
      }
      current = current->getNextNode();
    }
    
    bool adjacent = (current == firstOp2 && gap < maxAllowedGap);
    
    LLVM_DEBUG(llvm::dbgs() << "  IR adjacency check: gap=" << gap 
               << ", found=" << (current == firstOp2) 
               << " -> " << (adjacent ? "ADJACENT" : "NOT ADJACENT") << "\n");
    
    return adjacent;
  }
  
  // *** 新增：在fusion之前设置共享内存 ***
  void setupSharedMemory(SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
    LLVM_DEBUG(llvm::dbgs() << "\n=== Setting up Shared Memory for Adjacent Groups ===\n");
    
    for (size_t i = 1; i < fusibleGroups.size(); ++i) {
      auto& currentGroup = fusibleGroups[i];
      auto& prevGroup = fusibleGroups[i-1];
      
      // 如果当前组不需要输入复制，说明它可以复用前一个组的输出
      if (!currentGroup.needInputCopy) {
        // 前一个组的输出将作为当前组的输入
        currentGroup.sharedBatchedInput = prevGroup.sharedBatchedOutput;
        
        LLVM_DEBUG(llvm::dbgs() << "Group " << i << " will reuse output memory from Group " << (i-1) << "\n");
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "=== Shared Memory Setup Complete ===\n\n");
  }
  
  // 修改identifyFusibleGroups函数，确保正确的分组和相邻性检查
  void identifyFusibleGroups(func::FuncOp funcOp, SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
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
                               SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
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
      return fuseConvOperations(builder, group);
    } else if (group.opType == "MaxPool" || group.opType == "MaxPoolSingleOut") {
      return fuseMaxPoolOperations(builder, group);
    } else if (group.opType == "MatMul") {
      // 这里处理纯MatMul或者Gemm(bias=0)的融合
      return fuseMixedMatMulOperations(builder, group);
    } else if (group.opType == "Gemm") {
      // 这里处理有非零bias的Gemm融合
      return fuseGemmOperations(builder, group);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Unknown operation type for fusion: " << group.opType << "\n");
    return false;
  }
  

  // *** 修改：融合Conv操作 - 支持相邻组优化 ***
  bool fuseConvOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " Conv operations with optimizations\n");
    LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
               << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
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
    
    // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
    Value batchedInput;
    if (group.needInputCopy) {
      // 正常情况：分配新的输入内存
      batchedInput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), inputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
    } else {
      // 优化情况：复用前一个组的输出内存作为输入
      batchedInput = group.sharedBatchedInput;
      
      // *** 新增：类型兼容性检查 ***
      bool typeCompatible = false;
      if (batchedInput) {
        auto sharedType = batchedInput.getType().dyn_cast<MemRefType>();
        if (sharedType && sharedType.getShape() == inputMemRefType.getShape() && 
            sharedType.getElementType() == inputMemRefType.getElementType()) {
          typeCompatible = true;
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed - shared type vs required type mismatch\n");
          if (sharedType) {
            LLVM_DEBUG(llvm::dbgs() << "  Shared shape: [");
            for (size_t i = 0; i < sharedType.getShape().size(); ++i) {
              LLVM_DEBUG(llvm::dbgs() << sharedType.getShape()[i]);
              if (i < sharedType.getShape().size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
            }
            LLVM_DEBUG(llvm::dbgs() << "]\n");
            LLVM_DEBUG(llvm::dbgs() << "  Required shape: [");
            for (size_t i = 0; i < inputMemRefType.getShape().size(); ++i) {
              LLVM_DEBUG(llvm::dbgs() << inputMemRefType.getShape()[i]);
              if (i < inputMemRefType.getShape().size() - 1) LLVM_DEBUG(llvm::dbgs() << ", ");
            }
            LLVM_DEBUG(llvm::dbgs() << "]\n");
          }
        }
      }
      
      if (!batchedInput || !typeCompatible) {
        LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
        batchedInput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), inputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        // 重置为需要输入复制，因为我们现在需要分配新内存
        group.needInputCopy = true;
      }
    }
    
    // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
    Value batchedOutput;
    if (group.needOutputCopy) {
      // 正常情况：分配新的输出内存
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
    } else {
      // 优化情况：分配输出内存，但会被下一个组复用
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      // 设置为下一个组的共享输入（这个在analyzeAdjacentGroups中处理）
      group.sharedBatchedOutput = batchedOutput;
      LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
    }
    
    // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
    SmallVector<Value> individualOutputs;
    if (group.needOutputCopy) {
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
      LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
    }
    
    // 创建输入和输出子视图
    SmallVector<Value> inputViews;
    SmallVector<Value> outputViews;
    
    // *** 新增：验证 batchedInput 的有效性 ***
    if (!batchedInput) {
      LLVM_DEBUG(llvm::dbgs() << "ERROR: batchedInput is invalid, cannot create subviews\n");
      return false;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Creating subviews for " << group.operations.size() << " operations\n");
    LLVM_DEBUG(llvm::dbgs() << "Input shape: [" << fusedInputShape[0] << ", " << fusedInputShape[1] 
               << ", " << fusedInputShape[2] << ", " << fusedInputShape[3] << "]\n");
    LLVM_DEBUG(llvm::dbgs() << "Original batch size: " << originalBatchSize << "\n");
    
    for (size_t i = 0; i < group.operations.size(); ++i) {
      // 计算当前操作在batch中的偏移
      int64_t offset = i * originalBatchSize;
      
      LLVM_DEBUG(llvm::dbgs() << "Creating subview " << i << " with offset " << offset << "\n");
      
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
    
    // *** 修改：根据needInputCopy决定是否执行输入数据复制 ***
    // 注意：needInputCopy 可能在类型兼容性检查后被动态修改
    if (group.needInputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel)\n");
      
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
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
    }
    
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
    
    // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
    if (group.needOutputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
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
      
      // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        op->getResult(0).replaceAllUsesWith(splitResults[i]);
        LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
      }
      
      // 6. 立即删除操作（因为已经完成替换）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << "\n");
        op->erase();
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
      // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
      if (group.hasNextGroupDependency) {
        LLVM_DEBUG(llvm::dbgs() << "Adding operations to pending deletion (next group depends on this)\n");
        // 将操作添加到待删除列表，延迟到所有fusion完成后删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          group.pendingDeleteOps.push_back(op);
          LLVM_DEBUG(llvm::dbgs() << "Added operation " << i << " to pending deletion\n");
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting operations immediately\n");
        // 没有下一个组依赖，可以立即删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          LLVM_DEBUG(llvm::dbgs() << "Erasing operation " << i << " (no dependency)\n");
          op->erase();
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused Conv operations with optimizations\n");
    return true;
  }

  // *** 修改：融合MaxPool操作 - 支持相邻组优化（类似Conv的逻辑）***
  bool fuseMaxPoolOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " MaxPool operations with optimizations\n");
    LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
               << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
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
    
    // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
    Value batchedInput;
    if (group.needInputCopy) {
      batchedInput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), inputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
    } else {
      batchedInput = group.sharedBatchedInput;
      
      // *** 新增：类型兼容性检查 ***
      bool typeCompatible = false;
      if (batchedInput) {
        auto sharedType = batchedInput.getType().dyn_cast<MemRefType>();
        if (sharedType && sharedType.getShape() == inputMemRefType.getShape() && 
            sharedType.getElementType() == inputMemRefType.getElementType()) {
          typeCompatible = true;
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed for MaxPool\n");
        }
      }
      
      if (!batchedInput || !typeCompatible) {
        LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
        batchedInput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), inputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        group.needInputCopy = true;
      }
    }
    
    // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
    Value batchedOutput;
    if (group.needOutputCopy) {
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
    } else {
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      group.sharedBatchedOutput = batchedOutput;
      LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
    }
    
    // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
    SmallVector<Value> individualOutputs;
    if (group.needOutputCopy) {
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
      LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
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
    
    // *** 修改：根据needInputCopy决定是否执行输入数据复制 ***
    // 注意：needInputCopy 可能在类型兼容性检查后被动态修改
    if (group.needInputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel)\n");
      
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
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
    }
    
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
    
    // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
    if (group.needOutputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
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
      
      // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        op->getResult(0).replaceAllUsesWith(splitResults[i]);
        LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
      }
      
      // 6. 立即删除操作（因为已经完成替换）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << "\n");
        op->erase();
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
      // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
      if (group.hasNextGroupDependency) {
        LLVM_DEBUG(llvm::dbgs() << "Adding MaxPool operations to pending deletion (next group depends on this)\n");
        // 将操作添加到待删除列表，延迟到所有fusion完成后删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          group.pendingDeleteOps.push_back(op);
          LLVM_DEBUG(llvm::dbgs() << "Added MaxPool operation " << i << " to pending deletion\n");
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting MaxPool operations immediately\n");
        // 没有下一个组依赖，可以立即删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          LLVM_DEBUG(llvm::dbgs() << "Erasing MaxPool operation " << i << " (no dependency)\n");
          op->erase();
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused MaxPool operations with optimizations\n");
    return true;
  }

  // *** 修改：混合MatMul操作融合 - 支持相邻组优化（类似Conv的逻辑）***
  bool fuseMixedMatMulOperations(OpBuilder& builder, FusibleGroup& group) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing " << group.operations.size() << " mixed MatMul operations with optimizations\n");
    LLVM_DEBUG(llvm::dbgs() << "  needInputCopy: " << group.needInputCopy 
               << ", needOutputCopy: " << group.needOutputCopy << "\n");
    
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
    
    // *** 修改：根据是否需要输入复制来决定内存分配策略 ***
    Value batchedA;
    if (group.needInputCopy) {
      batchedA = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), aMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched input memory\n");
    } else {
      batchedA = group.sharedBatchedInput;
      
      // *** 新增：类型兼容性检查 ***
      bool typeCompatible = false;
      if (batchedA) {
        auto sharedType = batchedA.getType().dyn_cast<MemRefType>();
        if (sharedType && sharedType.getShape() == aMemRefType.getShape() && 
            sharedType.getElementType() == aMemRefType.getElementType()) {
          typeCompatible = true;
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check passed, reusing shared memory\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Type compatibility check failed for MatMul\n");
        }
      }
      
      if (!batchedA || !typeCompatible) {
        LLVM_DEBUG(llvm::dbgs() << "Shared memory not available or incompatible, falling back to allocation\n");
        batchedA = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), aMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        group.needInputCopy = true;
      }
    }
    
    // *** 修改：根据是否需要输出复制来决定输出内存分配策略 ***
    Value batchedOutput;
    if (group.needOutputCopy) {
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      LLVM_DEBUG(llvm::dbgs() << "Allocated new batched output memory\n");
    } else {
      batchedOutput = builder.create<memref::AllocOp>(
          builder.getUnknownLoc(), outputMemRefType, 
          ValueRange{}, builder.getI64IntegerAttr(16));
      group.sharedBatchedOutput = batchedOutput;
      LLVM_DEBUG(llvm::dbgs() << "Allocated batched output memory for sharing with next group\n");
    }
    
    // 为每个原始操作分配独立的输出内存（仅在需要输出复制时使用）
    SmallVector<Value> individualOutputs;
    if (group.needOutputCopy) {
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Value individualOutput = builder.create<memref::AllocOp>(
            builder.getUnknownLoc(), originalOutputMemRefType, 
            ValueRange{}, builder.getI64IntegerAttr(16));
        individualOutputs.push_back(individualOutput);
      }
      LLVM_DEBUG(llvm::dbgs() << "Allocated individual output memories\n");
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
    
    // *** 修改：根据needInputCopy决定是否执行输入数据复制 - 处理混合操作 ***
    if (group.needInputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing input data copying (parallel, mixed operations)\n");
      
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
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping input data copying (using shared memory)\n");
    }
    
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
    
    // *** 修改：根据needOutputCopy决定是否执行输出数据复制和替换 ***
    if (group.needOutputCopy) {
      LLVM_DEBUG(llvm::dbgs() << "Performing output data copying and result replacement\n");
      
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
      
      // 5. 先替换所有使用，然后再删除操作（避免dominance问题）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        op->getResult(0).replaceAllUsesWith(splitResults[i]);
        LLVM_DEBUG(llvm::dbgs() << "Replaced operation " << i << " result\n");
      }
      
      // 6. 立即删除操作（因为已经完成替换）
      for (size_t i = 0; i < group.operations.size(); ++i) {
        Operation* op = group.operations[i];
        LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << "\n");
        op->erase();
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Skipping output data copying (will be consumed by next group)\n");
      
      // *** 修改：如果有下一个组依赖，则延迟删除；否则可以立即删除 ***
      if (group.hasNextGroupDependency) {
        LLVM_DEBUG(llvm::dbgs() << "Adding mixed MatMul operations to pending deletion (next group depends on this)\n");
        // 将操作添加到待删除列表，延迟到所有fusion完成后删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          group.pendingDeleteOps.push_back(op);
          LLVM_DEBUG(llvm::dbgs() << "Added mixed MatMul operation " << i << " to pending deletion\n");
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "No dependency, deleting mixed MatMul operations immediately\n");
        // 没有下一个组依赖，可以立即删除
        for (size_t i = 0; i < group.operations.size(); ++i) {
          Operation* op = group.operations[i];
          LLVM_DEBUG(llvm::dbgs() << "Erasing mixed MatMul operation " << i << " (no dependency)\n");
          op->erase();
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused mixed MatMul operations with optimizations\n");
    return true;
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
    
    // *** 新增：复制优化标记和依赖关系 ***
    matmulGroup.needInputCopy = group.needInputCopy;
    matmulGroup.needOutputCopy = group.needOutputCopy;
    matmulGroup.sharedBatchedInput = group.sharedBatchedInput;
    matmulGroup.sharedBatchedOutput = group.sharedBatchedOutput;
    matmulGroup.dependsOnPrevGroup = group.dependsOnPrevGroup;
    matmulGroup.hasNextGroupDependency = group.hasNextGroupDependency;
    
    bool result = fuseMixedMatMulOperations(builder, matmulGroup);
    
    // *** 新增：将待删除操作复制回原组 ***
    group.pendingDeleteOps = std::move(matmulGroup.pendingDeleteOps);
    
    return result;
  }
  
  // 正常的Gemm融合
  bool fuseGemmNormal(OpBuilder& builder, FusibleGroup& group) {
    // 实现正常的Gemm融合逻辑（类似Conv和MaxPool的模式）
    // 这里简化实现，直接返回true
    LLVM_DEBUG(llvm::dbgs() << "Normal Gemm fusion not fully implemented\n");
    return true;
  }

  // 修改调试打印函数，显示更详细的融合信息
  void debugPrintFusibleGroups(const SmallVector<FusibleGroup, 4>& fusibleGroups) {  // 显式指定内联元素数量
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
      
      // *** 新增：显示优化信息和依赖关系 ***
      LLVM_DEBUG(llvm::dbgs() << " [InputCopy: " << (group.needInputCopy ? "YES" : "NO") 
                 << ", OutputCopy: " << (group.needOutputCopy ? "YES" : "NO")
                 << ", DependsOnPrev: " << (group.dependsOnPrevGroup ? "YES" : "NO")
                 << ", HasNextDep: " << (group.hasNextGroupDependency ? "YES" : "NO") << "]");
      
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