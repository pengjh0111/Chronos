// #include "mlir/Pass/Pass.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/Support/CommandLine.h"
// #include <vector>
// #include <string>
// #include <set>
// #include <sstream>
// #include <algorithm>

// using namespace mlir;

// #define DEBUG_TYPE "snn-structure-annotation"

// namespace {

// // Pass选项定义
// static llvm::cl::opt<std::string> networkStructure(
//     "network-structure",
//     llvm::cl::desc("Network structure specification (comma-separated layer types)"),
//     llvm::cl::value_desc("structure"),
//     llvm::cl::init(""));

// static llvm::cl::opt<int> timeSteps(
//     "time-steps", 
//     llvm::cl::desc("Number of time steps for SNN"),
//     llvm::cl::value_desc("steps"),
//     llvm::cl::init(1));

// class SNNStructureAnnotationPass
//     : public PassWrapper<SNNStructureAnnotationPass, OperationPass<func::FuncOp>> {
    
// private:
//   std::vector<std::string> networkStructure_; // 包含if的完整网络结构
//   int numTimeSteps_;
  
//   // 需要处理onnx_node_name的操作类型集合
//   std::set<std::string> targetOpTypes_ = {
//     "onnx.Conv", "onnx.MaxPoolSingleOut", "onnx.Flatten", 
//     "onnx.MatMul", "onnx.Gemm", "onnx.Add"
//   };

// public:
//   StringRef getArgument() const final { return "snn-structure-annotation"; }
//   StringRef getDescription() const final {
//     return "Annotate SNN network structure in onnx_node_name attributes based on user-provided structure";
//   }

//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
    
//     // 解析网络结构参数
//     if (networkStructure.empty()) {
//       LLVM_DEBUG(llvm::dbgs() << "No network structure specified, skipping pass\n");
//       return;
//     }
    
//     parseNetworkStructure();
//     numTimeSteps_ = timeSteps.getValue();
    
//     if (numTimeSteps_ <= 0) {
//       LLVM_DEBUG(llvm::dbgs() << "Invalid time steps: " << numTimeSteps_ << "\n");
//       signalPassFailure();
//       return;
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "=== SNN Structure Annotation Pass ===\n");
//     LLVM_DEBUG(llvm::dbgs() << "Network structure: " << networkStructure.getValue() << "\n");
//     LLVM_DEBUG(llvm::dbgs() << "Time steps: " << numTimeSteps_ << "\n");
    
//     if (failed(processOperationsWithStructure(funcOp))) {
//       signalPassFailure();
//     }
//   }

// private:
//   void parseNetworkStructure() {
//     networkStructure_.clear();
//     std::stringstream ss(networkStructure.getValue());
//     std::string item;
    
//     while (std::getline(ss, item, ',')) {
//       // 去除空格
//       item.erase(0, item.find_first_not_of(" \t"));
//       item.erase(item.find_last_not_of(" \t") + 1);
      
//       if (!item.empty()) {
//         networkStructure_.push_back(item);
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Parsed layer types: ");
//     for (size_t i = 0; i < networkStructure_.size(); ++i) {
//       LLVM_DEBUG(llvm::dbgs() << "[" << i << "]" << networkStructure_[i] << " ");
//     }
//     LLVM_DEBUG(llvm::dbgs() << "\n");
//   }

//   LogicalResult processOperationsWithStructure(func::FuncOp funcOp) {
//     Block& block = funcOp.getBody().front();
    
//     // 收集所有目标操作
//     std::vector<Operation*> targetOps;
//     for (Operation& op : block) {
//       if (isTargetOperation(&op)) {
//         targetOps.push_back(&op);
//       }
//     }
    
//     if (targetOps.empty()) {
//       LLVM_DEBUG(llvm::dbgs() << "No target operations found\n");
//       return success();
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Found " << targetOps.size() << " target operations\n");
    
//     return annotateOperationsWithStructure(targetOps);
//   }

//   bool isTargetOperation(Operation* op) {
//     std::string opName = op->getName().getStringRef().str();
//     return targetOpTypes_.count(opName) > 0;
//   }

//   LogicalResult annotateOperationsWithStructure(const std::vector<Operation*>& targetOps) {
//     int currentOpIndex = 0;
    
//     for (int timestep = 0; timestep < numTimeSteps_; ++timestep) {
//       LLVM_DEBUG(llvm::dbgs() << "\n--- Processing timestep " << timestep << " ---\n");
      
//       // 处理当前时间步的网络结构
//       for (size_t layerIndex = 0; layerIndex < networkStructure_.size(); ++layerIndex) {
//         const std::string& layerType = networkStructure_[layerIndex];
        
//         LLVM_DEBUG(llvm::dbgs() << "Processing layer " << layerIndex << " (" << layerType << ")\n");
        
//         // 特殊情况：第一层在后续时间步被跳过（复用第一个时间步的结果）
//         if (timestep > 0 && layerIndex == 0 && !isIfLayerType(layerType)) {
//           LLVM_DEBUG(llvm::dbgs() << "Skipping first layer in timestep " << timestep 
//                      << " (reusing from timestep 0)\n");
//           continue;
//         }
        
//         // 如果是if层，跳过（不需要处理任何实际的op，但占用层索引）
//         if (isIfLayerType(layerType)) {
//           LLVM_DEBUG(llvm::dbgs() << "Skipping IF layer at index " << layerIndex << "\n");
//           continue;
//         }
        
//         // 处理Add操作（时间步结束标记）
//         if (isAddLayerType(layerType)) {
//           if (currentOpIndex < targetOps.size()) {
//             Operation* op = findNextAddOperation(targetOps, currentOpIndex);
//             if (op) {
//               annotateAddOperation(op, timestep);
//               // 更新索引到找到的Add操作之后
//               currentOpIndex = findOpIndex(targetOps, op) + 1;
//               LLVM_DEBUG(llvm::dbgs() << "Processed Add operation for timestep " << timestep << "\n");
//             } else {
//               LLVM_DEBUG(llvm::dbgs() << "Warning: Could not find Add operation for timestep " << timestep << "\n");
//             }
//           }
//         }
//         // 处理普通计算层操作
//         else {
//           Operation* op = findNextMatchingOperation(targetOps, currentOpIndex, layerType);
//           if (op) {
//             annotateLayerOperation(op, layerIndex, timestep);
//             // 更新索引到找到的操作之后
//             currentOpIndex = findOpIndex(targetOps, op) + 1;
//             LLVM_DEBUG(llvm::dbgs() << "Processed " << layerType 
//                        << " operation (layer " << layerIndex 
//                        << ", timestep " << timestep << ")\n");
//           } else {
//             LLVM_DEBUG(llvm::dbgs() << "Warning: Could not find " << layerType 
//                        << " operation for layer " << layerIndex 
//                        << " in timestep " << timestep << "\n");
//           }
//         }
//       }
//     }
    
//     LLVM_DEBUG(llvm::dbgs() << "Annotation completed. Processed operations across " 
//                << numTimeSteps_ << " timesteps\n");
    
//     return success();
//   }

//   bool isIfLayerType(const std::string& layerType) {
//     std::string normalizedType = normalizeLayerType(layerType);
//     return normalizedType == "if";
//   }

//   bool isAddLayerType(const std::string& layerType) {
//     std::string normalizedType = normalizeLayerType(layerType);
//     return normalizedType == "add";
//   }

//   Operation* findNextAddOperation(const std::vector<Operation*>& targetOps, int startIndex) {
//     for (int i = startIndex; i < targetOps.size(); ++i) {
//       if (isAddOperation(targetOps[i])) {
//         return targetOps[i];
//       }
//     }
//     return nullptr;
//   }

//   Operation* findNextMatchingOperation(const std::vector<Operation*>& targetOps, 
//                                        int startIndex, 
//                                        const std::string& expectedType) {
//     for (int i = startIndex; i < targetOps.size(); ++i) {
//       Operation* op = targetOps[i];
      
//       // 跳过Add操作，它们用于时间步分界
//       if (isAddOperation(op)) {
//         continue;
//       }
      
//       if (matchesLayerType(op, expectedType)) {
//         return op;
//       }
//     }
//     return nullptr;
//   }

//   int findOpIndex(const std::vector<Operation*>& targetOps, Operation* target) {
//     for (int i = 0; i < targetOps.size(); ++i) {
//       if (targetOps[i] == target) {
//         return i;
//       }
//     }
//     return -1;
//   }

//   bool matchesLayerType(Operation* op, const std::string& expectedType) {
//     std::string opName = op->getName().getStringRef().str();
//     std::string opType = extractOpType(opName);
    
//     // 标准化比较
//     std::string normalizedExpected = normalizeLayerType(expectedType);
//     std::string normalizedActual = normalizeLayerType(opType);
    
//     return normalizedExpected == normalizedActual;
//   }

//   std::string extractOpType(const std::string& opName) {
//     // 从 "onnx.Conv" 提取 "Conv"
//     size_t dotPos = opName.find('.');
//     if (dotPos != std::string::npos && dotPos + 1 < opName.length()) {
//       return opName.substr(dotPos + 1);
//     }
//     return opName;
//   }

//   std::string normalizeLayerType(const std::string& type) {
//     std::string normalized = type;
//     std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
//     // 处理别名映射
//     if (normalized == "mp" || normalized == "maxpool") {
//       return "maxpoolsingleout";
//     } else if (normalized == "conv") {
//       return "conv";
//     } else if (normalized == "flatten") {
//       return "flatten";
//     } else if (normalized == "fc" || normalized == "matmul" || normalized == "gemm") {
//       // FC层可能使用MatMul或Gemm实现，都归类为相同类型
//       return "fc";
//     } else if (normalized == "add") {
//       return "add";
//     } else if (normalized == "if") {
//       return "if";
//     }
    
//     return normalized;
//   }

//   bool isAddOperation(Operation* op) {
//     return op->getName().getStringRef() == "onnx.Add";
//   }

//   void annotateLayerOperation(Operation* op, int layerIndex, int timestep) {
//     std::string opType = extractOpType(op->getName().getStringRef().str());
//     std::string newNodeName;
    
//     if (timestep == 0) {
//       newNodeName = "/layer/layer." + std::to_string(layerIndex) + "/" + opType;
//     } else {
//       newNodeName = "/layer/layer." + std::to_string(layerIndex) + "_" + std::to_string(timestep) + "/" + opType;
//     }
    
//     op->setAttr("onnx_node_name", StringAttr::get(op->getContext(), newNodeName));
    
//     LLVM_DEBUG(llvm::dbgs() << "  Annotated: " << op->getName().getStringRef() 
//                << " -> " << newNodeName << "\n");
//   }

//   void annotateAddOperation(Operation* op, int timestep) {
//     std::string newNodeName;
    
//     if (timestep == 0) {
//       newNodeName = "/Add";
//     } else {
//       newNodeName = "/Add_" + std::to_string(timestep);
//     }
    
//     op->setAttr("onnx_node_name", StringAttr::get(op->getContext(), newNodeName));
    
//     LLVM_DEBUG(llvm::dbgs() << "  Annotated Add: " << op->getName().getStringRef() 
//                << " -> " << newNodeName << "\n");
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {

// std::unique_ptr<Pass> createSNNStructureAnnotationPass() {
//   return std::make_unique<SNNStructureAnnotationPass>();
// }

// } // namespace onnx_mlir

// static mlir::PassRegistration<SNNStructureAnnotationPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <algorithm>

using namespace mlir;

#define DEBUG_TYPE "snn-structure-annotation"

namespace {

// Pass选项定义
static llvm::cl::opt<std::string> networkStructure(
    "network-structure",
    llvm::cl::desc("Network structure specification (comma-separated layer types)"),
    llvm::cl::value_desc("structure"),
    llvm::cl::init(""));

static llvm::cl::opt<int> timeSteps(
    "time-steps", 
    llvm::cl::desc("Number of time steps for SNN"),
    llvm::cl::value_desc("steps"),
    llvm::cl::init(1));

class SNNStructureAnnotationPass
    : public PassWrapper<SNNStructureAnnotationPass, OperationPass<func::FuncOp>> {
    
private:
  std::vector<std::string> networkStructure_; // 包含if的完整网络结构
  int numTimeSteps_;
  
  // 需要处理onnx_node_name的操作类型集合 - 添加新的操作类型
  std::set<std::string> targetOpTypes_ = {
    "onnx.Conv", "onnx.MaxPoolSingleOut", "onnx.Flatten", 
    "onnx.MatMul", "onnx.Gemm", "onnx.Add",
    "onnx.ReduceMeanV13", "onnx.ReduceSumV11", "onnx.Transpose", "onnx.Gather", "onnx.Div"  // 新添加的操作类型
  };

public:
  StringRef getArgument() const final { return "snn-structure-annotation"; }
  StringRef getDescription() const final {
    return "Annotate SNN network structure in onnx_node_name attributes based on user-provided structure";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    // 解析网络结构参数
    if (networkStructure.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No network structure specified, skipping pass\n");
      return;
    }
    
    parseNetworkStructure();
    numTimeSteps_ = timeSteps.getValue();
    
    if (numTimeSteps_ <= 0) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid time steps: " << numTimeSteps_ << "\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "=== SNN Structure Annotation Pass ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Network structure: " << networkStructure.getValue() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Time steps: " << numTimeSteps_ << "\n");
    
    if (failed(processOperationsWithStructure(funcOp))) {
      signalPassFailure();
    }
  }

private:
  void parseNetworkStructure() {
    networkStructure_.clear();
    std::stringstream ss(networkStructure.getValue());
    std::string item;
    
    while (std::getline(ss, item, ',')) {
      // 去除空格
      item.erase(0, item.find_first_not_of(" \t"));
      item.erase(item.find_last_not_of(" \t") + 1);
      
      if (!item.empty()) {
        networkStructure_.push_back(item);
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Parsed layer types: ");
    for (size_t i = 0; i < networkStructure_.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << "[" << i << "]" << networkStructure_[i] << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  LogicalResult processOperationsWithStructure(func::FuncOp funcOp) {
    Block& block = funcOp.getBody().front();
    
    // 收集所有目标操作
    std::vector<Operation*> targetOps;
    for (Operation& op : block) {
      if (isTargetOperation(&op)) {
        targetOps.push_back(&op);
      }
    }
    
    if (targetOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No target operations found\n");
      return success();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << targetOps.size() << " target operations\n");
    
    return annotateOperationsWithStructure(targetOps);
  }

  bool isTargetOperation(Operation* op) {
    std::string opName = op->getName().getStringRef().str();
    return targetOpTypes_.count(opName) > 0;
  }

  LogicalResult annotateOperationsWithStructure(const std::vector<Operation*>& targetOps) {
    int currentOpIndex = 0;
    
    for (int timestep = 0; timestep < numTimeSteps_; ++timestep) {
      LLVM_DEBUG(llvm::dbgs() << "\n--- Processing timestep " << timestep << " ---\n");
      
      // 处理当前时间步的网络结构
      for (size_t layerIndex = 0; layerIndex < networkStructure_.size(); ++layerIndex) {
        const std::string& layerType = networkStructure_[layerIndex];
        
        LLVM_DEBUG(llvm::dbgs() << "Processing layer " << layerIndex << " (" << layerType << ")\n");
        
        // 特殊情况：第一层在后续时间步被跳过（复用第一个时间步的结果）
        if (timestep > 0 && layerIndex == 0 && !isIfLayerType(layerType)) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping first layer in timestep " << timestep 
                     << " (reusing from timestep 0)\n");
          continue;
        }
        
        // 如果是if层，跳过（不需要处理任何实际的op，但占用层索引）
        if (isIfLayerType(layerType)) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping IF layer at index " << layerIndex << "\n");
          continue;
        }
        
        // 处理Add操作（时间步结束标记）
        if (isAddLayerType(layerType)) {
          if (currentOpIndex < targetOps.size()) {
            Operation* op = findNextAddOperation(targetOps, currentOpIndex);
            if (op) {
              annotateAddOperation(op, timestep);
              // 更新索引到找到的Add操作之后
              currentOpIndex = findOpIndex(targetOps, op) + 1;
              LLVM_DEBUG(llvm::dbgs() << "Processed Add operation for timestep " << timestep << "\n");
            } else {
              LLVM_DEBUG(llvm::dbgs() << "Warning: Could not find Add operation for timestep " << timestep << "\n");
            }
          }
        }
        // 处理普通计算层操作
        else {
          Operation* op = findNextMatchingOperation(targetOps, currentOpIndex, layerType);
          if (op) {
            annotateLayerOperation(op, layerIndex, timestep);
            // 更新索引到找到的操作之后
            currentOpIndex = findOpIndex(targetOps, op) + 1;
            LLVM_DEBUG(llvm::dbgs() << "Processed " << layerType 
                       << " operation (layer " << layerIndex 
                       << ", timestep " << timestep << ")\n");
          } else {
            LLVM_DEBUG(llvm::dbgs() << "Warning: Could not find " << layerType 
                       << " operation for layer " << layerIndex 
                       << " in timestep " << timestep << "\n");
          }
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Annotation completed. Processed operations across " 
               << numTimeSteps_ << " timesteps\n");
    
    return success();
  }

  bool isIfLayerType(const std::string& layerType) {
    std::string normalizedType = normalizeLayerType(layerType);
    return normalizedType == "if";
  }

  bool isAddLayerType(const std::string& layerType) {
    std::string normalizedType = normalizeLayerType(layerType);
    return normalizedType == "add";
  }

  Operation* findNextAddOperation(const std::vector<Operation*>& targetOps, int startIndex) {
    for (int i = startIndex; i < targetOps.size(); ++i) {
      if (isAddOperation(targetOps[i])) {
        return targetOps[i];
      }
    }
    return nullptr;
  }

  Operation* findNextMatchingOperation(const std::vector<Operation*>& targetOps, 
                                       int startIndex, 
                                       const std::string& expectedType) {
    for (int i = startIndex; i < targetOps.size(); ++i) {
      Operation* op = targetOps[i];
      
      // 跳过Add操作，它们用于时间步分界
      if (isAddOperation(op)) {
        continue;
      }
      
      if (matchesLayerType(op, expectedType)) {
        return op;
      }
    }
    return nullptr;
  }

  int findOpIndex(const std::vector<Operation*>& targetOps, Operation* target) {
    for (int i = 0; i < targetOps.size(); ++i) {
      if (targetOps[i] == target) {
        return i;
      }
    }
    return -1;
  }

  bool matchesLayerType(Operation* op, const std::string& expectedType) {
    std::string opName = op->getName().getStringRef().str();
    std::string opType = extractOpType(opName);
    
    // 标准化比较
    std::string normalizedExpected = normalizeLayerType(expectedType);
    std::string normalizedActual = normalizeLayerType(opType);
    
    return normalizedExpected == normalizedActual;
  }

  std::string extractOpType(const std::string& opName) {
    // 从 "onnx.Conv" 提取 "Conv"
    // 从 "onnx.ReduceMeanV13" 提取 "ReduceMeanV13"
    size_t dotPos = opName.find('.');
    if (dotPos != std::string::npos && dotPos + 1 < opName.length()) {
      return opName.substr(dotPos + 1);
    }
    return opName;
  }

  std::string normalizeLayerType(const std::string& type) {
    std::string normalized = type;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // 处理别名映射 - 添加新操作的映射
    if (normalized == "mp" || normalized == "maxpool") {
      return "maxpoolsingleout";
    } else if (normalized == "conv") {
      return "conv";
    } else if (normalized == "flatten") {
      return "flatten";
    } else if (normalized == "fc" || normalized == "matmul" || normalized == "gemm") {
      // FC层可能使用MatMul或Gemm实现，都归类为相同类型
      return "fc";
    } else if (normalized == "add") {
      return "add";
    } else if (normalized == "if") {
      return "if";
    } 
    // 新添加的操作类型映射
    else if (normalized == "reducemean" || normalized == "reducemeanv13") {
      return "reducemeanv13";
    } else if (normalized == "reducesum" || normalized == "reducesumv11") {
      return "reducesumv11";
    } else if (normalized == "transpose") {
      return "transpose";
    } else if (normalized == "gather") {
      return "gather";
    } else if (normalized == "div") {
      return "div";
    }
    
    return normalized;
  }

  bool isAddOperation(Operation* op) {
    return op->getName().getStringRef() == "onnx.Add";
  }

  void annotateLayerOperation(Operation* op, int layerIndex, int timestep) {
    std::string opType = extractOpType(op->getName().getStringRef().str());
    std::string newNodeName;
    
    if (timestep == 0) {
      newNodeName = "/layer/layer." + std::to_string(layerIndex) + "/" + opType;
    } else {
      newNodeName = "/layer/layer." + std::to_string(layerIndex) + "_" + std::to_string(timestep) + "/" + opType;
    }
    
    op->setAttr("onnx_node_name", StringAttr::get(op->getContext(), newNodeName));
    
    LLVM_DEBUG(llvm::dbgs() << "  Annotated: " << op->getName().getStringRef() 
               << " -> " << newNodeName << "\n");
  }

  void annotateAddOperation(Operation* op, int timestep) {
    std::string newNodeName;
    
    if (timestep == 0) {
      newNodeName = "/Add";
    } else {
      newNodeName = "/Add_" + std::to_string(timestep);
    }
    
    op->setAttr("onnx_node_name", StringAttr::get(op->getContext(), newNodeName));
    
    LLVM_DEBUG(llvm::dbgs() << "  Annotated Add: " << op->getName().getStringRef() 
               << " -> " << newNodeName << "\n");
  }
};

} // end anonymous namespace

namespace onnx_mlir {

std::unique_ptr<Pass> createSNNStructureAnnotationPass() {
  return std::make_unique<SNNStructureAnnotationPass>();
}

} // namespace onnx_mlir

static mlir::PassRegistration<SNNStructureAnnotationPass> pass;