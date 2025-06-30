#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <memory>

// Include our component headers
// In a real build system, these would be proper includes
#include "DependencyGraph.h"
#include "TopoLogicalSort.h"
#include "IrReorganization.h"
#include "EnhancedDependencyGraph.h"
#include "EnhancedIrReorganization.h"

using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "hierarchical-parallel-scheduler"

namespace {

struct HierarchicalParallelSchedulerPass
    : public PassWrapper<HierarchicalParallelSchedulerPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "hierarchical-parallel-scheduler"; }
  StringRef getDescription() const final {
    return "Hierarchical parallelize GPU kernels and loop nests based on dependencies(need stream-unification pass)";
  }
  
void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "Processing function: " << funcOp.getName() << "\n");
        
        if (funcOp.getBody().empty()) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping function " << funcOp.getName() << " (empty body)\n");
            continue;
        }
        
        auto enhancedGraph = buildEnhancedDependencyGraph(funcOp);
        
        LLVM_DEBUG(llvm::dbgs() << "Built enhanced dependency graph with " 
                << enhancedGraph->nodes.size() << " nodes\n");
        
        if (enhancedGraph->nodes.empty()) {
            LLVM_DEBUG(llvm::dbgs() << "No parallelizable nodes found in function " 
                    << funcOp.getName() << ", skipping\n");
            continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "Enhanced scheduling completed during graph building\n");
        
        try {
            reorganizeIRWithEnhancedScheduling_plus(funcOp, *enhancedGraph);
            LLVM_DEBUG(llvm::dbgs() << "Enhanced IR reorganization completed for function: " 
                      << funcOp.getName() << "\n");
        } catch (const std::exception& e) {
            llvm::errs() << "Error in enhanced IR reorganization for function " << funcOp.getName() 
                        << ": " << e.what() << "\n";
            return signalPassFailure();
        }
        
        try {
            reorganizeGPUModules_plus(moduleOp, *enhancedGraph);
            LLVM_DEBUG(llvm::dbgs() << "Enhanced GPU module reorganization completed\n");
        } catch (const std::exception& e) {
            llvm::errs() << "Error in enhanced GPU module reorganization: " << e.what() << "\n";
            return signalPassFailure();
        }
    }
}
};

} // end anonymous namespace

namespace onnx_mlir {
  
    std::unique_ptr<Pass> createHierarchicalParallelSchedulerPass() {
    return std::make_unique<HierarchicalParallelSchedulerPass>();
    }
  
  } // namespace onnx_mlir
  
static mlir::PassRegistration<HierarchicalParallelSchedulerPass> pass;