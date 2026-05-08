// //增加对batch维度的处理
// #include "mlir/Pass/Pass.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/SymbolTable.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// #include "src/Dialect/ONNX/ONNXOps.hpp"
// #include "src/Dialect/Krnl/KrnlOps.hpp"

// using namespace mlir;

// #define DEBUG_TYPE "convert-onnx-lif-to-krnl"

// // ═══════════════════════════════════════════════════════════════════════
// // 硬件限制
// // ═══════════════════════════════════════════════════════════════════════
// static constexpr int64_t kHardwareNeuronNum = 1024;

// namespace {

// static func::FuncOp getOrInsertFunc(OpBuilder &builder, ModuleOp module,
//                                     Location loc, StringRef name,
//                                     FunctionType funcType) {
//   if (auto existing = module.lookupSymbol<func::FuncOp>(name))
//     return existing;
//   OpBuilder::InsertionGuard guard(builder);
//   builder.setInsertionPointToStart(module.getBody());
//   auto funcOp = builder.create<func::FuncOp>(loc, name, funcType);
//   funcOp.setVisibility(SymbolTable::Visibility::Private);
//   return funcOp;
// }

// static void declareIntrinsics(OpBuilder &builder, ModuleOp module,
//                               Location loc) {
//   auto i32T = builder.getI32Type();
//   auto i64T = builder.getI64Type();
//   auto *ctx = builder.getContext();
//   getOrInsertFunc(builder, module, loc, "rvne_clear_neuron_data_1024",
//                   FunctionType::get(ctx, {}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_ncr",
//                   FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_leakage_integral_fire_1024",
//                   FunctionType::get(ctx, {}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_read_sor_group2",
//                   FunctionType::get(ctx, {i32T}, {i64T}));
//   getOrInsertFunc(builder, module, loc, "rvne_read_ncr_group2",
//                   FunctionType::get(ctx, {i32T}, {i64T}));
// }

// static Value tensorToMemref(OpBuilder &b, Location loc,
//                             Value tensor, MemRefType memrefType) {
//   return b.create<UnrealizedConversionCastOp>(
//       loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
// }

// static Value memrefToTensor(OpBuilder &b, Location loc,
//                             Value memref, RankedTensorType tensorType) {
//   return b.create<UnrealizedConversionCastOp>(
//       loc, TypeRange{tensorType}, ValueRange{memref}).getResult(0);
// }

// static Value castElemToI32ForNCR(OpBuilder &b, Location loc,
//                                   Value elem, Type elemTy) {
//   if (elemTy == b.getI32Type()) return elem;
//   if (elemTy.isF32())
//     return b.create<arith::BitcastOp>(loc, b.getI32Type(), elem);
//   llvm_unreachable("castElemToI32ForNCR: unsupported element type");
// }

// static Value castI32FromNCRToElemTy(OpBuilder &b, Location loc,
//                                      Value val_i32, Type dstTy) {
//   if (dstTy == b.getI32Type()) return val_i32;
//   if (dstTy.isF32())
//     return b.create<arith::BitcastOp>(loc, b.getF32Type(), val_i32);
//   llvm_unreachable("castI32FromNCRToElemTy: unsupported element type");
// }

// static Value castSpikeToElemTy(OpBuilder &b, Location loc,
//                                Value spike_i32, Type dstTy) {
//   if (dstTy == b.getI32Type()) return spike_i32;
//   if (dstTy.isF32())
//     return b.create<arith::SIToFPOp>(loc, b.getF32Type(), spike_i32);
//   llvm_unreachable("castSpikeToElemTy: unsupported spike element type");
// }

// static void buildKrnlLoop(
//     OpBuilder &builder, Location loc, Value lb, Value ub,
//     function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
//   auto defineOp = builder.create<KrnlDefineLoopsOp>(loc, 1);
//   Value loop = defineOp.getResults()[0];
//   auto iterOp = builder.create<KrnlIterateOp>(
//       loc, ValueRange{loop}, ValueRange{loop},
//       ValueRange{lb}, ValueRange{ub}, ValueRange{}, nullptr);
//   Block &body = iterOp.getBodyRegion().front();
//   Value iv = body.getArgument(0);
//   OpBuilder::InsertionGuard guard(builder);
//   builder.setInsertionPointToStart(&body);
//   bodyBuilder(builder, loc, iv);
// }

// // ═══════════════════════════════════════════════════════════════════════
// // 核心 Rewrite Pattern
// // ═══════════════════════════════════════════════════════════════════════
// struct ONNXLIFOpToKrnlConversion : public OpRewritePattern<ONNXLIFOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(ONNXLIFOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto module  = op->getParentOfType<ModuleOp>();

//     declareIntrinsics(rewriter, module, loc);

//     Value x          = op.getX();
//     Value v_combined = op.getVCombined();
//     auto xType  = cast<RankedTensorType>(x.getType());
//     auto vcType = cast<RankedTensorType>(v_combined.getType());

//     int64_t B  = xType.getDimSize(0);   // batch size = 8
//     int64_t N  = xType.getDimSize(1);   // neuron num = 640
//     int64_t N2 = vcType.getDimSize(1);  // 2*N = 1280
//     assert(N2 == 2 * N && N2 % 2 == 0);

//     // 分批参数（编译期确定）
//     // N=640 < 1024 → numBatches=1，只有一批
//     int64_t numBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

//     Type xElemTy      = xType.getElementType();
//     Type vcElemTy     = vcType.getElementType();
//     auto spikeResType = cast<RankedTensorType>(op.getResult(0).getType());
//     auto vcNextResType= cast<RankedTensorType>(op.getResult(1).getType());
//     Type spikeElemTy  = spikeResType.getElementType();
//     Type vcNextElemTy = vcNextResType.getElementType();

//     auto mkIndex = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantIndexOp>(loc, v);
//     };
//     auto mkI64 = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantOp>(
//           loc, rewriter.getI64IntegerAttr(v));
//     };

//     Value cst_0idx       = mkIndex(0);
//     Value cst_B          = mkIndex(B);
//     Value cst_64idx      = mkIndex(64);
//     Value cst_64_i64     = mkI64(64);
//     Value cst_1_i64      = mkI64(1);
//     Value cst_2_i64      = mkI64(2);
//     Value cst_32_i64     = mkI64(32);
//     Value cst_mask32_i64 = rewriter.create<arith::ConstantOp>(
//         loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

//     // 分配输出 memref
//     Value spikeBuf  = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({B, N},  spikeElemTy));
//     Value vcNextBuf = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({B, N2}, vcNextElemTy));

//     // tensor → memref
//     Value xBuf  = tensorToMemref(rewriter, loc, x,
//                       MemRefType::get({B, N},  xElemTy));
//     Value vcBuf = tensorToMemref(rewriter, loc, v_combined,
//                       MemRefType::get({B, N2}, vcElemTy));

//     // ─────────────────────────────────────────────────────────────────
//     // Batch 维循环
//     //
//     // 关键语义：
//     //   每个 batch 样本对应一次完整的硬件执行。
//     //   硬件没有 batch 概念，batch 就是串行地对同一硬件重复执行 B 次。
//     //   每次执行前必须 clear，因为硬件状态来自上一个 batch 的 v_combined
//     //   已经通过 rvne_write_ncr 注入，不需要保留上次的硬件残留。
//     //
//     //   对比 ConvolutionFunction：
//     //     外层 for(oc)  → 对应这里的 for(b)（串行处理每个独立样本）
//     //     内层 for(t)   → 对应这里的神经元分批循环
//     //     rvne_clear_neuron_data_1024 在每个 oc 批次开始时调用
//     //     → 这里在每个 batch 样本开始时调用
//     // ─────────────────────────────────────────────────────────────────
//     buildKrnlLoop(rewriter, loc, cst_0idx, cst_B,
//       [&](OpBuilder &b, Location loc, Value ib) {
//         // ib: 当前 batch 样本的 index（index 类型）

//         // 神经元分批循环（编译期展开，N=640时只有1批）
//         for (int64_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
//           int64_t neuronStart  = batchIdx * kHardwareNeuronNum;
//           int64_t neuronEnd    = std::min(neuronStart + kHardwareNeuronNum, N);
//           int64_t calN         = neuronEnd - neuronStart;
//           // calN 应为偶数（实际网络中神经元数总是2的倍数）
//           assert(calN % 2 == 0 && "calN must be even for NCR group2 reads");

//           // 每个 batch 样本的每个神经元批次开始前清空硬件状态
//           // 语义：丢弃硬件中可能残留的上一个 batch/批次的垃圾数据
//           //       本次计算所需的初始状态（V, I）通过下面的 write_ncr 注入
//           b.create<func::CallOp>(loc, "rvne_clear_neuron_data_1024",
//                                  TypeRange{}, ValueRange{});

//           // ── 写入膜电位 V → NCR[0 .. calN-1] ───────────────────
//           // vcBuf[ib][neuronStart .. neuronEnd-1] → NCR[0 .. calN-1]
//           // 硬件 NCR offset 从0开始（批内局部编号）
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(calN),
//             [&](OpBuilder &b, Location loc, Value local_n) {
//               // local_n: 批内局部下标 [0, calN)
//               // vcBuf 的全局列下标 = neuronStart + local_n
//               Value local_i64  = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), local_n);
//               Value global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(neuronStart));
//               Value global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), global_i64);

//               // 读取 v_combined[ib][global] 中的 V 值
//               Value v_elem = b.create<memref::LoadOp>(loc, vcBuf,
//                                  ValueRange{ib, global_idx});
//               Value v_i32  = castElemToI32ForNCR(b, loc, v_elem, vcElemTy);

//               // NCR offset = 局部下标（i32）
//               Value ncr_off = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), local_n);
//               b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{},
//                                      ValueRange{v_i32, ncr_off});
//             });

//           // ── 写入电流 I+x → NCR[calN .. 2*calN-1] ──────────────
//           // vcBuf[ib][N+neuronStart .. N+neuronEnd-1] 是 I_old
//           // xBuf[ib][neuronStart .. neuronEnd-1] 是本步输入 x
//           // I_input = I_old + x（软件叠加后写入 NCR[calN+local]）
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(calN),
//             [&](OpBuilder &b, Location loc, Value local_n) {
//               Value local_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), local_n);

//               // I_old 的全局列下标 = N + neuronStart + local_n
//               Value i_global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(N + neuronStart));
//               Value i_global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), i_global_i64);

//               // x 的全局列下标 = neuronStart + local_n
//               Value x_global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(neuronStart));
//               Value x_global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), x_global_i64);

//               Value i_elem = b.create<memref::LoadOp>(loc, vcBuf,
//                                  ValueRange{ib, i_global_idx});
//               Value x_elem = b.create<memref::LoadOp>(loc, xBuf,
//                                  ValueRange{ib, x_global_idx});

//               // 软件叠加
//               Value i_plus_x = vcElemTy.isF32()
//                   ? b.create<arith::AddFOp>(loc, i_elem, x_elem).getResult()
//                   : b.create<arith::AddIOp>(loc, i_elem, x_elem).getResult();

//               Value val_i32 = castElemToI32ForNCR(b, loc, i_plus_x, vcElemTy);

//               // NCR offset = calN + local_n（i64 → i32，截断安全）
//               Value ncr_off_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(calN));
//               Value ncr_off_i32 = b.create<arith::TruncIOp>(
//                   loc, b.getI32Type(), ncr_off_i64);
//               b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{},
//                                      ValueRange{val_i32, ncr_off_i32});
//             });

//           // ── 批量 LIF ─────────────────────────────────────────
//           b.create<func::CallOp>(loc, "rvne_leakage_integral_fire_1024",
//                                  TypeRange{}, ValueRange{});

//           // ── 读取 spike → spikeBuf[ib][neuronStart..neuronEnd-1]
//           int64_t numSpikeGroups = (calN + 63) / 64;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(numSpikeGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i32 = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), ig);
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               auto rc = b.create<func::CallOp>(
//                   loc, "rvne_read_sor_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ig_i32});
//               Value raw64 = rc.getResult(0);

//               buildKrnlLoop(b, loc, mkIndex(0), cst_64idx,
//                 [&](OpBuilder &b, Location loc, Value ibit) {
//                   Value ibit_i64 = b.create<arith::IndexCastOp>(
//                       loc, b.getI64Type(), ibit);

//                   // 批内局部编号
//                   Value local_n = b.create<arith::AddIOp>(
//                       loc,
//                       b.create<arith::MulIOp>(loc, ig_i64, cst_64_i64),
//                       ibit_i64);

//                   // 越界检查
//                   Value inBounds = b.create<arith::CmpIOp>(
//                       loc, arith::CmpIPredicate::ult,
//                       local_n, mkI64(calN));

//                   // 全局列下标 = neuronStart + local_n
//                   Value global_n = b.create<arith::AddIOp>(
//                       loc, local_n, mkI64(neuronStart));
//                   Value global_idx = b.create<arith::IndexCastOp>(
//                       loc, b.getIndexType(), global_n);

//                   // 位提取
//                   Value spike_i32 = b.create<arith::TruncIOp>(
//                       loc, b.getI32Type(),
//                       b.create<arith::AndIOp>(
//                           loc,
//                           b.create<arith::ShRUIOp>(loc, raw64, ibit_i64),
//                           cst_1_i64));
//                   Value spike_val = castSpikeToElemTy(
//                       b, loc, spike_i32, spikeElemTy);

//                   b.create<scf::IfOp>(loc, inBounds,
//                     [&](OpBuilder &b, Location loc) {
//                       b.create<memref::StoreOp>(loc, spike_val, spikeBuf,
//                                                 ValueRange{ib, global_idx});
//                       b.create<scf::YieldOp>(loc);
//                     });
//                 });
//             });

//           // ── 读回 V_next → vcNextBuf[ib][neuronStart..neuronEnd-1]
//           // NCR[0 .. calN-1] 存放更新后的 V
//           int64_t vNCRGroups = calN / 2;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(vNCRGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i32 = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), ig);
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               auto rb = b.create<func::CallOp>(
//                   loc, "rvne_read_ncr_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ig_i32});
//               Value pair64 = rb.getResult(0);

//               Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::AndIOp>(loc, pair64, cst_mask32_i64));
//               Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::ShRUIOp>(loc, pair64, cst_32_i64));

//               // 局部下标
//               Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst_2_i64);
//               Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst_1_i64);

//               // 全局 V 列下标 = neuronStart + local
//               Value glo_lo = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_lo, mkI64(neuronStart)));
//               Value glo_hi = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_hi, mkI64(neuronStart)));

//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_lo});
//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_hi});
//             });

//           // ── 读回 I_next → vcNextBuf[ib][N+neuronStart..N+neuronEnd-1]
//           // NCR[calN .. 2*calN-1] 存放更新后的 I
//           // ncrg2_index 从 calN/2 开始
//           int64_t iNCRStart  = calN / 2;
//           int64_t iNCRGroups = calN / 2;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(iNCRGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               // ncrg2_index = iNCRStart + ig
//               Value ncr_grp_i32 = b.create<arith::TruncIOp>(
//                   loc, b.getI32Type(),
//                   b.create<arith::AddIOp>(loc, ig_i64, mkI64(iNCRStart)));

//               auto rb = b.create<func::CallOp>(
//                   loc, "rvne_read_ncr_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ncr_grp_i32});
//               Value pair64 = rb.getResult(0);

//               Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::AndIOp>(loc, pair64, cst_mask32_i64));
//               Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::ShRUIOp>(loc, pair64, cst_32_i64));

//               // 局部 I 下标
//               Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst_2_i64);
//               Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst_1_i64);

//               // 全局 I 列下标 = N + neuronStart + local
//               Value glo_lo = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_lo, mkI64(N + neuronStart)));
//               Value glo_hi = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_hi, mkI64(N + neuronStart)));

//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_lo});
//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_hi});
//             });

//         } // end for batchIdx（神经元分批，编译期展开）
//       }); // end buildKrnlLoop（batch 样本串行循环）

//     Value spikeOut  = memrefToTensor(rewriter, loc, spikeBuf,  spikeResType);
//     Value vcNextOut = memrefToTensor(rewriter, loc, vcNextBuf, vcNextResType);
//     rewriter.replaceOp(op, ValueRange{spikeOut, vcNextOut});
//     return success();
//   }
// };

// class ConvertONNXSNNToKrnlPass
//     : public PassWrapper<ConvertONNXSNNToKrnlPass,
//                          OperationPass<func::FuncOp>> {
// public:
//   StringRef getArgument()    const final { return "convert-onnx-lif-to-krnl"; }
//   StringRef getDescription() const final {
//     return "Convert onnx.LIF to krnl + hardware intrinsics. "
//            "Batch dimension is serialized (hardware has no batch concept). "
//            "Each batch sample is one full hardware execution.";
//   }
//   void runOnOperation() override {
//     func::FuncOp funcOp = getOperation();
//     MLIRContext *ctx = &getContext();
//     RewritePatternSet patterns(ctx);
//     patterns.add<ONNXLIFOpToKrnlConversion>(ctx);
//     if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
//       signalPassFailure();
//   }
// };

// } // end anonymous namespace

// namespace onnx_mlir {
// std::unique_ptr<Pass> createConvertONNXSNNToKrnlPass() {
//   return std::make_unique<ConvertONNXSNNToKrnlPass>();
// }
// } // namespace onnx_mlir

// static mlir::PassRegistration<ConvertONNXSNNToKrnlPass> pass;

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

#define DEBUG_TYPE "convert-onnx-lif-to-krnl"

static constexpr int64_t kHardwareNeuronNum = 1024;

namespace {

// ═══════════════════════════════════════════════════════════════════════
// 辅助工具函数
// ═══════════════════════════════════════════════════════════════════════

static func::FuncOp getOrInsertFunc(OpBuilder &builder, ModuleOp module,
                                    Location loc, StringRef name,
                                    FunctionType funcType) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<func::FuncOp>(loc, name, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  return funcOp;
}

static void declareIntrinsics(OpBuilder &builder, ModuleOp module, Location loc) {
  auto i32T = builder.getI32Type();
  auto i64T = builder.getI64Type();
  auto *ctx = builder.getContext();
  getOrInsertFunc(builder, module, loc, "rvne_clear_neuron_data_1024", FunctionType::get(ctx, {}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_write_ncr", FunctionType::get(ctx, {i32T, i32T}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_leakage_integral_fire_1024", FunctionType::get(ctx, {}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_read_sor_group2", FunctionType::get(ctx, {i32T}, {i64T}));
  getOrInsertFunc(builder, module, loc, "rvne_read_ncr_group2", FunctionType::get(ctx, {i32T}, {i64T}));
}

static void declareSNNFCIntrinsics(OpBuilder &builder, ModuleOp module, Location loc) {
  auto i32T = builder.getI32Type();
  auto i64T = builder.getI64Type();
  auto *ctx = builder.getContext();
  getOrInsertFunc(builder, module, loc, "rvne_spike_propagation_1024", FunctionType::get(ctx, {}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_write_svr", FunctionType::get(ctx, {i32T, i32T}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_write_wvr", FunctionType::get(ctx, {i32T, i32T}, {}));
  getOrInsertFunc(builder, module, loc, "rvne_current_acc_128", FunctionType::get(ctx, {i32T, i32T, i32T}, {}));
}


// static void declareSNNConvIntrinsics(OpBuilder &builder, ModuleOp module, Location loc) {
//   auto i32T = builder.getI32Type();
//   auto *ctx = builder.getContext();

//   auto outPtrTy = MemRefType::get({1}, i32T);

//   getOrInsertFunc(builder, module, loc, "rvne_set_reset_current", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_set_reset_voltage", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_set_neuron_type_32", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_set_voltage_threshold", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_save_current_32", FunctionType::get(ctx, {i32T, outPtrTy}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_set_leakage_index",
//                   FunctionType::get(ctx, {i32T, i32T, i32T, i32T, i32T, i32T,
//                                           i32T, i32T, i32T, i32T, i32T, i32T}, {}));

//   getOrInsertFunc(builder, module, loc, "rvne_write_svr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_sor", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_ncr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_nvr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_wvr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_clear_neuron_data_1024", FunctionType::get(ctx, {}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_clear_neuron_state_1024", FunctionType::get(ctx, {}, {}));

//   getOrInsertFunc(builder, module, loc, "rvne_current_acc_32", FunctionType::get(ctx, {i32T, i32T, i32T}, {}));
// }

// static void declareSNNConvIntrinsics(OpBuilder &builder, ModuleOp module, Location loc) {
//   auto i32T = builder.getI32Type();
//   auto i64T = builder.getI64Type();
//   auto *ctx = builder.getContext();
//   getOrInsertFunc(builder, module, loc, "rvne_spike_propagation_1024", FunctionType::get(ctx, {}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_svr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_write_wvr", FunctionType::get(ctx, {i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_read_nsr_group8", FunctionType::get(ctx, {i32T}, {i64T}));
//   getOrInsertFunc(builder, module, loc, "rvne_current_acc_128", FunctionType::get(ctx, {i32T, i32T, i32T}, {}));
//   getOrInsertFunc(builder, module, loc, "rvne_clear_neuron_state_1024",
//                 FunctionType::get(ctx, {}, {}));
// }



static Value tensorToMemref(OpBuilder &b, Location loc, Value tensor, MemRefType memrefType) {
  return b.create<UnrealizedConversionCastOp>(loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
}

static Value memrefToTensor(OpBuilder &b, Location loc, Value memref, RankedTensorType tensorType) {
  return b.create<UnrealizedConversionCastOp>(loc, TypeRange{tensorType}, ValueRange{memref}).getResult(0);
}

static Value castElemToI32ForNCR(OpBuilder &b, Location loc, Value elem, Type elemTy) {
  if (elemTy == b.getI32Type()) return elem;
  if (elemTy.isF32()) return b.create<arith::BitcastOp>(loc, b.getI32Type(), elem);
  if (elemTy.isIndex()) {
      Value i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), elem);
      return b.create<arith::TruncIOp>(loc, b.getI32Type(), i64);
  }
  return b.create<arith::TruncIOp>(loc, b.getI32Type(), elem);
}

static Value castI32FromNCRToElemTy(OpBuilder &b, Location loc, Value val_i32, Type dstTy) {
  if (dstTy == b.getI32Type()) return val_i32;
  if (dstTy.isF32()) return b.create<arith::BitcastOp>(loc, b.getF32Type(), val_i32);
  return b.create<arith::ExtSIOp>(loc, dstTy, val_i32);
}

static Value castSpikeToElemTy(OpBuilder &b, Location loc, Value spike_i32, Type dstTy) {
  if (dstTy == b.getI32Type()) return spike_i32;
  if (dstTy.isF32()) return b.create<arith::SIToFPOp>(loc, b.getF32Type(), spike_i32);
  return b.create<arith::ExtSIOp>(loc, dstTy, spike_i32);
}

static void buildKrnlLoop(OpBuilder &builder, Location loc, Value lb, Value ub,
                          function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  auto defineOp = builder.create<KrnlDefineLoopsOp>(loc, 1);
  Value loop = defineOp.getResults()[0];
  auto iterOp = builder.create<KrnlIterateOp>(loc, ValueRange{loop}, ValueRange{loop},
                                             ValueRange{lb}, ValueRange{ub}, ValueRange{}, nullptr);
  Block &body = iterOp.getBodyRegion().front();
  Value iv = body.getArgument(0);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);
  bodyBuilder(builder, loc, iv);
}

// struct ONNXLIFOpToKrnlConversion : public OpRewritePattern<ONNXLIFOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(ONNXLIFOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto module  = op->getParentOfType<ModuleOp>();

//     declareIntrinsics(rewriter, module, loc);

//     Value x          = op.getX();
//     Value v_combined = op.getVCombined();
//     auto xType  = cast<RankedTensorType>(x.getType());
//     auto vcType = cast<RankedTensorType>(v_combined.getType());

//     int64_t B  = xType.getDimSize(0);   // batch size = 8
//     int64_t N  = xType.getDimSize(1);   // neuron num = 640
//     int64_t N2 = vcType.getDimSize(1);  // 2*N = 1280
//     assert(N2 == 2 * N && N2 % 2 == 0);

//     // 分批参数（编译期确定）
//     // N=640 < 1024 → numBatches=1，只有一批
//     int64_t numBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

//     Type xElemTy      = xType.getElementType();
//     Type vcElemTy     = vcType.getElementType();
//     auto spikeResType = cast<RankedTensorType>(op.getResult(0).getType());
//     auto vcNextResType= cast<RankedTensorType>(op.getResult(1).getType());
//     Type spikeElemTy  = spikeResType.getElementType();
//     Type vcNextElemTy = vcNextResType.getElementType();

//     auto mkIndex = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantIndexOp>(loc, v);
//     };
//     auto mkI64 = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantOp>(
//           loc, rewriter.getI64IntegerAttr(v));
//     };

//     Value cst_0idx       = mkIndex(0);
//     Value cst_B          = mkIndex(B);
//     Value cst_64idx      = mkIndex(64);
//     Value cst_64_i64     = mkI64(64);
//     Value cst_1_i64      = mkI64(1);
//     Value cst_2_i64      = mkI64(2);
//     Value cst_32_i64     = mkI64(32);
//     Value cst_mask32_i64 = rewriter.create<arith::ConstantOp>(
//         loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

//     // 分配输出 memref
//     Value spikeBuf  = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({B, N},  spikeElemTy));
//     Value vcNextBuf = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({B, N2}, vcNextElemTy));

//     // tensor → memref
//     Value xBuf  = tensorToMemref(rewriter, loc, x,
//                       MemRefType::get({B, N},  xElemTy));
//     Value vcBuf = tensorToMemref(rewriter, loc, v_combined,
//                       MemRefType::get({B, N2}, vcElemTy));

//     // ─────────────────────────────────────────────────────────────────
//     // Batch 维循环
//     //
//     // 关键语义：
//     //   每个 batch 样本对应一次完整的硬件执行。
//     //   硬件没有 batch 概念，batch 就是串行地对同一硬件重复执行 B 次。
//     //   每次执行前必须 clear，因为硬件状态来自上一个 batch 的 v_combined
//     //   已经通过 rvne_write_ncr 注入，不需要保留上次的硬件残留。
//     //
//     //   对比 ConvolutionFunction：
//     //     外层 for(oc)  → 对应这里的 for(b)（串行处理每个独立样本）
//     //     内层 for(t)   → 对应这里的神经元分批循环
//     //     rvne_clear_neuron_data_1024 在每个 oc 批次开始时调用
//     //     → 这里在每个 batch 样本开始时调用
//     // ─────────────────────────────────────────────────────────────────
//     buildKrnlLoop(rewriter, loc, cst_0idx, cst_B,
//       [&](OpBuilder &b, Location loc, Value ib) {
//         // ib: 当前 batch 样本的 index（index 类型）

//         // 神经元分批循环（编译期展开，N=640时只有1批）
//         for (int64_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
//           int64_t neuronStart  = batchIdx * kHardwareNeuronNum;
//           int64_t neuronEnd    = std::min(neuronStart + kHardwareNeuronNum, N);
//           int64_t calN         = neuronEnd - neuronStart;
//           // calN 应为偶数（实际网络中神经元数总是2的倍数）
//           assert(calN % 2 == 0 && "calN must be even for NCR group2 reads");

//           // 每个 batch 样本的每个神经元批次开始前清空硬件状态
//           // 语义：丢弃硬件中可能残留的上一个 batch/批次的垃圾数据
//           //       本次计算所需的初始状态（V, I）通过下面的 write_ncr 注入
//           b.create<func::CallOp>(loc, "rvne_clear_neuron_data_1024",
//                                  TypeRange{}, ValueRange{});

//           // ── 写入膜电位 V → NCR[0 .. calN-1] ───────────────────
//           // vcBuf[ib][neuronStart .. neuronEnd-1] → NCR[0 .. calN-1]
//           // 硬件 NCR offset 从0开始（批内局部编号）
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(calN),
//             [&](OpBuilder &b, Location loc, Value local_n) {
//               // local_n: 批内局部下标 [0, calN)
//               // vcBuf 的全局列下标 = neuronStart + local_n
//               Value local_i64  = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), local_n);
//               Value global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(neuronStart));
//               Value global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), global_i64);

//               // 读取 v_combined[ib][global] 中的 V 值
//               Value v_elem = b.create<memref::LoadOp>(loc, vcBuf,
//                                  ValueRange{ib, global_idx});
//               Value v_i32  = castElemToI32ForNCR(b, loc, v_elem, vcElemTy);

//               // NCR offset = 局部下标（i32）
//               Value ncr_off = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), local_n);
//               b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{},
//                                      ValueRange{v_i32, ncr_off});
//             });

//           // ── 写入电流 I+x → NCR[calN .. 2*calN-1] ──────────────
//           // vcBuf[ib][N+neuronStart .. N+neuronEnd-1] 是 I_old
//           // xBuf[ib][neuronStart .. neuronEnd-1] 是本步输入 x
//           // I_input = I_old + x（软件叠加后写入 NCR[calN+local]）
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(calN),
//             [&](OpBuilder &b, Location loc, Value local_n) {
//               Value local_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), local_n);

//               // I_old 的全局列下标 = N + neuronStart + local_n
//               Value i_global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(N + neuronStart));
//               Value i_global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), i_global_i64);

//               // x 的全局列下标 = neuronStart + local_n
//               Value x_global_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(neuronStart));
//               Value x_global_idx = b.create<arith::IndexCastOp>(
//                   loc, b.getIndexType(), x_global_i64);

//               Value i_elem = b.create<memref::LoadOp>(loc, vcBuf,
//                                  ValueRange{ib, i_global_idx});
//               Value x_elem = b.create<memref::LoadOp>(loc, xBuf,
//                                  ValueRange{ib, x_global_idx});

//               // 软件叠加
//               Value i_plus_x = vcElemTy.isF32()
//                   ? b.create<arith::AddFOp>(loc, i_elem, x_elem).getResult()
//                   : b.create<arith::AddIOp>(loc, i_elem, x_elem).getResult();

//               Value val_i32 = castElemToI32ForNCR(b, loc, i_plus_x, vcElemTy);

//               // NCR offset = calN + local_n（i64 → i32，截断安全）
//               Value ncr_off_i64 = b.create<arith::AddIOp>(
//                   loc, local_i64, mkI64(calN));
//               Value ncr_off_i32 = b.create<arith::TruncIOp>(
//                   loc, b.getI32Type(), ncr_off_i64);
//               b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{},
//                                      ValueRange{val_i32, ncr_off_i32});
//             });

//           // ── 批量 LIF ─────────────────────────────────────────
//           b.create<func::CallOp>(loc, "rvne_leakage_integral_fire_1024",
//                                  TypeRange{}, ValueRange{});

//           // ── 读取 spike → spikeBuf[ib][neuronStart..neuronEnd-1]
//           int64_t numSpikeGroups = (calN + 63) / 64;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(numSpikeGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i32 = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), ig);
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               auto rc = b.create<func::CallOp>(
//                   loc, "rvne_read_sor_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ig_i32});
//               Value raw64 = rc.getResult(0);

//               buildKrnlLoop(b, loc, mkIndex(0), cst_64idx,
//                 [&](OpBuilder &b, Location loc, Value ibit) {
//                   Value ibit_i64 = b.create<arith::IndexCastOp>(
//                       loc, b.getI64Type(), ibit);

//                   // 批内局部编号
//                   Value local_n = b.create<arith::AddIOp>(
//                       loc,
//                       b.create<arith::MulIOp>(loc, ig_i64, cst_64_i64),
//                       ibit_i64);

//                   // 越界检查
//                   Value inBounds = b.create<arith::CmpIOp>(
//                       loc, arith::CmpIPredicate::ult,
//                       local_n, mkI64(calN));

//                   // 全局列下标 = neuronStart + local_n
//                   Value global_n = b.create<arith::AddIOp>(
//                       loc, local_n, mkI64(neuronStart));
//                   Value global_idx = b.create<arith::IndexCastOp>(
//                       loc, b.getIndexType(), global_n);

//                   // 位提取
//                   Value spike_i32 = b.create<arith::TruncIOp>(
//                       loc, b.getI32Type(),
//                       b.create<arith::AndIOp>(
//                           loc,
//                           b.create<arith::ShRUIOp>(loc, raw64, ibit_i64),
//                           cst_1_i64));
//                   Value spike_val = castSpikeToElemTy(
//                       b, loc, spike_i32, spikeElemTy);

//                   b.create<scf::IfOp>(loc, inBounds,
//                     [&](OpBuilder &b, Location loc) {
//                       b.create<memref::StoreOp>(loc, spike_val, spikeBuf,
//                                                 ValueRange{ib, global_idx});
//                       b.create<scf::YieldOp>(loc);
//                     });
//                 });
//             });

//           // ── 读回 V_next → vcNextBuf[ib][neuronStart..neuronEnd-1]
//           // NCR[0 .. calN-1] 存放更新后的 V
//           int64_t vNCRGroups = calN / 2;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(vNCRGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i32 = b.create<arith::IndexCastOp>(
//                   loc, b.getI32Type(), ig);
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               auto rb = b.create<func::CallOp>(
//                   loc, "rvne_read_ncr_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ig_i32});
//               Value pair64 = rb.getResult(0);

//               Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::AndIOp>(loc, pair64, cst_mask32_i64));
//               Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::ShRUIOp>(loc, pair64, cst_32_i64));

//               // 局部下标
//               Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst_2_i64);
//               Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst_1_i64);

//               // 全局 V 列下标 = neuronStart + local
//               Value glo_lo = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_lo, mkI64(neuronStart)));
//               Value glo_hi = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_hi, mkI64(neuronStart)));

//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_lo});
//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_hi});
//             });

//           // ── 读回 I_next → vcNextBuf[ib][N+neuronStart..N+neuronEnd-1]
//           // NCR[calN .. 2*calN-1] 存放更新后的 I
//           // ncrg2_index 从 calN/2 开始
//           int64_t iNCRStart  = calN / 2;
//           int64_t iNCRGroups = calN / 2;
//           buildKrnlLoop(b, loc, mkIndex(0), mkIndex(iNCRGroups),
//             [&](OpBuilder &b, Location loc, Value ig) {
//               Value ig_i64 = b.create<arith::IndexCastOp>(
//                   loc, b.getI64Type(), ig);

//               // ncrg2_index = iNCRStart + ig
//               Value ncr_grp_i32 = b.create<arith::TruncIOp>(
//                   loc, b.getI32Type(),
//                   b.create<arith::AddIOp>(loc, ig_i64, mkI64(iNCRStart)));

//               auto rb = b.create<func::CallOp>(
//                   loc, "rvne_read_ncr_group2",
//                   TypeRange{b.getI64Type()}, ValueRange{ncr_grp_i32});
//               Value pair64 = rb.getResult(0);

//               Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::AndIOp>(loc, pair64, cst_mask32_i64));
//               Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(),
//                   b.create<arith::ShRUIOp>(loc, pair64, cst_32_i64));

//               // 局部 I 下标
//               Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst_2_i64);
//               Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst_1_i64);

//               // 全局 I 列下标 = N + neuronStart + local
//               Value glo_lo = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_lo, mkI64(N + neuronStart)));
//               Value glo_hi = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
//                   b.create<arith::AddIOp>(loc, local_hi, mkI64(N + neuronStart)));

//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_lo});
//               b.create<memref::StoreOp>(loc,
//                   castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy),
//                   vcNextBuf, ValueRange{ib, glo_hi});
//             });

//         } // end for batchIdx（神经元分批，编译期展开）
//       }); // end buildKrnlLoop（batch 样本串行循环）

//     Value spikeOut  = memrefToTensor(rewriter, loc, spikeBuf,  spikeResType);
//     Value vcNextOut = memrefToTensor(rewriter, loc, vcNextBuf, vcNextResType);
//     rewriter.replaceOp(op, ValueRange{spikeOut, vcNextOut});
//     return success();
//   }
// };



struct ONNXLIFOpToKrnlConversion : public OpRewritePattern<ONNXLIFOp> {
  using OpRewritePattern::OpRewritePattern;

  struct CHWIndex {
    Value cIdx; // index
    Value hIdx; // index
    Value wIdx; // index
    Value cI64; // i64 (optional, for computing C+c)
  };

  static CHWIndex flatToCHW(OpBuilder &b, Location loc, Value flatN_i64,
                            int64_t H, int64_t W) {
    auto i64Ty = b.getI64Type();
    auto idxTy = b.getIndexType();

    Value cstW = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(W));
    Value cstH = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(H));

    // NOTE:
    // flatN_i64 is derived from loop indices (non-negative), so signed div/rem
    // are equivalent to unsigned div/rem here. This avoids arith::RemUIOp/DivUIOp.

    // w = n % W
    Value w_i64 = b.create<arith::RemSIOp>(loc, flatN_i64, cstW);
    // tmp = n / W
    Value tmp_i64 = b.create<arith::DivSIOp>(loc, flatN_i64, cstW);
    // h = tmp % H
    Value h_i64 = b.create<arith::RemSIOp>(loc, tmp_i64, cstH);
    // c = tmp / H
    Value c_i64 = b.create<arith::DivSIOp>(loc, tmp_i64, cstH);

    CHWIndex out;
    out.cI64 = c_i64;
    out.cIdx = b.create<arith::IndexCastOp>(loc, idxTy, c_i64);
    out.hIdx = b.create<arith::IndexCastOp>(loc, idxTy, h_i64);
    out.wIdx = b.create<arith::IndexCastOp>(loc, idxTy, w_i64);
    return out;
  }

  LogicalResult matchAndRewrite(ONNXLIFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module  = op->getParentOfType<ModuleOp>();
    declareIntrinsics(rewriter, module, loc);

    Value x          = op.getX();
    Value v_combined = op.getVCombined();

    auto xType  = cast<RankedTensorType>(x.getType());
    auto vcType = cast<RankedTensorType>(v_combined.getType());
    int64_t xRank  = xType.getRank();
    int64_t vcRank = vcType.getRank();

    if (!((xRank == 2 && vcRank == 2) || (xRank == 4 && vcRank == 4)))
      return rewriter.notifyMatchFailure(op, "LIF expects x/v_combined rank (2,2) or (4,4)");

    // static shape only (per your statement)
    int64_t B = xType.getDimSize(0);

    int64_t C = 0, H = 0, W = 0;
    int64_t N = 0;

    if (xRank == 2) {
      N = xType.getDimSize(1);
      int64_t N2 = vcType.getDimSize(1);
      if (N2 != 2 * N)
        return rewriter.notifyMatchFailure(op, "2D: v_combined must be (B, 2N)");
    } else {
      C = xType.getDimSize(1);
      H = xType.getDimSize(2);
      W = xType.getDimSize(3);

      int64_t C2 = vcType.getDimSize(1);
      int64_t H2 = vcType.getDimSize(2);
      int64_t W2 = vcType.getDimSize(3);

      if (H2 != H || W2 != W)
        return rewriter.notifyMatchFailure(op, "4D: v_combined H/W must match x H/W");
      if (C2 != 2 * C)
        return rewriter.notifyMatchFailure(op, "4D: v_combined must be (B, 2C, H, W)");

      N = C * H * W;
    }

    // hardware batching
    int64_t numBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

    Type xElemTy  = xType.getElementType();
    Type vcElemTy = vcType.getElementType();

    auto spikeResType   = cast<RankedTensorType>(op.getResult(0).getType());
    auto vcNextResType  = cast<RankedTensorType>(op.getResult(1).getType());
    Type spikeElemTy    = spikeResType.getElementType();
    Type vcNextElemTy   = vcNextResType.getElementType();

    // ---------------------------
    // alloc output buffers (rank must equal output rank)
    // ---------------------------
    Value spikeBuf, vcNextBuf;
    if (xRank == 2) {
      spikeBuf  = rewriter.create<memref::AllocOp>(loc, MemRefType::get({B, N}, spikeElemTy));
      vcNextBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({B, 2 * N}, vcNextElemTy));
    } else {
      spikeBuf  = rewriter.create<memref::AllocOp>(loc, MemRefType::get({B, C, H, W}, spikeElemTy));
      vcNextBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({B, 2 * C, H, W}, vcNextElemTy));
    }

    // ---------------------------
    // tensor -> memref (rank一致)
    // ---------------------------
    Value xBuf, vcBuf;
    if (xRank == 2) {
      xBuf  = tensorToMemref(rewriter, loc, x,          MemRefType::get({B, N}, xElemTy));
      vcBuf = tensorToMemref(rewriter, loc, v_combined, MemRefType::get({B, 2 * N}, vcElemTy));
    } else {
      xBuf  = tensorToMemref(rewriter, loc, x,          MemRefType::get({B, C, H, W}, xElemTy));
      vcBuf = tensorToMemref(rewriter, loc, v_combined, MemRefType::get({B, 2 * C, H, W}, vcElemTy));
    }

    // ---------------------------
    // helpers: all constants created INSIDE the lambda builder
    // ---------------------------
    auto mkIndex = [&](OpBuilder &b, int64_t v) -> Value {
      return b.create<arith::ConstantIndexOp>(loc, v);
    };
    auto mkI64 = [&](OpBuilder &b, int64_t v) -> Value {
      return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(v));
    };

    auto loadXByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64) -> Value {
      if (xRank == 2) {
        Value nIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flat_n_i64);
        return b.create<memref::LoadOp>(loc, xBuf, ValueRange{ib, nIdx});
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      return b.create<memref::LoadOp>(loc, xBuf, ValueRange{ib, idx.cIdx, idx.hIdx, idx.wIdx});
    };

    auto loadVC_V_ByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64) -> Value {
      if (xRank == 2) {
        Value nIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flat_n_i64);
        return b.create<memref::LoadOp>(loc, vcBuf, ValueRange{ib, nIdx});
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      return b.create<memref::LoadOp>(loc, vcBuf, ValueRange{ib, idx.cIdx, idx.hIdx, idx.wIdx});
    };

    auto loadVC_I_ByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64) -> Value {
      if (xRank == 2) {
        Value cstN = mkI64(b, N);
        Value col_i64 = b.create<arith::AddIOp>(loc, flat_n_i64, cstN);
        Value col_idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), col_i64);
        return b.create<memref::LoadOp>(loc, vcBuf, ValueRange{ib, col_idx});
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      Value cstC = mkI64(b, C);
      Value c2_i64 = b.create<arith::AddIOp>(loc, idx.cI64, cstC);
      Value c2_idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), c2_i64);
      return b.create<memref::LoadOp>(loc, vcBuf, ValueRange{ib, c2_idx, idx.hIdx, idx.wIdx});
    };

    auto storeSpikeByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64, Value spike_val) {
      if (xRank == 2) {
        Value nIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flat_n_i64);
        b.create<memref::StoreOp>(loc, spike_val, spikeBuf, ValueRange{ib, nIdx});
        return;
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      b.create<memref::StoreOp>(loc, spike_val, spikeBuf, ValueRange{ib, idx.cIdx, idx.hIdx, idx.wIdx});
    };

    auto storeVCNext_V_ByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64, Value v_val) {
      if (xRank == 2) {
        Value nIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flat_n_i64);
        b.create<memref::StoreOp>(loc, v_val, vcNextBuf, ValueRange{ib, nIdx});
        return;
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      b.create<memref::StoreOp>(loc, v_val, vcNextBuf, ValueRange{ib, idx.cIdx, idx.hIdx, idx.wIdx});
    };

    auto storeVCNext_I_ByFlatN = [&](OpBuilder &b, Value ib, Value flat_n_i64, Value i_val) {
      if (xRank == 2) {
        Value cstN = mkI64(b, N);
        Value col_i64 = b.create<arith::AddIOp>(loc, flat_n_i64, cstN);
        Value col_idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), col_i64);
        b.create<memref::StoreOp>(loc, i_val, vcNextBuf, ValueRange{ib, col_idx});
        return;
      }
      auto idx = flatToCHW(b, loc, flat_n_i64, H, W);
      Value cstC = mkI64(b, C);
      Value c2_i64 = b.create<arith::AddIOp>(loc, idx.cI64, cstC);
      Value c2_idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), c2_i64);
      b.create<memref::StoreOp>(loc, i_val, vcNextBuf, ValueRange{ib, c2_idx, idx.hIdx, idx.wIdx});
    };

    // ---------------------------
    // outer loop over batch B
    // IMPORTANT: all loop bounds constants created with *rewriter* here are OK,
    // but to be ultra-safe with KrnlToAffine, create bounds via builder inside lambda.
    // We'll use mkIndex(b, ...) inside the loop creators below.
    // ---------------------------
    buildKrnlLoop(rewriter, loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
                  rewriter.create<arith::ConstantIndexOp>(loc, B),
      [&](OpBuilder &b0, Location loc, Value ib) {

        // compile-time batches over neurons (C*H*W flattened)
        for (int64_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
          int64_t neuronStart = batchIdx * kHardwareNeuronNum;
          int64_t neuronEnd   = std::min(neuronStart + kHardwareNeuronNum, N);
          int64_t calN        = neuronEnd - neuronStart;

          // Ensure even for group2 reads (same as your original assumption)
          assert(calN % 2 == 0 && "calN must be even for NCR group2 reads");

          b0.create<func::CallOp>(loc, "rvne_clear_neuron_data_1024", TypeRange{}, ValueRange{});

          // ---- write V -> NCR[0..calN-1]
          buildKrnlLoop(b0, loc, mkIndex(b0, 0), mkIndex(b0, calN),
            [&](OpBuilder &b, Location loc, Value local_n) {
              Value local_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), local_n);
              Value cstStart  = mkI64(b, neuronStart);
              Value global_i64 = b.create<arith::AddIOp>(loc, local_i64, cstStart);

              Value v_elem = loadVC_V_ByFlatN(b, ib, global_i64);
              Value v_i32  = castElemToI32ForNCR(b, loc, v_elem, vcElemTy);

              Value ncr_off = b.create<arith::IndexCastOp>(loc, b.getI32Type(), local_n);
              b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{}, ValueRange{v_i32, ncr_off});
            });

          // ---- write I_old + x -> NCR[calN..2*calN-1]
          buildKrnlLoop(b0, loc, mkIndex(b0, 0), mkIndex(b0, calN),
            [&](OpBuilder &b, Location loc, Value local_n) {
              Value local_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), local_n);
              Value cstStart  = mkI64(b, neuronStart);
              Value global_i64 = b.create<arith::AddIOp>(loc, local_i64, cstStart);

              Value i_elem = loadVC_I_ByFlatN(b, ib, global_i64);
              Value x_elem = loadXByFlatN(b, ib, global_i64);

              Value i_plus_x = vcElemTy.isF32()
                  ? b.create<arith::AddFOp>(loc, i_elem, x_elem).getResult()
                  : b.create<arith::AddIOp>(loc, i_elem, x_elem).getResult();

              Value val_i32 = castElemToI32ForNCR(b, loc, i_plus_x, vcElemTy);

              Value cstCalN = mkI64(b, calN);
              Value ncr_off_i64 = b.create<arith::AddIOp>(loc, local_i64, cstCalN);
              Value ncr_off_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), ncr_off_i64);

              b.create<func::CallOp>(loc, "rvne_write_ncr", TypeRange{}, ValueRange{val_i32, ncr_off_i32});
            });

          // ---- execute hardware
          b0.create<func::CallOp>(loc, "rvne_leakage_integral_fire_1024", TypeRange{}, ValueRange{});

          // ---- read spike
          int64_t numSpikeGroups = (calN + 63) / 64;
          buildKrnlLoop(b0, loc, mkIndex(b0, 0), mkIndex(b0, numSpikeGroups),
            [&](OpBuilder &b, Location loc, Value ig) {
              Value ig_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), ig);
              Value ig_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), ig);

              auto rc = b.create<func::CallOp>(loc, "rvne_read_sor_group2",
                                               TypeRange{b.getI64Type()}, ValueRange{ig_i32});
              Value raw64 = rc.getResult(0);

              buildKrnlLoop(b, loc, mkIndex(b, 0), mkIndex(b, 64),
                [&](OpBuilder &b2, Location loc, Value ibit) {
                  Value ibit_i64 = b2.create<arith::IndexCastOp>(loc, b2.getI64Type(), ibit);

                  Value cst64 = mkI64(b2, 64);
                  Value base = b2.create<arith::MulIOp>(loc, ig_i64, cst64);
                  Value local_n = b2.create<arith::AddIOp>(loc, base, ibit_i64);

                  Value cstCalN = mkI64(b2, calN);
                  Value inBounds = b2.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_n, cstCalN);

                  // global_n = neuronStart + local_n
                  Value cstStart = mkI64(b2, neuronStart);
                  Value global_n = b2.create<arith::AddIOp>(loc, local_n, cstStart);

                  // extract bit
                  Value shr = b2.create<arith::ShRUIOp>(loc, raw64, ibit_i64);
                  Value cst1 = mkI64(b2, 1);
                  Value bit = b2.create<arith::AndIOp>(loc, shr, cst1);
                  Value spike_i32 = b2.create<arith::TruncIOp>(loc, b2.getI32Type(), bit);
                  Value spike_val = castSpikeToElemTy(b2, loc, spike_i32, spikeElemTy);

                  b2.create<scf::IfOp>(loc, inBounds,
                    [&](OpBuilder &b3, Location loc) {
                      storeSpikeByFlatN(b3, ib, global_n, spike_val);
                      b3.create<scf::YieldOp>(loc);
                    });
                });
            });

          // ---- read back V_next from NCR[0..calN-1]
          int64_t vNCRGroups = calN / 2;
          buildKrnlLoop(b0, loc, mkIndex(b0, 0), mkIndex(b0, vNCRGroups),
            [&](OpBuilder &b, Location loc, Value ig) {
              Value ig_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), ig);
              Value ig_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), ig);

              auto rb = b.create<func::CallOp>(loc, "rvne_read_ncr_group2",
                                               TypeRange{b.getI64Type()}, ValueRange{ig_i32});
              Value pair64 = rb.getResult(0);

              Value cstMask = mkI64(b, 0xFFFFFFFFLL);
              Value lo64 = b.create<arith::AndIOp>(loc, pair64, cstMask);
              Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), lo64);

              Value cst32 = mkI64(b, 32);
              Value hi64 = b.create<arith::ShRUIOp>(loc, pair64, cst32);
              Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), hi64);

              Value cst2 = mkI64(b, 2);
              Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst2);
              Value cst1 = mkI64(b, 1);
              Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst1);

              Value cstStart = mkI64(b, neuronStart);
              Value global_lo = b.create<arith::AddIOp>(loc, local_lo, cstStart);
              Value global_hi = b.create<arith::AddIOp>(loc, local_hi, cstStart);

              Value v_lo = castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy);
              Value v_hi = castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy);

              storeVCNext_V_ByFlatN(b, ib, global_lo, v_lo);
              storeVCNext_V_ByFlatN(b, ib, global_hi, v_hi);
            });

          // ---- read back I_next from NCR[calN..2*calN-1]
          // group2 index offset = calN/2
          int64_t iNCRStart  = calN / 2;
          int64_t iNCRGroups = calN / 2;

          buildKrnlLoop(b0, loc, mkIndex(b0, 0), mkIndex(b0, iNCRGroups),
            [&](OpBuilder &b, Location loc, Value ig) {
              Value ig_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), ig);

              // ncr_group = iNCRStart + ig
              Value cstStartGrp = mkI64(b, iNCRStart);
              Value grp_i64 = b.create<arith::AddIOp>(loc, ig_i64, cstStartGrp);
              Value grp_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), grp_i64);

              auto rb = b.create<func::CallOp>(loc, "rvne_read_ncr_group2",
                                               TypeRange{b.getI64Type()}, ValueRange{grp_i32});
              Value pair64 = rb.getResult(0);

              Value cstMask = mkI64(b, 0xFFFFFFFFLL);
              Value lo64 = b.create<arith::AndIOp>(loc, pair64, cstMask);
              Value lo_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), lo64);

              Value cst32 = mkI64(b, 32);
              Value hi64 = b.create<arith::ShRUIOp>(loc, pair64, cst32);
              Value hi_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), hi64);

              Value cst2 = mkI64(b, 2);
              Value local_lo = b.create<arith::MulIOp>(loc, ig_i64, cst2);
              Value cst1 = mkI64(b, 1);
              Value local_hi = b.create<arith::AddIOp>(loc, local_lo, cst1);

              Value cstNeuronStart = mkI64(b, neuronStart);
              Value global_lo = b.create<arith::AddIOp>(loc, local_lo, cstNeuronStart);
              Value global_hi = b.create<arith::AddIOp>(loc, local_hi, cstNeuronStart);

              Value i_lo = castI32FromNCRToElemTy(b, loc, lo_i32, vcNextElemTy);
              Value i_hi = castI32FromNCRToElemTy(b, loc, hi_i32, vcNextElemTy);

              storeVCNext_I_ByFlatN(b, ib, global_lo, i_lo);
              storeVCNext_I_ByFlatN(b, ib, global_hi, i_hi);
            });
        } // end neuron batchIdx
      });

    // ---------------------------
    // finalize: memref -> tensor (rank must match)
    // ---------------------------
    Value spikeOut  = memrefToTensor(rewriter, loc, spikeBuf,  spikeResType);
    Value vcNextOut = memrefToTensor(rewriter, loc, vcNextBuf, vcNextResType);
    rewriter.replaceOp(op, ValueRange{spikeOut, vcNextOut});
    return success();
  }
};






// // ═══════════════════════════════════════════════════════════════════════
// // 多步 LIF 转换 Pattern：直接拦截 onnx.Custom 节点
// // 解决 T 维度循环、硬件状态持久化、以及 index 类型转换问题
// // ═══════════════════════════════════════════════════════════════════════
// struct ONNXMultiStepLIFToKrnlConversion : public OpRewritePattern<ONNXCustomOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(ONNXCustomOp op, PatternRewriter &rewriter) const override {
//     // 1. 依据 function_name 属性拦截，确保只处理 MultiStepLIF
//     if (op.getFunctionName() != "MultiStepLIF")
//       return failure();

//     Location loc = op.getLoc();
//     auto module = op->getParentOfType<ModuleOp>();
//     declareIntrinsics(rewriter, module, loc);

//     // 2. 提取输入 Operands (T, N) 和 (1, 2N)
//     Value x_seq = op.getOperands()[0];
//     Value vc_init = op.getOperands()[1];

//     auto xType = llvm::cast<RankedTensorType>(x_seq.getType());
//     auto vcType = llvm::cast<RankedTensorType>(vc_init.getType());
//     Type xElemTy = xType.getElementType();
//     Type vcElemTy = vcType.getElementType();

//     // 提取输出的预期 Tensor 类型，用于最后的 replaceOp
//     auto resType0 = llvm::cast<RankedTensorType>(op.getResult(0).getType());
//     auto resType1 = llvm::cast<RankedTensorType>(op.getResult(1).getType());

//     int64_t T = xType.getDimSize(0);
//     int64_t N = xType.getDimSize(1);
//     int64_t N2 = vcType.getDimSize(1);
//     int64_t numBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

//     // 常量辅助
//     auto mkIndex = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
//     auto mkI64 = [&](int64_t v) { return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(v)); };
//     Value cst_0idx = mkIndex(0);
//     Value cst_64_i64 = mkI64(64);
//     Value cst_mask32_i64 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

//     // 3. 分配内存 Buffer
//     // spikeBuf 存储全时序结果 (T, N)，vcFinalBuf 存储最终硬件状态 (1, 2N)
//     Value spikeBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({T, N}, resType0.getElementType()));
//     Value vcFinalBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1, N2}, resType1.getElementType()));

//     Value xMem = tensorToMemref(rewriter, loc, x_seq, MemRefType::get({T, N}, xElemTy));
//     Value vcInitMem = tensorToMemref(rewriter, loc, vc_init, MemRefType::get({1, N2}, vcElemTy));

//     // 4. 初始化：将初始状态拷贝到持久化 Buffer vcFinalBuf 中
//     buildKrnlLoop(rewriter, loc, cst_0idx, mkIndex(N2), [&](OpBuilder &b, Location l, Value i) {
//       Value val = b.create<memref::LoadOp>(l, vcInitMem, ValueRange{cst_0idx, i});
//       b.create<memref::StoreOp>(l, val, vcFinalBuf, ValueRange{cst_0idx, i});
//     });

//     // 5. 核心：显式时间步 T 维度循环
//     // 使用 [&] 捕获当前作用域所有变量，解决 vcElemTy 等作用域报错
//     buildKrnlLoop(rewriter, loc, cst_0idx, mkIndex(T), [&](OpBuilder &b, Location l, Value it) {
      
//       for (int64_t bi = 0; bi < numBatches; ++bi) {
//         int64_t nStart = bi * kHardwareNeuronNum;
//         int64_t calN = std::min(kHardwareNeuronNum, N - nStart);

//         b.create<func::CallOp>(l, "rvne_clear_neuron_data_1024", TypeRange{}, ValueRange{});

//         // --- 写入膜电位 V ---
//         buildKrnlLoop(b, l, mkIndex(0), mkIndex(calN), [&](OpBuilder &b2, Location l2, Value ln) {
//           Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
//           Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart)));
          
//           Value v_val = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{cst_0idx, gn_idx});
//           Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ln_i64); // 正确转换 index->i64->i32
//           b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{}, 
//                                   ValueRange{castElemToI32ForNCR(b2, l2, v_val, vcElemTy), ncr_off});
//         });

//         // --- 写入 I + x (x 来自当前时间步 it) ---
//         buildKrnlLoop(b, l, mkIndex(0), mkIndex(calN), [&](OpBuilder &b2, Location l2, Value ln) {
//           Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
//           Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart)));
//           Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(N + nStart)));

//           Value i_old = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{cst_0idx, gn_I_idx});
//           Value x_val = b2.create<memref::LoadOp>(l2, xMem, ValueRange{it, gn_idx});
          
//           Value i_new = vcElemTy.isF32() ? b2.create<arith::AddFOp>(l2, i_old, x_val).getResult() 
//                                          : b2.create<arith::AddIOp>(l2, i_old, x_val).getResult();

//           Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(calN)));
//           b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{}, 
//                                   ValueRange{castElemToI32ForNCR(b2, l2, i_new, vcElemTy), ncr_off});
//         });

//         // 硬件发射
//         b.create<func::CallOp>(l, "rvne_leakage_integral_fire_1024", TypeRange{}, ValueRange{});

//         // --- 读回 Spike (SOR) ---
//         int64_t numGroups = (calN + 63) / 64;
//         buildKrnlLoop(b, l, mkIndex(0), mkIndex(numGroups), [&](OpBuilder &b2, Location l2, Value ig) {
//           Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
//           auto rc = b2.create<func::CallOp>(l2, "rvne_read_sor_group2", TypeRange{b2.getI64Type()}, 
//                                             ValueRange{b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64)});
//           Value raw64 = rc.getResult(0);

//           buildKrnlLoop(b2, l2, mkIndex(0), mkIndex(64), [&](OpBuilder &b3, Location l3, Value ibit) {
//             Value ibit_i64 = b3.create<arith::IndexCastOp>(l3, b3.getI64Type(), ibit);
//             Value local_n = b3.create<arith::AddIOp>(l3, b3.create<arith::MulIOp>(l3, ig_i64, cst_64_i64), ibit_i64);
//             Value inBounds = b3.create<arith::CmpIOp>(l3, arith::CmpIPredicate::ult, local_n, mkI64(calN));
//             Value gn_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), b3.create<arith::AddIOp>(l3, local_n, mkI64(nStart)));

//             Value s_i32 = b3.create<arith::TruncIOp>(l3, b3.getI32Type(), 
//                                                      b3.create<arith::AndIOp>(l3, b3.create<arith::ShRUIOp>(l3, raw64, ibit_i64), mkI64(1)));
//             b3.create<scf::IfOp>(l3, inBounds, [&](OpBuilder &b4, Location l4) {
//               b4.create<memref::StoreOp>(l4, castSpikeToElemTy(b4, l4, s_i32, resType0.getElementType()), spikeBuf, ValueRange{it, gn_idx});
//               b4.create<scf::YieldOp>(l4);
//             });
//           });
//         });

//         // --- 读回更新后的 V/I (NCR) 并持久化到 vcFinalBuf ---
//         int64_t ncrGroups = calN / 2;
//         buildKrnlLoop(b, l, mkIndex(0), mkIndex(ncrGroups), [&](OpBuilder &b2, Location l2, Value ig) {
//           Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
//           Value ig_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64);

//           // 读回 V
//           auto rb_V = b2.create<func::CallOp>(l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{ig_i32});
//           Value pair_V = rb_V.getResult(0);
          
//           // 读回 I (偏移量为 calN/2)
//           Value off_I_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AddIOp>(l2, ig_i64, mkI64(calN / 2)));
//           auto rb_I = b2.create<func::CallOp>(l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{off_I_i32});
//           Value pair_I = rb_I.getResult(0);

//           // 解析并存储每一对 (Lo, Hi)
//           for (int i = 0; i < 2; ++i) {
//             Value bitOff = (i == 0) ? mkI64(0) : mkI64(32);
//             Value val_V = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AndIOp>(l2, b2.create<arith::ShRUIOp>(l2, pair_V, bitOff), cst_mask32_i64));
//             Value val_I = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AndIOp>(l2, b2.create<arith::ShRUIOp>(l2, pair_I, bitOff), cst_mask32_i64));

//             Value l_idx = b2.create<arith::AddIOp>(l2, b2.create<arith::MulIOp>(l2, ig_i64, mkI64(2)), mkI64(i));
//             Value gn_V = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, l_idx, mkI64(nStart)));
//             Value gn_I = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, l_idx, mkI64(N + nStart)));

//             b2.create<memref::StoreOp>(l2, castI32FromNCRToElemTy(b2, l2, val_V, vcElemTy), vcFinalBuf, ValueRange{cst_0idx, gn_V});
//             b2.create<memref::StoreOp>(l2, castI32FromNCRToElemTy(b2, l2, val_I, vcElemTy), vcFinalBuf, ValueRange{cst_0idx, gn_I});
//           }
//         });
//       }
//     });

//     // 6. 替换输出，完成 Lowering
//     Value finalSpikeTensor = memrefToTensor(rewriter, loc, spikeBuf, resType0);
//     Value finalStateTensor = memrefToTensor(rewriter, loc, vcFinalBuf, resType1);
//     rewriter.replaceOp(op, ValueRange{finalSpikeTensor, finalStateTensor});

//     return success();
//   }
// };



// // ═══════════════════════════════════════════════════════════════════════
// // 多步 LIF 转换 Pattern：直接拦截 onnx.Custom 节点 (TBCHW)
// // x_seq:   (T, B, C, H, W)
// // vc_init: (B, 2 * (C*H*W))，每个 batch 一份 [V || I]
// // 输出:
// // spike:   (T, B, C, H, W)
// // vc_next: (B, 2 * (C*H*W))
// // ═══════════════════════════════════════════════════════════════════════

// struct ONNXMultiStepLIFToKrnlConversion : public OpRewritePattern<ONNXCustomOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(ONNXCustomOp op,
//                                 PatternRewriter &rewriter) const override {
//     // 1) 仅拦截 MultiStepLIF
//     if (op.getFunctionName() != "MultiStepLIF")
//       return failure();

//     Location loc = op.getLoc();
//     auto module = op->getParentOfType<ModuleOp>();
//     declareIntrinsics(rewriter, module, loc);

//     // 2) 取输入
//     Value x_seq = op.getOperands()[0];
//     Value vc_init = op.getOperands()[1];

//     auto xType = llvm::cast<RankedTensorType>(x_seq.getType());
//     auto vcType = llvm::cast<RankedTensorType>(vc_init.getType());
//     Type xElemTy = xType.getElementType();
//     Type vcElemTy = vcType.getElementType();

//     // 输出类型
//     auto resType0 = llvm::cast<RankedTensorType>(op.getResult(0).getType()); // spike
//     auto resType1 = llvm::cast<RankedTensorType>(op.getResult(1).getType()); // vc_next

//     // 3) 解析维度：x (T,B,C,H,W)
//     if (xType.getRank() != 5)
//       return rewriter.notifyMatchFailure(op, "x_seq must be rank-5 (T,B,C,H,W)");

//     int64_t T = xType.getDimSize(0);
//     int64_t B = xType.getDimSize(1);
//     int64_t C = xType.getDimSize(2);
//     int64_t H = xType.getDimSize(3);
//     int64_t W = xType.getDimSize(4);

//     int64_t N = C * H * W;   // 每个 batch 的 neuron 数
//     int64_t N2 = 2 * N;      // [V || I]

//     // vc (B, 2N)
//     if (vcType.getRank() != 2)
//       return rewriter.notifyMatchFailure(op, "vc_init must be rank-2 (B, 2N)");

//     if (vcType.getDimSize(0) != B)
//       return rewriter.notifyMatchFailure(op, "vc_init dim0 must equal B");

//     if (vcType.getDimSize(1) != N2)
//       return rewriter.notifyMatchFailure(op, "vc_init dim1 must equal 2*(C*H*W)");

//     int64_t numHwBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

//     // 常量辅助（尽量沿用你的写法）
//     auto mkIndex = [&](int64_t v) {
//       return rewriter.create<arith::ConstantIndexOp>(loc, v);
//     };
//     auto mkI64 = [&](int64_t v) {
//       return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(v));
//     };

//     Value cst_0idx = mkIndex(0);
//     Value cst_64_i64 = mkI64(64);
//     Value cst_mask32_i64 =
//         rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

//     // 4) 分配 buffer
//     Value spikeBuf = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({T, B, C, H, W}, resType0.getElementType()));

//     Value vcFinalBuf = rewriter.create<memref::AllocOp>(
//         loc, MemRefType::get({B, N2}, resType1.getElementType()));

//     // tensor -> memref
//     Value xMem = tensorToMemref(rewriter, loc, x_seq,
//                                 MemRefType::get({T, B, C, H, W}, xElemTy));
//     Value vcInitMem = tensorToMemref(rewriter, loc, vc_init,
//                                      MemRefType::get({B, N2}, vcElemTy));

//     // 5) 初始化：vcInitMem -> vcFinalBuf
//     buildKrnlLoop(rewriter, loc, mkIndex(0), mkIndex(B),
//                   [&](OpBuilder &b0, Location l0, Value ib) {
//       buildKrnlLoop(b0, l0, mkIndex(0), mkIndex(N2),
//                     [&](OpBuilder &b1, Location l1, Value i) {
//         Value val = b1.create<memref::LoadOp>(l1, vcInitMem, ValueRange{ib, i});
//         b1.create<memref::StoreOp>(l1, val, vcFinalBuf, ValueRange{ib, i});
//       });
//     });

//     // 6) 时间步循环 it
//     buildKrnlLoop(rewriter, loc, mkIndex(0), mkIndex(T),
//                   [&](OpBuilder &b, Location l, Value it) {

//       // batch 循环 ib
//       buildKrnlLoop(b, l, mkIndex(0), mkIndex(B),
//                     [&](OpBuilder &bb, Location lb, Value ib) {

//         for (int64_t hwb = 0; hwb < numHwBatches; ++hwb) {
//           int64_t nStart = hwb * kHardwareNeuronNum;
//           int64_t calN = std::min<int64_t>(kHardwareNeuronNum, N - nStart);

//           bb.create<func::CallOp>(lb, "rvne_clear_neuron_data_1024", TypeRange{}, ValueRange{});

//           // -------------------------
//           // 写入 V：NCR[0..calN-1]
//           // -------------------------
//           buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(calN),
//                         [&](OpBuilder &b2, Location l2, Value ln) {
//             Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
//             Value gn_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart));
//             Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_i64);

//             Value v_val = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{ib, gn_idx});
//             Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ln_i64);

//             b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{},
//                                     ValueRange{castElemToI32ForNCR(b2, l2, v_val, vcElemTy), ncr_off});
//           });

//           // -------------------------
//           // 写入 I+x：NCR[calN..2*calN-1]
//           // 其中 x 来自 xMem[it, ib, c, h, w] (gn 反解到 CHW)
//           // -------------------------
//           buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(calN),
//                         [&](OpBuilder &b2, Location l2, Value ln) {
//             Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
//             Value gn_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart));

//             // gn -> (c,h,w)
//             // c = gn / (H*W), rem = gn % (H*W), h = rem / W, w = rem % W
//             int64_t HW = H * W;
//             // Value c_i64 = b2.create<arith::DivUIOp>(l2, gn_i64, mkI64(HW));
//             // Value rem_i64 = b2.create<arith::RemUIOp>(l2, gn_i64, mkI64(HW));
//             // Value h_i64 = b2.create<arith::DivUIOp>(l2, rem_i64, mkI64(W));
//             // Value w_i64 = b2.create<arith::RemUIOp>(l2, rem_i64, mkI64(W));

//             Value c_i64 = b2.create<arith::DivSIOp>(l2, gn_i64, mkI64(HW));
//             Value rem_i64 = b2.create<arith::RemSIOp>(l2, gn_i64, mkI64(HW));
//             Value h_i64 = b2.create<arith::DivSIOp>(l2, rem_i64, mkI64(W));
//             Value w_i64 = b2.create<arith::RemSIOp>(l2, rem_i64, mkI64(W));

//             Value c_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), c_i64);
//             Value h_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), h_i64);
//             Value w_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), w_i64);

//             Value gn_I_i64 = b2.create<arith::AddIOp>(l2, gn_i64, mkI64(N));
//             Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_I_i64);

//             Value i_old = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{ib, gn_I_idx});
//             Value x_val = b2.create<memref::LoadOp>(l2, xMem, ValueRange{it, ib, c_idx, h_idx, w_idx});

//             Value i_new = vcElemTy.isF32()
//                               ? b2.create<arith::AddFOp>(l2, i_old, x_val).getResult()
//                               : b2.create<arith::AddIOp>(l2, i_old, x_val).getResult();

//             Value ncr_off_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(calN));
//             Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ncr_off_i64);

//             b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{},
//                                     ValueRange{castElemToI32ForNCR(b2, l2, i_new, vcElemTy), ncr_off});
//           });

//           // 硬件发射
//           bb.create<func::CallOp>(lb, "rvne_leakage_integral_fire_1024", TypeRange{}, ValueRange{});

//           // -------------------------
//           // 读回 Spike (SOR)，写入 spikeBuf[it,ib,c,h,w]
//           // -------------------------
//           int64_t numGroups = (calN + 63) / 64;
//           buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(numGroups),
//                         [&](OpBuilder &b2, Location l2, Value ig) {
//             Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);

//             auto rc = b2.create<func::CallOp>(
//                 l2, "rvne_read_sor_group2", TypeRange{b2.getI64Type()},
//                 ValueRange{b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64)});
//             Value raw64 = rc.getResult(0);

//             buildKrnlLoop(b2, l2, mkIndex(0), mkIndex(64),
//                           [&](OpBuilder &b3, Location l3, Value ibit) {
//               Value ibit_i64 = b3.create<arith::IndexCastOp>(l3, b3.getI64Type(), ibit);
//               Value local_n = b3.create<arith::AddIOp>(
//                   l3, b3.create<arith::MulIOp>(l3, ig_i64, cst_64_i64), ibit_i64);

//               Value inBounds = b3.create<arith::CmpIOp>(
//                   l3, arith::CmpIPredicate::ult, local_n, mkI64(calN));

//               Value gn_i64 = b3.create<arith::AddIOp>(l3, local_n, mkI64(nStart));

//               // bit extraction
//               Value bit = b3.create<arith::AndIOp>(
//                   l3, b3.create<arith::ShRUIOp>(l3, raw64, ibit_i64), mkI64(1));
//               Value s_i32 = b3.create<arith::TruncIOp>(l3, b3.getI32Type(), bit);

//               // gn -> (c,h,w)
//               int64_t HW = H * W;
//               // Value c_i64 = b3.create<arith::DivUIOp>(l3, gn_i64, mkI64(HW));
//               // Value rem_i64 = b3.create<arith::RemUIOp>(l3, gn_i64, mkI64(HW));
//               // Value h_i64 = b3.create<arith::DivUIOp>(l3, rem_i64, mkI64(W));
//               // Value w_i64 = b3.create<arith::RemUIOp>(l3, rem_i64, mkI64(W));

//               Value c_i64 = b2.create<arith::DivSIOp>(l2, gn_i64, mkI64(HW));
//               Value rem_i64 = b2.create<arith::RemSIOp>(l2, gn_i64, mkI64(HW));
//               Value h_i64 = b2.create<arith::DivSIOp>(l2, rem_i64, mkI64(W));
//               Value w_i64 = b2.create<arith::RemSIOp>(l2, rem_i64, mkI64(W));              

//               Value c_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), c_i64);
//               Value h_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), h_i64);
//               Value w_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), w_i64);

//               b3.create<scf::IfOp>(l3, inBounds, [&](OpBuilder &b4, Location l4) {
//                 b4.create<memref::StoreOp>(
//                     l4,
//                     castSpikeToElemTy(b4, l4, s_i32, resType0.getElementType()),
//                     spikeBuf,
//                     ValueRange{it, ib, c_idx, h_idx, w_idx});
//                 b4.create<scf::YieldOp>(l4);
//               });
//             });
//           });

//           // -------------------------
//           // 读回更新后的 V/I (NCR) 并持久化到 vcFinalBuf
//           // 关键：支持 calN 为奇数 —— 对每个 lane 做 inBounds 判断
//           // -------------------------
//           int64_t ncrGroups = (calN + 1) / 2; // ceil(calN/2)

//           buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(ncrGroups),
//                         [&](OpBuilder &b2, Location l2, Value ig) {
//             Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
//             Value ig_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64);

//             // 读回 V pair
//             auto rb_V = b2.create<func::CallOp>(
//                 l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{ig_i32});
//             Value pair_V = rb_V.getResult(0);

//             // 读回 I pair：偏移 ncrGroups（因为写入时 I 区从 calN 开始；读回按 group2 的 group 索引组织）
//             // 这里保持你原先“V groups + I groups”的布局，只是 groups 由 ceil(calN/2) 决定
//             Value off_I_i64 = b2.create<arith::AddIOp>(l2, ig_i64, mkI64(ncrGroups));
//             Value off_I_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), off_I_i64);

//             auto rb_I = b2.create<func::CallOp>(
//                 l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{off_I_i32});
//             Value pair_I = rb_I.getResult(0);

//             // lane 0/1
//             for (int lane = 0; lane < 2; ++lane) {
//               Value bitOff = (lane == 0) ? mkI64(0) : mkI64(32);

//               // local index within this hw-batch: l_idx = ig*2 + lane
//               Value l_idx_i64 = b2.create<arith::AddIOp>(
//                   l2, b2.create<arith::MulIOp>(l2, ig_i64, mkI64(2)), mkI64(lane));

//               // inBounds: l_idx < calN
//               Value inBounds = b2.create<arith::CmpIOp>(
//                   l2, arith::CmpIPredicate::ult, l_idx_i64, mkI64(calN));

//               // extract 32-bit
//               Value val_V = b2.create<arith::TruncIOp>(
//                   l2, b2.getI32Type(),
//                   b2.create<arith::AndIOp>(
//                       l2, b2.create<arith::ShRUIOp>(l2, pair_V, bitOff), cst_mask32_i64));

//               Value val_I = b2.create<arith::TruncIOp>(
//                   l2, b2.getI32Type(),
//                   b2.create<arith::AndIOp>(
//                       l2, b2.create<arith::ShRUIOp>(l2, pair_I, bitOff), cst_mask32_i64));

//               // global neuron id: gn = nStart + l_idx
//               Value gn_i64 = b2.create<arith::AddIOp>(l2, l_idx_i64, mkI64(nStart));
//               Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_i64);

//               Value gn_I_i64 = b2.create<arith::AddIOp>(l2, gn_i64, mkI64(N));
//               Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_I_i64);

//               // if(inBounds) store
//               b2.create<scf::IfOp>(l2, inBounds, [&](OpBuilder &b3, Location l3) {
//                 b3.create<memref::StoreOp>(
//                     l3,
//                     castI32FromNCRToElemTy(b3, l3, val_V, vcElemTy),
//                     vcFinalBuf,
//                     ValueRange{ib, gn_idx});

//                 b3.create<memref::StoreOp>(
//                     l3,
//                     castI32FromNCRToElemTy(b3, l3, val_I, vcElemTy),
//                     vcFinalBuf,
//                     ValueRange{ib, gn_I_idx});
//                 b3.create<scf::YieldOp>(l3);
//               });
//             }
//           });
//         } // end hw batches
//       }); // end B loop
//     }); // end T loop

//     // 7) 输出替换
//     Value finalSpikeTensor = memrefToTensor(rewriter, loc, spikeBuf, resType0);
//     Value finalStateTensor = memrefToTensor(rewriter, loc, vcFinalBuf, resType1);
//     rewriter.replaceOp(op, ValueRange{finalSpikeTensor, finalStateTensor});
//     return success();
//   }
// };


struct ONNXMultiStepLIFToKrnlConversion : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCustomOp op, PatternRewriter &rewriter) const override {
    auto xType = llvm::cast<RankedTensorType>(op.getOperands()[0].getType());
    bool is5D = (xType.getRank() == 5);

    if (is5D) {
      // ═══════════════════════════════════════════════════════════════════════
      // 多步 LIF 转换 Pattern：直接拦截 onnx.Custom 节点 (TBCHW)
      // x_seq:   (T, B, C, H, W)
      // vc_init: (B, 2 * (C*H*W))，每个 batch 一份 [V || I]
      // 输出:
      // spike:   (T, B, C, H, W)
      // vc_next: (B, 2 * (C*H*W))
      // ═══════════════════════════════════════════════════════════════════════

      Location loc = op.getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      declareIntrinsics(rewriter, module, loc);

      // 2) 取输入
      Value x_seq = op.getOperands()[0];
      Value vc_init = op.getOperands()[1];

      auto xType = llvm::cast<RankedTensorType>(x_seq.getType());
      auto vcType = llvm::cast<RankedTensorType>(vc_init.getType());
      Type xElemTy = xType.getElementType();
      Type vcElemTy = vcType.getElementType();

      // 输出类型
      auto resType0 = llvm::cast<RankedTensorType>(op.getResult(0).getType()); // spike
      auto resType1 = llvm::cast<RankedTensorType>(op.getResult(1).getType()); // vc_next

      // 3) 解析维度：x (T,B,C,H,W)
      if (xType.getRank() != 5)
        return rewriter.notifyMatchFailure(op, "x_seq must be rank-5 (T,B,C,H,W)");

      int64_t T = xType.getDimSize(0);
      int64_t B = xType.getDimSize(1);
      int64_t C = xType.getDimSize(2);
      int64_t H = xType.getDimSize(3);
      int64_t W = xType.getDimSize(4);

      int64_t N = C * H * W;   // 每个 batch 的 neuron 数
      int64_t N2 = 2 * N;      // [V || I]

      // vc (B, 2N)
      if (vcType.getRank() != 2)
        return rewriter.notifyMatchFailure(op, "vc_init must be rank-2 (B, 2N)");

      if (vcType.getDimSize(0) != B)
        return rewriter.notifyMatchFailure(op, "vc_init dim0 must equal B");

      if (vcType.getDimSize(1) != N2)
        return rewriter.notifyMatchFailure(op, "vc_init dim1 must equal 2*(C*H*W)");

      int64_t numHwBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

      // 常量辅助（尽量沿用你的写法）
      auto mkIndex = [&](int64_t v) {
        return rewriter.create<arith::ConstantIndexOp>(loc, v);
      };
      auto mkI64 = [&](int64_t v) {
        return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(v));
      };

      Value cst_0idx = mkIndex(0);
      Value cst_64_i64 = mkI64(64);
      Value cst_mask32_i64 =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

      // 4) 分配 buffer
      Value spikeBuf = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get({T, B, C, H, W}, resType0.getElementType()));

      Value vcFinalBuf = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get({B, N2}, resType1.getElementType()));

      // tensor -> memref
      Value xMem = tensorToMemref(rewriter, loc, x_seq,
                                  MemRefType::get({T, B, C, H, W}, xElemTy));
      Value vcInitMem = tensorToMemref(rewriter, loc, vc_init,
                                       MemRefType::get({B, N2}, vcElemTy));

      // 5) 初始化：vcInitMem -> vcFinalBuf
      buildKrnlLoop(rewriter, loc, mkIndex(0), mkIndex(B),
                    [&](OpBuilder &b0, Location l0, Value ib) {
        buildKrnlLoop(b0, l0, mkIndex(0), mkIndex(N2),
                      [&](OpBuilder &b1, Location l1, Value i) {
          Value val = b1.create<memref::LoadOp>(l1, vcInitMem, ValueRange{ib, i});
          b1.create<memref::StoreOp>(l1, val, vcFinalBuf, ValueRange{ib, i});
        });
      });

      // 6) 时间步循环 it
      buildKrnlLoop(rewriter, loc, mkIndex(0), mkIndex(T),
                    [&](OpBuilder &b, Location l, Value it) {

        // batch 循环 ib
        buildKrnlLoop(b, l, mkIndex(0), mkIndex(B),
                      [&](OpBuilder &bb, Location lb, Value ib) {

          for (int64_t hwb = 0; hwb < numHwBatches; ++hwb) {
            int64_t nStart = hwb * kHardwareNeuronNum;
            int64_t calN = std::min<int64_t>(kHardwareNeuronNum, N - nStart);

            bb.create<func::CallOp>(lb, "rvne_clear_neuron_data_1024", TypeRange{}, ValueRange{});

            // -------------------------
            // 写入 V：NCR[0..calN-1]
            // -------------------------
            buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(calN),
                          [&](OpBuilder &b2, Location l2, Value ln) {
              Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
              Value gn_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart));
              Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_i64);

              Value v_val = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{ib, gn_idx});
              Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ln_i64);

              b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{},
                                      ValueRange{castElemToI32ForNCR(b2, l2, v_val, vcElemTy), ncr_off});
            });

            // -------------------------
            // 写入 I+x：NCR[calN..2*calN-1]
            // 其中 x 来自 xMem[it, ib, c, h, w] (gn 反解到 CHW)
            // -------------------------
            buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(calN),
                          [&](OpBuilder &b2, Location l2, Value ln) {
              Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
              Value gn_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart));

              int64_t HW = H * W;
              Value c_i64 = b2.create<arith::DivSIOp>(l2, gn_i64, mkI64(HW));
              Value rem_i64 = b2.create<arith::RemSIOp>(l2, gn_i64, mkI64(HW));
              Value h_i64 = b2.create<arith::DivSIOp>(l2, rem_i64, mkI64(W));
              Value w_i64 = b2.create<arith::RemSIOp>(l2, rem_i64, mkI64(W));

              Value c_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), c_i64);
              Value h_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), h_i64);
              Value w_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), w_i64);

              Value gn_I_i64 = b2.create<arith::AddIOp>(l2, gn_i64, mkI64(N));
              Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_I_i64);

              Value i_old = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{ib, gn_I_idx});
              Value x_val = b2.create<memref::LoadOp>(l2, xMem, ValueRange{it, ib, c_idx, h_idx, w_idx});

              Value i_new = vcElemTy.isF32()
                                ? b2.create<arith::AddFOp>(l2, i_old, x_val).getResult()
                                : b2.create<arith::AddIOp>(l2, i_old, x_val).getResult();

              Value ncr_off_i64 = b2.create<arith::AddIOp>(l2, ln_i64, mkI64(calN));
              Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ncr_off_i64);

              b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{},
                                      ValueRange{castElemToI32ForNCR(b2, l2, i_new, vcElemTy), ncr_off});
            });

            // 硬件发射
            bb.create<func::CallOp>(lb, "rvne_leakage_integral_fire_1024", TypeRange{}, ValueRange{});

            // -------------------------
            // 读回 Spike (SOR)，写入 spikeBuf[it,ib,c,h,w]
            // -------------------------
            int64_t numGroups = (calN + 63) / 64;
            buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(numGroups),
                          [&](OpBuilder &b2, Location l2, Value ig) {
              Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);

              auto rc = b2.create<func::CallOp>(
                  l2, "rvne_read_sor_group2", TypeRange{b2.getI64Type()},
                  ValueRange{b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64)});
              Value raw64 = rc.getResult(0);

              buildKrnlLoop(b2, l2, mkIndex(0), mkIndex(64),
                            [&](OpBuilder &b3, Location l3, Value ibit) {
                Value ibit_i64 = b3.create<arith::IndexCastOp>(l3, b3.getI64Type(), ibit);
                Value local_n = b3.create<arith::AddIOp>(
                    l3, b3.create<arith::MulIOp>(l3, ig_i64, cst_64_i64), ibit_i64);

                Value inBounds = b3.create<arith::CmpIOp>(
                    l3, arith::CmpIPredicate::ult, local_n, mkI64(calN));

                Value gn_i64 = b3.create<arith::AddIOp>(l3, local_n, mkI64(nStart));

                int64_t HW = H * W;
                Value c_i64 = b3.create<arith::DivSIOp>(l3, gn_i64, mkI64(HW));
                Value rem_i64 = b3.create<arith::RemSIOp>(l3, gn_i64, mkI64(HW));
                Value h_i64 = b3.create<arith::DivSIOp>(l3, rem_i64, mkI64(W));
                Value w_i64 = b3.create<arith::RemSIOp>(l3, rem_i64, mkI64(W));              

                Value c_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), c_i64);
                Value h_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), h_i64);
                Value w_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), w_i64);

                Value s_i32 = b3.create<arith::TruncIOp>(l3, b3.getI32Type(), 
                                                         b3.create<arith::AndIOp>(l3, b3.create<arith::ShRUIOp>(l3, raw64, ibit_i64), mkI64(1)));

                b3.create<scf::IfOp>(l3, inBounds, [&](OpBuilder &b4, Location l4) {
                  b4.create<memref::StoreOp>(
                      l4,
                      castSpikeToElemTy(b4, l4, s_i32, resType0.getElementType()),
                      spikeBuf,
                      ValueRange{it, ib, c_idx, h_idx, w_idx});
                  b4.create<scf::YieldOp>(l4);
                });
              });
            });

            // -------------------------
            // 读回更新后的 V/I (NCR) 并持久化到 vcFinalBuf
            // -------------------------
            int64_t ncrGroups = (calN + 1) / 2; // ceil(calN/2)

            buildKrnlLoop(bb, lb, mkIndex(0), mkIndex(ncrGroups),
                          [&](OpBuilder &b2, Location l2, Value ig) {
              Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
              Value ig_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64);

              // 读回 V pair
              auto rb_V = b2.create<func::CallOp>(
                  l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{ig_i32});
              Value pair_V = rb_V.getResult(0);

              // 读回 I pair
              Value off_I_i64 = b2.create<arith::AddIOp>(l2, ig_i64, mkI64(ncrGroups));
              Value off_I_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), off_I_i64);

              auto rb_I = b2.create<func::CallOp>(
                  l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{off_I_i32});
              Value pair_I = rb_I.getResult(0);

              // lane 0/1
              for (int lane = 0; lane < 2; ++lane) {
                Value bitOff = (lane == 0) ? mkI64(0) : mkI64(32);

                Value l_idx_i64 = b2.create<arith::AddIOp>(
                    l2, b2.create<arith::MulIOp>(l2, ig_i64, mkI64(2)), mkI64(lane));

                Value inBounds = b2.create<arith::CmpIOp>(
                    l2, arith::CmpIPredicate::ult, l_idx_i64, mkI64(calN));

                Value val_V = b2.create<arith::TruncIOp>(
                    l2, b2.getI32Type(),
                    b2.create<arith::AndIOp>(
                        l2, b2.create<arith::ShRUIOp>(l2, pair_V, bitOff), cst_mask32_i64));

                Value val_I = b2.create<arith::TruncIOp>(
                    l2, b2.getI32Type(),
                    b2.create<arith::AndIOp>(
                        l2, b2.create<arith::ShRUIOp>(l2, pair_I, bitOff), cst_mask32_i64));

                Value gn_i64 = b2.create<arith::AddIOp>(l2, l_idx_i64, mkI64(nStart));
                Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_i64);

                Value gn_I_i64 = b2.create<arith::AddIOp>(l2, gn_i64, mkI64(N));
                Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), gn_I_i64);

                b2.create<scf::IfOp>(l2, inBounds, [&](OpBuilder &b3, Location l3) {
                  b3.create<memref::StoreOp>(
                      l3,
                      castI32FromNCRToElemTy(b3, l3, val_V, vcElemTy),
                      vcFinalBuf,
                      ValueRange{ib, gn_idx});

                  b3.create<memref::StoreOp>(
                      l3,
                      castI32FromNCRToElemTy(b3, l3, val_I, vcElemTy),
                      vcFinalBuf,
                      ValueRange{ib, gn_I_idx});
                  b3.create<scf::YieldOp>(l3);
                });
              }
            });
          } // end hw batches
        }); // end B loop
      }); // end T loop

      // 7) 输出替换
      Value finalSpikeTensor = memrefToTensor(rewriter, loc, spikeBuf, resType0);
      Value finalStateTensor = memrefToTensor(rewriter, loc, vcFinalBuf, resType1);
      rewriter.replaceOp(op, ValueRange{finalSpikeTensor, finalStateTensor});
      return success();
    }
    else
    {
      // ═══════════════════════════════════════════════════════════════════════
      // 多步 LIF 转换 Pattern：直接拦截 onnx.Custom 节点
      // 解决 T 维度循环、硬件状态持久化、以及 index 类型转换问题
      // ═══════════════════════════════════════════════════════════════════════
      Location loc = op.getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      declareIntrinsics(rewriter, module, loc);

      // 2. 提取输入 Operands (T, N) 和 (1, 2N)
      Value x_seq = op.getOperands()[0];
      Value vc_init = op.getOperands()[1];

      auto xType = llvm::cast<RankedTensorType>(x_seq.getType());
      auto vcType = llvm::cast<RankedTensorType>(vc_init.getType());
      Type xElemTy = xType.getElementType();
      Type vcElemTy = vcType.getElementType();

      // 提取输出的预期 Tensor 类型，用于最后的 replaceOp
      auto resType0 = llvm::cast<RankedTensorType>(op.getResult(0).getType());
      auto resType1 = llvm::cast<RankedTensorType>(op.getResult(1).getType());

      int64_t T = xType.getDimSize(0);
      int64_t N = xType.getDimSize(1);
      int64_t N2 = vcType.getDimSize(1);
      int64_t numBatches = (N + kHardwareNeuronNum - 1) / kHardwareNeuronNum;

      // 常量辅助
      auto mkIndex = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
      auto mkI64 = [&](int64_t v) { return rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(v)); };
      Value cst_0idx = mkIndex(0);
      Value cst_64_i64 = mkI64(64);
      Value cst_mask32_i64 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0xFFFFFFFFLL));

      // 3. 分配内存 Buffer
      Value spikeBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({T, N}, resType0.getElementType()));
      Value vcFinalBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1, N2}, resType1.getElementType()));

      Value xMem = tensorToMemref(rewriter, loc, x_seq, MemRefType::get({T, N}, xElemTy));
      Value vcInitMem = tensorToMemref(rewriter, loc, vc_init, MemRefType::get({1, N2}, vcElemTy));

      // 4. 初始化：将初始状态拷贝到持久化 Buffer vcFinalBuf 中
      buildKrnlLoop(rewriter, loc, cst_0idx, mkIndex(N2), [&](OpBuilder &b, Location l, Value i) {
        Value val = b.create<memref::LoadOp>(l, vcInitMem, ValueRange{cst_0idx, i});
        b.create<memref::StoreOp>(l, val, vcFinalBuf, ValueRange{cst_0idx, i});
      });

      // 5. 核心：显式时间步 T 维度循环
      buildKrnlLoop(rewriter, loc, cst_0idx, mkIndex(T), [&](OpBuilder &b, Location l, Value it) {
        for (int64_t bi = 0; bi < numBatches; ++bi) {
          int64_t nStart = bi * kHardwareNeuronNum;
          int64_t calN = std::min(kHardwareNeuronNum, N - nStart);

          b.create<func::CallOp>(l, "rvne_clear_neuron_data_1024", TypeRange{}, ValueRange{});

          // --- 写入膜电位 V ---
          buildKrnlLoop(b, l, mkIndex(0), mkIndex(calN), [&](OpBuilder &b2, Location l2, Value ln) {
            Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
            Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart)));
            Value v_val = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{cst_0idx, gn_idx});
            Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ln_i64); // 正确转换 index->i64->i32
            b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{}, 
                                    ValueRange{castElemToI32ForNCR(b2, l2, v_val, vcElemTy), ncr_off});
          });

          // --- 写入 I + x (x 来自当前时间步 it) ---
          buildKrnlLoop(b, l, mkIndex(0), mkIndex(calN), [&](OpBuilder &b2, Location l2, Value ln) {
            Value ln_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ln);
            Value gn_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(nStart)));
            Value gn_I_idx = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(N + nStart)));

            Value i_old = b2.create<memref::LoadOp>(l2, vcFinalBuf, ValueRange{cst_0idx, gn_I_idx});
            Value x_val = b2.create<memref::LoadOp>(l2, xMem, ValueRange{it, gn_idx});
            
            Value i_new = vcElemTy.isF32() ? b2.create<arith::AddFOp>(l2, i_old, x_val).getResult() 
                                           : b2.create<arith::AddIOp>(l2, i_old, x_val).getResult();

            Value ncr_off = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AddIOp>(l2, ln_i64, mkI64(calN)));
            b2.create<func::CallOp>(l2, "rvne_write_ncr", TypeRange{}, 
                                    ValueRange{castElemToI32ForNCR(b2, l2, i_new, vcElemTy), ncr_off});
          });

          // 硬件发射
          b.create<func::CallOp>(l, "rvne_leakage_integral_fire_1024", TypeRange{}, ValueRange{});

          // --- 读回 Spike (SOR) ---
          int64_t numGroups = (calN + 63) / 64;
          buildKrnlLoop(b, l, mkIndex(0), mkIndex(numGroups), [&](OpBuilder &b2, Location l2, Value ig) {
            Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
            auto rc = b2.create<func::CallOp>(l2, "rvne_read_sor_group2", TypeRange{b2.getI64Type()}, 
                                              ValueRange{b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64)});
            Value raw64 = rc.getResult(0);

            buildKrnlLoop(b2, l2, mkIndex(0), mkIndex(64), [&](OpBuilder &b3, Location l3, Value ibit) {
              Value ibit_i64 = b3.create<arith::IndexCastOp>(l3, b3.getI64Type(), ibit);
              Value local_n = b3.create<arith::AddIOp>(l3, b3.create<arith::MulIOp>(l3, ig_i64, cst_64_i64), ibit_i64);
              Value inBounds = b3.create<arith::CmpIOp>(l3, arith::CmpIPredicate::ult, local_n, mkI64(calN));
              Value gn_idx = b3.create<arith::IndexCastOp>(l3, b3.getIndexType(), b3.create<arith::AddIOp>(l3, local_n, mkI64(nStart)));

              Value s_i32 = b3.create<arith::TruncIOp>(l3, b3.getI32Type(), 
                                                       b3.create<arith::AndIOp>(l3, b3.create<arith::ShRUIOp>(l3, raw64, ibit_i64), mkI64(1)));
              b3.create<scf::IfOp>(l3, inBounds, [&](OpBuilder &b4, Location l4) {
                b4.create<memref::StoreOp>(l4, castSpikeToElemTy(b4, l4, s_i32, resType0.getElementType()), spikeBuf, ValueRange{it, gn_idx});
                b4.create<scf::YieldOp>(l4);
              });
            });
          });

          // --- 读回更新后的 V/I (NCR) 并持久化到 vcFinalBuf ---
          int64_t ncrGroups = calN / 2;
          buildKrnlLoop(b, l, mkIndex(0), mkIndex(ncrGroups), [&](OpBuilder &b2, Location l2, Value ig) {
            Value ig_i64 = b2.create<arith::IndexCastOp>(l2, b2.getI64Type(), ig);
            Value ig_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), ig_i64);

            // 读回 V
            auto rb_V = b2.create<func::CallOp>(l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{ig_i32});
            Value pair_V = rb_V.getResult(0);
            
            // 读回 I (偏移量为 calN/2)
            Value off_I_i32 = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AddIOp>(l2, ig_i64, mkI64(calN / 2)));
            auto rb_I = b2.create<func::CallOp>(l2, "rvne_read_ncr_group2", TypeRange{b2.getI64Type()}, ValueRange{off_I_i32});
            Value pair_I = rb_I.getResult(0);

            // 解析并存储每一对 (Lo, Hi)
            for (int i = 0; i < 2; ++i) {
              Value bitOff = (i == 0) ? mkI64(0) : mkI64(32);
              Value val_V = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AndIOp>(l2, b2.create<arith::ShRUIOp>(l2, pair_V, bitOff), cst_mask32_i64));
              Value val_I = b2.create<arith::TruncIOp>(l2, b2.getI32Type(), b2.create<arith::AndIOp>(l2, b2.create<arith::ShRUIOp>(l2, pair_I, bitOff), cst_mask32_i64));

              Value l_idx = b2.create<arith::AddIOp>(l2, b2.create<arith::MulIOp>(l2, ig_i64, mkI64(2)), mkI64(i));
              Value gn_V = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, l_idx, mkI64(nStart)));
              Value gn_I = b2.create<arith::IndexCastOp>(l2, b2.getIndexType(), b2.create<arith::AddIOp>(l2, l_idx, mkI64(N + nStart)));

              b2.create<memref::StoreOp>(l2, castI32FromNCRToElemTy(b2, l2, val_V, vcElemTy), vcFinalBuf, ValueRange{cst_0idx, gn_V});
              b2.create<memref::StoreOp>(l2, castI32FromNCRToElemTy(b2, l2, val_I, vcElemTy), vcFinalBuf, ValueRange{cst_0idx, gn_I});
            }
          });
        }
      });

      // 6. 替换输出，完成 Lowering
      Value finalSpikeTensor = memrefToTensor(rewriter, loc, spikeBuf, resType0);
      Value finalStateTensor = memrefToTensor(rewriter, loc, vcFinalBuf, resType1);
      rewriter.replaceOp(op, ValueRange{finalSpikeTensor, finalStateTensor});

      return success();
    }
    return success();
  }
};


struct ONNXDataToVectorOpToKrnlConversion
    : public OpRewritePattern<ONNXDataToVectorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXDataToVectorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();
    auto inType = cast<RankedTensorType>(input.getType());
    auto outType = cast<RankedTensorType>(op.getResult().getType());

    int64_t B = inType.getDimSize(0);
    int64_t N = inType.getDimSize(1);

    int64_t bitWidth = 32;
    if (auto bwAttr = op.getBitWidthAttr())
      bitWidth = bwAttr.getInt();
    if (bitWidth != 32)
      return op.emitError("Only bit_width=32 is supported for now"), failure();

    // 固定输出 32 words（对齐 C: int input_vector[32]）
    int64_t M = 32;

    // 输出必须是 [B,32]
    if (outType.getRank() != 2 || outType.getDimSize(0) != B ||
        outType.getDimSize(1) != M)
      return op.emitError("Output shape must be [B, 32]"), failure();

    Type inElemTy = inType.getElementType();   // f32
    Type outElemTy = outType.getElementType(); // f32 (承载 bitpattern)

    auto mkIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto mkI32 = [&](int32_t v) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, v, rewriter.getI32Type());
    };
    auto mkF32 = [&](float v) -> Value {
      return rewriter.create<arith::ConstantFloatOp>(loc, APFloat(v), rewriter.getF32Type());
    };

    Value c0 = mkIndex(0);
    Value cB = mkIndex(B);
    Value cN = mkIndex(N);
    Value cM = mkIndex(M);

    Value c0_i32 = mkI32(0);
    Value c1_i32 = mkI32(1);

    // alloc out [B,32]
    Value outBuf = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get({B, M}, outElemTy));

    Value inBuf = tensorToMemref(
        rewriter, loc, input, MemRefType::get({B, N}, inElemTy));

    buildKrnlLoop(rewriter, loc, c0, cB,
      [&](OpBuilder &b, Location loc, Value ib) {

        // (1) clear outBuf[ib][0..31] = 0
        buildKrnlLoop(b, loc, mkIndex(0), cM,
          [&](OpBuilder &b, Location loc, Value im) {
            if (outElemTy.isF32())
              b.create<memref::StoreOp>(loc, mkF32(0.0f), outBuf, ValueRange{ib, im});
            else
              b.create<memref::StoreOp>(loc, b.create<arith::ConstantIntOp>(loc, 0, outElemTy),
                                        outBuf, ValueRange{ib, im});
          });

        // (2) pack first N spikes into first ceil(N/32) words, rest stay 0
        buildKrnlLoop(b, loc, mkIndex(0), cN,
          [&](OpBuilder &b, Location loc, Value in) {
            Value x = b.create<memref::LoadOp>(loc, inBuf, ValueRange{ib, in});

            // isSpike = (x != 0)
            Value isSpike;
            if (inElemTy.isF32()) {
              Value z = mkF32(0.0f);
              isSpike = b.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::ONE, x, z);
            } else if (inElemTy.isInteger(32)) {
              isSpike = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ne, x, c0_i32);
            } else {
              b.getInsertionBlock()->getParentOp()->emitError(
                  "Unsupported input element type for DataToVector");
              return;
            }

            // idx = in / 32, bit = in % 32
            Value idx = b.create<arith::DivSIOp>(loc, in, mkIndex(32));
            Value bit = b.create<arith::RemSIOp>(loc, in, mkIndex(32));

            Value idx_i64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
            Value c32_i64 = b.create<arith::ConstantIntOp>(loc, 32, b.getI64Type());

            // guard: idx < 32 (防止 N > 1024 越界)
            // Value idxInRange = b.create<arith::CmpIOp>(
            //     loc, arith::CmpIPredicate::ult, idx, mkIndex(32));
            Value idxInRange = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ult, idx_i64, c32_i64);

            // if (idx < 32) { ... do or-store ... }
            b.create<scf::IfOp>(loc, idxInRange, [&](OpBuilder &b, Location loc) {
              Value bit_i32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), bit);
              Value mask = b.create<arith::ShLIOp>(loc, c1_i32, bit_i32);

              Value oldRaw = b.create<memref::LoadOp>(loc, outBuf, ValueRange{ib, idx});

              Value old_i32;
              if (outElemTy.isF32())
                old_i32 = b.create<arith::BitcastOp>(loc, b.getI32Type(), oldRaw);
              else if (outElemTy.isInteger(32))
                old_i32 = oldRaw;
              else
                old_i32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), oldRaw);

              Value addMask = b.create<arith::SelectOp>(loc, isSpike, mask, c0_i32);
              Value new_i32 = b.create<arith::OrIOp>(loc, old_i32, addMask);

              if (outElemTy.isF32()) {
                Value newRaw = b.create<arith::BitcastOp>(loc, b.getF32Type(), new_i32);
                b.create<memref::StoreOp>(loc, newRaw, outBuf, ValueRange{ib, idx});
              } else {
                b.create<memref::StoreOp>(loc, new_i32, outBuf, ValueRange{ib, idx});
              }
              b.create<scf::YieldOp>(loc);
            });
          });
      });

    Value outTensor = memrefToTensor(rewriter, loc, outBuf, outType);
    rewriter.replaceOp(op, outTensor);
    return success();
  }
};



struct ONNXSNNFCOpToKrnlConversion : public OpRewritePattern<ONNXSNNFCOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXSNNFCOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    declareSNNFCIntrinsics(rewriter, module, loc);

    Value inputVector = op.getInputVector();   // [B, M]
    Value inputWeight = op.getInputweight();   // [K1, NeuronNum]
    Value lsmWeight   = op.getLsmweight();     // [K2, NeuronNum]
    Value anchor      = op.getShapeAnchor();   // [B, NeuronNum]

    auto vecType = dyn_cast<RankedTensorType>(inputVector.getType());
    auto iwType  = dyn_cast<RankedTensorType>(inputWeight.getType());
    auto lwType  = dyn_cast<RankedTensorType>(lsmWeight.getType());
    auto aType   = dyn_cast<RankedTensorType>(anchor.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!vecType || !iwType || !lwType || !aType || !outType)
      return op.emitError("All operands/results must be RankedTensorType"), failure();

    if (vecType.getRank() != 2 || iwType.getRank() != 2 || lwType.getRank() != 2 || aType.getRank() != 2)
      return op.emitError("All inputs must be rank-2"), failure();

    // Output type must equal anchor type (per your importer).
    if (outType != aType)
      return op.emitError("result type must equal shape_anchor type"), failure();

    Type vecElemTy = vecType.getElementType();
    Type iwElemTy  = iwType.getElementType();
    Type lwElemTy  = lwType.getElementType();
    Type outElemTy = outType.getElementType();

    auto i32Ty = rewriter.getI32Type();
    auto mkIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto c0 = mkIndex(0);

    auto isStatic = [](int64_t d) { return d != ShapedType::kDynamic; };

    // MemRef types can keep dynamic dims.
    auto makeMrTy = [&](int64_t d0, int64_t d1, Type elemTy) -> MemRefType {
      return MemRefType::get({d0, d1}, elemTy);
    };

    int64_t B_s  = vecType.getDimSize(0);
    int64_t M_s  = vecType.getDimSize(1);
    int64_t K1_s = iwType.getDimSize(0);
    int64_t N_s  = iwType.getDimSize(1);
    int64_t K2_s = lwType.getDimSize(0);
    int64_t N2_s = lwType.getDimSize(1);

    int64_t aB_s = aType.getDimSize(0);
    int64_t aN_s = aType.getDimSize(1);

    // Static consistency checks when known.
    if (isStatic(N_s) && isStatic(N2_s) && N_s != N2_s)
      return op.emitError("inputweight/lsmweight NeuronNum mismatch"), failure();
    if (isStatic(B_s) && isStatic(aB_s) && B_s != aB_s)
      return op.emitError("batch mismatch between input_vector and anchor"), failure();
    if (isStatic(N_s) && isStatic(aN_s) && N_s != aN_s)
      return op.emitError("NeuronNum mismatch between inputweight and anchor"), failure();

    Value vecBuf = tensorToMemref(rewriter, loc, inputVector, makeMrTy(B_s,  M_s,  vecElemTy));
    Value iwBuf  = tensorToMemref(rewriter, loc, inputWeight, makeMrTy(K1_s, N_s,  iwElemTy));
    Value lwBuf  = tensorToMemref(rewriter, loc, lsmWeight,   makeMrTy(K2_s, N2_s, lwElemTy));

    // Allocate output [B, NeuronNum] using anchor shape. Dynamic sizes come from inputs.
    MemRefType outMrTy = makeMrTy(aB_s, aN_s, outElemTy);
    SmallVector<Value, 2> outSizes;
    if (!isStatic(aB_s))
      outSizes.push_back(rewriter.create<memref::DimOp>(loc, vecBuf, 0));  // B
    if (!isStatic(aN_s))
      outSizes.push_back(rewriter.create<memref::DimOp>(loc, iwBuf, 1));   // NeuronNum
    Value outBuf = rewriter.create<memref::AllocOp>(loc, outMrTy, outSizes);

    // Loop bounds as Values.
    Value B  = isStatic(aB_s) ? mkIndex(aB_s) : rewriter.create<memref::DimOp>(loc, outBuf, 0);
    Value N  = isStatic(aN_s) ? mkIndex(aN_s) : rewriter.create<memref::DimOp>(loc, outBuf, 1);
    Value M  = isStatic(M_s)  ? mkIndex(M_s)  : rewriter.create<memref::DimOp>(loc, vecBuf, 1);
    Value K1 = isStatic(K1_s) ? mkIndex(K1_s) : rewriter.create<memref::DimOp>(loc, iwBuf, 0);
    Value K2 = isStatic(K2_s) ? mkIndex(K2_s) : rewriter.create<memref::DimOp>(loc, lwBuf, 0);

    // lanesPerAcc：每次 rvne_current_acc_128 覆盖多少个权重。
    // 建议：做成 op attribute。这里先给默认 16；如有 attr 可替换读取。
    int64_t lanesPerAcc = 16;

    // Compute acc iteration counts:
    // accIters = ceil_div(K, lanesPerAcc)
    auto ceilDivIndex = [&](OpBuilder &b, Location l, Value x) -> Value {
      Value lanesV = b.create<arith::ConstantIndexOp>(l, lanesPerAcc);
      Value lanesMinus1 = b.create<arith::ConstantIndexOp>(l, lanesPerAcc - 1);
      Value num = b.create<arith::AddIOp>(l, x, lanesMinus1);
      return b.create<arith::DivSIOp>(l, num, lanesV);
    };

    // Cast helpers.
    auto castScalarToI32 = [&](OpBuilder &b, Location l, Value scalar, Type elemTy) -> Value {
      if (elemTy.isInteger(32))
        return scalar;
      if (elemTy.isInteger(1) || elemTy.isInteger(8) || elemTy.isInteger(16))
        return b.create<arith::ExtUIOp>(l, i32Ty, scalar);
      if (elemTy.isInteger(64))
        return b.create<arith::TruncIOp>(l, i32Ty, scalar);
      if (elemTy.isF32())
        return b.create<arith::BitcastOp>(l, i32Ty, scalar); // packed bitpattern
      if (elemTy.isa<FloatType>())
        return b.create<arith::FPToSIOp>(l, i32Ty, scalar);
      b.getInsertionBlock()->getParentOp()->emitError("Unsupported elem type for SNNFC");
      return Value();
    };
    auto idxToI32 = [&](OpBuilder &b, Location l, Value idx) -> Value {
      return b.create<arith::IndexCastOp>(l, i32Ty, idx);
    };

    // Call helpers.
    auto call0 = [&](OpBuilder &b, Location l, StringRef name) {
      b.create<func::CallOp>(l, name, TypeRange{}, ValueRange{});
    };
    auto call2 = [&](OpBuilder &b, Location l, StringRef name, Value a0, Value a1) {
      b.create<func::CallOp>(l, name, TypeRange{}, ValueRange{a0, a1});
    };
    auto call3 = [&](OpBuilder &b, Location l, StringRef name, Value a0, Value a1, Value a2) {
      b.create<func::CallOp>(l, name, TypeRange{}, ValueRange{a0, a1, a2});
    };

    // for b in [0..B)
    buildKrnlLoop(rewriter, loc, c0, B,
      [&](OpBuilder &b, Location l, Value ib) {

        // 1) write input_vector[ib][j] into svr
        buildKrnlLoop(b, l, c0, M,
          [&](OpBuilder &b, Location l, Value j) {
            Value v = b.create<memref::LoadOp>(l, vecBuf, ValueRange{ib, j});
            Value v_i32 = castScalarToI32(b, l, v, vecElemTy);
            call2(b, l, "rvne_write_svr", v_i32, idxToI32(b, l, j));
          });

        // Compute acc loops for input weights / lsm weights (per batch).
        // These are scalars used as loop upper bounds.
        Value accInputIters = ceilDivIndex(b, l, K1);
        Value accLsmIters   = ceilDivIndex(b, l, K2);

        // 2) per neuron: write inputweight, then acc over ceil(K1/lanes)
        buildKrnlLoop(b, l, c0, N,
          [&](OpBuilder &b, Location l, Value j) {

            buildKrnlLoop(b, l, c0, K1,
              [&](OpBuilder &b, Location l, Value k) {
                Value w = b.create<memref::LoadOp>(l, iwBuf, ValueRange{k, j});
                Value w_i32 = castScalarToI32(b, l, w, iwElemTy);
                call2(b, l, "rvne_write_wvr", w_i32, idxToI32(b, l, k));
              });

            // for kk in [0..accInputIters): rvne_current_acc_128(kk, kk, j)
            buildKrnlLoop(b, l, c0, accInputIters,
              [&](OpBuilder &b, Location l, Value kk) {
                Value kk_i32 = idxToI32(b, l, kk);
                call3(b, l, "rvne_current_acc_128", kk_i32, kk_i32, idxToI32(b, l, j));
              });
          });

        // 3) spike propagation
        call0(b, l, "rvne_spike_propagation_1024");

        // 4) per neuron: write lsmweight, then acc over ceil(K2/lanes)
        buildKrnlLoop(b, l, c0, N,
          [&](OpBuilder &b, Location l, Value j) {

            buildKrnlLoop(b, l, c0, K2,
              [&](OpBuilder &b, Location l, Value k) {
                Value w = b.create<memref::LoadOp>(l, lwBuf, ValueRange{k, j});
                Value w_i32 = castScalarToI32(b, l, w, lwElemTy);
                call2(b, l, "rvne_write_wvr", w_i32, idxToI32(b, l, k));
              });

            // for kk in [0..accLsmIters): rvne_current_acc_128(kk, kk, j)
            buildKrnlLoop(b, l, c0, accLsmIters,
              [&](OpBuilder &b, Location l, Value kk) {
                Value kk_i32 = idxToI32(b, l, kk);
                call3(b, l, "rvne_current_acc_128", kk_i32, kk_i32, idxToI32(b, l, j));
              });

            // output writeback placeholder
            Value zero;
            if (outElemTy.isF32())
              zero = b.create<arith::ConstantFloatOp>(l, APFloat(0.0f), b.getF32Type());
            else if (outElemTy.isInteger(32))
              zero = b.create<arith::ConstantIntOp>(l, 0, i32Ty);
            else
              zero = b.create<arith::ConstantIntOp>(l, 0, outElemTy);

            b.create<memref::StoreOp>(l, zero, outBuf, ValueRange{ib, j});
          });
      });

    Value outTensor = memrefToTensor(rewriter, loc, outBuf, outType);
    rewriter.replaceOp(op, outTensor);
    return success();
  }
};




//===----------------------------------------------------------------------===//
// 1) Helpers: declare intrinsic bridge(s)
//===----------------------------------------------------------------------===//

static bool isStaticShape(ArrayRef<int64_t> shape) {
  for (int64_t d : shape)
    if (d == ShapedType::kDynamic)
      return false;
  return true;
}

static std::optional<int64_t>
computeOutDimStatic(int64_t in, int64_t padding, int64_t k, int64_t stride) {
  if (in == ShapedType::kDynamic)
    return std::nullopt;
  // (in + 2P - K) / S + 1
  int64_t num = in + 2 * padding - k;
  if (num < 0)
    return std::nullopt;
  if (stride <= 0)
    return std::nullopt;
  // Require exact divisibility to keep it "obviously" consistent.
  if (num % stride != 0)
    return std::nullopt;
  return num / stride + 1;
}

//===----------------------------------------------------------------------===//
// A) Default intrinsic entry: pointer ABI (single fixed signature)
//===----------------------------------------------------------------------===//

static void declareConvIntrinsicPtrDefault(OpBuilder &builder, ModuleOp module,
                                          Location loc) {
  MLIRContext *ctx = builder.getContext();
  Type i32T = builder.getI32Type();
  Type i64T = builder.getI64Type();
  Type f32T = builder.getF32Type();

  // emitc.ptr<f32>
  Type f32PtrT = emitc::PointerType::get(f32T);

  auto fnTy = FunctionType::get(
      ctx,
      {/*dims*/ i32T, i32T, i32T, i32T,
       /*params*/ i32T, i32T, i32T, i32T,
       /*input*/ f32PtrT, i64T,
       /*weight*/ f32PtrT, i64T,
       /*out*/ f32PtrT, i64T},
      {i32T});

  getOrInsertFunc(builder, module, loc,
                  "conv_compute_currents_from_f32_ptr_defaultNP_f32out", fnTy);
}

// Keep old entry point: declare default ptr entry always.
static void declareConvIntrinsicBridge(OpBuilder &builder, ModuleOp module,
                                       Location loc) {
  declareConvIntrinsicPtrDefault(builder, module, loc);
}

//===----------------------------------------------------------------------===//
// B) Wrapper generator: per-static-shape wrapper with static memref signature
//    Wrapper extracts raw pointers and calls the default ptr entry.
//===----------------------------------------------------------------------===//

static std::string buildConvWrapperName(MemRefType xMrTy, MemRefType wMrTy,
                                        MemRefType outLinearTy,
                                        int64_t outElems) {
  auto xs = xMrTy.getShape(); // [T,CHI,HI,WI]
  auto ws = wMrTy.getShape(); // [CHO,WCHI,KH,KW]
  std::string name = "conv_wrapper_to_ptr_defaultNP_f32out";
  name += "_T" + std::to_string(xs[0]);
  name += "_CHI" + std::to_string(xs[1]);
  name += "_HI" + std::to_string(xs[2]);
  name += "_WI" + std::to_string(xs[3]);
  name += "_CHO" + std::to_string(ws[0]);
  name += "_WCHI" + std::to_string(ws[1]);
  name += "_KH" + std::to_string(ws[2]);
  name += "_KW" + std::to_string(ws[3]);
  name += "_OUTE" + std::to_string(outElems);
  return name;
}

static func::FuncOp getOrCreateConvWrapperToPtr(OpBuilder &builder,
                                                ModuleOp module, Location loc,
                                                MemRefType xMrTy,
                                                MemRefType wMrTy,
                                                MemRefType outLinearTy,
                                                int64_t outElemsSS) {
  MLIRContext *ctx = builder.getContext();
  Type i32T = builder.getI32Type();
  Type i64T = builder.getI64Type();
  Type f32T = builder.getF32Type();
  Type f32PtrT = emitc::PointerType::get(f32T);

  std::string wrapperName =
      buildConvWrapperName(xMrTy, wMrTy, outLinearTy, outElemsSS);

  if (auto existing = module.lookupSymbol<func::FuncOp>(wrapperName))
    return existing;

  // Ensure default ptr entry exists.
  declareConvIntrinsicPtrDefault(builder, module, loc);

  // Wrapper signature keeps memrefs static to avoid static->dynamic cast issues.
  auto fnTy = FunctionType::get(
      ctx,
      {/*dims*/ i32T, i32T, i32T, i32T,
       /*params*/ i32T, i32T, i32T, i32T,
       /*input*/ xMrTy, i64T,
       /*weight*/ wMrTy, i64T,
       /*out*/ outLinearTy, i64T},
      {i32T});

  auto fn = func::FuncOp::create(loc, wrapperName, fnTy);
  // module.push_back(fn);
  module.getBody()->push_front(fn);

  OpBuilder::InsertionGuard g(builder);
  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // args
  Value T_i32   = entry->getArgument(0);
  Value CHI_i32 = entry->getArgument(1);
  Value HI_i32  = entry->getArgument(2);
  Value WI_i32  = entry->getArgument(3);

  Value CHO_i32 = entry->getArgument(4);
  Value K_i32   = entry->getArgument(5);
  Value S_i32   = entry->getArgument(6);
  Value P_i32   = entry->getArgument(7);

  Value xMem = entry->getArgument(8);
  Value inElemsI64 = entry->getArgument(9);

  Value wMem = entry->getArgument(10);
  Value wElemsI64 = entry->getArgument(11);

  Value outMem = entry->getArgument(12);
  Value outElemsI64 = entry->getArgument(13);

  // memref -> raw pointer (as index)
  auto extractPtrAsIndex = [&](Value mem) -> Value {
    return builder
        .create<memref::ExtractAlignedPointerAsIndexOp>(loc, mem)
        .getResult();
  };

  Value xPtrIdx   = extractPtrAsIndex(xMem);
  Value wPtrIdx   = extractPtrAsIndex(wMem);
  Value outPtrIdx = extractPtrAsIndex(outMem);

  // index -> emitc.ptr<f32>
  // If your EmitC does not support this CastOp signature, compilation will fail;
  // report the error and we'll replace with a call_opaque-based cast.
  Value xPtr = builder.create<emitc::CastOp>(loc, f32PtrT, xPtrIdx).getResult();
  Value wPtr = builder.create<emitc::CastOp>(loc, f32PtrT, wPtrIdx).getResult();
  Value oPtr = builder.create<emitc::CastOp>(loc, f32PtrT, outPtrIdx).getResult();

  // Call the single default entry (ptr ABI).
  auto call = builder.create<func::CallOp>(
      loc, "conv_compute_currents_from_f32_ptr_defaultNP_f32out",
      TypeRange{i32T},
      ValueRange{
          T_i32, CHI_i32, HI_i32, WI_i32,
          CHO_i32, K_i32, S_i32, P_i32,
          xPtr, inElemsI64,
          wPtr, wElemsI64,
          oPtr, outElemsI64});

  builder.create<func::ReturnOp>(loc, call.getResult(0));
  return fn;
}

//===----------------------------------------------------------------------===//
// 2) Helper: read i64 array attrs (pads/strides/dilations/kernel_shape)
//===----------------------------------------------------------------------===//

static LogicalResult getI64ArrayAttr(ONNXConvOp op, StringRef name,
                                    SmallVectorImpl<int64_t> &out) {
  auto arr = op->getAttrOfType<ArrayAttr>(name);
  if (!arr)
    return failure();
  out.clear();
  out.reserve(arr.size());
  for (Attribute a : arr) {
    auto i = dyn_cast<IntegerAttr>(a);
    if (!i)
      return failure();
    out.push_back(i.getInt());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// 3) Pattern: ONNXConvOp -> func.call intrinsic + linear buffer + copy-back
//===----------------------------------------------------------------------===//

struct ONNXConvOpToKrnlIntrinsicCall final
    : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    declareConvIntrinsicBridge(rewriter, module, loc);

    // Operands
    Value X = op.getX();
    Value W = op.getW();
    Value B = op.getB();

    // Types
    auto xType = dyn_cast<RankedTensorType>(X.getType());
    auto wType = dyn_cast<RankedTensorType>(W.getType());
    auto yType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!xType || !wType || !yType)
      return op.emitError("X/W/Y must be RankedTensorType"), failure();

    if (xType.getRank() != 4 || wType.getRank() != 4 || yType.getRank() != 4)
      return op.emitError("Only rank-4 conv supported (NCHW)"), failure();

    if (!xType.getElementType().isF32() || !wType.getElementType().isF32() ||
        !yType.getElementType().isF32())
      return op.emitError("Only f32 X/W/Y supported"), failure();

    // Bias must be none
    if (B && !isa<NoneType>(B.getType()))
      return op.emitError("Bias is not supported by this lowering"), failure();

    // Attributes (force group=1)
    SmallVector<int64_t, 4> pads, strides, dilations, kernelShape;
    if (failed(getI64ArrayAttr(op, "pads", pads)))
      return op.emitError("Missing/invalid pads"), failure();
    if (failed(getI64ArrayAttr(op, "strides", strides)))
      return op.emitError("Missing/invalid strides"), failure();

    if (failed(getI64ArrayAttr(op, "dilations", dilations)))
      dilations.assign({1, 1});

    if (failed(getI64ArrayAttr(op, "kernel_shape", kernelShape))) {
      int64_t KH = wType.getDimSize(2);
      int64_t KW = wType.getDimSize(3);
      if (KH == ShapedType::kDynamic || KW == ShapedType::kDynamic)
        return op.emitError("kernel_shape missing and cannot infer"), failure();
      kernelShape.assign({KH, KW});
    }

    // Constraints
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1)
      return op.emitError("Only dilations=[1,1] supported"), failure();
    if (strides.size() != 2)
      return op.emitError("strides must have 2 elements"), failure();
    if (strides[0] != strides[1])
      return op.emitError("Only stride_h == stride_w supported"), failure();
    if (pads.size() != 4)
      return op.emitError("pads must have 4 elements"), failure();
    if (!(pads[0] == pads[1] && pads[0] == pads[2] && pads[0] == pads[3]))
      return op.emitError("Only symmetric pads=[p,p,p,p] supported"), failure();
    if (kernelShape.size() != 2 || kernelShape[0] != kernelShape[1])
      return op.emitError("Only square kernel supported"), failure();

    int64_t K = kernelShape[0];
    int64_t STRIDE = strides[0];
    int64_t PADDING = pads[0];

    auto isStatic = [](int64_t d) { return d != ShapedType::kDynamic; };

    // Static dims (may be dynamic)
    int64_t T_s   = xType.getDimSize(0);
    int64_t CHI_s = xType.getDimSize(1);
    int64_t HI_s  = xType.getDimSize(2);
    int64_t WI_s  = xType.getDimSize(3);

    int64_t CHO_s  = wType.getDimSize(0);
    int64_t WCHI_s = wType.getDimSize(1);
    int64_t KH_s   = wType.getDimSize(2);
    int64_t KW_s   = wType.getDimSize(3);

    int64_t yT_s = yType.getDimSize(0);
    int64_t yC_s = yType.getDimSize(1);
    int64_t HO_s = yType.getDimSize(2);
    int64_t WO_s = yType.getDimSize(3);

    // consistency checks
    if (isStatic(WCHI_s) && isStatic(CHI_s) && WCHI_s != CHI_s)
      return op.emitError("weight CHI mismatch"), failure();
    if (isStatic(KH_s) && KH_s != K)
      return op.emitError("weight KH mismatch"), failure();
    if (isStatic(KW_s) && KW_s != K)
      return op.emitError("weight KW mismatch"), failure();
    if (isStatic(yT_s) && isStatic(T_s) && yT_s != T_s)
      return op.emitError("Y batch mismatch"), failure();
    if (isStatic(yC_s) && isStatic(CHO_s) && yC_s != CHO_s)
      return op.emitError("Y channel mismatch"), failure();

    // tensor -> memref
    Type f32Ty = rewriter.getF32Type();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();

    MemRefType xMrTy = MemRefType::get({T_s, CHI_s, HI_s, WI_s}, f32Ty);
    MemRefType wMrTy = MemRefType::get({CHO_s, WCHI_s, KH_s, KW_s}, f32Ty);

    Value xBuf = tensorToMemref(rewriter, loc, X, xMrTy);
    Value wBuf = tensorToMemref(rewriter, loc, W, wMrTy);

    // Helpers
    auto cIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    auto dim = [&](Value mem, int64_t axis) -> Value {
      return rewriter.create<memref::DimOp>(loc, mem, axis).getResult();
    };
    auto cI32 = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, v, i32Ty).getResult();
    };
    auto idxToI32 = [&](Value idx) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, i32Ty, idx).getResult();
    };
    auto idxToI64 = [&](Value idx) -> Value {
      return rewriter.create<arith::IndexCastOp>(loc, i64Ty, idx).getResult();
    };
    auto cI64 = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, v, i64Ty).getResult();
    };

    // runtime dims (index)
    Value T_v   = isStatic(T_s)   ? cIndex(T_s)   : dim(xBuf, 0);
    Value CHI_v = isStatic(CHI_s) ? cIndex(CHI_s) : dim(xBuf, 1);
    Value HI_v  = isStatic(HI_s)  ? cIndex(HI_s)  : dim(xBuf, 2);
    Value WI_v  = isStatic(WI_s)  ? cIndex(WI_s)  : dim(xBuf, 3);
    Value CHO_v = isStatic(CHO_s) ? cIndex(CHO_s) : dim(wBuf, 0);

    // Static HO/WO if possible (for fully static outLinear length)
    std::optional<int64_t> HO_static =
        computeOutDimStatic(HI_s, PADDING, K, STRIDE);
    std::optional<int64_t> WO_static =
        computeOutDimStatic(WI_s, PADDING, K, STRIDE);

    // HO/WO values (index) used for copy-back loops
    Value HO_v = isStatic(HO_s) ? cIndex(HO_s) : [&]() -> Value {
      Value twoP = cIndex(2 * PADDING);
      Value Kc   = cIndex(K);
      Value Str  = cIndex(STRIDE);
      Value num = rewriter.create<arith::SubIOp>(
          loc,
          rewriter.create<arith::AddIOp>(loc, HI_v, twoP), Kc).getResult();
      Value div = rewriter.create<arith::DivSIOp>(loc, num, Str).getResult();
      return rewriter.create<arith::AddIOp>(loc, div, cIndex(1)).getResult();
    }();

    Value WO_v = isStatic(WO_s) ? cIndex(WO_s) : [&]() -> Value {
      Value twoP = cIndex(2 * PADDING);
      Value Kc   = cIndex(K);
      Value Str  = cIndex(STRIDE);
      Value num = rewriter.create<arith::SubIOp>(
          loc,
          rewriter.create<arith::AddIOp>(loc, WI_v, twoP), Kc).getResult();
      Value div = rewriter.create<arith::DivSIOp>(loc, num, Str).getResult();
      return rewriter.create<arith::AddIOp>(loc, div, cIndex(1)).getResult();
    }();

    // elems (index)
    Value inElemsIdx =
        rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, T_v, CHI_v).getResult(),
                HI_v).getResult(),
            WI_v).getResult();

    Value wElemsIdx =
        rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, CHO_v, CHI_v).getResult(),
                cIndex(K)).getResult(),
            cIndex(K)).getResult();

    Value outElemsIdx =
        rewriter.create<arith::MulIOp>(
            loc,
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::MulIOp>(loc, T_v, CHO_v).getResult(),
                HO_v).getResult(),
            WO_v).getResult();

    Value inElemsI64  = idxToI64(inElemsIdx);
    Value wElemsI64   = idxToI64(wElemsIdx);
    Value outElemsI64 = idxToI64(outElemsIdx);

    // i32 args
    Value T_i32   = idxToI32(T_v);
    Value CHI_i32 = idxToI32(CHI_v);
    Value HI_i32  = idxToI32(HI_v);
    Value WI_i32  = idxToI32(WI_v);
    Value CHO_i32 = idxToI32(CHO_v);

    Value K_i32       = cI32(K);
    Value STRIDE_i32  = cI32(STRIDE);
    Value PADDING_i32 = cI32(PADDING);

    // Decide wrapper-vs-direct-ptr call.
    bool xStatic4 = isStaticShape(xMrTy.getShape());
    bool wStatic4 = isStaticShape(wMrTy.getShape());
    bool canStaticOut =
        xStatic4 && wStatic4 && HO_static.has_value() && WO_static.has_value();

    Value outLinear;
    StringRef calleeName;
    SmallVector<Value, 16> callArgs;

    if (canStaticOut) {
      int64_t HO_ss = *HO_static;
      int64_t WO_ss = *WO_static;
      int64_t outElemsSS = T_s * CHO_s * HO_ss * WO_ss;

      MemRefType outLinearTy = MemRefType::get({outElemsSS}, f32Ty);
      outLinear = rewriter.create<memref::AllocOp>(loc, outLinearTy).getResult();

      func::FuncOp wrapper = getOrCreateConvWrapperToPtr(
          rewriter, module, loc, xMrTy, wMrTy, outLinearTy, outElemsSS);

      calleeName = wrapper.getName();

      callArgs = {
          T_i32, CHI_i32, HI_i32, WI_i32,
          CHO_i32, K_i32, STRIDE_i32, PADDING_i32,
          xBuf, inElemsI64,
          wBuf, wElemsI64,
          outLinear, cI64(outElemsSS)};
    } else {
      // Dynamic outLinear
      MemRefType dyn1f32 = MemRefType::get({ShapedType::kDynamic}, f32Ty);
      outLinear =
          rewriter.create<memref::AllocOp>(loc, dyn1f32, ValueRange{outElemsIdx})
              .getResult();

      // Direct call default ptr entry.
      calleeName = "conv_compute_currents_from_f32_ptr_defaultNP_f32out";

      auto extractPtrAsIndex = [&](Value mem) -> Value {
        return rewriter
            .create<memref::ExtractAlignedPointerAsIndexOp>(loc, mem)
            .getResult();
      };

      Type f32PtrT = emitc::PointerType::get(f32Ty);
      Value xPtr =
          rewriter.create<emitc::CastOp>(loc, f32PtrT, extractPtrAsIndex(xBuf))
              .getResult();
      Value wPtr =
          rewriter.create<emitc::CastOp>(loc, f32PtrT, extractPtrAsIndex(wBuf))
              .getResult();
      Value oPtr = rewriter
                       .create<emitc::CastOp>(loc, f32PtrT,
                                             extractPtrAsIndex(outLinear))
                       .getResult();

      callArgs = {
          T_i32, CHI_i32, HI_i32, WI_i32,
          CHO_i32, K_i32, STRIDE_i32, PADDING_i32,
          xPtr, inElemsI64,
          wPtr, wElemsI64,
          oPtr, outElemsI64};
    }

    // Call (wrapper or default ptr entry)
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{i32Ty}, callArgs);

    // Copy linear -> out4 and return tensor (unchanged)
    MemRefType out4Ty = MemRefType::get({T_s, CHO_s, HO_s, WO_s}, f32Ty);
    SmallVector<Value, 4> outDynSizes;
    if (!isStatic(T_s))   outDynSizes.push_back(T_v);
    if (!isStatic(CHO_s)) outDynSizes.push_back(CHO_v);
    if (!isStatic(HO_s))  outDynSizes.push_back(HO_v);
    if (!isStatic(WO_s))  outDynSizes.push_back(WO_v);

    Value out4 =
        rewriter.create<memref::AllocOp>(loc, out4Ty, outDynSizes).getResult();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();

    // lin = (((t*CHO + c)*HO + h)*WO + w)
    buildKrnlLoop(rewriter, loc, c0, T_v, [&](OpBuilder &b, Location l, Value t) {
      buildKrnlLoop(b, l, c0, CHO_v, [&](OpBuilder &b, Location l, Value c) {
        buildKrnlLoop(b, l, c0, HO_v, [&](OpBuilder &b, Location l, Value h) {
          buildKrnlLoop(b, l, c0, WO_v, [&](OpBuilder &b, Location l, Value w) {
            Value tCHO = b.create<arith::MulIOp>(l, t, CHO_v).getResult();
            Value tCHOc = b.create<arith::AddIOp>(l, tCHO, c).getResult();
            Value tCHOcHO = b.create<arith::MulIOp>(l, tCHOc, HO_v).getResult();
            Value tCHOcHOh = b.create<arith::AddIOp>(l, tCHOcHO, h).getResult();
            Value tCHOcHOhWO =
                b.create<arith::MulIOp>(l, tCHOcHOh, WO_v).getResult();
            Value lin = b.create<arith::AddIOp>(l, tCHOcHOhWO, w).getResult();

            Value v =
                b.create<memref::LoadOp>(l, outLinear, ValueRange{lin});
            b.create<memref::StoreOp>(l, v, out4, ValueRange{t, c, h, w});
          });
        });
      });
    });

    Value outTensor = memrefToTensor(rewriter, loc, out4, yType);
    rewriter.replaceOp(op, outTensor);
    return success();
  }
};


// //===----------------------------------------------------------------------===//
// // 1) Helpers: declare intrinsic bridge(s)
// //===----------------------------------------------------------------------===//

// static bool isStaticShape(ArrayRef<int64_t> shape) {
//   for (int64_t d : shape)
//     if (d == ShapedType::kDynamic)
//       return false;
//   return true;
// }

// static std::optional<int64_t>
// computeOutDimStatic(int64_t in, int64_t padding, int64_t k, int64_t stride) {
//   if (in == ShapedType::kDynamic)
//     return std::nullopt;
//   // (in + 2P - K) / S + 1
//   int64_t num = in + 2 * padding - k;
//   if (num < 0)
//     return std::nullopt;
//   if (stride <= 0)
//     return std::nullopt;
//   // Require exact divisibility to keep it "obviously" consistent.
//   if (num % stride != 0)
//     return std::nullopt;
//   return num / stride + 1;
// }

// static std::string buildStaticConvBridgeName(MemRefType xMrTy, MemRefType wMrTy,
//                                              int64_t outElems) {
//   auto xs = xMrTy.getShape(); // [T,CHI,HI,WI]
//   auto ws = wMrTy.getShape(); // [CHO,WCHI,KH,KW]
//   std::string name = "conv_compute_currents_from_f32_ptr_defaultNP_f32out";
//   name += "_T" + std::to_string(xs[0]);
//   name += "_CHI" + std::to_string(xs[1]);
//   name += "_HI" + std::to_string(xs[2]);
//   name += "_WI" + std::to_string(xs[3]);
//   name += "_CHO" + std::to_string(ws[0]);
//   name += "_WCHI" + std::to_string(ws[1]);
//   name += "_KH" + std::to_string(ws[2]);
//   name += "_KW" + std::to_string(ws[3]);
//   name += "_OUTE" + std::to_string(outElems);
//   return name;
// }

// static void declareConvIntrinsicBridgeDynamic(OpBuilder &builder, ModuleOp module,
//                                               Location loc) {
//   MLIRContext *ctx = builder.getContext();
//   Type i32T = builder.getI32Type();
//   Type i64T = builder.getI64Type();
//   Type f32T = builder.getF32Type();

//   auto dyn4f32 =
//       MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic,
//                        ShapedType::kDynamic, ShapedType::kDynamic},
//                       f32T);
//   auto dyn1f32 = MemRefType::get({ShapedType::kDynamic}, f32T);

//   auto fnTy = FunctionType::get(
//       ctx,
//       {/*dims*/ i32T, i32T, i32T, i32T,
//        /*params*/ i32T, i32T, i32T, i32T,
//        /*input*/ dyn4f32, i64T,
//        /*weight*/ dyn4f32, i64T,
//        /*out*/ dyn1f32, i64T},
//       {i32T});

//   getOrInsertFunc(builder, module, loc,
//                   "conv_compute_currents_from_f32_ptr_defaultNP_f32out", fnTy);
// }

// // Static bridge: input/weight/outLinear are all static memrefs.
// static StringRef declareConvIntrinsicBridgeStatic(OpBuilder &builder,
//                                                   ModuleOp module, Location loc,
//                                                   MemRefType xMrTy,
//                                                   MemRefType wMrTy,
//                                                   MemRefType outLinearTy,
//                                                   int64_t outElems) {
//   MLIRContext *ctx = builder.getContext();
//   Type i32T = builder.getI32Type();
//   Type i64T = builder.getI64Type();

//   std::string fnName = buildStaticConvBridgeName(xMrTy, wMrTy, outElems);

//   auto fnTy = FunctionType::get(
//       ctx,
//       {/*dims*/ i32T, i32T, i32T, i32T,
//        /*params*/ i32T, i32T, i32T, i32T,
//        /*input*/ xMrTy, i64T,
//        /*weight*/ wMrTy, i64T,
//        /*out*/ outLinearTy, i64T},
//       {i32T});

//   getOrInsertFunc(builder, module, loc, fnName, fnTy);
//   return builder.getStringAttr(fnName).getValue();
// }

// // Keep old entry point: declare dynamic bridge always (fallback).
// static void declareConvIntrinsicBridge(OpBuilder &builder, ModuleOp module,
//                                        Location loc) {
//   declareConvIntrinsicBridgeDynamic(builder, module, loc);
// }

// //===----------------------------------------------------------------------===//
// // 2) Helper: read i64 array attrs (pads/strides/dilations/kernel_shape)
// //===----------------------------------------------------------------------===//

// static LogicalResult getI64ArrayAttr(ONNXConvOp op, StringRef name,
//                                     SmallVectorImpl<int64_t> &out) {
//   auto arr = op->getAttrOfType<ArrayAttr>(name);
//   if (!arr)
//     return failure();
//   out.clear();
//   out.reserve(arr.size());
//   for (Attribute a : arr) {
//     auto i = dyn_cast<IntegerAttr>(a);
//     if (!i)
//       return failure();
//     out.push_back(i.getInt());
//   }
//   return success();
// }

// //===----------------------------------------------------------------------===//
// // 3) Pattern: ONNXConvOp -> func.call intrinsic + linear buffer + copy-back
// //===----------------------------------------------------------------------===//

// struct ONNXConvOpToKrnlIntrinsicCall final : public OpRewritePattern<ONNXConvOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(ONNXConvOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     ModuleOp module = op->getParentOfType<ModuleOp>();
//     declareConvIntrinsicBridge(rewriter, module, loc);

//     // Operands
//     Value X = op.getX();
//     Value W = op.getW();
//     Value B = op.getB();

//     // Types
//     auto xType = dyn_cast<RankedTensorType>(X.getType());
//     auto wType = dyn_cast<RankedTensorType>(W.getType());
//     auto yType = dyn_cast<RankedTensorType>(op.getResult().getType());
//     if (!xType || !wType || !yType)
//       return op.emitError("X/W/Y must be RankedTensorType"), failure();

//     if (xType.getRank() != 4 || wType.getRank() != 4 || yType.getRank() != 4)
//       return op.emitError("Only rank-4 conv supported (NCHW)"), failure();

//     if (!xType.getElementType().isF32() || !wType.getElementType().isF32() ||
//         !yType.getElementType().isF32())
//       return op.emitError("Only f32 X/W/Y supported"), failure();

//     // Bias must be none
//     if (B && !isa<NoneType>(B.getType()))
//       return op.emitError("Bias is not supported by this lowering"), failure();

//     // Attributes (force group=1)
//     SmallVector<int64_t, 4> pads, strides, dilations, kernelShape;
//     if (failed(getI64ArrayAttr(op, "pads", pads)))
//       return op.emitError("Missing/invalid pads"), failure();
//     if (failed(getI64ArrayAttr(op, "strides", strides)))
//       return op.emitError("Missing/invalid strides"), failure();

//     if (failed(getI64ArrayAttr(op, "dilations", dilations)))
//       dilations.assign({1, 1});

//     if (failed(getI64ArrayAttr(op, "kernel_shape", kernelShape))) {
//       int64_t KH = wType.getDimSize(2);
//       int64_t KW = wType.getDimSize(3);
//       if (KH == ShapedType::kDynamic || KW == ShapedType::kDynamic)
//         return op.emitError("kernel_shape missing and cannot infer"), failure();
//       kernelShape.assign({KH, KW});
//     }

//     // Constraints
//     if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1)
//       return op.emitError("Only dilations=[1,1] supported"), failure();
//     if (strides.size() != 2)
//       return op.emitError("strides must have 2 elements"), failure();
//     if (strides[0] != strides[1])
//       return op.emitError("Only stride_h == stride_w supported"), failure();
//     if (pads.size() != 4)
//       return op.emitError("pads must have 4 elements"), failure();
//     if (!(pads[0] == pads[1] && pads[0] == pads[2] && pads[0] == pads[3]))
//       return op.emitError("Only symmetric pads=[p,p,p,p] supported"), failure();
//     if (kernelShape.size() != 2 || kernelShape[0] != kernelShape[1])
//       return op.emitError("Only square kernel supported"), failure();

//     int64_t K = kernelShape[0];
//     int64_t STRIDE = strides[0];
//     int64_t PADDING = pads[0];

//     auto isStatic = [](int64_t d) { return d != ShapedType::kDynamic; };

//     // Static dims (may be dynamic)
//     int64_t T_s   = xType.getDimSize(0);
//     int64_t CHI_s = xType.getDimSize(1);
//     int64_t HI_s  = xType.getDimSize(2);
//     int64_t WI_s  = xType.getDimSize(3);

//     int64_t CHO_s  = wType.getDimSize(0);
//     int64_t WCHI_s = wType.getDimSize(1);
//     int64_t KH_s   = wType.getDimSize(2);
//     int64_t KW_s   = wType.getDimSize(3);

//     int64_t yT_s = yType.getDimSize(0);
//     int64_t yC_s = yType.getDimSize(1);
//     int64_t HO_s = yType.getDimSize(2);
//     int64_t WO_s = yType.getDimSize(3);

//     // consistency checks
//     if (isStatic(WCHI_s) && isStatic(CHI_s) && WCHI_s != CHI_s)
//       return op.emitError("weight CHI mismatch"), failure();
//     if (isStatic(KH_s) && KH_s != K)
//       return op.emitError("weight KH mismatch"), failure();
//     if (isStatic(KW_s) && KW_s != K)
//       return op.emitError("weight KW mismatch"), failure();
//     if (isStatic(yT_s) && isStatic(T_s) && yT_s != T_s)
//       return op.emitError("Y batch mismatch"), failure();
//     if (isStatic(yC_s) && isStatic(CHO_s) && yC_s != CHO_s)
//       return op.emitError("Y channel mismatch"), failure();

//     // tensor -> memref
//     Type f32Ty = rewriter.getF32Type();
//     Type i32Ty = rewriter.getI32Type();
//     Type i64Ty = rewriter.getI64Type();

//     MemRefType xMrTy = MemRefType::get({T_s, CHI_s, HI_s, WI_s}, f32Ty);
//     MemRefType wMrTy = MemRefType::get({CHO_s, WCHI_s, KH_s, KW_s}, f32Ty);

//     Value xBuf = tensorToMemref(rewriter, loc, X, xMrTy);
//     Value wBuf = tensorToMemref(rewriter, loc, W, wMrTy);

//     // Helpers
//     auto cIndex = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
//     };
//     auto dim = [&](Value mem, int64_t axis) -> Value {
//       return rewriter.create<memref::DimOp>(loc, mem, axis).getResult();
//     };
//     auto cI32 = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantIntOp>(loc, v, i32Ty).getResult();
//     };
//     auto idxToI32 = [&](Value idx) -> Value {
//       return rewriter.create<arith::IndexCastOp>(loc, i32Ty, idx).getResult();
//     };
//     auto idxToI64 = [&](Value idx) -> Value {
//       return rewriter.create<arith::IndexCastOp>(loc, i64Ty, idx).getResult();
//     };
//     auto cI64 = [&](int64_t v) -> Value {
//       return rewriter.create<arith::ConstantIntOp>(loc, v, i64Ty).getResult();
//     };

//     // runtime dims (index)
//     Value T_v   = isStatic(T_s)   ? cIndex(T_s)   : dim(xBuf, 0);
//     Value CHI_v = isStatic(CHI_s) ? cIndex(CHI_s) : dim(xBuf, 1);
//     Value HI_v  = isStatic(HI_s)  ? cIndex(HI_s)  : dim(xBuf, 2);
//     Value WI_v  = isStatic(WI_s)  ? cIndex(WI_s)  : dim(xBuf, 3);
//     Value CHO_v = isStatic(CHO_s) ? cIndex(CHO_s) : dim(wBuf, 0);

//     // Static HO/WO if possible (for fully static outLinear length)
//     std::optional<int64_t> HO_static = computeOutDimStatic(HI_s, PADDING, K, STRIDE);
//     std::optional<int64_t> WO_static = computeOutDimStatic(WI_s, PADDING, K, STRIDE);

//     // HO/WO values (index) used for copy-back loops
//     Value HO_v = isStatic(HO_s) ? cIndex(HO_s) : [&]() -> Value {
//       Value twoP = cIndex(2 * PADDING);
//       Value Kc   = cIndex(K);
//       Value Str  = cIndex(STRIDE);
//       Value num = rewriter.create<arith::SubIOp>(
//           loc, rewriter.create<arith::AddIOp>(loc, HI_v, twoP), Kc).getResult();
//       Value div = rewriter.create<arith::DivSIOp>(loc, num, Str).getResult();
//       return rewriter.create<arith::AddIOp>(loc, div, cIndex(1)).getResult();
//     }();

//     Value WO_v = isStatic(WO_s) ? cIndex(WO_s) : [&]() -> Value {
//       Value twoP = cIndex(2 * PADDING);
//       Value Kc   = cIndex(K);
//       Value Str  = cIndex(STRIDE);
//       Value num = rewriter.create<arith::SubIOp>(
//           loc, rewriter.create<arith::AddIOp>(loc, WI_v, twoP), Kc).getResult();
//       Value div = rewriter.create<arith::DivSIOp>(loc, num, Str).getResult();
//       return rewriter.create<arith::AddIOp>(loc, div, cIndex(1)).getResult();
//     }();

//     // elems (index) for dynamic path and general use
//     Value inElemsIdx =
//         rewriter.create<arith::MulIOp>(
//             loc,
//             rewriter.create<arith::MulIOp>(
//                 loc,
//                 rewriter.create<arith::MulIOp>(loc, T_v, CHI_v).getResult(),
//                 HI_v)
//                 .getResult(),
//             WI_v)
//             .getResult();

//     Value wElemsIdx =
//         rewriter.create<arith::MulIOp>(
//             loc,
//             rewriter.create<arith::MulIOp>(
//                 loc,
//                 rewriter.create<arith::MulIOp>(loc, CHO_v, CHI_v).getResult(),
//                 cIndex(K))
//                 .getResult(),
//             cIndex(K))
//             .getResult();

//     Value outElemsIdx =
//         rewriter.create<arith::MulIOp>(
//             loc,
//             rewriter.create<arith::MulIOp>(
//                 loc,
//                 rewriter.create<arith::MulIOp>(loc, T_v, CHO_v).getResult(),
//                 HO_v)
//                 .getResult(),
//             WO_v)
//             .getResult();

//     Value inElemsI64  = idxToI64(inElemsIdx);
//     Value wElemsI64   = idxToI64(wElemsIdx);
//     Value outElemsI64 = idxToI64(outElemsIdx);

//     // Decide static-vs-dynamic bridge.
//     bool xStatic4 = isStaticShape(xMrTy.getShape());
//     bool wStatic4 = isStaticShape(wMrTy.getShape());

//     // For fully-static outLinear, we also need HO/WO static and outElems fits in int64.
//     bool canStaticOut =
//         xStatic4 && wStatic4 && HO_static.has_value() && WO_static.has_value();

//     StringRef calleeName;
//     Value xArg;
//     Value wArg;
//     Value outLinear;
//     Value outElemsArgI64;

//     if (canStaticOut) {
//       int64_t HO_ss = *HO_static;
//       int64_t WO_ss = *WO_static;
//       int64_t outElemsSS = T_s * CHO_s * HO_ss * WO_ss;

//       // Static outLinear type and alloc.
//       MemRefType outLinearTy = MemRefType::get({outElemsSS}, f32Ty);
//       outLinear = rewriter.create<memref::AllocOp>(loc, outLinearTy).getResult();

//       // Declare and call the per-shape static bridge.
//       calleeName = declareConvIntrinsicBridgeStatic(
//           rewriter, module, loc, xMrTy, wMrTy, outLinearTy, outElemsSS);

//       xArg = xBuf;
//       wArg = wBuf;
//       outElemsArgI64 = cI64(outElemsSS);
//     } else {
//       // Fallback dynamic bridge.
//       calleeName = "conv_compute_currents_from_f32_ptr_defaultNP_f32out";

//       // Cast to dyn4 for bridge signature (same rank: OK)
//       MemRefType dyn4f32 =
//           MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic,
//                            ShapedType::kDynamic, ShapedType::kDynamic},
//                           f32Ty);
//       xArg = rewriter.create<memref::CastOp>(loc, dyn4f32, xBuf).getResult();
//       wArg = rewriter.create<memref::CastOp>(loc, dyn4f32, wBuf).getResult();

//       // Dynamic outLinear alloc.
//       MemRefType dyn1f32 = MemRefType::get({ShapedType::kDynamic}, f32Ty);
//       outLinear = rewriter.create<memref::AllocOp>(loc, dyn1f32,
//                                                   ValueRange{outElemsIdx})
//                       .getResult();
//       outElemsArgI64 = outElemsI64;
//     }

//     // i32 args
//     Value T_i32   = idxToI32(T_v);
//     Value CHI_i32 = idxToI32(CHI_v);
//     Value HI_i32  = idxToI32(HI_v);
//     Value WI_i32  = idxToI32(WI_v);
//     Value CHO_i32 = idxToI32(CHO_v);

//     Value K_i32       = cI32(K);
//     Value STRIDE_i32  = cI32(STRIDE);
//     Value PADDING_i32 = cI32(PADDING);

//     // Call intrinsic bridge
//     rewriter.create<func::CallOp>(
//         loc, calleeName, TypeRange{i32Ty},
//         ValueRange{
//             T_i32, CHI_i32, HI_i32, WI_i32,
//             CHO_i32, K_i32, STRIDE_i32, PADDING_i32,
//             xArg, inElemsI64,
//             wArg, wElemsI64,
//             outLinear, outElemsArgI64});

//     // Copy linear -> out4 and return tensor (unchanged)
//     MemRefType out4Ty = MemRefType::get({T_s, CHO_s, HO_s, WO_s}, f32Ty);
//     SmallVector<Value, 4> outDynSizes;
//     if (!isStatic(T_s))   outDynSizes.push_back(T_v);
//     if (!isStatic(CHO_s)) outDynSizes.push_back(CHO_v);
//     if (!isStatic(HO_s))  outDynSizes.push_back(HO_v);
//     if (!isStatic(WO_s))  outDynSizes.push_back(WO_v);

//     Value out4 = rewriter.create<memref::AllocOp>(loc, out4Ty, outDynSizes).getResult();

//     Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();

//     // lin = (((t*CHO + c)*HO + h)*WO + w)
//     buildKrnlLoop(rewriter, loc, c0, T_v, [&](OpBuilder &b, Location l, Value t) {
//       buildKrnlLoop(b, l, c0, CHO_v, [&](OpBuilder &b, Location l, Value c) {
//         buildKrnlLoop(b, l, c0, HO_v, [&](OpBuilder &b, Location l, Value h) {
//           buildKrnlLoop(b, l, c0, WO_v, [&](OpBuilder &b, Location l, Value w) {
//             Value tCHO = b.create<arith::MulIOp>(l, t, CHO_v).getResult();
//             Value tCHOc = b.create<arith::AddIOp>(l, tCHO, c).getResult();
//             Value tCHOcHO = b.create<arith::MulIOp>(l, tCHOc, HO_v).getResult();
//             Value tCHOcHOh = b.create<arith::AddIOp>(l, tCHOcHO, h).getResult();
//             Value tCHOcHOhWO = b.create<arith::MulIOp>(l, tCHOcHOh, WO_v).getResult();
//             Value lin = b.create<arith::AddIOp>(l, tCHOcHOhWO, w).getResult();

//             Value v = b.create<memref::LoadOp>(l, outLinear, ValueRange{lin});
//             b.create<memref::StoreOp>(l, v, out4, ValueRange{t, c, h, w});
//           });
//         });
//       });
//     });

//     Value outTensor = memrefToTensor(rewriter, loc, out4, yType);
//     rewriter.replaceOp(op, outTensor);
//     return success();
//   }
// };





// ═══════════════════════════════════════════════════════════════════════
// Pass 定义
// ═══════════════════════════════════════════════════════════════════════

class ConvertONNXSNNToKrnlPass : public PassWrapper<ConvertONNXSNNToKrnlPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "convert-onnx-lif-to-krnl"; }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // 同时注册单步和多步 Pattern
    patterns.add<ONNXLIFOpToKrnlConversion>(ctx);
    patterns.add<ONNXMultiStepLIFToKrnlConversion>(ctx);
    patterns.add<ONNXDataToVectorOpToKrnlConversion>(ctx);
    patterns.add<ONNXSNNFCOpToKrnlConversion>(ctx);
    patterns.add<ONNXConvOpToKrnlIntrinsicCall>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createConvertONNXSNNToKrnlPass() { return std::make_unique<ConvertONNXSNNToKrnlPass>(); }
}
static mlir::PassRegistration<ConvertONNXSNNToKrnlPass> pass;