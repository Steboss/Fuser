// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <string.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <exception>
#include <utility>

namespace nvfuser {

// The name ping-pong comes from Warp-Specialized Persistent Ping-Pong kernels
// for GEMM in CUTLASS. Here it is generalized to represent any warp specialized
// kernels with multiple computation warp groups, includes GEMM and non-GEMM
// kernels.
//
// For non-GEMM kernel there is no ping-pong switching between tensor cores and
// cuda cores, but multiple warp groups work on different tiles and are
// scheduled and synchronized separately on the same SM.
using PingPongCircularBufferingParams = std::tuple<bool, int>;
class PingPongCircularBuffering
    : public NVFuserFixtureParamTest<PingPongCircularBufferingParams> {
 public:
  void SetUp() override {
    opt_guard_ = std::make_unique<EnableOptionsGuard>();
    if (std::get<1>(GetParam())) {
      EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
    } else {
      EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
    }
    NVFuserTest::SetUp();
  }

 protected:
  // This keeps the guard alive until all tests are done.
  std::unique_ptr<EnableOptionsGuard> opt_guard_;
};

TEST_P(PingPongCircularBuffering, StageSlicePositionComputeAt) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t rows_per_stage = 4;
  constexpr int64_t compute_warp_groups = 2;
  constexpr int64_t circular_loop = 12;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  constexpr int64_t dim1 = 128;
  constexpr int64_t stages = 6;

  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion->addInput(tv0);

  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  fusion->addOutput(tv3);

  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  tv1->setMemoryType(MemoryType::Shared);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, rows_per_stage);
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);

    // [I, 2, 132, 4]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->axis(3)->parallelize(ParallelType::Unroll);
  }

  tv1->axis(1)->parallelize(ParallelType::Serial);
  tv1->axis(4)->parallelize(ParallelType::Bulk);

  for (auto tv : {tv2, tv3}) {
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(4)->parallelize(ParallelType::TIDx);
  }

  // [I, 2, 132, 4] --> [132, I, 2, 4]
  std::unordered_map<int64_t, int64_t> reorder_map = {{0, 1}, {1, 2}, {2, 0}};
  for (auto tv : {tv1, tv2, tv3}) {
    tv->reorder(reorder_map);
  }

  inlineSelectedAt({tv1}, tv2, /*reference_pos=*/2);
  inlineSelectedAt({tv2}, tv2, /*reference_pos=*/3);

  auto [_, stage_slice_position] = GetParam();
  tv1->circularBuffer(
      stages,
      stages - 1,
      WarpSpecialized(ParallelType::TIDy, stage_slice_position));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = t0 + t0;

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (stage_slice_position == -1) {
      const char* error_msg = R"(Slice position must be non-negative integer)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position < 2) {
      const char* error_msg =
          R"(Expected outer_most_circular_buffer_position <= inner_most_circular_buffer_position)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position == 5) {
      const char* error_msg =
          R"(Detected an iterDomain with ParallelType::Bulk to the left of stage slice position.)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    } else if (stage_slice_position == 6) {
      const char* error_msg =
          R"(Slice position must be inside TensorView nDims.)";
      const char* str_match_pointer = strstr(e.what(), error_msg);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    }

    throw;
  }

  auto cg_outputs = ke.run({t0});
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    Single,
    PingPongCircularBuffering,
    ::testing::Combine(testing::Values(false), ::testing::Range(-1, 7)),
    [](const testing::TestParamInfo<PingPongCircularBufferingParams>& info) {
      std::stringstream ss;
      ss << "IdModel_" << std::get<0>(info.param);
      ss << "_stage_slice_position_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

TEST_P(PingPongCircularBuffering, TwoTmaLoads) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t rows_per_stage = 4;
  constexpr int64_t compute_warp_groups = 2;
  constexpr int64_t circular_loop = 12;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t dim0 =
      rows_per_stage * compute_warp_groups * sm_count * circular_loop;
  constexpr int64_t dim1 = 128;
  constexpr int64_t stages = 6;

  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  TensorView* tv1 = makeContigConcreteTensor({dim0, dim1}, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  TensorView* tv6 = add(tv0, tv1);
  fusion->addOutput(tv6);

  // Create Cache Tensors
  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  TensorView* tv3 = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  TensorView* tv4 = tv2->cacheAfter();
  TensorView* tv5 = tv2->cacheAfter();
  TensorView* tv7 = tv6->cacheBefore();

  // Move first level cache to shared memory
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  for (TensorView* tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->split(0, rows_per_stage);
    tv->split(0, sm_count);
    tv->split(0, compute_warp_groups);
    // [I, 2, 132, 4]
    tv->axis(0)->parallelize(ParallelType::Serial);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::BIDy);
    tv->axis(3)->parallelize(ParallelType::Unroll);
    tv->axis(4)->parallelize(ParallelType::TIDx);
    if (tv == tv2 || tv == tv3) {
      tv->axis(1)->parallelize(ParallelType::Serial);
      tv->axis(4)->parallelize(ParallelType::Bulk);
    }
  }

  // Reorder Axes
  // [I, 2, 132, 4] --> [132, I, 2, 4]
  std::unordered_map<int64_t, int64_t> reorder_map;
  reorder_map[0] = 1;
  reorder_map[1] = 2;
  reorder_map[2] = 0;
  for (TensorView* tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->reorder(reorder_map);
  }

  // Inline Step
  for (TensorView* tv : {tv2, tv3}) {
    inlineSelectedAt({tv}, tv, 2);
  }

  for (TensorView* tv : {tv4, tv5, tv6, tv7}) {
    inlineSelectedAt({tv}, tv, 3);
  }

  auto [_, stage_slice_position] = GetParam();
  for (auto tv : {tv2, tv3}) {
    tv->circularBuffer(
        stages,
        stages - 1,
        WarpSpecialized(ParallelType::TIDy, stage_slice_position));
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = at::randn({dim0, dim1}, options);
  at::Tensor t2 = t0 + t1;

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});
  auto cg_outputs = ke.run({t0, t1});
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    Sibling,
    PingPongCircularBuffering,
    ::testing::Combine(testing::Bool(), testing::Range(2, 4)),
    [](const testing::TestParamInfo<PingPongCircularBufferingParams>& info) {
      std::stringstream ss;
      ss << "IdModel_" << std::get<0>(info.param);
      ss << "_stage_slice_position_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser
