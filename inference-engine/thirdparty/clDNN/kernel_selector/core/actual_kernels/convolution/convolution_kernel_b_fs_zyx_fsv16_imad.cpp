// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "convolution_kernel_b_fs_zyx_fsv16_imad.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

//
// Kernel specific constants
//
static constexpr size_t fsv = 16;
static constexpr size_t simd = 16;

static size_t getOutBlock_X(const size_t output_size_x, const size_t stride_x, const size_t filter_size_x, const size_t dilation_x) {
    // Calculate number of variables needed to hold minimum input width.
    // Equation for input block width: (output_block - 1) * stride + (filter_size - 1) * dilation + 1
    // Result for one output_block gives minimum size of input width.
    size_t min_in_block_size = (filter_size_x - 1) * dilation_x + 1;
    // Input block is spread across sub-group, so ceil-divide by simd size.
    size_t min_in_block_simds = kernel_selector::CeilDiv(min_in_block_size, simd);

    size_t output_block_width = 0;
    size_t max_block_size = std::min((min_in_block_simds * simd - 1 - (filter_size_x - 1) * dilation_x) / stride_x + 1, output_size_x);

    if (output_size_x <= max_block_size)
        return output_size_x;

    for (size_t block = 4; block <= max_block_size; ++block) {
        if (output_size_x % block == 0)
            output_block_width = block;
    }
    if (output_block_width == 0 && output_size_x < max_block_size * 3) {
        size_t min_overhang = max_block_size;
        for (size_t block = 4; block <= max_block_size; ++block) {
            size_t overhang = block - output_size_x % block;
            if (overhang <= min_overhang) {
                min_overhang = overhang;
                output_block_width = block;
            }
        }
    }

    if (output_block_width == 0) {
        output_block_width = max_block_size;
    }
    return output_block_width;
}

namespace kernel_selector {

Convolution_kernel_b_fs_zyx_fsv16_imad::BlockParams
Convolution_kernel_b_fs_zyx_fsv16_imad::GetBlockParams(const convolution_params& params) const {

    size_t max_block_width = getOutBlock_X(params.output.X().v, params.stride.x, params.filterSize.x, params.dilation.x);
    size_t max_in_block_width = (max_block_width - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;

    size_t block_width = max_block_width;
    if (max_block_width > 1) {
        for (size_t w = max_block_width; w >= CeilDiv(max_block_width, 2); w -= 1) {
            if (params.output.X().v % w == 0) {
                block_width = w;
                break;
            }
        }
    }

    size_t in_block_width = (block_width - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;
    size_t block_features = simd;
    size_t feature_slm_split = 1;
    size_t block_height = 1;
    size_t block_depth = 1;
    size_t in_block_height = 1;
    size_t in_block_depth = 1;

    auto test_block_params = BlockParams{ block_width, 1, 1, simd, in_block_width, 1, 1, 1 };
    auto best_block_params_ratio = EstimateBlockParamsRatio(params, test_block_params);

    for (size_t w = 0; w < 2; w++) {
        size_t temp_block_width = block_width;
        size_t temp_in_block_width = in_block_width;

        if (w == 1) {
            if (max_block_width > 1) {
                temp_block_width = max_block_width;
                temp_in_block_width = max_in_block_width;
            } else {
                break;
            }
        }
        for (size_t split = 1; split <= params.engineInfo.maxWorkGroupSize / simd; ++split) {
            for (size_t temp_block_features = simd; temp_block_features <= simd * 2; temp_block_features += simd) {
                for (size_t d = 1; d < 16; ++d) {
                    if (params.output.Z().v % d != 0)
                        continue;
                    for (size_t h = 1; h < 16; ++h) {
                        if (params.output.Y().v % h != 0)
                            continue;

                        bool c_ifm_mul = CeilDiv(params.weights.IFM().v, fsv) % split == 0;
                        bool c_mul_f = params.weights.OFM().v % temp_block_features == 0;

                        size_t temp_block_height = 1;
                        size_t temp_block_depth = 1;
                        size_t temp_in_block_height = 1;
                        size_t temp_in_block_depth = 1;

                        if (h != 1) {
                            temp_block_height = h;
                            temp_block_depth = d;
                            temp_in_block_height = (h - 1) * params.stride.y + (params.filterSize.y - 1) * params.dilation.y + 1;
                            temp_in_block_depth = (d - 1) * params.stride.z + (params.filterSize.z - 1) * params.dilation.z + 1;
                        }

                        test_block_params = BlockParams{ temp_block_width, temp_block_height, temp_block_depth, temp_block_features,
                                                         temp_in_block_width, temp_in_block_height, temp_in_block_depth, split };
                        auto block_params_ratio = EstimateBlockParamsRatio(params, test_block_params);

                        // Try to increase block_params_ratio
                        if (c_ifm_mul && c_mul_f && block_params_ratio > best_block_params_ratio) {
                            best_block_params_ratio = block_params_ratio;

                            // Update block params if current ratio is better than previous
                            block_width = temp_block_width;
                            block_height = temp_block_height;
                            block_depth = temp_block_depth;
                            block_features = temp_block_features;

                            in_block_width = temp_in_block_width;
                            in_block_height = temp_in_block_height;
                            in_block_depth = temp_in_block_depth;
                            feature_slm_split = split;
                        }
                    }
                }
            }
            if (split * fsv >= params.weights.IFM().v)
                break;
        }
    }

// #define DEBUG_BLOCK_PARAMS_RATIO

#ifdef DEBUG_BLOCK_PARAMS_RATIO
    float occupancy = EstimateOccupancy(params, BlockParams{ block_width, block_height, block_depth, block_features,
                                                             in_block_width, in_block_height, in_block_depth, feature_slm_split });
    float slm_usage = EstimateSLMUsage(params, BlockParams{ block_width, block_height, block_depth, block_features,
                                                            in_block_width, in_block_height, in_block_depth, feature_slm_split });
    float reg_pressure = EstimateRegPressure(params, BlockParams{ block_width, block_height, block_depth, block_features,
                                                                  in_block_width, in_block_height, in_block_depth, feature_slm_split });

    printf("output ratio factors: occupancy = %f slm_usage = %f reg_pressure = %f\n", occupancy, slm_usage, reg_pressure);
    printf("logical size: %d\n", (int)params.output.LogicalSize());
    printf("XYZFB: %d %d %d %d %d\n", (int)params.output.X().v, (int)params.output.Y().v, (int)params.output.Z().v,
           (int)params.output.Feature().v, (int)params.output.Batch().v);
    printf("output ratio: %f\n", best_block_params_ratio);
    printf("output params: %d %d %d %d %d %d %d %d\n\n\n", (int)block_width, (int)block_height, (int)block_depth, (int)block_features,
           (int)in_block_width, (int)in_block_height, (int)in_block_depth, (int)feature_slm_split);
#endif // DEBUG_BLOCK_PARAMS_RATIO
    /*const auto& output = params.output;

    // densenet-161
    if (output.X().v == 56 && output.Y().v == 56 && output.Z().v == 1 && output.Feature().v == 192)
        return BlockParams{ 14, 1, 1, 32, 14, 1, 1, 1 };
    else*/
    return BlockParams{ block_width, block_height, block_depth, block_features, in_block_width, in_block_height, in_block_depth, feature_slm_split };
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateBlockParamsRatio(const convolution_params& params, const BlockParams& block) const {
    constexpr size_t max_threads_per_compute_unit = 7;
    size_t max_compute_units_per_device = params.engineInfo.computeUnitsCount;
    size_t max_threads_per_device = max_compute_units_per_device * max_threads_per_compute_unit;
    bool increase_max_reg_pressure = static_cast<float>(params.output.LogicalSize() / max_threads_per_device) >= 595.f;
    bool double_increase_max_reg_pressure = static_cast<float>(params.output.LogicalSize() / max_threads_per_device) >= 595.f * 2.f;

    float max_reg_pressure = double_increase_max_reg_pressure ? 0.785f : increase_max_reg_pressure ? 0.75f : 0.7f;

    constexpr float max_slm_usage = 1.f;
    constexpr float max_occupancy = 2.f;
    float reduce_occupancy = 0.f;

    float occupancy = EstimateOccupancy(params, block);
    float slm_usage = EstimateSLMUsage(params, block);
    float reg_pressure = EstimateRegPressure(params, block);

    if (occupancy > max_occupancy) { reduce_occupancy = log10f(occupancy - max_occupancy); occupancy = max_occupancy; }

    // Estimate current block_params_ratio
    float block_params_ratio = logf(occupancy) + slm_usage + reg_pressure - reduce_occupancy;

    // Check all restrictions
    bool bad_block_params = reg_pressure > max_reg_pressure || slm_usage > max_slm_usage;

    return bad_block_params ? -10.f : block_params_ratio;
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateRegPressure(const convolution_params& params, const BlockParams& block) const {
    size_t bytes_used = 0;

    // Accumulator
    size_t accumulator_elements = block.output_block_width * block.output_block_height * block.output_block_depth * block.output_block_features;
    bytes_used += accumulator_elements * BytesPerElement(GetAccumulatorType(params));

    // Input block
    size_t input_block_elements = block.input_block_depth * block.input_block_height * Align(block.input_block_width, simd) * fsv;
    bytes_used += input_block_elements * BytesPerElement(params.inputs[0].GetDType());

    // Weights block
    size_t weights_block_elements = block.output_block_features * fsv;
    bytes_used += weights_block_elements * BytesPerElement(params.weights.GetDType());

    // Experimentally selected number of registers needed for extra variables (eg. out_x, out_y, out_z, filter_idx, etc.)
    constexpr size_t experimental_extra_regs = 8 * 32;
    bytes_used += experimental_extra_regs;

    // Experimentally selected number of registers needed for slm handling
    constexpr size_t experimental_slm_regs = 4 * 32;
    if (block.feature_slm_split != 1) {
        bytes_used += experimental_slm_regs;
    }

    constexpr size_t reg_num = 128;
    constexpr size_t bytes_per_reg = 32;
    constexpr size_t max_reg_bytes = reg_num * bytes_per_reg;

    return static_cast<float>(bytes_used) / static_cast<float>(max_reg_bytes);
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateOccupancy(const convolution_params& params, const BlockParams& block) const {
    size_t blocks_w = CeilDiv(params.output.X().v, block.output_block_width);
    size_t blocks_h = CeilDiv(params.output.Y().v, block.output_block_height);
    size_t blocks_d = CeilDiv(params.output.Z().v, block.output_block_depth);
    size_t blocks_f = CeilDiv(params.weights.OFM().v, block.output_block_features) * params.groups * block.feature_slm_split;
    size_t block_b = params.output.Batch().v;

    auto threads = blocks_w * blocks_h * blocks_d * blocks_f * block_b;
    constexpr size_t max_threads_per_cu = 7;
    size_t compute_units = params.engineInfo.computeUnitsCount;
    size_t max_threads = compute_units * max_threads_per_cu;

    return static_cast<float>(threads) / static_cast<float>(max_threads);
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateSLMUsage(const convolution_params& params, const BlockParams& block) const {
    if (block.feature_slm_split == 1)
        return 0.f;

    size_t slm_elements_per_work_group = block.output_block_width * block.output_block_height * block.output_block_depth *
                                         block.output_block_features * (block.feature_slm_split - 1);
    size_t slm_bytes_per_work_group = slm_elements_per_work_group * BytesPerElement(GetAccumulatorType(params));

    size_t max_slm_bytes_per_sub_slice = params.engineInfo.maxLocalMemSize;
    if (slm_bytes_per_work_group > max_slm_bytes_per_sub_slice)
        return 0.f;

    const auto& output = params.output;
    size_t work_groups_number = CeilDiv(output.X().v, block.output_block_width) *
                                CeilDiv(output.Y().v, block.output_block_height) *
                                CeilDiv(output.Z().v, block.output_block_depth) *
                                output.Batch().v *
                                CeilDiv(params.weights.OFM().v, block.output_block_features) *
                                params.groups;

    constexpr size_t max_threads_per_compute_unit = 7;
    constexpr size_t max_compute_units_per_sub_slice = 8;
    constexpr size_t max_work_groups_per_sub_slice = 16;
    size_t max_sub_slices_per_device = params.engineInfo.computeUnitsCount / max_compute_units_per_sub_slice;
    size_t max_work_groups_per_device = max_sub_slices_per_device * max_work_groups_per_sub_slice;
    if (work_groups_number > max_work_groups_per_device * 100)
        return 0.f;

    size_t threads_per_work_group = block.feature_slm_split;
    size_t threads_per_sub_slice = max_threads_per_compute_unit * max_compute_units_per_sub_slice;
    size_t current_max_work_groups_per_sub_slice = threads_per_sub_slice / threads_per_work_group;
    while (current_max_work_groups_per_sub_slice * slm_bytes_per_work_group > max_slm_bytes_per_sub_slice)
        current_max_work_groups_per_sub_slice--;

    if (current_max_work_groups_per_sub_slice == 1)
        return 1.0;

    float max_slm_bytes_per_work_group = static_cast<float>(max_slm_bytes_per_sub_slice) / static_cast<float>(current_max_work_groups_per_sub_slice);
    max_slm_bytes_per_work_group = static_cast<float>(Align(static_cast<size_t>(max_slm_bytes_per_work_group), 1024));
    if (max_slm_bytes_per_work_group * static_cast<float>(current_max_work_groups_per_sub_slice) > static_cast<float>(max_slm_bytes_per_sub_slice))
        max_slm_bytes_per_work_group -= 1024.0;

    return static_cast<float>(slm_bytes_per_work_group) / static_cast<float>(max_slm_bytes_per_work_group);
}

ParamsKey Convolution_kernel_b_fs_zyx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDilation();
    k.DisableTuning();
    return k;
}

KernelsData Convolution_kernel_b_fs_zyx_fsv16_imad::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants Convolution_kernel_b_fs_zyx_fsv16_imad::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    auto block_params = GetBlockParams(params);

    bool unroll_filter_y = block_params.output_block_height != 1;
    bool unroll_filter_z = block_params.output_block_depth != 1;

    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block_params.output_block_width));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", block_params.input_block_width));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block_params.output_block_height));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_HEIGHT", block_params.input_block_height));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_DEPTH", block_params.output_block_depth));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_DEPTH", block_params.input_block_depth));
    mem_consts.AddConstant(MakeJitConstant("FILTER_SIZE_Y_UNROLL", unroll_filter_y ? params.filterSize.y : 1));
    mem_consts.AddConstant(MakeJitConstant("FILTER_SIZE_Z_UNROLL", unroll_filter_z ? params.filterSize.z : 1));
    mem_consts.AddConstant(MakeJitConstant("OFM_BLOCKS_PER_SIMD", static_cast<int>(std::ceil(block_params.output_block_features / simd))));
    mem_consts.AddConstant(MakeJitConstant("OFM_SIZE_PER_SIMD", block_params.output_block_features));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_SLM_SPLIT", block_params.feature_slm_split));
    mem_consts.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    mem_consts.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = { "out_b", "(out_f + ofb * 16)", "(out_y + oh)", "(out_x + ow)" };
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = { "out_b", "(out_f + ofb * 16)", "(out_z + od)", "(out_y + oh)", "(out_x + ow)" };
        }

        std::vector<Tensor::DataChannelName> loop_axes = { Tensor::DataChannelName::X };

        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            if (block_params.output_block_depth != 1) {
                loop_axes.push_back(Tensor::DataChannelName::Z);
            } else {
                idx_order[idx_order.size() - 3] = "out_z";
            }
        }

        if (block_params.output_block_height != 1) {
            loop_axes.push_back(Tensor::DataChannelName::Y);
        } else {
            idx_order[idx_order.size() - 2] = "out_y";
        }

        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              idx_order,
                                              "dequantized_val",
                                              input_dt,
                                              1,
                                              LoadType::LT_UNALIGNED,
                                              BoundaryCheck::DISABLED };
        conf_scalar.SetLoopAxes(loop_axes, true);

        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_zyx_fsv16_imad::SetDefault(const convolution_params& params,
                                                                                       int) const {
    DispatchData dispatchData;
    const auto& output = params.output;
    const auto& weights = params.weights;
    auto block_params = GetBlockParams(params);

    dispatchData.gws[0] = CeilDiv(output.X().v, block_params.output_block_width);
    dispatchData.gws[1] = CeilDiv(output.Y().v, block_params.output_block_height) * CeilDiv(output.Z().v, block_params.output_block_depth);
    dispatchData.gws[2] = output.Batch().v * CeilDiv(weights.OFM().v, block_params.output_block_features) * params.groups * simd * block_params.feature_slm_split;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = simd * block_params.feature_slm_split;

    dispatchData.cldnnStyle = {0, 0, 0, 0, 0};
    dispatchData.gemmStyle = {0, 0, 0, 0, 0, 0};

    dispatchData.efficiency = FORCE_PRIORITY_2;
    if (static_cast<float>(params.weights.IFM().v) / static_cast<float>(Align(params.weights.IFM().v, fsv)) < 0.5f)
        dispatchData.efficiency = FORCE_PRIORITY_4;

    return dispatchData;
}  // SetDefault

bool Convolution_kernel_b_fs_zyx_fsv16_imad::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (newParams.split != 1)
        return false;

    return true;
}
}  // namespace kernel_selector
