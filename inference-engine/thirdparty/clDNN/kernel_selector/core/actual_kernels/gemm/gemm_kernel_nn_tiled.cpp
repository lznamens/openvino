/*
// Copyright (c) 2018-2020 Intel Corporation
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
*/

#include "gemm_kernel_nn_tiled.h"
#include <iostream>

namespace kernel_selector {
ParamsKey GemmKernelTiled::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();

    return k;
}

GemmKernelBase::DispatchData GemmKernelTiled::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;

    DispatchData kd;
    GemmTuningData td = SetTuningParams(params);

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    std::vector<size_t> global = { output.X().v, output.Y().v, total_batches };

    kd.gws0 = Align(global[0], td.tile_n_size) / (td.tile_n_size / td.simd_size);
    kd.gws1 = Align(global[1], td.tile_m_size) / td.tile_m_size;
    kd.gws2 = global[2];

    kd.lws0 = td.simd_size;
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

GemmKernelTiled::GemmTuningData GemmKernelTiled::SetTuningParams(const gemm_params& params) const {
    const auto& output = params.output;

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    tuning_data.simd_size = 8;

    if (output.X().v >= 8) {
        tuning_data.tile_n_size = tuning_data.simd_size;

        while (tuning_data.tile_n_size < 64 && output.X().v / (tuning_data.tile_n_size * 2) >= 1) {
            tuning_data.tile_n_size *= 2;
        }
    }

    // tuning_data.tile_k_size must be the same as simd_size when k % tile_k != 0
    tuning_data.tile_k_size = tuning_data.simd_size;
    tuning_data.tile_m_size = 8;

    if ((output.Y().v % tuning_data.tile_m_size) != 0 || (params.inputs[0].X().v % tuning_data.tile_k_size) != 0 ||
        (output.X().v % tuning_data.tile_n_size) != 0 || total_batches > 1 || params.transpose_input1)
    {
        tuning_data.simd_size = 16;
        tuning_data.tile_n_size = tuning_data.simd_size;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = 16;
    }

    // Hardcoded one bert case for better performance
    // N == 768 && M == 128 && K == 768
    if (output.X().v == 768 && output.Y().v == 128 && params.inputs[0].X().v == 768) {
        tuning_data.simd_size = 16;
        tuning_data.tile_n_size = tuning_data.simd_size;
        tuning_data.tile_k_size = tuning_data.simd_size;
        tuning_data.tile_m_size = 16;
    }

    return tuning_data;
}

JitConstants GemmKernelTiled::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    auto leftover_m = params.output.Y().v % tuning_data.tile_m_size;
    auto leftover_n = params.inputs[1].X().v % tuning_data.tile_n_size;
    auto leftover_k = params.output.X().v % tuning_data.tile_k_size;
    auto b_vec_size = tuning_data.tile_n_size / tuning_data.simd_size;

    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType(), "ACCUMULATOR"));

    jit.AddConstants({
        MakeJitConstant("M", params.output.Y().v),
        MakeJitConstant("K", params.inputs[0].X().v),
        MakeJitConstant("N", params.output.X().v),
        MakeJitConstant("SIMD_WIDTH", tuning_data.simd_size),
        MakeJitConstant("TILE_M", tuning_data.tile_m_size),
        MakeJitConstant("TILE_N", tuning_data.tile_n_size),
        MakeJitConstant("TILE_K", tuning_data.tile_k_size),
        MakeJitConstant("K_FULL_ITERATIONS", params.inputs[0].X().v / tuning_data.tile_k_size ),
        MakeJitConstant("TILE_M_NOT_DIVISIBLE", leftover_m != 0),
        MakeJitConstant("TILE_K_NOT_DIVISIBLE", leftover_k != 0),
        MakeJitConstant("TILE_N_NOT_DIVISIBLE", leftover_n != 0),
        MakeJitConstant("TILE_M_LEFTOVER", leftover_m),
        MakeJitConstant("TILE_K_LEFTOVER", leftover_k),
        MakeJitConstant("TILE_N_LEFTOVER", leftover_n),
    });

    if (tuning_data.tile_k_size > tuning_data.simd_size) {
        jit.AddConstants({
            MakeJitConstant("A_VEC_SIZE", tuning_data.tile_k_size / tuning_data.simd_size),
            MakeJitConstant("A_FLOATN", std::string("UNIT_TYPE") + std::to_string(tuning_data.tile_k_size / tuning_data.simd_size)),
        });
    }
    else {
        jit.AddConstants({
            MakeJitConstant("A_VEC_SIZE", 1),
            MakeJitConstant("A_FLOATN", std::string("UNIT_TYPE")),
        });
    }

    if (tuning_data.tile_n_size > tuning_data.simd_size) {
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", b_vec_size),
            MakeJitConstant("B_FLOATN", std::string("UNIT_TYPE") + std::to_string(b_vec_size)),
        });
    }
    else {
        b_vec_size = 1;
        jit.AddConstants({
            MakeJitConstant("B_VEC_SIZE", 1),
            MakeJitConstant("B_FLOATN", std::string("UNIT_TYPE")),
        });
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC", {"b", "f", "(y + write_id)", "x"},
                                           "dequantized",
                                           input_dt,
                                           b_vec_size,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::Y };
        FusedOpsConfiguration conf_scalar = { "_SCALAR", {"b", "f", "(y + write_id)", "x"},
                                               "dequantized",
                                               input_dt,
                                               1,
                                               LoadType::LT_ALIGNED_READ,
                                               BoundaryCheck::ENABLED,
                                               IndexType::TENSOR_COORD,
                                               Tensor::DataChannelName::Y };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec, conf_scalar }));
    }

    return jit;
}

KernelsData GemmKernelTiled::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_3);
}

bool GemmKernelTiled::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);

    // if (gmm_params.transpose_input0 == true || gmm_params.transpose_input1 == true)
    //    return false;
    if (gmm_params.transpose_input1 && (gmm_params.inputs[1].X().v % 16 || gmm_params.inputs[1].Y().v % 16))
        return false;

    if (gmm_params.transpose_input0) 
        return false;

    return true;
}
}  // namespace kernel_selector
