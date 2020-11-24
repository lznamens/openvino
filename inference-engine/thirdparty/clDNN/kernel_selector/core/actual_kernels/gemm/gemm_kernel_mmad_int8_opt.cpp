/*
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
*/

#include "gemm_kernel_mmad_int8_opt.h"

namespace kernel_selector {
ParamsKey GemmKernelMMADint8opt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();
    k.EnableQuantization(QuantizationType::SYMMETRIC);

    return k;
}

JitConstants GemmKernelMMADint8opt::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);
    GemmTuningData td = SetTuningParams(params);

    // size_t size_n_leftovers_factor = td.big_block_leftovers ? 4 : td.output_block_size;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", td.simd_size));
    jit.Merge(MakeTypeJitConstants(Datatype::INT32, "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT0"));
    jit.Merge(MakeTypeJitConstants(params.inputs[1].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT1"));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_M", td.output_block_size_y));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_N", td.simd_size * td.output_block_size_x));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_K", td.simd_size * td.pack_size));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_SIZE_X", td.output_block_size_x));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_SIZE_Y", td.output_block_size_y));
    // jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_M", td.size_m % td.simd_size));
    // jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_N", td.size_n % (td.simd_size * size_n_leftovers_factor)));
    // jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS_K", td.size_k % (td.simd_size * td.pack_size)));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "f", "output_y", "output_x"},
                                              "dequantized",
                                              input_dt,
                                              1 };
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           { "b", "f", "output_y", "output_x" },
                                           "dequantized",
                                           input_dt,
                                           td.output_block_size_x,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::DISABLED };
        conf_vec.SetLoopAxes({ Tensor::DataChannelName::Y }, true);
        conf_vec.SetVectorAxis(Tensor::DataChannelName::X);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_scalar, conf_vec }));
    }

    return jit;
}

GemmKernelBase::DispatchData GemmKernelMMADint8opt::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;
    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);

    DispatchData kd;
    GemmTuningData td = SetTuningParams(params);

    std::vector<size_t> global = { Align(output.X().v / td.output_block_size_x, td.simd_size),
                                   Align(output.Y().v, td.output_block_size_y) / td.output_block_size_y,
                                   total_batches };

    std::vector<size_t> local = { td.simd_size, 1, 1 };

    kd.gws = { global[0], global[1], global[2] };
    kd.lws = { local[0], local[1], local[2] };

    return kd;
}

GemmKernelMMADint8opt::GemmTuningData GemmKernelMMADint8opt::InitGemmTuningData(const gemm_params& params) const {
    GemmTuningData tuning_data;

    tuning_data.size_m = params.output.Y().v;
    tuning_data.size_n = params.output.X().v;
    tuning_data.size_k = params.inputs[0].X().v;
    printf("m = %d n = %d k = %d\n", (int)tuning_data.size_m, (int)tuning_data.size_n, (int)tuning_data.size_k);
    return tuning_data;
}

inline size_t GemmKernelMMADint8opt::GetMmadOperationsNumber(const GemmTuningData& tuning_data) const {
    return tuning_data.size_m * tuning_data.size_n * tuning_data.size_k;
}

inline bool GemmKernelMMADint8opt::HasLeftovers(const GemmTuningData& tuning_data) const {
    return (tuning_data.size_m % tuning_data.output_block_size_y ||
            tuning_data.size_n % (tuning_data.output_block_size_x * tuning_data.simd_size) ||
            tuning_data.size_k % (tuning_data.simd_size * tuning_data.pack_size));
}

GemmKernelMMADint8opt::GemmTuningData GemmKernelMMADint8opt::SetTuningParams(const gemm_params& params) const {
    GemmTuningData tuning_data = InitGemmTuningData(params);
    // auto mmad_operations_number = GetMmadOperationsNumber(tuning_data);

    size_t simd_size = 8;
    size_t output_block_size_x = 1;
    size_t output_block_size_y = 1;

    tuning_data.simd_size = simd_size;
    tuning_data.output_block_size_x = output_block_size_x;
    tuning_data.output_block_size_y = output_block_size_y;

    return tuning_data;
}

KernelsData GemmKernelMMADint8opt::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto run_info = GemmKernelMMADint8opt::SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     run_info,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(params));

    GemmTuningData tuning_data = InitGemmTuningData(prim_params);
    auto mmad_operations_number = GetMmadOperationsNumber(tuning_data);

    k_data.estimatedTime = mmad_operations_number < 4096 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_1;

    return {k_data};
}

bool GemmKernelMMADint8opt::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);
    auto input0_type = gmm_params.inputs[0].GetDType();
    auto input1_type = gmm_params.inputs[1].GetDType();

    GemmTuningData tuning_data = InitGemmTuningData(gmm_params);
    if (HasLeftovers(tuning_data))
        return false;

    if (gmm_params.transpose_input0 || gmm_params.transpose_input1)
        return false;

    if ((input0_type != Datatype::UINT8 && input0_type != Datatype::INT8) ||
        (input1_type != Datatype::UINT8 && input1_type != Datatype::INT8))
        return false;

    return true;
}
}  // namespace kernel_selector
