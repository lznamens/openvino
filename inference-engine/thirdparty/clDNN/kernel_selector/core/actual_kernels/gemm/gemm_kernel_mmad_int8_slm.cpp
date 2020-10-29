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

#include "gemm_kernel_mmad_int8_slm.h"

namespace kernel_selector {
ParamsKey GemmKernelMMADslmInt8::GetSupportedKey() const {
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

JitConstants GemmKernelMMADslmInt8::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);
    GemmTuningData td = SetTuningParams(params);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", td.simd_size));
    jit.AddConstant(MakeJitConstant("PACK_SIZE", td.pack_size));
    jit.Merge(MakeTypeJitConstants(Datatype::INT32, "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(params.inputs[0].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT0"));
    jit.Merge(MakeTypeJitConstants(params.inputs[1].GetDType() == Datatype::INT8 ? Datatype::INT32 : Datatype::UINT32, "PACKED_INPUT1"));
    jit.AddConstant(MakeJitConstant("SLM_TILE_SIZE_N", td.slm_tile_size_n));
    jit.AddConstant(MakeJitConstant("SLM_TILE_SIZE_K", td.slm_tile_size_k));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_SIZE", td.output_block_size));
    jit.AddConstant(MakeJitConstant("SLM_DECIMATION_FACTOR", td.slm_decimation_factor));
    if (td.size_k <= td.max_slm_preloading_size) jit.AddConstant(MakeJitConstant("PRELOADING_SLM", 1));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = {  "_VEC",
                                            { "b", "f", "output_y", "output_x" },
                                            "dequantized",
                                            input_dt,
                                            td.output_block_size,
                                            LoadType::LT_ALIGNED_READ,
                                            BoundaryCheck::DISABLED };
        conf_vec.SetLoopAxes({ Tensor::DataChannelName::Y }, true);
        conf_vec.SetVectorAxis(Tensor::DataChannelName::X);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec }));
    }

    return jit;
}

GemmKernelBase::DispatchData GemmKernelMMADslmInt8::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;
    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);

    DispatchData kd;
    GemmTuningData td = SetTuningParams(params);

    std::vector<size_t> global = { td.size_n / td.output_block_size, output.Y().v / td.simd_size, total_batches };
    std::vector<size_t> local = { td.slm_tile_size_n / td.output_block_size, td.slm_tile_size_k / td.simd_size, 1 };

    kd.gws = { global[0], global[1], global[2] };
    kd.lws = { local[0], local[1], local[2] };

    return kd;
}

GemmKernelMMADslmInt8::GemmTuningData GemmKernelMMADslmInt8::InitGemmTuningData(const gemm_params& params) const {
    GemmTuningData tuning_data;

    tuning_data.size_m = params.output.Y().v;
    tuning_data.size_n = params.output.X().v;
    tuning_data.size_k = params.inputs[0].X().v;

    return tuning_data;
}

inline size_t GemmKernelMMADslmInt8::GetMmadOperationsNumber(const GemmTuningData& tuning_data) const {
    return tuning_data.size_m * tuning_data.size_n * tuning_data.size_k;
}

inline bool GemmKernelMMADslmInt8::HasLeftovers(const GemmTuningData& tuning_data) const {
    return tuning_data.size_m % tuning_data.simd_size || tuning_data.size_n % tuning_data.slm_tile_size_n ||
           tuning_data.size_k % (tuning_data.slm_tile_size_k * tuning_data.slm_decimation_factor);
}

GemmKernelMMADslmInt8::GemmTuningData GemmKernelMMADslmInt8::SetTuningParams(const gemm_params& params) const {
    GemmTuningData tuning_data = InitGemmTuningData(params);

    tuning_data.slm_decimation_factor = tuning_data.size_k <= tuning_data.max_slm_preloading_size ?
                                        tuning_data.size_k / tuning_data.slm_tile_size_k : 2;
    return tuning_data;
}

KernelsData GemmKernelMMADslmInt8::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto run_info = GemmKernelMMADslmInt8::SetDefault(prim_params);
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

    if ((tuning_data.size_m == 384 && tuning_data.size_k == 384 && tuning_data.size_n == 64) ||
        (tuning_data.size_m == 384 && tuning_data.size_k == 64 && tuning_data.size_n == 384))
        k_data.estimatedTime = FORCE_PRIORITY_2;
    else if (mmad_operations_number <= 65536 || tuning_data.size_k <= 64)
        k_data.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    else
        k_data.estimatedTime = FORCE_PRIORITY_5;

    return {k_data};
}

bool GemmKernelMMADslmInt8::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& gmm_params = static_cast<const gemm_params&>(params);
    auto input0_type = gmm_params.inputs[0].GetDType();
    auto input1_type = gmm_params.inputs[1].GetDType();

    if (gmm_params.transpose_input0 || gmm_params.transpose_input1)
        return false;

    GemmTuningData tuning_data = InitGemmTuningData(gmm_params);
    if (HasLeftovers(tuning_data))
        return false;

    if ((input0_type != Datatype::UINT8 && input0_type != Datatype::INT8) ||
        (input1_type != Datatype::UINT8 && input1_type != Datatype::INT8))
        return false;

    return true;
}
}  // namespace kernel_selector
