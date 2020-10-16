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

#include "include/include_all.cl"
#include "include/mmad.cl"

#define AS_TYPE(type, val)          CAT(as_, type)(val)
#define ACCUMULATOR_TYPE_VEC        CAT(ACCUMULATOR_TYPE, SUB_GROUP_SIZE)
#define ACTIVATION_TYPE_VEC         CAT(ACTIVATION_TYPE, OUTPUT_BLOCK_SIZE)
#define TO_ACTIVATION_TYPE_VEC(val) CAT(convert_, ACTIVATION_TYPE_VEC)(val)
#define INPUT2_TYPE_VEC             CAT(INPUT2_TYPE, OUTPUT_BLOCK_SIZE)
#define AS_INPUT2_TYPE_VEC          CAT(as_, INPUT2_TYPE_VEC)
#define PACKED_INPUT0_TYPE_VEC      CAT(PACKED_INPUT0_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT1_TYPE_VEC      CAT(PACKED_INPUT1_TYPE, SUB_GROUP_SIZE)
#define BLOCK_READ(ptr)             intel_sub_group_block_read((const __global uint*)(ptr))

#ifdef INPUT2_TYPE
#if INPUT2_TYPE_SIZE == 1
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(BLOCK_READ_UC_4((__global uchar*)(ptr)))
#elif INPUT2_TYPE_SIZE == 2
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(intel_sub_group_block_read_us4((__global ushort*)(ptr)))
#elif INPUT2_TYPE_SIZE == 4        
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(intel_sub_group_block_read4((__global uint*)(ptr)))
#else
#   error gemm_mmad_int8_slm.cl : unsupported input2 type
#endif // INPUT2_TYPE_SIZE == 1
#endif // INPUT2_TYPE

#if OUTPUT_TYPE_SIZE == 1
#   define BLOCK_WRITE(ptr, offset, val)    BLOCK_WRITE_UC_4((__global uchar*)(ptr) + (offset), as_uchar4(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#else
#   error gemm_mmad_int8_slm.cl : unsupported output type
#endif // OUTPUT_TYPE_SIZE == 1

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_mmad_int8_slm.cl : unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_mmad_int8_slm.cl : unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_mmad_int8_slm.cl : unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_mmad_int8_slm.cl : unsupported output format
#endif // OUTPUT_SIMPLE
}

// GEMM int8 kernel using SLM and MMAD macro. Without transpositions of input matrices and without leftovers
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, PACK_SIZE, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(gemm_mmad_int8_slm)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif // INPUT2_TYPE
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif // HAS_FUSED_OPS_DECLS
    )
{
    // Indices
    const uint output_x_tile = (uint)get_global_id(0) * OUTPUT_BLOCK_SIZE / SLM_TILE_SIZE_N;
    const uint output_y_tile = (uint)get_global_id(1);

    uint batch = get_global_id(2);
    const uint lid0 = (uint)get_local_id(0);
    const uint lid1 = (uint)get_local_id(1);
    const uint gid0 = (uint)get_group_id(0);
    const uint gid1 = (uint)get_group_id(1);

    const uint z = batch % OUTPUT_SIZE_Z;
    batch /= OUTPUT_SIZE_Z;
    const uint w = batch % OUTPUT_SIZE_W;
    batch /= OUTPUT_SIZE_W;
    const uint f = batch % OUTPUT_FEATURE_NUM;
    batch /= OUTPUT_FEATURE_NUM;
    const uint b = batch % OUTPUT_BATCH_NUM;

    // Batch offsets
    const uint batch_offset_input0 = FUNC_CALL(get_input0_batch_offset)(b, f, w, z);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_batch_offset)(b, f, w, z);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(b, f, w, z);
#endif // INPUT2_TYPE
    const uint batch_offset_output = FUNC_CALL(get_output_batch_offset)(b, f, w, z);

    // Pointer for loading the matrix B from the global memory
    __global PACKED_INPUT1_TYPE_VEC* input1_pnt = (__global PACKED_INPUT1_TYPE_VEC*)input1;

    // SLM tile of the matrix B
    __local PACKED_INPUT1_TYPE_VEC slm_tile_input1[SLM_TILE_SIZE_K * SLM_DECIMATION_FACTOR];

    // Pointer for loading the matrix B from SLM to registry chunks (GRF)
    __local INPUT1_TYPE* slm_tile_input1_pnt = (__local INPUT1_TYPE*)slm_tile_input1;

    // Registry chunks of input matrices (A, B)
    PACKED_INPUT0_TYPE_VEC reg_tile_input0;
    PACKED_INPUT1_TYPE_VEC reg_tile_input1;

    // Registry chunks of the output matrix (C)
    ACCUMULATOR_TYPE_VEC reg_tile_output[OUTPUT_BLOCK_SIZE] = { (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO) };

    // Pointer to the result array (the matrix C)
    ACCUMULATOR_TYPE* reg_tile_output_pnt = (ACCUMULATOR_TYPE*)reg_tile_output;

    // Calculating positions for loading input matrices from the global memory
    const uint wg_offset_y = lid1 * SUB_GROUP_SIZE + lid0;
    const uint input1_index = (batch_offset_input1 + wg_offset_y * INPUT1_SIZE_X) / SLM_TILE_SIZE_N + gid0;
    const uint common_input0_offset = batch_offset_input0 + (gid1 * SUB_GROUP_SIZE * PACK_SIZE + lid1 * SUB_GROUP_SIZE) * INPUT0_SIZE_X;

#ifdef PRELOADING_SLM
    for (uint i = 0; i < SLM_DECIMATION_FACTOR; i++) {
        slm_tile_input1[i * SLM_TILE_SIZE_K + wg_offset_y] = input1_pnt[input1_index + i * INPUT1_SIZE_X];
    }

    // Synchronization; waiting until all work items will finish loading the matrix B from the global memory to SLM
    barrier(CLK_LOCAL_MEM_FENCE);
#endif // PRELOADING_SLM

    // Loop by "k" tiles
    for (uint k = 0; k < INPUT0_SIZE_X / SLM_TILE_SIZE_K; k++) {

        // Loading the matrix A from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            reg_tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ(input0 + common_input0_offset + k * SLM_TILE_SIZE_K + i * INPUT0_SIZE_X));
        }

#ifndef PRELOADING_SLM
        // Loading the matrix B to SLM
        if (k % SLM_DECIMATION_FACTOR == 0) {

            // Synchronization
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < SLM_DECIMATION_FACTOR; i++) {
                slm_tile_input1[i * SLM_TILE_SIZE_K + wg_offset_y] = input1_pnt[input1_index + (k + i) * INPUT1_SIZE_X];
            }

            // Synchronization; waiting until all work items will finish loading the matrix B from the global memory to SLM
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#endif // PRELOADING_SLM

        // Loading the matrix B from SLM to GRF and calculating the matrix C
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1;

        // Here is OUTPUT_BLOCK_SIZE iterations in the extern loop because we should calculate OUTPUT_BLOCK_SIZE chunks of the matrix C
        for (uint i = 0; i < OUTPUT_BLOCK_SIZE; i++) {
            const uint common_offset = (k % SLM_DECIMATION_FACTOR) * SLM_TILE_SIZE_K * SLM_TILE_SIZE_N + i * SUB_GROUP_SIZE + lid0;

            for (uint j = 0; j < SUB_GROUP_SIZE; j++) {
                temp_input1.s0 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE_N * (j * PACK_SIZE + 0)];
                temp_input1.s1 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE_N * (j * PACK_SIZE + 1)];
                temp_input1.s2 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE_N * (j * PACK_SIZE + 2)];
                temp_input1.s3 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE_N * (j * PACK_SIZE + 3)];

                reg_tile_input1[j] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1);
            }

        // Calculating one chunk of the matrix C
        reg_tile_output[i] = MMAD_8x8(reg_tile_input0, reg_tile_input1, reg_tile_output[i]);
        }
    } // End of the loop by "k"

#if HAS_FUSED_OPS
    uint output_x = output_x_tile * SLM_TILE_SIZE_N;
    uint output_y = output_y_tile * SUB_GROUP_SIZE;
#if FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
#endif // HAS_FUSED_OPS

    // Last calculations and writing result in the global memory
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) reg_tile_output_temp;
        for (uint j = 0; j < OUTPUT_BLOCK_SIZE; j++) {
            reg_tile_output_temp[j] = reg_tile_output_pnt[j * SUB_GROUP_SIZE + i];
        }

        ACTIVATION_TYPE_VEC dequantized = TO_ACTIVATION_TYPE(ALPHA) * TO_ACTIVATION_TYPE_VEC(reg_tile_output_temp);

#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * TO_ACTIVATION_TYPE_VEC(BLOCK_READ_INPUT2(input2 + batch_offset_input2 +
                       (output_y_tile * SUB_GROUP_SIZE + i) * OUTPUT_SIZE_X + output_x_tile * SLM_TILE_SIZE_N));
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_VEC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_BLOCK_SIZE) res = FUSED_OPS_RESULT_VEC;
        BLOCK_WRITE(output, batch_offset_output + (output_y_tile * SUB_GROUP_SIZE + i) * OUTPUT_SIZE_X + output_x_tile * SLM_TILE_SIZE_N, res);
        output_y++;
#else // HAS_FUSED_OPS
        BLOCK_WRITE(output, batch_offset_output + (output_y_tile * SUB_GROUP_SIZE + i) * OUTPUT_SIZE_X + output_x_tile * SLM_TILE_SIZE_N, dequantized);
#endif // HAS_FUSED_OPS
    }
}

#undef AS_TYPE
#undef ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef TO_ACTIVATION_TYPE_VEC
#undef INPUT2_TYPE_VEC
#undef AS_INPUT2_TYPE_VEC
#undef PACKED_INPUT0_TYPE_VEC
#undef PACKED_INPUT1_TYPE_VEC
#undef BLOCK_READ
#undef BLOCK_READ_INPUT2
#undef BLOCK_WRITE
