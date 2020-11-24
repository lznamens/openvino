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

#define PACK_SIZE                   4

#define AS_TYPE(type, val)          CAT(as_, type)(val)
#define ACCUMULATOR_TYPE_VEC        CAT(ACCUMULATOR_TYPE, SUB_GROUP_SIZE)
#define ACTIVATION_TYPE_VEC         CAT(ACTIVATION_TYPE, OUTPUT_BLOCK_SIZE)
#define TO_ACTIVATION_TYPE_VEC(val) CAT(convert_, ACTIVATION_TYPE_VEC)(val)
#define INPUT2_TYPE_VEC             CAT(INPUT2_TYPE, OUTPUT_BLOCK_SIZE)
#define AS_INPUT2_TYPE_VEC          CAT(as_, INPUT2_TYPE_VEC)
#define PACKED_INPUT0_TYPE_VEC      CAT(PACKED_INPUT0_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT1_TYPE_VEC      CAT(PACKED_INPUT1_TYPE, SUB_GROUP_SIZE)
#define INPUT1_TYPE_VEC             CAT(INPUT1_TYPE, OUTPUT_BLOCK_SIZE)
#define BLOCK_READ_INT(ptr)         intel_sub_group_block_read((const __global uint*)(ptr))
#define BLOCK_READ_CHAR(ptr)        BLOCK_READ_UC_4((__global uchar*)(ptr))
#define BLOCK_SHUFFLE               intel_sub_group_shuffle

#ifdef INPUT2_TYPE
#if INPUT2_TYPE_SIZE == 1
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(BLOCK_READ_UC_4((__global uchar*)(ptr)))
#elif INPUT2_TYPE_SIZE == 2
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(intel_sub_group_block_read_us4((__global ushort*)(ptr)))
#elif INPUT2_TYPE_SIZE == 4
#   define BLOCK_READ_INPUT2(ptr)   AS_INPUT2_TYPE_VEC(intel_sub_group_block_read4((__global uint*)(ptr)))
#else
#   error gemm_mmad_int8.cl : unsupported input2 type
#endif // INPUT2_TYPE_SIZE == 1
#endif // INPUT2_TYPE

#if OUTPUT_TYPE_SIZE == 1
#   define BLOCK_WRITE(ptr, offset, val)    BLOCK_WRITE_UC_4((__global uchar*)(ptr) + (offset), as_uchar4(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define BLOCK_WRITE(ptr, offset, val)    intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#else
#   error gemm_mmad_int8.cl : unsupported output type
#endif // OUTPUT_TYPE_SIZE == 1

#if SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_8x8
#else // SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_16x16
#endif // SUB_GROUP_SIZE == 8

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_mmad_int8.cl : unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_mmad_int8.cl : unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_mmad_int8.cl : unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_mmad_int8.cl : unsupported output format
#endif // OUTPUT_SIMPLE
}

inline uint FUNC(get_common_input1_offset)(uint batch_offset_input1, uint k, uint i, uint output_x_tile, uint lid) {
#if !TRANSPOSE_INPUT1
    return batch_offset_input1 + (k * TILE_SIZE_K + i * PACK_SIZE) * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;
#else // !TRANSPOSE_INPUT1
    return batch_offset_input1 + (output_x_tile * TILE_SIZE_N + lid) * INPUT1_SIZE_X + k * TILE_SIZE_K + i * PACK_SIZE;
#endif // !TRANSPOSE_INPUT1
}

inline uint FUNC(get_current_input1_offset)(uint common_input1_offset, uint i, uint lid) {
#if !TRANSPOSE_INPUT1
    return common_input1_offset + INPUT1_SIZE_X * i + lid;
#else // !TRANSPOSE_INPUT1
    return common_input1_offset + i;
#endif // !TRANSPOSE_INPUT1
}

inline uint FUNC(get_common_input0_offset)(uint batch_offset_input0, uint k, uint i, uint output_y_tile, uint lid) {
#if !TRANSPOSE_INPUT0
    return batch_offset_input0 + (output_y_tile * TILE_SIZE_M + i) * INPUT0_SIZE_X + k * TILE_SIZE_K;
#else // !TRANSPOSE_INPUT0
    return batch_offset_input0 + (k * TILE_SIZE_K + lid * PACK_SIZE) * INPUT0_SIZE_X + output_y_tile * TILE_SIZE_M + i;
#endif // !TRANSPOSE_INPUT0
}

inline uint FUNC(get_current_input0_offset)(uint common_input0_offset, uint i, uint lid) {
#if !TRANSPOSE_INPUT0
    return common_input0_offset + lid * PACK_SIZE + i;
#else // !TRANSPOSE_INPUT0
    return common_input0_offset + INPUT0_SIZE_X * i;
#endif // !TRANSPOSE_INPUT0
}

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(gemm_mmad_int8)(
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

// ***************************************************************************************** //
// Kernel with leftovers for all sizes of input matrices and all transposition combinations. //
// ***************************************************************************************** //

#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
{
    // Indices
    const uint output_x_tile = (uint)get_global_id(0) / TILE_SIZE_N;
    const uint output_y_tile = (uint)get_global_id(1);

    uint batch = get_global_id(2);
    const uint lid = (uint)get_local_id(0);

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

    // Chunks of input matrices
    PACKED_INPUT0_TYPE_VEC tile_input0;
    PACKED_INPUT1_TYPE_VEC tile_input1;
#ifdef INPUT2_TYPE
    MAKE_VECTOR_TYPE(ACTIVATION_TYPE, SUB_GROUP_SIZE) tile_input2;
#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        if (output_y_tile * TILE_SIZE_M + i >= OUTPUT_SIZE_Y) continue;
        if (output_x_tile * TILE_SIZE_N + lid >= OUTPUT_SIZE_X) continue;

        tile_input2[i] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X +
                                                   output_x_tile * TILE_SIZE_N + lid]);
    }
#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        tile_input2[i] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X +
                                                   output_x_tile * TILE_SIZE_N + lid]);
    }
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
#endif // INPUT2_TYPE

    // One chunk of the output matrix (C)
    ACCUMULATOR_TYPE_VEC tile_output = (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO);

#if !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = (INPUT0_SIZE_X - 1) / TILE_SIZE_K + 1;
    const uint K_SIZE = INPUT0_SIZE_X;
#else // !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = (INPUT0_SIZE_Y - 1) / TILE_SIZE_K + 1;
    const uint K_SIZE = INPUT0_SIZE_Y;
#endif // !TRANSPOSE_INPUT0

    // Loop by "k" tiles
    for (uint k = 0; k < K_BLOCK_NUM; k++) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input0[SUB_GROUP_SIZE];
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1[SUB_GROUP_SIZE];

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            const uint common_input1_offset = FUNC_CALL(get_common_input1_offset)(batch_offset_input1, k, i, output_x_tile, lid);

#if OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
            const uint cur_n = output_x_tile * TILE_SIZE_N + lid;
            const uint cur_k = k * TILE_SIZE_K + i * PACK_SIZE;

            temp_input1[i] = 0;

            if (cur_n < OUTPUT_SIZE_X) {
                if (cur_k + 3 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                    temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
                    temp_input1[i].s3 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 3, lid)];
                } else if (cur_k + 2 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                    temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
                } else if (cur_k + 1 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                } else if (cur_k < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                }
            }
#else // OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
            temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
            temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
            temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
            temp_input1[i].s3 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 3, lid)];
#endif // OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K

            tile_input1[i] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1[i]);
        }

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            const uint common_input0_offset = FUNC_CALL(get_common_input0_offset)(batch_offset_input0, k, i, output_y_tile, lid);

#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K
            const uint cur_m = output_y_tile * TILE_SIZE_M + i;
            const uint cur_k = k * TILE_SIZE_K + lid * PACK_SIZE;

            temp_input0[i] = 0;

            if (cur_m < OUTPUT_SIZE_Y) {
                if (cur_k + 3 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                    temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
                    temp_input0[i].s3 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 3, lid)];
                } else if (cur_k + 2 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                    temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
                } else if (cur_k + 1 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                } else if (cur_k < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                }
            }

            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K

#if !TRANSPOSE_INPUT0
            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ_INT(input0 + common_input0_offset));
#else // !TRANSPOSE_INPUT0
            temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
            temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
            temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
            temp_input0[i].s3 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 3, lid)];

            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
#endif // !TRANSPOSE_INPUT0

#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K
        }

        // Calculating one chunk of the matrix C
        tile_output = MMAD(tile_input0, tile_input1, tile_output);
    }

#if HAS_FUSED_OPS
    const uint output_x = (uint)get_global_id(0);
    uint output_y = output_y_tile * TILE_SIZE_M;
#if FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
#endif // HAS_FUSED_OPS

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
        if (output_y_tile * TILE_SIZE_M + i >= OUTPUT_SIZE_Y) continue;
        if (output_x_tile * TILE_SIZE_N + lid >= OUTPUT_SIZE_X) continue;
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N

        ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(tile_output[i]);
        dequantized *= TO_ACTIVATION_TYPE(ALPHA);

#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * tile_input2[i];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_SCALAR;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
        OUTPUT_TYPE res = FUSED_OPS_RESULT_SCALAR;
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = res;
        output_y++;
#else // HAS_FUSED_OPS
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = dequantized;
#endif // HAS_FUSED_OPS
    }
}

// ******************************************************************* //
// Optimized kernel without leftovers for different tiling parameters. //
// ******************************************************************* //

#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
{
    // Indices
    const uint output_x_tile = (uint)get_global_id(0) * OUTPUT_BLOCK_SIZE / TILE_SIZE_N;
    const uint output_y_tile = (uint)get_global_id(1);

    const uint lid = (uint)get_local_id(0);

    uint batch = get_global_id(2);
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

#if !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1
    // Register chunks of the output matrix (C)
    ACCUMULATOR_TYPE_VEC tile_output[OUTPUT_BLOCK_SIZE] = { (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                            (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                            (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                            (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO) };

    // Pointer to the result array (the matrix C)
    ACCUMULATOR_TYPE* tile_output_pnt = (ACCUMULATOR_TYPE*)tile_output;
#else // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

    // One register chunk of the output matrix (C)
    ACCUMULATOR_TYPE_VEC tile_output = (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO);
#endif // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

#if !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = INPUT0_SIZE_X / TILE_SIZE_K;
#else // !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = INPUT0_SIZE_Y / TILE_SIZE_K;
#endif // !TRANSPOSE_INPUT0

    // Loop by "k" tiles
    for (uint k = 0; k < K_BLOCK_NUM; k++) {
        PACKED_INPUT0_TYPE_VEC tile_input0;

#if !TRANSPOSE_INPUT1 && !TRANSPOSE_INPUT0
        PACKED_INPUT1_TYPE_VEC tile_input1[OUTPUT_BLOCK_SIZE];
        PACKED_INPUT1_TYPE* tile_input1_pnt = (PACKED_INPUT1_TYPE*)tile_input1;
        INPUT1_TYPE_VEC temp_input1[PACK_SIZE];
        INPUT1_TYPE* temp_input1_pnt = (INPUT1_TYPE*)temp_input1;

        const uint common_input1_offset = batch_offset_input1 + k * TILE_SIZE_K * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {

            // Loading the matrix B from the global memory to GRF
            for (uint j = 0; j < PACK_SIZE; j++) {
                temp_input1[j] = AS_TYPE(INPUT1_TYPE_VEC, BLOCK_READ_CHAR(input1 + common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + j * INPUT1_SIZE_X));
            }

            for (uint j = 0; j < OUTPUT_BLOCK_SIZE; j++) {
                INPUT1_TYPE_VEC temp_input1_pack;
                temp_input1_pack.s0 = temp_input1_pnt[j];
                temp_input1_pack.s1 = temp_input1_pnt[j + OUTPUT_BLOCK_SIZE];
                temp_input1_pack.s2 = temp_input1_pnt[j + 2 * OUTPUT_BLOCK_SIZE];
                temp_input1_pack.s3 = temp_input1_pnt[j + 3 * OUTPUT_BLOCK_SIZE];

                tile_input1_pnt[i + j * SUB_GROUP_SIZE] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1_pack);
            }
        }
#elif !TRANSPOSE_INPUT1
        PACKED_INPUT1_TYPE_VEC tile_input1;
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1[SUB_GROUP_SIZE];

        const uint common_input1_offset = batch_offset_input1 + k * TILE_SIZE_K * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;

        // Loading the matrix B from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            temp_input1[i].s0 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + lid];
            temp_input1[i].s1 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + INPUT1_SIZE_X + lid];
            temp_input1[i].s2 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + 2 * INPUT1_SIZE_X + lid];
            temp_input1[i].s3 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + 3 * INPUT1_SIZE_X + lid];

            tile_input1[i] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1[i]);
        }
#else // !TRANSPOSE_INPUT1 && !TRANSPOSE_INPUT0
        PACKED_INPUT1_TYPE_VEC tile_input1;

        const uint common_input1_offset = batch_offset_input1 + output_x_tile * TILE_SIZE_N * INPUT1_SIZE_X + k * TILE_SIZE_K;

        // Loading the matrix B from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            tile_input1[i] = AS_TYPE(PACKED_INPUT1_TYPE, BLOCK_READ_INT(input1 + common_input1_offset  + i * INPUT1_SIZE_X));
        }

        PACKED_INPUT1_TYPE_VEC tile_input1_col0 = BLOCK_SHUFFLE(tile_input1, 0);
        PACKED_INPUT1_TYPE_VEC tile_input1_col1 = BLOCK_SHUFFLE(tile_input1, 1);
        PACKED_INPUT1_TYPE_VEC tile_input1_col2 = BLOCK_SHUFFLE(tile_input1, 2);
        PACKED_INPUT1_TYPE_VEC tile_input1_col3 = BLOCK_SHUFFLE(tile_input1, 3);
        PACKED_INPUT1_TYPE_VEC tile_input1_col4 = BLOCK_SHUFFLE(tile_input1, 4);
        PACKED_INPUT1_TYPE_VEC tile_input1_col5 = BLOCK_SHUFFLE(tile_input1, 5);
        PACKED_INPUT1_TYPE_VEC tile_input1_col6 = BLOCK_SHUFFLE(tile_input1, 6);
        PACKED_INPUT1_TYPE_VEC tile_input1_col7 = BLOCK_SHUFFLE(tile_input1, 7);
#if SUB_GROUP_SIZE == 16
        PACKED_INPUT1_TYPE_VEC tile_input1_col8 = BLOCK_SHUFFLE(tile_input1, 8);
        PACKED_INPUT1_TYPE_VEC tile_input1_col9 = BLOCK_SHUFFLE(tile_input1, 9);
        PACKED_INPUT1_TYPE_VEC tile_input1_col10 = BLOCK_SHUFFLE(tile_input1, 10);
        PACKED_INPUT1_TYPE_VEC tile_input1_col11 = BLOCK_SHUFFLE(tile_input1, 11);
        PACKED_INPUT1_TYPE_VEC tile_input1_col12 = BLOCK_SHUFFLE(tile_input1, 12);
        PACKED_INPUT1_TYPE_VEC tile_input1_col13 = BLOCK_SHUFFLE(tile_input1, 13);
        PACKED_INPUT1_TYPE_VEC tile_input1_col14 = BLOCK_SHUFFLE(tile_input1, 14);
        PACKED_INPUT1_TYPE_VEC tile_input1_col15 = BLOCK_SHUFFLE(tile_input1, 15);
#endif // SUB_GROUP_SIZE == 16

        tile_input1.s0 = tile_input1_col0[lid];
        tile_input1.s1 = tile_input1_col1[lid];
        tile_input1.s2 = tile_input1_col2[lid];
        tile_input1.s3 = tile_input1_col3[lid];
        tile_input1.s4 = tile_input1_col4[lid];
        tile_input1.s5 = tile_input1_col5[lid];
        tile_input1.s6 = tile_input1_col6[lid];
        tile_input1.s7 = tile_input1_col7[lid];
#if SUB_GROUP_SIZE == 16
        tile_input1.s8 = tile_input1_col8[lid];
        tile_input1.s9 = tile_input1_col9[lid];
        tile_input1.sa = tile_input1_col10[lid];
        tile_input1.sb = tile_input1_col11[lid];
        tile_input1.sc = tile_input1_col12[lid];
        tile_input1.sd = tile_input1_col13[lid];
        tile_input1.se = tile_input1_col14[lid];
        tile_input1.sf = tile_input1_col15[lid];
#endif // SUB_GROUP_SIZE == 16

#endif // !TRANSPOSE_INPUT1 && !TRANSPOSE_INPUT0

#if !TRANSPOSE_INPUT0
        const uint common_input0_offset = batch_offset_input0 + output_y_tile * TILE_SIZE_M * INPUT0_SIZE_X + k * TILE_SIZE_K;

        // Loading the matrix A from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ_INT(input0 + common_input0_offset + i * INPUT0_SIZE_X));
        }

#else // !TRANSPOSE_INPUT0
        MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input0[SUB_GROUP_SIZE];
        const uint common_input0_offset = batch_offset_input0 + (k * TILE_SIZE_K + lid * PACK_SIZE) * INPUT0_SIZE_X + output_y_tile * TILE_SIZE_M;

        // Loading the matrix A from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            temp_input0[i].s0 = input0[common_input0_offset + i];
            temp_input0[i].s1 = input0[common_input0_offset + 1 * INPUT0_SIZE_X + i];
            temp_input0[i].s2 = input0[common_input0_offset + 2 * INPUT0_SIZE_X + i];
            temp_input0[i].s3 = input0[common_input0_offset + 3 * INPUT0_SIZE_X + i];

            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
        }

#endif // !TRANSPOSE_INPUT0

#if !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1
        // We should calculate OUTPUT_BLOCK_SIZE chunks of the matrix C
        for (uint i = 0; i < OUTPUT_BLOCK_SIZE; i++) {
            tile_output[i] = MMAD(tile_input0, tile_input1[i], tile_output[i]);
        }
#else // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

        // Calculating one chunk of the matrix C
        tile_output = MMAD(tile_input0, tile_input1, tile_output);
#endif // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1
    }

#if !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

#if HAS_FUSED_OPS
    uint output_x = output_x_tile * TILE_SIZE_N;
    uint output_y = output_y_tile * TILE_SIZE_M;
#if FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
#endif // HAS_FUSED_OPS

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4) tile_output_temp;
        for (uint j = 0; j < OUTPUT_BLOCK_SIZE; j++) {
            tile_output_temp[j] = tile_output_pnt[j * SUB_GROUP_SIZE + i];
        }

        ACTIVATION_TYPE_VEC dequantized = TO_ACTIVATION_TYPE(ALPHA) * TO_ACTIVATION_TYPE_VEC(tile_output_temp);

#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * TO_ACTIVATION_TYPE_VEC(BLOCK_READ_INPUT2(input2 + batch_offset_input2 +
                       (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N));
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_VEC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_BLOCK_SIZE) res = FUSED_OPS_RESULT_VEC;
        BLOCK_WRITE(output, batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N, res);
        output_y++;
#else // HAS_FUSED_OPS
        BLOCK_WRITE(output, batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N, dequantized);
#endif // HAS_FUSED_OPS
    }

#else // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

#if HAS_FUSED_OPS
    uint output_x = (uint)get_global_id(0);
    uint output_y = output_y_tile * TILE_SIZE_M;
#if FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
#endif // HAS_FUSED_OPS

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(tile_output[i]);
        dequantized *= TO_ACTIVATION_TYPE(ALPHA);
#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * TO_ACTIVATION_TYPE(input2[batch_offset_input2 +
                       (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid]);
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_SCALAR;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD

        OUTPUT_TYPE res = FUSED_OPS_RESULT_SCALAR;
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = res;
        output_y++;
#else // HAS_FUSED_OPS
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = dequantized;
#endif // HAS_FUSED_OPS
    }

#endif // !TRANSPOSE_INPUT0 && !TRANSPOSE_INPUT1

}
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K

#undef PACK_SIZE
#undef AS_TYPE
#undef ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef TO_ACTIVATION_TYPE_VEC
#undef INPUT2_TYPE_VEC
#undef AS_INPUT2_TYPE_VEC
#undef PACKED_INPUT0_TYPE_VEC
#undef PACKED_INPUT1_TYPE_VEC
#undef INPUT1_TYPE_VEC
#undef BLOCK_READ_INT
#undef BLOCK_READ_CHAR
#undef BLOCK_READ_INPUT2
#undef BLOCK_WRITE
#undef BLOCK_SHUFFLE
#undef MMAD
