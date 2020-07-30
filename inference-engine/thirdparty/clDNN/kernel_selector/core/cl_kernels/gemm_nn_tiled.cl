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

#include "include/common.cl"
#include "include/fetch.cl"
#include "include/unit_type.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

#if TILE_K > SIMD_WIDTH
    #define BLOCK_READ_A(ptr, offset) CAT(UNIT_BLOCK_READ, A_VEC_SIZE)(ptr, offset)
#else // TILE_K > SIMD_WIDTH
    #define BLOCK_READ_A(ptr, offset) UNIT_BLOCK_READ(ptr, offset)
#endif // TILE_K > SIMD_WIDTH

#if TILE_N > SIMD_WIDTH
    #define BLOCK_READ_B(ptr, offset) CAT(UNIT_BLOCK_READ, B_VEC_SIZE)(ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) CAT(UNIT_BLOCK_WRITE, B_VEC_SIZE)(ptr, offset, data)
#else // TILE_N > SIMD_WIDTH
    #define BLOCK_READ_B(ptr, offset) UNIT_BLOCK_READ(ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) UNIT_BLOCK_WRITE(ptr, offset, data)
#endif // TILE_N > SIMD_WIDTH

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_nn_tiled.cl : Unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_nn_tiled.cl : Unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_nn_tiled.cl : Unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_nn_tiled.cl : Unsupported output format
#endif // OUTPUT_SIMPLE
}

__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(SIMD_WIDTH, 1, 1)))
KERNEL(gemm_nn_tiled)(
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
    const uint tile_n_num = (uint)get_group_id(0);
    const uint tile_m_num = (uint)get_group_id(1);
    const uint tile_m_size = (uint)get_global_size(1);
    uint batch_number = (uint)get_global_id(2);
    const uint sglid = (uint)get_sub_group_local_id();
#if B_VEC_SIZE == 1
    const uint x = (uint)get_global_id(0);
#else // B_VEC_SIZE == 1
    const uint x = (uint)get_group_id(0) * SIMD_WIDTH * B_VEC_SIZE;
#endif // B_VEC_SIZE == 1
    uint y = tile_m_num * TILE_M;

#if TILE_M_NOT_DIVISIBLE
    const uint tile_m_iterations = tile_m_num == (tile_m_size - 1) ? TILE_M_LEFTOVER : TILE_M;
#else // TILE_M_NOT_DIVISIBLE
    const uint tile_m_iterations = TILE_M;
#endif // TILE_M_NOT_DIVISIBLE

    const uint z = batch_number % OUTPUT_SIZE_Z;
    batch_number /= OUTPUT_SIZE_Z;
    const uint w = batch_number % OUTPUT_SIZE_W;
    batch_number /= OUTPUT_SIZE_W;
    const uint f = batch_number % OUTPUT_FEATURE_NUM;
    batch_number /= OUTPUT_FEATURE_NUM;
    const uint b = batch_number % OUTPUT_BATCH_NUM;

    // Batch offsets
    const uint batch_offset_input0 = FUNC_CALL(get_input0_batch_offset)(b, f, w, z);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_batch_offset)(b, f, w, z);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(b, f, w, z);
#endif // INPUT2_TYPE
    const uint batch_offset_output = FUNC_CALL(get_output_batch_offset)(b, f, w, z);

    // Start pointers offsets
    const __global INPUT0_TYPE* a_ptr = input0 + batch_offset_input0 + tile_m_num * TILE_M * K;
    const __global INPUT1_TYPE* b_ptr = input1 + batch_offset_input1 + tile_n_num * TILE_N;
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* c_ptr = input2 + batch_offset_input2 + tile_m_num * TILE_M * N + tile_n_num * TILE_N;
#endif // INPUT2_TYPE
    __global OUTPUT_TYPE* d_ptr = output + batch_offset_output + tile_m_num * TILE_M * N + tile_n_num * TILE_N;

    uint b_raw_global_id = tile_n_num * TILE_N + sglid;

    B_FLOATN b_tile[TILE_K];
    B_FLOATN c_tile[TILE_M];

    for (uint i = 0; i < TILE_M; i++) {
        c_tile[i] = (B_FLOATN)(ACCUMULATOR_VAL_ZERO);
    }

    // Full tile calculation
    for (uint k = 0; k < K_FULL_ITERATIONS; k++) {

        // Loading B tile
        unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
#if TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
#else // TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
#endif // TILE_N_NOT_DIVISIBLE
            b_ptr += N;
        } // Loading B tile end

        // Loading A tile
        for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
#if TILE_K_NOT_DIVISIBLE
            A_FLOATN a_read = a_ptr[dot_id * K + sglid];
#else // TILE_K_NOT_DIVISIBLE
            A_FLOATN a_read = BLOCK_READ_A(a_ptr, (dot_id * K));
#endif // TILE_K_NOT_DIVISIBLE

            unroll_for (uint subtile_k_id = 0; subtile_k_id < TILE_K / SIMD_WIDTH; subtile_k_id++) {
                unroll_for (uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++) {
#if TILE_K > SIMD_WIDTH
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read[subtile_k_id], simd_local_id)),
                                         b_tile[subtile_k_id * SIMD_WIDTH + simd_local_id], c_tile[dot_id]);
#else // TILE_K > SIMD_WIDTH
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_local_id)), b_tile[simd_local_id], c_tile[dot_id]);
#endif // TILE_K > SIMD_WIDTH
                }
            }
        } // Loading A tile end

        a_ptr += TILE_K;
    } // Full tile calculation end

#if TILE_K_NOT_DIVISIBLE
    // Loading leftovers of the matrix B
    unroll_for (uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++) {
#if TILE_N_NOT_DIVISIBLE
        b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
#else // TILE_N_NOT_DIVISIBLE
        b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
#endif // TILE_N_NOT_DIVISIBLE
        b_ptr += N;
    }

    // Loading leftovers of the matrix A and calculating
    unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
        INPUT0_TYPE a_read = a_ptr[dot_id * K + sglid];

        unroll_for (uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++) {
            c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_id)), b_tile[simd_id], c_tile[dot_id]);
        }
    }
#endif // TILE_K_NOT_DIVISIBLE

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
#if TILE_N_NOT_DIVISIBLE
    FUSED_OPS_PRELOAD_SCALAR;
#else // TILE_N_NOT_DIVISIBLE
    FUSED_OPS_PRELOAD_VEC;
#endif // TILE_N_NOT_DIVISIBLE
#endif // HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD

    // Writing result in the global memory
    unroll_for (uint write_id = 0; write_id < tile_m_iterations; write_id++) {
#if TILE_N_NOT_DIVISIBLE
        if (b_raw_global_id < N) {
#ifdef INPUT2_TYPE
            OUTPUT_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
#else // INPUT2_TYPE
            OUTPUT_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC_SCALAR;
#else // FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
            OUTPUT_TYPE res = FUSED_OPS_RESULT_SCALAR;
            d_ptr[sglid] = res;
#else // HAS_FUSED_OPS
            d_ptr[sglid] = dequantized;
#endif // HAS_FUSED_OPS
        }

#else // TILE_N_NOT_DIVISIBLE

#ifdef INPUT2_TYPE
        B_FLOATN c_val = BLOCK_READ_B(c_ptr, 0);
        B_FLOATN dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_val;
#else // INPUT2_TYPE
        B_FLOATN dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_VEC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
        B_FLOATN res = FUSED_OPS_RESULT_VEC;
        BLOCK_WRITE_C(d_ptr, 0, res);
#else // HAS_FUSED_OPS
        BLOCK_WRITE_C(d_ptr, 0, dequantized);
#endif // HAS_FUSED_OPS

#endif // TILE_N_NOT_DIVISIBLE
        d_ptr += N;
#ifdef INPUT2_TYPE
        c_ptr += N;
#endif // INPUT2_TYPE
    }
}

#undef unroll_for
#undef BLOCK_READ_A
#undef BLOCK_READ_B
#undef BLOCK_WRITE_C
