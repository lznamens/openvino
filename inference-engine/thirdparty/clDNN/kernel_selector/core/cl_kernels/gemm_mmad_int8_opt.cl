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

#define INPUT1_TYPE_VEC             CAT(INPUT1_TYPE, OUTPUT_BLOCK_SIZE_X)
#define INPUT2_TYPE_VEC             CAT(INPUT2_TYPE, OUTPUT_BLOCK_SIZE_X)
#define ACTIVATION_TYPE_VEC         CAT(ACTIVATION_TYPE, OUTPUT_BLOCK_SIZE_X)

#define AS_INPUT2_TYPE_VEC          CAT(as_, INPUT2_TYPE_VEC)
#define TO_ACTIVATION_TYPE_VEC(val) CAT(convert_, ACTIVATION_TYPE_VEC)(val)

#if OUTPUT_BLOCK_SIZE_Y > 1
#   define PACKED_INPUT0_TYPE_VEC   CAT(PACKED_INPUT0_TYPE, OUTPUT_BLOCK_SIZE_Y)
#   define ACCUMULATOR_TYPE_VEC     CAT(ACCUMULATOR_TYPE, OUTPUT_BLOCK_SIZE_Y)
#else
#   define PACKED_INPUT0_TYPE_VEC   PACKED_INPUT0_TYPE
#   define ACCUMULATOR_TYPE_VEC     ACCUMULATOR_TYPE
#endif

#define PACKED_INPUT1_TYPE_VEC      CAT(PACKED_INPUT1_TYPE, SUB_GROUP_SIZE)

#define BLOCK_READ_US_1(ptr)        intel_sub_group_block_read_us(ptr)
#define BLOCK_READ_US_2(ptr)        intel_sub_group_block_read_us2(ptr)
#define BLOCK_READ_US_4(ptr)        intel_sub_group_block_read_us4(ptr)
#define BLOCK_READ_US_8(ptr)        intel_sub_group_block_read_us8(ptr)

#define BLOCK_READ_UI_1(ptr)        intel_sub_group_block_read(ptr)
#define BLOCK_READ_UI_2(ptr)        intel_sub_group_block_read2(ptr)
#define BLOCK_READ_UI_4(ptr)        intel_sub_group_block_read4(ptr)
#define BLOCK_READ_UI_8(ptr)        intel_sub_group_block_read8(ptr)

#define BLOCK_READ_UC_N(ptr)        CAT(BLOCK_READ_UC_, OUTPUT_BLOCK_SIZE_X)(ptr)
#define BLOCK_READ_US_N(ptr)        CAT(BLOCK_READ_US_, OUTPUT_BLOCK_SIZE_X)(ptr)
#define BLOCK_READ_UI_N(ptr)        CAT(BLOCK_READ_UI_, OUTPUT_BLOCK_SIZE_X)(ptr)

#define BLOCK_WRITE_US_1(ptr, val)      intel_sub_group_block_write_us(ptr, as_ushort(val))
#define BLOCK_WRITE_US_2(ptr, val)      intel_sub_group_block_write_us2(ptr, as_ushort2(val))
#define BLOCK_WRITE_US_4(ptr, val)      intel_sub_group_block_write_us4(ptr, as_ushort4(val))
#define BLOCK_WRITE_US_8(ptr, val)      intel_sub_group_block_write_us8(ptr, as_ushort8(val))

#define BLOCK_WRITE_UI_1(ptr, val)      intel_sub_group_block_write(ptr, as_uint(val))
#define BLOCK_WRITE_UI_2(ptr, val)      intel_sub_group_block_write2(ptr, as_uint2(val))
#define BLOCK_WRITE_UI_4(ptr, val)      intel_sub_group_block_write4(ptr, as_uint4(val))
#define BLOCK_WRITE_UI_8(ptr, val)      intel_sub_group_block_write8(ptr, as_uint8(val))

#define BLOCK_WRITE_UC_N(ptr, val)      CAT(BLOCK_WRITE_UC_, OUTPUT_BLOCK_SIZE_X)(ptr, val)
#define BLOCK_WRITE_US_N(ptr, val)      CAT(BLOCK_WRITE_US_, OUTPUT_BLOCK_SIZE_X)(ptr, val)
#define BLOCK_WRITE_UI_N(ptr, val)      CAT(BLOCK_WRITE_UI_, OUTPUT_BLOCK_SIZE_X)(ptr, val)

#define BLOCK_READ_INPUT0(ptr)          AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ_UI_1((__global uint*)(ptr)))
#define BLOCK_READ_INPUT1(ptr)          AS_TYPE(INPUT1_TYPE_VEC, BLOCK_READ_UC_N((__global uchar*)(ptr)))

#ifdef INPUT2_TYPE
#if INPUT2_TYPE_SIZE == 1
#   define BLOCK_READ_INPUT2(ptr)       AS_INPUT2_TYPE_VEC(BLOCK_READ_UC_N((__global uchar*)(ptr)))
#elif INPUT2_TYPE_SIZE == 2
#   define BLOCK_READ_INPUT2(ptr)       AS_INPUT2_TYPE_VEC(BLOCK_READ_US_N((__global ushort*)(ptr)))
#elif INPUT2_TYPE_SIZE == 4
#   define BLOCK_READ_INPUT2(ptr)       AS_INPUT2_TYPE_VEC(BLOCK_READ_UI_N((__global uint*)(ptr)))
#else
#   error gemm_mmad_int8_opt.cl : unsupported input2 type
#endif // INPUT2_TYPE_SIZE == 1
#endif // INPUT2_TYPE

#if OUTPUT_TYPE_SIZE == 1
#   define BLOCK_WRITE_OUTPUT(ptr, offset, val)     BLOCK_WRITE_UC_N((__global uchar*)(ptr) + (offset), val)
#elif OUTPUT_TYPE_SIZE == 2
#   define BLOCK_WRITE_OUTPUT(ptr, offset, val)     BLOCK_WRITE_US_N((__global ushort*)(ptr) + (offset), val)
#elif OUTPUT_TYPE_SIZE == 4
#   define BLOCK_WRITE_OUTPUT(ptr, offset, val)     BLOCK_WRITE_UI_N((__global uint*)(ptr) + (offset), val)
#else
#   error gemm_mmad_int8_opt.cl : unsupported output type
#endif // OUTPUT_TYPE_SIZE == 1

#if SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_8
#else // SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_16
#endif // SUB_GROUP_SIZE == 8

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_mmad_int8_opt.cl : unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_mmad_int8_opt.cl : unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_mmad_int8_opt.cl : unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_mmad_int8_opt.cl : unsupported output format
#endif // OUTPUT_SIMPLE
}

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(gemm_mmad_int8_opt)(
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
    const uint output_x_tile = (uint)get_global_id(0) * OUTPUT_BLOCK_SIZE_X / TILE_SIZE_N;
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

    // Register chunks of the output matrix (C)
    ACCUMULATOR_TYPE_VEC tile_output[OUTPUT_BLOCK_SIZE_X] = { (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO) };

    // Pointer to the result array (the matrix C)
    ACCUMULATOR_TYPE* tile_output_pnt = (ACCUMULATOR_TYPE*)tile_output;

    const uint K_BLOCK_NUM = INPUT0_SIZE_X / TILE_SIZE_K;

    // Loop by "k" tiles
    for (uint k = 0; k < K_BLOCK_NUM; k++) {
        //PACKED_INPUT0_TYPE_VEC tile_input0;
        PACKED_INPUT0_TYPE tile_input0[OUTPUT_BLOCK_SIZE_Y];

        PACKED_INPUT1_TYPE_VEC tile_input1[OUTPUT_BLOCK_SIZE_X];
        PACKED_INPUT1_TYPE* tile_input1_pnt = (PACKED_INPUT1_TYPE*)tile_input1;
        INPUT1_TYPE_VEC temp_input1[PACK_SIZE];
        INPUT1_TYPE* temp_input1_pnt = (INPUT1_TYPE*)temp_input1;

        const uint common_input1_offset = batch_offset_input1 + k * TILE_SIZE_K * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {

            // Loading the matrix B from the global memory to GRF
            for (uint j = 0; j < PACK_SIZE; j++) {
                temp_input1[j] = BLOCK_READ_INPUT1(input1 + common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + j * INPUT1_SIZE_X);
                //printf("x = %d y = %d lid = %d temp_input1 = %02x\n", (uint)get_global_id(0), (uint)get_global_id(1), (uint)lid, temp_input1[j]);
            }

            for (uint j = 0; j < OUTPUT_BLOCK_SIZE_X; j++) {
                MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1_pack;
                temp_input1_pack.s0 = temp_input1_pnt[j];
                temp_input1_pack.s1 = temp_input1_pnt[j + OUTPUT_BLOCK_SIZE_X];
                temp_input1_pack.s2 = temp_input1_pnt[j + 2 * OUTPUT_BLOCK_SIZE_X];
                temp_input1_pack.s3 = temp_input1_pnt[j + 3 * OUTPUT_BLOCK_SIZE_X];

                tile_input1_pnt[i + j * SUB_GROUP_SIZE] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1_pack);
                // printf("i = %d x = %d y = %d lid = %d tile_input1_pnt = %02x\n", i, (uint)get_global_id(0), (uint)get_global_id(1), (uint)lid, tile_input1_pnt[i + j * SUB_GROUP_SIZE]);
            }
        }

        const uint common_input0_offset = batch_offset_input0 + output_y_tile * TILE_SIZE_M * INPUT0_SIZE_X + k * TILE_SIZE_K;

        // Loading the matrix A from the global memory to GRF
        for (uint i = 0; i < OUTPUT_BLOCK_SIZE_Y; i++) {
            tile_input0[i] = BLOCK_READ_INPUT0(input0 + common_input0_offset + i * INPUT0_SIZE_X);
            // printf("x = %d y = %d lid = %d tile_input0 = %08x\n", (uint)get_global_id(0), (uint)get_global_id(1), (uint)lid, tile_input0);
        }

        // We should calculate OUTPUT_BLOCK_SIZE_X chunks of the matrix C
        for (uint i = 0; i < OUTPUT_BLOCK_SIZE_X; i++) {
            MAKE_VECTOR_TYPE(PACKED_INPUT0_TYPE_VEC, SUB_GROUP_SIZE) temp_input0;

            temp_input0.s0 = sub_group_broadcast(tile_input0[i], 0);
            temp_input0.s1 = sub_group_broadcast(tile_input0[i], 1);
            temp_input0.s2 = sub_group_broadcast(tile_input0[i], 2);
            temp_input0.s3 = sub_group_broadcast(tile_input0[i], 3);
            temp_input0.s4 = sub_group_broadcast(tile_input0[i], 4);
            temp_input0.s5 = sub_group_broadcast(tile_input0[i], 5);
            temp_input0.s6 = sub_group_broadcast(tile_input0[i], 6);
            temp_input0.s7 = sub_group_broadcast(tile_input0[i], 7);

            //printf("x = %d y = %d lid = %d tile_input0 = %08x\n", (uint)get_global_id(0), (uint)get_global_id(1), (uint)lid, temp_input0);
            tile_output[i] = MMAD(temp_input0, tile_input1[i], tile_output[i]);
            //printf("x = %d y = %d lid = %d temp_input0 = %08x   tile_input1[i] = %08x    tile_output[i] = %08x\n", (uint)get_global_id(0), (uint)get_global_id(1), (uint)lid, temp_input0, tile_input1[i], tile_output[i]);
        }
    }

#if HAS_FUSED_OPS
    uint output_x = output_x_tile * TILE_SIZE_N;
    uint output_y = output_y_tile * TILE_SIZE_M;
#if FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
#endif // HAS_FUSED_OPS

    for (uint i = 0; i < OUTPUT_BLOCK_SIZE_Y; i++) {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, OUTPUT_BLOCK_SIZE_X) tile_output_temp;
        for (uint j = 0; j < OUTPUT_BLOCK_SIZE_X; j++) {
            tile_output_temp[j] = tile_output_pnt[j * OUTPUT_BLOCK_SIZE_Y + i];
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

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_BLOCK_SIZE_X) res = FUSED_OPS_RESULT_VEC;
        BLOCK_WRITE_OUTPUT(output, batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N, res);
        output_y++;
#else // HAS_FUSED_OPS
        BLOCK_WRITE_OUTPUT(output, batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N, dequantized);
#endif // HAS_FUSED_OPS
    }
}

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

