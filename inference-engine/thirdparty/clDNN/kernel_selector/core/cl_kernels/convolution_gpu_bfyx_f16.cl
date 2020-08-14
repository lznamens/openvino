// Copyright (c) 2016-2020 Intel Corporation
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

#define INPUT_TYPE          INPUT0_TYPE
#define INPUT_TYPE2         MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4         MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8         MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

#define FILTER_TYPE8        MAKE_VECTOR_TYPE(FILTER_TYPE, 8)

#define AS_INPUT_TYPE       CAT(as_, INPUT_TYPE)
#define AS_INPUT_TYPE2      CAT(as_, INPUT_TYPE2)
#define AS_INPUT_TYPE4      CAT(as_, INPUT_TYPE4)
#define AS_INPUT_TYPE8      CAT(as_, INPUT_TYPE8)

#define AS_FILTER_TYPE8     CAT(as_, FILTER_TYPE8)

#if INPUT0_TYPE_SIZE == 2
#   define INPUT_BLOCK_READ(ptr, offset)    AS_INPUT_TYPE(intel_sub_group_block_read_us((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ2(ptr, offset)   AS_INPUT_TYPE2(intel_sub_group_block_read_us2((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ4(ptr, offset)   AS_INPUT_TYPE4(intel_sub_group_block_read_us4((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ8(ptr, offset)   AS_INPUT_TYPE8(intel_sub_group_block_read_us8((__global ushort*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ16(ptr, offset)  AS_INPUT_TYPE16(intel_sub_group_block_read8((__global uint*)(ptr) + (offset)))
#elif INPUT0_TYPE_SIZE == 4
#   define INPUT_BLOCK_READ(ptr, offset)    AS_INPUT_TYPE(intel_sub_group_block_read((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ2(ptr, offset)   AS_INPUT_TYPE2(intel_sub_group_block_read2((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ4(ptr, offset)   AS_INPUT_TYPE4(intel_sub_group_block_read4((__global uint*)(ptr) + (offset)))
#   define INPUT_BLOCK_READ8(ptr, offset)   AS_INPUT_TYPE8(intel_sub_group_block_read8((__global uint*)(ptr) + (offset)))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported input type.
#endif

#if FILTER_TYPE_SIZE == 2
#   define FILTER_BLOCK_READ8(ptr, offset) AS_FILTER_TYPE8(intel_sub_group_block_read_us8((__global ushort*)(ptr) + (offset)))
#elif FILTER_TYPE_SIZE == 4
#   define FILTER_BLOCK_READ8(ptr, offset) AS_FILTER_TYPE8(intel_sub_group_block_read8((__global uint*)(ptr) + (offset)))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported filter type.
#endif

#if OUTPUT_TYPE_SIZE == 1
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)     CAT(BLOCK_WRITE_UC_, OUTPUT_X_BLOCK_SIZE)((__global uchar*)(ptr) + (offset), \
                                                    CAT(as_uchar, OUTPUT_X_BLOCK_SIZE)(val))
#   define OUTPUT_SINGLE_WRITE(ptr, offset, val)    BLOCK_WRITE_UC_1((__global uchar*)(ptr) + (offset), as_uchar(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)     CAT(intel_sub_group_block_write_us, OUTPUT_X_BLOCK_SIZE)((__global ushort*)(ptr) + (offset), \
                                                    CAT(as_ushort, OUTPUT_X_BLOCK_SIZE)(val))
#   define OUTPUT_SINGLE_WRITE(ptr, offset, val)    intel_sub_group_block_write_us((__global ushort*)(ptr) + (offset), as_ushort(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE(ptr, offset, val)     CAT(intel_sub_group_block_write, OUTPUT_X_BLOCK_SIZE)((__global uint*)(ptr) + (offset), \
                                                    CAT(as_uint, OUTPUT_X_BLOCK_SIZE)(val))
#   define OUTPUT_SINGLE_WRITE(ptr, offset, val)    intel_sub_group_block_write((__global uint*)(ptr) + (offset), as_uint(val))
#else
#error convolution_gpu_bfyx_f16.cl - unsupported output type.
#endif // OUTPUT_TYPE_SIZE == 1

#if OUTPUT_TYPE_SIZE == 1
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   BLOCK_WRITE_UC_2((__global uchar*)(ptr) + (offset), as_uchar2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   BLOCK_WRITE_UC_4((__global uchar*)(ptr) + (offset), as_uchar4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   BLOCK_WRITE_UC_8((__global uchar*)(ptr) + (offset), as_uchar8(val))
#elif OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write_us2((__global ushort*)(ptr) + (offset), as_ushort2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write_us4((__global ushort*)(ptr) + (offset), as_ushort4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write_us8((__global ushort*)(ptr) + (offset), as_ushort8(val))
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE2(ptr, offset, val)   intel_sub_group_block_write2((__global uint*)(ptr) + (offset), as_uint2(val))
#   define OUTPUT_BLOCK_WRITE4(ptr, offset, val)   intel_sub_group_block_write4((__global uint*)(ptr) + (offset), as_uint4(val))
#   define OUTPUT_BLOCK_WRITE8(ptr, offset, val)   intel_sub_group_block_write8((__global uint*)(ptr) + (offset), as_uint8(val))
#else
#   error convolution_gpu_bfyx_f16.cl - unsupported output type.
#endif

#if INPUT0_TYPE_SIZE == 2
#   define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
#   define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#   define GET_SRC(data, id)    AS_INPUT_SRC(intel_sub_group_shuffle(AS_US_SRC(data), id))
#else
#   define GET_SRC(data, id)    intel_sub_group_shuffle(data, id)
#endif // INPUT0_TYPE_SIZE == 2

#define FEATURE_SLICE_SIZE 16
#define FILTER_OFM_NUM_ALIGNED (((FILTER_OFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
#define FILTER_IFM_NUM_ALIGNED (((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#if BATCH_IS_ONE != 1
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE, 1)))
#else
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
#endif
KERNEL(convolution_bfyx_f16)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx) {
#if GROUPED
#if BATCH_IS_ONE != 1
    const int f_block = (int)get_group_id(1);
#else
    const int f_block = (int)get_group_id(0);
#endif // BATCH_IS_ONE != 1   
    const int group = (f_block * FEATURE_SLICE_SIZE) / FILTER_OFM_NUM;
    const int prev_group_leftover = (FILTER_OFM_NUM * (group + 1)) - (f_block * FEATURE_SLICE_SIZE);
    int groups_per_sub_group = 1;
    if (prev_group_leftover < 16)
        groups_per_sub_group += ((FEATURE_SLICE_SIZE - prev_group_leftover - 1) / FILTER_OFM_NUM) + 1;
#else
#if BATCH_IS_ONE != 1
    const int f_block = (int)get_group_id(1);
#else
    const int f_block = (int)get_group_id(0);
#endif // BATCH_IS_ONE != 1
    const int group = split_idx;
    const int groups_per_sub_group = 1;
#endif // GROUPED

#if BATCH_IS_ONE != 1
    const int lid = get_sub_group_local_id();
    const int b = (int)get_global_id(2);

    const int xy = (int)get_global_id(0);
    int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM;
    const int y = (xy / X_BLOCKS);
#else
    const int lid = get_sub_group_local_id();
    const int b = 0;

    int x = (int)get_global_id(1) * OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM;
    const int y = get_global_id(2);
#endif // BATCH_IS_ONE != 1   

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    uint output_offset = b * output_b_pitch +
                         (f_block + output_fs_pad_before) * output_fs_pitch +
                         (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                         (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;
    vec_t dst[OUTPUT_X_BLOCK_NUM];
    INPUT0_TYPE* dst_scalar = (INPUT0_TYPE*)dst;

#if BIAS_TERM
    uint bias_offset = f_block * FEATURE_SLICE_SIZE;
    dst[0] = (vec_t)(INPUT_BLOCK_READ(biases, bias_offset));

    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
    for (uint i = 1; i < OUTPUT_X_BLOCK_NUM; i++) { 
        dst[i] = dst[i - 1];
    }    
#else
    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
    for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
        dst[i] = INPUT0_VAL_ZERO;
    }
#endif // BIAS_TERM

#if MULTIPLE_GROUPS_INPUT_PRELOAD
    const uint in_split_offset = f_block * input_fs_pitch;
    const uint g = lid / (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint ofm_in_group = lid % (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint grouped_filter_offset = (group + g) * FILTER_GROUPS_PITCH;
#else
#if GROUPED
    for (uint g = group; g < group + groups_per_sub_group; g++) {
        const uint in_split_offset = g * input_fs_pitch * (FILTER_IFM_NUM / FEATURE_SLICE_SIZE);
        const uint filter_split_offset = g * FILTER_GROUPS_PITCH;
        const uint filter_offset = (f_block % (FILTER_OFM_NUM / FEATURE_SLICE_SIZE)) * filter_os_pitch;
#else
        const uint in_split_offset = 0;
        const uint filter_split_offset = 0;
        const uint filter_offset = f_block * filter_os_pitch;
#endif  // GROUPED
        const uint grouped_filter_offset = filter_offset + filter_split_offset;
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD

        const uint grouped_input_offset = input_offset + in_split_offset;

        for (uint icb = 0; icb < IC_BLOCKS; icb++) {
            __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
            for (int kh = 0; kh < FILTER_SIZE_Y; kh++) {
                int calc_range_test = input_y + kh * DILATION_SIZE_Y;
                if (calc_range_test < 0 || calc_range_test >= INPUT0_SIZE_Y)
                    continue;

                INPUT_TYPE line_cache[INPUT_LINE_SIZE];

#if INPUT_LEFTOVERS
                if ((icb + 1) * FEATURE_SLICE_SIZE >= FILTER_IFM_NUM)
                {
                    for (int xb = 0; xb < INPUT_LINE_SIZE; xb++)
                    {
                        if (icb * FEATURE_SLICE_SIZE + lid >= FILTER_IFM_NUM)
                            line_cache[xb] = 0;
                        else
                            line_cache[xb] = input[grouped_input_offset +
                                                   icb * input_fs_pitch +
                                                   kh * DILATION_SIZE_Y * input_y_pitch +
                                                   xb * input_x_pitch +
                                                   lid];
                    }
                }
                else
#endif // INPUT_LEFTOVERS
                {
                    int xb = 0;   
                    const int common_line_cache_offset = grouped_input_offset + icb * input_fs_pitch + kh * DILATION_SIZE_Y * input_y_pitch;

                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        INPUT_TYPE8 vv = INPUT_BLOCK_READ8(input, common_line_cache_offset + xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                        line_cache[xb + 4] = vv[4];
                        line_cache[xb + 5] = vv[5];
                        line_cache[xb + 6] = vv[6];
                        line_cache[xb + 7] = vv[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        INPUT_TYPE4 vv = INPUT_BLOCK_READ4(input, common_line_cache_offset + xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                    }
                    for (; xb + 2 <= INPUT_LINE_SIZE; xb += 2) {
                        INPUT_TYPE2 vv = INPUT_BLOCK_READ2(input, common_line_cache_offset + xb * input_x_pitch);

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = INPUT_BLOCK_READ(input, common_line_cache_offset + xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                for (int kw = 0; kw < FILTER_SIZE_X; kw++) {
                    vec_t src[OUTPUT_X_BLOCK_NUM];

                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
                    for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {     
#if FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                        src[i].s0 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 0];
                        src[i].s1 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 1];
#if OUTPUT_X_BLOCK_SIZE > 2                        
                        src[i].s2 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 2];
                        src[i].s3 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 3];
#endif
#if OUTPUT_X_BLOCK_SIZE > 4                        
                        src[i].s4 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 4];
                        src[i].s5 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 5];
                        src[i].s6 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 6];
                        src[i].s7 = line_cache[i * OUTPUT_X_BLOCK_SIZE + 7];
#endif                        
#else
                        src[i].s0 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 0)];
                        src[i].s1 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 1)];
#if OUTPUT_X_BLOCK_SIZE > 2
                        src[i].s2 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 2)];
                        src[i].s3 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 3)];
#endif
#if OUTPUT_X_BLOCK_SIZE > 4                        
                        src[i].s4 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 4)];
                        src[i].s5 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 5)];
                        src[i].s6 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 6)];
                        src[i].s7 = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * (i * OUTPUT_X_BLOCK_SIZE + 7)];
#endif                        
#endif // FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                    }

#if MULTIPLE_GROUPS_INPUT_PRELOAD
                    typedef MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_IFM_NUM) ifm_vec_t;
                    ifm_vec_t wei0 = FILTER_VAL_ZERO;

                    for (int ifm = 0; ifm < FILTER_IFM_NUM; ifm++) {
                        wei0[ifm] = weights[grouped_filter_offset +
                                            ofm_in_group +
                                            ifm * filter_isv_pitch +
                                            kh * filter_y_pitch +
                                            kw * filter_x_pitch];
                    }                             

                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
                    for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
#if FILTER_IFM_NUM == 2
                        dst[i] = mad(wei0.s0, GET_SRC(src[i], g * FILTER_IFM_NUM + 0), dst[i]);
                        dst[i] = mad(wei0.s1, GET_SRC(src[i], g * FILTER_IFM_NUM + 1), dst[i]);
#elif FILTER_IFM_NUM == 4
                        dst[i] = mad(wei0.s0, GET_SRC(src[i], g * FILTER_IFM_NUM + 0), dst[i]);
                        dst[i] = mad(wei0.s1, GET_SRC(src[i], g * FILTER_IFM_NUM + 1), dst[i]);
                        dst[i] = mad(wei0.s2, GET_SRC(src[i], g * FILTER_IFM_NUM + 2), dst[i]);
                        dst[i] = mad(wei0.s3, GET_SRC(src[i], g * FILTER_IFM_NUM + 3), dst[i]);
#elif FILTER_IFM_NUM == 8
                        dst[i] = mad(wei0.s0, GET_SRC(src[i], g * FILTER_IFM_NUM + 0), dst[i]);
                        dst[i] = mad(wei0.s1, GET_SRC(src[i], g * FILTER_IFM_NUM + 1), dst[i]);
                        dst[i] = mad(wei0.s2, GET_SRC(src[i], g * FILTER_IFM_NUM + 2), dst[i]);
                        dst[i] = mad(wei0.s3, GET_SRC(src[i], g * FILTER_IFM_NUM + 3), dst[i]);
                        dst[i] = mad(wei0.s4, GET_SRC(src[i], g * FILTER_IFM_NUM + 4), dst[i]);
                        dst[i] = mad(wei0.s5, GET_SRC(src[i], g * FILTER_IFM_NUM + 5), dst[i]);
                        dst[i] = mad(wei0.s6, GET_SRC(src[i], g * FILTER_IFM_NUM + 6), dst[i]);
                        dst[i] = mad(wei0.s7, GET_SRC(src[i], g * FILTER_IFM_NUM + 7), dst[i]);
#else                    
#error Unsupported input feature size for multiple groups input preload
#endif // FILTER_IFM_NUM
                    }

#else
                    int wei_offset = grouped_filter_offset + 
                                     icb * filter_is_pitch +
                                     kh * filter_y_pitch +
                                     kw * filter_x_pitch;

                    FILTER_TYPE8 wei0 = FILTER_BLOCK_READ8(weights, wei_offset);
                    FILTER_TYPE8 wei1 = FILTER_BLOCK_READ8(weights, wei_offset + 8 * filter_isv_pitch);

                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
                    for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
                        dst[i] = mad(wei0.s0, GET_SRC(src[i], 0), dst[i]);
                        dst[i] = mad(wei0.s1, GET_SRC(src[i], 1), dst[i]);
                        dst[i] = mad(wei0.s2, GET_SRC(src[i], 2), dst[i]);
                        dst[i] = mad(wei0.s3, GET_SRC(src[i], 3), dst[i]);
                        dst[i] = mad(wei0.s4, GET_SRC(src[i], 4), dst[i]);
                        dst[i] = mad(wei0.s5, GET_SRC(src[i], 5), dst[i]);
                        dst[i] = mad(wei0.s6, GET_SRC(src[i], 6), dst[i]);
                        dst[i] = mad(wei0.s7, GET_SRC(src[i], 7), dst[i]);
                        dst[i] = mad(wei1.s0, GET_SRC(src[i], 8), dst[i]);
                        dst[i] = mad(wei1.s1, GET_SRC(src[i], 9), dst[i]);
                        dst[i] = mad(wei1.s2, GET_SRC(src[i], 10), dst[i]);
                        dst[i] = mad(wei1.s3, GET_SRC(src[i], 11), dst[i]);
                        dst[i] = mad(wei1.s4, GET_SRC(src[i], 12), dst[i]);
                        dst[i] = mad(wei1.s5, GET_SRC(src[i], 13), dst[i]);
                        dst[i] = mad(wei1.s6, GET_SRC(src[i], 14), dst[i]);
                        dst[i] = mad(wei1.s7, GET_SRC(src[i], 15), dst[i]);
                    }    

#endif // MULTIPLE_GROUPS_INPUT_PRELOAD
                }
            }
        }
#if GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD
    }
#endif  // GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD

    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
    for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
        dst[i] = ACTIVATION(dst[i], ACTIVATION_PARAMS);
    } 

    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
    out_vec_t res[OUTPUT_X_BLOCK_NUM];
    OUTPUT_TYPE* res_scalar = (OUTPUT_TYPE*)res;

#if OUTPUT_X_BLOCK_SIZE == 8 && OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM) != 0
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) dst0123 = dst[(OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM)) / 8].lo;
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 2) dst01 = (dst[(OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM)) / 8].hi).lo;
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 1) dst0 = ((dst[(OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM)) / 8].hi).hi).lo;

    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) res0123;
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) res01;
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 1) res0;
#endif // OUTPUT_X_BLOCK_SIZE == 8 && OUTPUT_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM) != 0

#if OUTPUT_LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {       
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            res_scalar[i] = FUSED_OPS_RESULT_SCALAR;
#else
            res_scalar[i] = TO_OUTPUT_TYPE(dst_scalar[i]);
#endif
            if ((f_block * FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i * output_x_pitch + lid] = res_scalar[i];
            }
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    { 

#if OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM) == 0
#if HAS_FUSED_OPS
        { FUSED_OPS_VEC8_0; res[0] = FUSED_OPS_RESULT_VEC8_0; }
#if OUTPUT_X_BLOCK_NUM >= 2
        { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_1; res[1] = FUSED_OPS_RESULT_VEC8_1; }
#endif // OUTPUT_X_BLOCK_NUM >= 2
#if OUTPUT_X_BLOCK_NUM == 4
        { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_2; res[2] = FUSED_OPS_RESULT_VEC8_2; }
        { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_3; res[3] = FUSED_OPS_RESULT_VEC8_3; }
#endif // OUTPUT_X_BLOCK_NUM == 4
#else
        __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
        for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
            res[i] = dst[i];
        }     
#endif // HAS_FUSED_OPS
        __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
        for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
            OUTPUT_BLOCK_WRITE(output, output_offset + OUTPUT_X_BLOCK_SIZE * output_x_pitch * i, res[i]);
        }
#else
        if (x + OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM <= OUTPUT_SIZE_X) {            
#if HAS_FUSED_OPS
            { FUSED_OPS_VEC8_0; res[0] = FUSED_OPS_RESULT_VEC8_0; }
#if OUTPUT_X_BLOCK_NUM >= 2
            { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_1; res[1] = FUSED_OPS_RESULT_VEC8_1; }
#endif // OUTPUT_X_BLOCK_NUM >= 2
#if OUTPUT_X_BLOCK_NUM == 4
            { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_2; res[2] = FUSED_OPS_RESULT_VEC8_2; }
            { x += OUTPUT_X_BLOCK_SIZE; FUSED_OPS_VEC8_3; res[3] = FUSED_OPS_RESULT_VEC8_3; }
#endif // OUTPUT_X_BLOCK_NUM == 4
#else
            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
            for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
                res[i] = dst[i];
            }     
#endif // HAS_FUSED_OPS
            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_NUM)))
            for (uint i = 0; i < OUTPUT_X_BLOCK_NUM; i++) {
                OUTPUT_BLOCK_WRITE(output, output_offset + OUTPUT_X_BLOCK_SIZE * output_x_pitch * i, res[i]);
            }

        } else {

            const uint x_div_x_block = OUTPUT_SIZE_X % (OUTPUT_X_BLOCK_SIZE * OUTPUT_X_BLOCK_NUM);

//#if OUTPUT_X_BLOCK_SIZE != 8
            __attribute__((opencl_unroll_hint(x_div_x_block)))
            for (int i = 0; i < x_div_x_block; i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                res_scalar[i] = FUSED_OPS_RESULT_SCALAR;
#else
                res_scalar[i] = TO_OUTPUT_TYPE(dst_scalar[i]);
#endif
                OUTPUT_SINGLE_WRITE(output, output_offset + i * output_x_pitch, res_scalar[i]);
            }
//#else

/*#if OUTPUT_X_BLOCK_NUM == 4
#if HAS_FUSED_OPS
            if (x_div_x_block & 16) { FUSED_OPS_VEC8_0; res[0] = FUSED_OPS_RESULT_VEC8_0; x += 8; }
                                    { FUSED_OPS_VEC8_1; res[1] = FUSED_OPS_RESULT_VEC8_1; x += 8; }  
            if (x_div_x_block & 8) { FUSED_OPS_VEC8_2; res[2] = FUSED_OPS_RESULT_VEC8_2; x += 8; }
            if (x_div_x_block & 4) { FUSED_OPS_VEC4; res0123 = FUSED_OPS_RESULT_VEC4; x += 4; }
            if (x_div_x_block & 2) { FUSED_OPS_VEC2; res01 = FUSED_OPS_RESULT_VEC2; x += 2; }
            if (x_div_x_block & 1) { FUSED_OPS_VEC1; res0 = FUSED_OPS_RESULT_VEC1; }
#endif // HAS_FUSED_OPS
            if (x_div_x_block & 16) {
                OUTPUT_BLOCK_WRITE8(output, output_offset, res[0]);
                OUTPUT_BLOCK_WRITE8(output, output_offset + OUTPUT_X_BLOCK_SIZE * output_x_pitch, res[1]);  
            }    
            if (x_div_x_block & 8) OUTPUT_BLOCK_WRITE8(output, output_offset + (x_div_x_block & ~15) * output_x_pitch, res[2]);
            if (x_div_x_block & 4) OUTPUT_BLOCK_WRITE4(output, output_offset + (x_div_x_block & ~7) * output_x_pitch, res0123);
            if (x_div_x_block & 2) OUTPUT_BLOCK_WRITE2(output, output_offset + (x_div_x_block & ~3) * output_x_pitch, res01);
            if (x_div_x_block & 1) OUTPUT_SINGLE_WRITE(output, output_offset + (x_div_x_block & ~1) * output_x_pitch, res0);
#elif OUTPUT_X_BLOCK_NUM == 2
#if HAS_FUSED_OPS
            if (x_div_x_block & 8) { FUSED_OPS_VEC8_0; res[0] = FUSED_OPS_RESULT_VEC8_0; x += 8; }
            if (x_div_x_block & 4) { FUSED_OPS_VEC4; res0123 = FUSED_OPS_RESULT_VEC4; x += 4; }
            if (x_div_x_block & 2) { FUSED_OPS_VEC2; res01 = FUSED_OPS_RESULT_VEC2; x += 2; }
            if (x_div_x_block & 1) { FUSED_OPS_VEC1; res0 = FUSED_OPS_RESULT_VEC1; }
#endif // HAS_FUSED_OPS 
            if (x_div_x_block & 8) OUTPUT_BLOCK_WRITE8(output, output_offset, res[0]);
            if (x_div_x_block & 4) OUTPUT_BLOCK_WRITE4(output, output_offset + (x_div_x_block & ~7) * output_x_pitch, res0123);
            if (x_div_x_block & 2) OUTPUT_BLOCK_WRITE2(output, output_offset + (x_div_x_block & ~3) * output_x_pitch, res01);
            if (x_div_x_block & 1) OUTPUT_SINGLE_WRITE(output, output_offset + (x_div_x_block & ~1) * output_x_pitch, res0);
#else
#if HAS_FUSED_OPS
            if (x_div_x_block & 4) { FUSED_OPS_VEC4; res0123 = FUSED_OPS_RESULT_VEC4; x += 4; }
            if (x_div_x_block & 2) { FUSED_OPS_VEC2; res01 = FUSED_OPS_RESULT_VEC2; x += 2; }
            if (x_div_x_block & 1) { FUSED_OPS_VEC1; res0 = FUSED_OPS_RESULT_VEC1; }
#endif // HAS_FUSED_OPS 
            if (x_div_x_block & 4) OUTPUT_BLOCK_WRITE4(output, output_offset, res0123);
            if (x_div_x_block & 2) OUTPUT_BLOCK_WRITE2(output, output_offset + (x_div_x_block & ~3) * output_x_pitch, res01);
            if (x_div_x_block & 1) OUTPUT_SINGLE_WRITE(output, output_offset + (x_div_x_block & ~1) * output_x_pitch, res0);
#endif // OUTPUT_X_BLOCK_NUM == 1

#endif // OUTPUT_X_BLOCK_SIZE != 8*/ 
        }
#endif
    }
}

#undef AS_INPUT_SRC
#undef AS_US_SRC
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef FILTER_OFM_NUM_ALIGNED
#undef FILTER_IFM_NUM_ALIGNED

#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8

#undef FILTER_TYPE8

#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE8

#undef AS_FILTER_TYPE8

#undef INPUT_BLOCK_READ
#undef INPUT_BLOCK_READ2
#undef INPUT_BLOCK_READ4
#undef INPUT_BLOCK_READ8

#undef FILTER_BLOCK_READ8

#undef OUTPUT_BLOCK_WRITE
#undef OUTPUT_SINGLE_WRITE
