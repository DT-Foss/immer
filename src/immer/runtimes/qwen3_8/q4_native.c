/*
 * IMMER row-addressable Q4_0/Q8_0 CPU kernels.
 *
 * The 32-value block encodings are wire-compatible with ggml Q4_0 and Q8_0:
 *   Q4_0: fp16 scale + 16 packed nibbles, value = (nibble - 8) * scale
 *   Q8_0: fp16 scale + 32 signed bytes, value = q * scale
 *
 * The AVX2 dot-product structure follows the public Q4_0/Q8_0 formulation
 * used by llama.cpp (MIT); this file is an independent, narrow C ABI for
 * IMMER's own Qwen attention/runtime rather than a model backend.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define IMMER_EXPORT __declspec(dllexport)
#else
#define IMMER_EXPORT __attribute__((visibility("default")))
#endif

#define IMMER_Q4_ABI 2u
#define IMMER_QK 32
#define IMMER_FORMAT_Q4_0 4
#define IMMER_FORMAT_Q8_0 8

#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
typedef struct
#if !defined(_MSC_VER)
    __attribute__((packed))
#endif
{
    uint16_t d;
    uint8_t qs[16];
} immer_block_q4_0;

typedef struct
#if !defined(_MSC_VER)
    __attribute__((packed))
#endif
{
    uint16_t d;
    int8_t qs[32];
} immer_block_q8_0;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif

typedef char immer_q4_size_check[(sizeof(immer_block_q4_0) == 18) ? 1 : -1];
typedef char immer_q8_size_check[(sizeof(immer_block_q8_0) == 34) ? 1 : -1];

static float immer_half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    int32_t exponent = (int32_t) (((uint32_t) h >> 10) & 0x1fu);
    uint32_t fraction = (uint32_t) h & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                --exponent;
            }
            fraction &= 0x03ffu;
            bits = sign | ((uint32_t) (exponent + 112) << 23) | (fraction << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((uint32_t) (exponent + 112) << 23) | (fraction << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t immer_float_to_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t absolute = bits & 0x7fffffffu;
    if (absolute >= 0x7f800000u) {
        return (uint16_t) (sign | (absolute > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    const int32_t exponent = (int32_t) ((absolute >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = absolute & 0x7fffffu;
    if (exponent >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t) sign;
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (
            remainder > halfway
            || (remainder == halfway && (half_mantissa & 1u))
        ) {
            ++half_mantissa;
        }
        return (uint16_t) (sign | half_mantissa);
    }
    uint32_t half = sign | ((uint32_t) exponent << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
    }
    return (uint16_t) half;
}

static void immer_quantize_q8_row(const float *x, immer_block_q8_0 *out, int64_t cols) {
    const int64_t blocks = cols / IMMER_QK;
    for (int64_t block = 0; block < blocks; ++block) {
        const float *values = x + block * IMMER_QK;
        float absolute_max = 0.0f;
        for (int index = 0; index < IMMER_QK; ++index) {
            const float absolute = fabsf(values[index]);
            if (absolute > absolute_max) absolute_max = absolute;
        }
        const float scale = absolute_max / 127.0f;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[block].d = immer_float_to_half(scale);
        for (int index = 0; index < IMMER_QK; ++index) {
            int quantized = (int) roundf(values[index] * inverse);
            if (quantized < -127) quantized = -127;
            if (quantized > 127) quantized = 127;
            out[block].qs[index] = (int8_t) quantized;
        }
    }
}

static void immer_quantize_q4_row(const float *x, immer_block_q4_0 *out, int64_t cols) {
    const int64_t blocks = cols / IMMER_QK;
    for (int64_t block = 0; block < blocks; ++block) {
        const float *values = x + block * IMMER_QK;
        float absolute_max = 0.0f;
        float signed_max = 0.0f;
        for (int index = 0; index < IMMER_QK; ++index) {
            const float absolute = fabsf(values[index]);
            if (absolute > absolute_max) {
                absolute_max = absolute;
                signed_max = values[index];
            }
        }
        const float scale = signed_max / -8.0f;
        const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
        out[block].d = immer_float_to_half(scale);
        for (int index = 0; index < IMMER_QK / 2; ++index) {
            int low = (int8_t) (values[index] * inverse + 8.5f);
            int high = (int8_t) (values[index + IMMER_QK / 2] * inverse + 8.5f);
            if (low < 0) low = 0;
            if (low > 15) low = 15;
            if (high < 0) high = 0;
            if (high > 15) high = 15;
            out[block].qs[index] = (uint8_t) (low | (high << 4));
        }
    }
}

#if defined(__AVX2__)
static inline __m256i immer_unpack_nibbles(const uint8_t *packed) {
    const __m128i raw = _mm_loadu_si128((const __m128i *) packed);
    const __m256i both = _mm256_insertf128_si256(
        _mm256_castsi128_si256(raw), _mm_srli_epi16(raw, 4), 1
    );
    return _mm256_and_si256(both, _mm256_set1_epi8(0x0f));
}

static inline __m256 immer_mul_sum_i8_pairs_float(__m256i left, __m256i right) {
    const __m256i absolute_left = _mm256_sign_epi8(left, left);
    const __m256i signed_right = _mm256_sign_epi8(right, left);
    const __m256i products = _mm256_maddubs_epi16(absolute_left, signed_right);
    const __m256i pairs = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(pairs);
}

static inline float immer_horizontal_sum(__m256 value) {
    __m128 sum = _mm_add_ps(
        _mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1)
    );
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static float immer_dot_q4_q8_avx2(
    const immer_block_q4_0 *weight,
    const immer_block_q8_0 *input,
    int64_t blocks
) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int64_t block = 0; block < blocks; ++block) {
        __m256i qweight = immer_unpack_nibbles(weight[block].qs);
        qweight = _mm256_sub_epi8(qweight, _mm256_set1_epi8(8));
        const __m256i qinput = _mm256_loadu_si256(
            (const __m256i *) input[block].qs
        );
        const float scale = immer_half_to_float(weight[block].d)
            * immer_half_to_float(input[block].d);
        accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(scale),
            immer_mul_sum_i8_pairs_float(qweight, qinput),
            accumulator
        );
    }
    return immer_horizontal_sum(accumulator);
}

static float immer_dot_q8_q8_avx2(
    const immer_block_q8_0 *weight,
    const immer_block_q8_0 *input,
    int64_t blocks
) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int64_t block = 0; block < blocks; ++block) {
        const __m256i qweight = _mm256_loadu_si256(
            (const __m256i *) weight[block].qs
        );
        const __m256i qinput = _mm256_loadu_si256(
            (const __m256i *) input[block].qs
        );
        const float scale = immer_half_to_float(weight[block].d)
            * immer_half_to_float(input[block].d);
        accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(scale),
            immer_mul_sum_i8_pairs_float(qweight, qinput),
            accumulator
        );
    }
    return immer_horizontal_sum(accumulator);
}
#endif

static float immer_dot_q4_q8_scalar(
    const immer_block_q4_0 *weight,
    const immer_block_q8_0 *input,
    int64_t blocks
) {
    float sum = 0.0f;
    for (int64_t block = 0; block < blocks; ++block) {
        int32_t integer_sum = 0;
        for (int index = 0; index < 16; ++index) {
            const uint8_t packed = weight[block].qs[index];
            integer_sum += ((int32_t) (packed & 0x0f) - 8)
                * (int32_t) input[block].qs[index];
            integer_sum += ((int32_t) (packed >> 4) - 8)
                * (int32_t) input[block].qs[index + 16];
        }
        sum += (float) integer_sum
            * immer_half_to_float(weight[block].d)
            * immer_half_to_float(input[block].d);
    }
    return sum;
}

static float immer_dot_q8_q8_scalar(
    const immer_block_q8_0 *weight,
    const immer_block_q8_0 *input,
    int64_t blocks
) {
    float sum = 0.0f;
    for (int64_t block = 0; block < blocks; ++block) {
        int32_t integer_sum = 0;
        for (int index = 0; index < 32; ++index) {
            integer_sum += (int32_t) weight[block].qs[index]
                * (int32_t) input[block].qs[index];
        }
        sum += (float) integer_sum
            * immer_half_to_float(weight[block].d)
            * immer_half_to_float(input[block].d);
    }
    return sum;
}

static float immer_dot_packed_q8(
    const uint8_t *weight,
    int format,
    const immer_block_q8_0 *input,
    int64_t blocks
) {
    if (format == IMMER_FORMAT_Q4_0) {
#if defined(__AVX2__)
        return immer_dot_q4_q8_avx2(
            (const immer_block_q4_0 *) weight, input, blocks
        );
#else
        return immer_dot_q4_q8_scalar(
            (const immer_block_q4_0 *) weight, input, blocks
        );
#endif
    }
#if defined(__AVX2__)
    return immer_dot_q8_q8_avx2(
        (const immer_block_q8_0 *) weight, input, blocks
    );
#else
    return immer_dot_q8_q8_scalar(
        (const immer_block_q8_0 *) weight, input, blocks
    );
#endif
}

IMMER_EXPORT uint32_t immer_q4_abi(void) {
    return IMMER_Q4_ABI;
}

IMMER_EXPORT int immer_q4_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

IMMER_EXPORT int64_t immer_q4_row_bytes(int format, int64_t cols) {
    if (cols <= 0 || cols % IMMER_QK != 0) return -1;
    if (format == IMMER_FORMAT_Q4_0) {
        return (cols / IMMER_QK) * (int64_t) sizeof(immer_block_q4_0);
    }
    if (format == IMMER_FORMAT_Q8_0) {
        return (cols / IMMER_QK) * (int64_t) sizeof(immer_block_q8_0);
    }
    return -1;
}

IMMER_EXPORT int immer_q4_quantize_f32(
    const float *input,
    int64_t rows,
    int64_t cols,
    int format,
    uint8_t *output,
    int threads
) {
    const int64_t row_bytes = immer_q4_row_bytes(format, cols);
    if (!input || !output || rows <= 0 || row_bytes <= 0 || threads <= 0) return 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        if (format == IMMER_FORMAT_Q4_0) {
            immer_quantize_q4_row(
                input + row * cols,
                (immer_block_q4_0 *) (output + row * row_bytes),
                cols
            );
        } else {
            immer_quantize_q8_row(
                input + row * cols,
                (immer_block_q8_0 *) (output + row * row_bytes),
                cols
            );
        }
    }
    return 0;
}

IMMER_EXPORT int immer_q4_dequantize_rows_f32(
    const uint8_t *weights,
    int format,
    int64_t rows,
    int64_t cols,
    const int64_t *row_ids,
    int64_t selected_rows,
    float *output,
    int threads
) {
    const int64_t row_bytes = immer_q4_row_bytes(format, cols);
    if (!weights || !row_ids || !output || rows <= 0 || selected_rows < 0
        || row_bytes <= 0 || threads <= 0) return 1;
    for (int64_t index = 0; index < selected_rows; ++index) {
        if (row_ids[index] < 0 || row_ids[index] >= rows) return 2;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t selected = 0; selected < selected_rows; ++selected) {
        const uint8_t *row = weights + row_ids[selected] * row_bytes;
        float *target = output + selected * cols;
        const int64_t blocks = cols / IMMER_QK;
        if (format == IMMER_FORMAT_Q4_0) {
            const immer_block_q4_0 *source = (const immer_block_q4_0 *) row;
            for (int64_t block = 0; block < blocks; ++block) {
                const float scale = immer_half_to_float(source[block].d);
                for (int index = 0; index < 16; ++index) {
                    const uint8_t packed = source[block].qs[index];
                    target[block * 32 + index] =
                        ((int) (packed & 0x0f) - 8) * scale;
                    target[block * 32 + index + 16] =
                        ((int) (packed >> 4) - 8) * scale;
                }
            }
        } else {
            const immer_block_q8_0 *source = (const immer_block_q8_0 *) row;
            for (int64_t block = 0; block < blocks; ++block) {
                const float scale = immer_half_to_float(source[block].d);
                for (int index = 0; index < 32; ++index) {
                    target[block * 32 + index] = source[block].qs[index] * scale;
                }
            }
        }
    }
    return 0;
}

IMMER_EXPORT int immer_q4_linear_f32(
    const float *input,
    int64_t input_rows,
    int64_t input_cols,
    const uint8_t *weights,
    int format,
    int64_t output_rows,
    float *output,
    int threads
) {
    const int64_t row_bytes = immer_q4_row_bytes(format, input_cols);
    if (!input || !weights || !output || input_rows <= 0 || output_rows <= 0
        || row_bytes <= 0 || threads <= 0) return 1;
    const int64_t blocks = input_cols / IMMER_QK;
    immer_block_q8_0 *quantized_input = (immer_block_q8_0 *) malloc(
        (size_t) input_rows * (size_t) blocks * sizeof(immer_block_q8_0)
    );
    if (!quantized_input) return 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t row = 0; row < input_rows; ++row) {
        immer_quantize_q8_row(
            input + row * input_cols,
            quantized_input + row * blocks,
            input_cols
        );
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint8_t *weight_row = weights + output_row * row_bytes;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            const immer_block_q8_0 *active_input =
                quantized_input + input_row * blocks;
            output[input_row * output_rows + output_row] = immer_dot_packed_q8(
                weight_row, format, active_input, blocks
            );
        }
    }
    free(quantized_input);
    return 0;
}

IMMER_EXPORT int immer_q4_linear_group_f32(
    const float *input,
    int64_t input_rows,
    int64_t input_cols,
    const uint8_t * const *weights,
    const int *formats,
    const int64_t *output_rows,
    float * const *outputs,
    int tensor_count,
    int threads
) {
    if (
        !input || !weights || !formats || !output_rows || !outputs
        || input_rows <= 0 || input_cols <= 0 || input_cols % IMMER_QK != 0
        || tensor_count <= 0 || threads <= 0
    ) return 1;
    int64_t total_output_rows = 0;
    int64_t *prefix = (int64_t *) malloc(
        (size_t) (tensor_count + 1) * sizeof(int64_t)
    );
    if (!prefix) return 3;
    prefix[0] = 0;
    for (int tensor = 0; tensor < tensor_count; ++tensor) {
        if (
            !weights[tensor] || !outputs[tensor] || output_rows[tensor] <= 0
            || immer_q4_row_bytes(formats[tensor], input_cols) <= 0
        ) {
            free(prefix);
            return 1;
        }
        total_output_rows += output_rows[tensor];
        prefix[tensor + 1] = total_output_rows;
    }
    const int64_t blocks = input_cols / IMMER_QK;
    immer_block_q8_0 *quantized_input = (immer_block_q8_0 *) malloc(
        (size_t) input_rows * (size_t) blocks * sizeof(immer_block_q8_0)
    );
    if (!quantized_input) {
        free(prefix);
        return 3;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t row = 0; row < input_rows; ++row) {
        immer_quantize_q8_row(
            input + row * input_cols,
            quantized_input + row * blocks,
            input_cols
        );
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t global_row = 0; global_row < total_output_rows; ++global_row) {
        int tensor = 0;
        while (global_row >= prefix[tensor + 1]) ++tensor;
        const int64_t local_row = global_row - prefix[tensor];
        const int64_t row_bytes = immer_q4_row_bytes(
            formats[tensor], input_cols
        );
        const uint8_t *weight_row = weights[tensor] + local_row * row_bytes;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            const immer_block_q8_0 *active_input =
                quantized_input + input_row * blocks;
            outputs[tensor][input_row * output_rows[tensor] + local_row] =
                immer_dot_packed_q8(
                    weight_row, formats[tensor], active_input, blocks
                );
        }
    }
    free(quantized_input);
    free(prefix);
    return 0;
}
