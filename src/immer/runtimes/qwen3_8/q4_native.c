#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

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

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

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

#define IMMER_Q4_ABI 6u
#define IMMER_QK 32
#define IMMER_MLP_PAGE_NEURONS 64
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
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
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
#endif
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

static float immer_round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x00007fffu + ((bits >> 16) & 1u);
    } else if (bits & 0x007fffffu) {
        bits |= 0x00400000u;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float immer_bf16_bits_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline void immer_quantize_q8_block(
    const float *values,
    immer_block_q8_0 *out,
    double *energy
) {
    float absolute_max = 0.0f;
    for (int index = 0; index < IMMER_QK; ++index) {
        const float value = values[index];
        const float absolute = fabsf(value);
        if (absolute > absolute_max) absolute_max = absolute;
        if (energy) *energy += (double) value * (double) value;
    }
    const float scale = absolute_max / 127.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    out->d = immer_float_to_half(scale);
    for (int index = 0; index < IMMER_QK; ++index) {
        int quantized = (int) roundf(values[index] * inverse);
        if (quantized < -127) quantized = -127;
        if (quantized > 127) quantized = 127;
        out->qs[index] = (int8_t) quantized;
    }
}

static void immer_quantize_q8_row(
    const float *x,
    immer_block_q8_0 *out,
    int64_t cols
) {
    const int64_t blocks = cols / IMMER_QK;
    for (int64_t block = 0; block < blocks; ++block) {
        immer_quantize_q8_block(
            x + block * IMMER_QK,
            out + block,
            NULL
        );
    }
}

typedef struct {
    double score;
    int64_t page;
} immer_mlp_page_energy;

static int immer_compare_mlp_page_energy(const void *left, const void *right) {
    const immer_mlp_page_energy *a = (const immer_mlp_page_energy *) left;
    const immer_mlp_page_energy *b = (const immer_mlp_page_energy *) right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->page < b->page) return -1;
    if (a->page > b->page) return 1;
    return 0;
}

static int immer_quantize_q8_row_with_page_topk(
    const float *x,
    immer_block_q8_0 *out,
    int64_t cols,
    int64_t k,
    int64_t *top_page_ids,
    double *top_page_scores,
    double *total_page_score,
    immer_mlp_page_energy *energies
) {
    const int64_t blocks = cols / IMMER_QK;
    const int64_t page_count =
        (blocks - 1) / (IMMER_MLP_PAGE_NEURONS / IMMER_QK) + 1;
    int64_t page = 0;
    double page_energy = 0.0;
    for (int64_t block = 0; block < blocks; ++block) {
        immer_quantize_q8_block(
            x + block * IMMER_QK,
            out + block,
            &page_energy
        );
        if (
            (block + 1) % (IMMER_MLP_PAGE_NEURONS / IMMER_QK) != 0
            && block + 1 != blocks
        ) continue;
        if (!isfinite(page_energy)) return 0;
        energies[page].score = page_energy;
        energies[page].page = page;
        ++page;
        page_energy = 0.0;
    }
    if (page != page_count) return 0;
    double total_energy = 0.0;
    for (int64_t index = 0; index < page_count; ++index) {
        total_energy += energies[index].score;
    }
    if (!isfinite(total_energy)) return 0;
    *total_page_score = total_energy;
    qsort(
        energies,
        (size_t) page_count,
        sizeof(immer_mlp_page_energy),
        immer_compare_mlp_page_energy
    );
    for (int64_t index = 0; index < k; ++index) {
        top_page_ids[index] = energies[index].page;
        top_page_scores[index] = energies[index].score;
    }
    return 1;
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

#if defined(__AVX2__)
static float immer_dot_selected_q4_q8_avx2(
    const immer_block_q4_0 *weight,
    const immer_block_q8_0 *input,
    const int64_t *block_ids,
    int64_t selected_blocks
) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int64_t selected = 0; selected < selected_blocks; ++selected) {
        const immer_block_q4_0 *weight_block = weight + block_ids[selected];
        __m256i qweight = immer_unpack_nibbles(weight_block->qs);
        qweight = _mm256_sub_epi8(qweight, _mm256_set1_epi8(8));
        const __m256i qinput = _mm256_loadu_si256(
            (const __m256i *) input[selected].qs
        );
        const float scale = immer_half_to_float(weight_block->d)
            * immer_half_to_float(input[selected].d);
        accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(scale),
            immer_mul_sum_i8_pairs_float(qweight, qinput),
            accumulator
        );
    }
    return immer_horizontal_sum(accumulator);
}

static float immer_dot_selected_q8_q8_avx2(
    const immer_block_q8_0 *weight,
    const immer_block_q8_0 *input,
    const int64_t *block_ids,
    int64_t selected_blocks
) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int64_t selected = 0; selected < selected_blocks; ++selected) {
        const immer_block_q8_0 *weight_block = weight + block_ids[selected];
        const __m256i qweight = _mm256_loadu_si256(
            (const __m256i *) weight_block->qs
        );
        const __m256i qinput = _mm256_loadu_si256(
            (const __m256i *) input[selected].qs
        );
        const float scale = immer_half_to_float(weight_block->d)
            * immer_half_to_float(input[selected].d);
        accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(scale),
            immer_mul_sum_i8_pairs_float(qweight, qinput),
            accumulator
        );
    }
    return immer_horizontal_sum(accumulator);
}
#endif

static float immer_dot_selected_q4_q8_scalar(
    const immer_block_q4_0 *weight,
    const immer_block_q8_0 *input,
    const int64_t *block_ids,
    int64_t selected_blocks
) {
    float sum = 0.0f;
    for (int64_t selected = 0; selected < selected_blocks; ++selected) {
        const immer_block_q4_0 *weight_block = weight + block_ids[selected];
        int32_t integer_sum = 0;
        for (int index = 0; index < 16; ++index) {
            const uint8_t packed = weight_block->qs[index];
            integer_sum += ((int32_t) (packed & 0x0f) - 8)
                * (int32_t) input[selected].qs[index];
            integer_sum += ((int32_t) (packed >> 4) - 8)
                * (int32_t) input[selected].qs[index + 16];
        }
        sum += (float) integer_sum
            * immer_half_to_float(weight_block->d)
            * immer_half_to_float(input[selected].d);
    }
    return sum;
}

static float immer_dot_selected_q8_q8_scalar(
    const immer_block_q8_0 *weight,
    const immer_block_q8_0 *input,
    const int64_t *block_ids,
    int64_t selected_blocks
) {
    float sum = 0.0f;
    for (int64_t selected = 0; selected < selected_blocks; ++selected) {
        const immer_block_q8_0 *weight_block = weight + block_ids[selected];
        int32_t integer_sum = 0;
        for (int index = 0; index < 32; ++index) {
            integer_sum += (int32_t) weight_block->qs[index]
                * (int32_t) input[selected].qs[index];
        }
        sum += (float) integer_sum
            * immer_half_to_float(weight_block->d)
            * immer_half_to_float(input[selected].d);
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

static float immer_dot_selected_packed_q8(
    const uint8_t *weight,
    int format,
    const immer_block_q8_0 *input,
    const int64_t *block_ids,
    int64_t selected_blocks
) {
    if (format == IMMER_FORMAT_Q4_0) {
#if defined(__AVX2__)
        return immer_dot_selected_q4_q8_avx2(
            (const immer_block_q4_0 *) weight,
            input,
            block_ids,
            selected_blocks
        );
#else
        return immer_dot_selected_q4_q8_scalar(
            (const immer_block_q4_0 *) weight,
            input,
            block_ids,
            selected_blocks
        );
#endif
    }
#if defined(__AVX2__)
    return immer_dot_selected_q8_q8_avx2(
        (const immer_block_q8_0 *) weight,
        input,
        block_ids,
        selected_blocks
    );
#else
    return immer_dot_selected_q8_q8_scalar(
        (const immer_block_q8_0 *) weight,
        input,
        block_ids,
        selected_blocks
    );
#endif
}

static float immer_dot_sparse_q4_f32(
    const immer_block_q4_0 *weight,
    const float *values,
    const int64_t *coordinates,
    int64_t sparse_count
) {
    float sum = 0.0f;
    for (int64_t sparse = 0; sparse < sparse_count; ++sparse) {
        const int64_t coordinate = coordinates[sparse];
        const immer_block_q4_0 *weight_block =
            weight + coordinate / IMMER_QK;
        const int index = (int) (coordinate % IMMER_QK);
        const int packed_index = index < IMMER_QK / 2
            ? index
            : index - IMMER_QK / 2;
        const uint8_t packed = weight_block->qs[packed_index];
        const int quantized = index < IMMER_QK / 2
            ? (int) (packed & 0x0f) - 8
            : (int) (packed >> 4) - 8;
        sum += values[sparse] * (float) quantized
            * immer_half_to_float(weight_block->d);
    }
    return sum;
}

static float immer_dot_sparse_q8_f32(
    const immer_block_q8_0 *weight,
    const float *values,
    const int64_t *coordinates,
    int64_t sparse_count
) {
    float sum = 0.0f;
    for (int64_t sparse = 0; sparse < sparse_count; ++sparse) {
        const int64_t coordinate = coordinates[sparse];
        const immer_block_q8_0 *weight_block =
            weight + coordinate / IMMER_QK;
        const int index = (int) (coordinate % IMMER_QK);
        sum += values[sparse] * (float) weight_block->qs[index]
            * immer_half_to_float(weight_block->d);
    }
    return sum;
}

static float immer_dot_sparse_packed_f32(
    const uint8_t *weight,
    int format,
    const float *values,
    const int64_t *coordinates,
    int64_t sparse_count
) {
    if (format == IMMER_FORMAT_Q4_0) {
        return immer_dot_sparse_q4_f32(
            (const immer_block_q4_0 *) weight,
            values,
            coordinates,
            sparse_count
        );
    }
    return immer_dot_sparse_q8_f32(
        (const immer_block_q8_0 *) weight,
        values,
        coordinates,
        sparse_count
    );
}

static int64_t immer_discard_read_pages(const uint8_t *start, size_t length) {
#if defined(_WIN32)
    (void) start;
    (void) length;
    return 0;
#else
    if (!start || length == 0) return 0;
    long raw_page = sysconf(_SC_PAGESIZE);
    if (raw_page <= 0) return 0;
    const uintptr_t page = (uintptr_t) raw_page;
    const uintptr_t begin = (uintptr_t) start;
    const uintptr_t aligned = begin - begin % page;
    if (length > (size_t) (UINTPTR_MAX - begin)) return 0;
    const uintptr_t end = begin + (uintptr_t) length;
    if (end > UINTPTR_MAX - (page - 1u)) return 0;
    const uintptr_t aligned_end = ((end + page - 1u) / page) * page;
    const size_t aligned_length = (size_t) (aligned_end - aligned);
    return madvise((void *) aligned, aligned_length, MADV_DONTNEED) == 0
        ? (int64_t) length
        : 0;
#endif
}

static float immer_silu_f32(float value) {
    if (value >= 0.0f) {
        return value / (1.0f + expf(-value));
    }
    const float exponent = expf(value);
    return value * exponent / (1.0f + exponent);
}

static float immer_sigmoid_f32(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float exponent = expf(value);
    return exponent / (1.0f + exponent);
}

static float immer_softplus_f32(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

static double immer_route_feature(float gate, float up) {
    double clipped = (double) gate;
    if (clipped < -60.0) clipped = -60.0;
    if (clipped > 60.0) clipped = 60.0;
    const double activation = (
        (double) gate / (1.0 + exp(-clipped))
    ) * (double) up;
    return activation * activation;
}

static int immer_size_product_fits(
    int64_t first,
    int64_t second,
    size_t element_size
) {
    if (first < 0 || second < 0 || element_size == 0) return 0;
    if (first == 0 || second == 0) return 1;
    const size_t maximum_elements = (size_t) -1 / element_size;
    return (uint64_t) first
        <= (uint64_t) maximum_elements / (uint64_t) second;
}

static int immer_size_product3_fits(
    int64_t first,
    int64_t second,
    int64_t third,
    size_t element_size
) {
    if (first < 0 || second < 0 || third < 0 || element_size == 0) return 0;
    if (first == 0 || second == 0 || third == 0) return 1;
    const uint64_t maximum_elements =
        (uint64_t) ((size_t) -1 / element_size);
    if (
        (uint64_t) first
        > maximum_elements / (uint64_t) second
    ) return 0;
    const uint64_t first_second =
        (uint64_t) first * (uint64_t) second;
    return (uint64_t) third <= maximum_elements / first_second;
}

static int immer_byte_ranges_overlap(
    const void *left,
    size_t left_bytes,
    const void *right,
    size_t right_bytes
) {
    const uintptr_t left_start = (uintptr_t) left;
    const uintptr_t right_start = (uintptr_t) right;
    if (
        left_bytes > UINTPTR_MAX - left_start
        || right_bytes > UINTPTR_MAX - right_start
    ) return 1;
    return left_start < right_start + right_bytes
        && right_start < left_start + left_bytes;
}

static int immer_checked_row_layout(
    int format,
    int64_t cols,
    int64_t *blocks,
    size_t *row_bytes
) {
    if (!blocks || !row_bytes || cols <= 0 || cols % IMMER_QK != 0) return 0;
    size_t block_bytes;
    if (format == IMMER_FORMAT_Q4_0) {
        block_bytes = sizeof(immer_block_q4_0);
    } else if (format == IMMER_FORMAT_Q8_0) {
        block_bytes = sizeof(immer_block_q8_0);
    } else {
        return 0;
    }
    const int64_t block_count = cols / IMMER_QK;
    if ((uint64_t) block_count > (uint64_t) ((size_t) -1 / block_bytes)) {
        return 0;
    }
    *blocks = block_count;
    *row_bytes = (size_t) block_count * block_bytes;
    return 1;
}

static int immer_f32_values_are_finite(const float *values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
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

IMMER_EXPORT int immer_q4_topk_bf16_f32(
    const float *input,
    int64_t input_rows,
    int64_t input_cols,
    const uint8_t *weights,
    int format,
    int64_t output_rows,
    int64_t k,
    int64_t block_rows,
    float *top_values,
    int64_t *top_ids,
    int64_t *discarded_bytes_out,
    int64_t *discard_calls_out,
    int discard_consumed,
    int threads
) {
    int64_t packed_blocks;
    size_t row_bytes;
    if (
        !input || !weights || !top_values || !top_ids
        || !discarded_bytes_out || !discard_calls_out
        || input_rows <= 0 || input_cols <= 0 || output_rows <= 0
        || k <= 0 || k > output_rows || k > 256
        || block_rows <= 0 || threads <= 0
        || (discard_consumed != 0 && discard_consumed != 1)
        || !immer_checked_row_layout(
            format, input_cols, &packed_blocks, &row_bytes
        )
        || !immer_size_product_fits(
            input_rows, packed_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(input_rows, k, sizeof(float))
        || !immer_size_product_fits(input_rows, k, sizeof(int64_t))
        || !immer_size_product_fits(
            input_rows,
            block_rows < output_rows ? block_rows : output_rows,
            sizeof(float)
        )
    ) return 1;
    const size_t input_count = (size_t) input_rows * (size_t) input_cols;
    if (!immer_f32_values_are_finite(input, input_count)) return 2;

    const size_t quantized_count =
        (size_t) input_rows * (size_t) packed_blocks;
    const int64_t chunk_capacity = block_rows < output_rows
        ? block_rows
        : output_rows;
    immer_block_q8_0 *quantized = (immer_block_q8_0 *) malloc(
        quantized_count * sizeof(immer_block_q8_0)
    );
    float *chunk = (float *) malloc(
        (size_t) input_rows * (size_t) chunk_capacity * sizeof(float)
    );
    if (!quantized || !chunk) {
        free(quantized);
        free(chunk);
        return 3;
    }

    for (int64_t row = 0; row < input_rows; ++row) {
        immer_quantize_q8_row(
            input + (size_t) row * (size_t) input_cols,
            quantized + (size_t) row * (size_t) packed_blocks,
            input_cols
        );
        for (int64_t block = 0; block < packed_blocks; ++block) {
            if (!isfinite(immer_half_to_float(
                quantized[(size_t) row * (size_t) packed_blocks + block].d
            ))) {
                free(quantized);
                free(chunk);
                return 2;
            }
        }
    }
    for (int64_t row = 0; row < input_rows; ++row) {
        for (int64_t index = 0; index < k; ++index) {
            top_values[(size_t) row * (size_t) k + index] = -INFINITY;
            top_ids[(size_t) row * (size_t) k + index] = INT64_MAX;
        }
    }
    *discarded_bytes_out = 0;
    *discard_calls_out = 0;
    int numeric_error = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        for (int64_t start = 0; start < output_rows; start += block_rows) {
            const int64_t count = output_rows - start < block_rows
                ? output_rows - start
                : block_rows;
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t local = 0; local < count; ++local) {
                const int64_t output_row = start + local;
                const uint8_t *weight_row = weights
                    + (size_t) output_row * row_bytes;
                for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                    const immer_block_q8_0 *active_input = quantized
                        + (size_t) input_row * (size_t) packed_blocks;
                    float value = immer_dot_packed_q8(
                        weight_row, format, active_input, packed_blocks
                    );
                    value = immer_round_bf16(value);
                    if (!isfinite(value)) numeric_error = 1;
                    chunk[(size_t) input_row * (size_t) chunk_capacity + local] = value;
                }
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                if (!numeric_error) {
                    for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                        float *active_values = top_values
                            + (size_t) input_row * (size_t) k;
                        int64_t *active_ids = top_ids
                            + (size_t) input_row * (size_t) k;
                        for (int64_t local = 0; local < count; ++local) {
                            const float value = chunk[
                                (size_t) input_row * (size_t) chunk_capacity + local
                            ];
                            const int64_t token_id = start + local;
                            int64_t insert = 0;
                            while (
                                insert < k
                                && (
                                    active_values[insert] > value
                                    || (
                                        active_values[insert] == value
                                        && active_ids[insert] < token_id
                                    )
                                )
                            ) ++insert;
                            if (insert < k) {
                                for (
                                    int64_t position = k - 1;
                                    position > insert;
                                    --position
                                ) {
                                    active_values[position] = active_values[position - 1];
                                    active_ids[position] = active_ids[position - 1];
                                }
                                active_values[insert] = value;
                                active_ids[insert] = token_id;
                            }
                        }
                    }
                    if (discard_consumed) {
                        const size_t consumed = (size_t) count * row_bytes;
                        const int64_t discarded = immer_discard_read_pages(
                            weights + (size_t) start * row_bytes,
                            consumed
                        );
                        if (discarded > 0) {
                            *discarded_bytes_out += discarded;
                            *discard_calls_out += 1;
                        }
                    }
                }
            }
        }
    }
    free(quantized);
    free(chunk);
    return numeric_error ? 2 : 0;
}

IMMER_EXPORT int immer_q4_linear_rows_f32(
    const float *input,
    int64_t input_rows,
    int64_t input_cols,
    const uint8_t *weights,
    int format,
    int64_t output_rows,
    const int64_t *row_ids,
    int64_t selected_output_rows,
    float *output,
    int threads
) {
    int64_t blocks;
    size_t row_bytes;
    if (
        !input || !weights || !row_ids || !output
        || input_rows <= 0 || output_rows <= 0 || selected_output_rows < 0
        || threads <= 0
        || !immer_checked_row_layout(
            format, input_cols, &blocks, &row_bytes
        )
        || !immer_size_product_fits(
            input_rows, input_cols, sizeof(float)
        )
        || !immer_size_product_fits(
            output_rows, 1, row_bytes
        )
        || !immer_size_product_fits(
            selected_output_rows, 1, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, selected_output_rows, sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, blocks, sizeof(immer_block_q8_0)
        )
    ) return 1;
    for (int64_t selected = 0; selected < selected_output_rows; ++selected) {
        if (row_ids[selected] < 0 || row_ids[selected] >= output_rows) return 2;
    }
    if (selected_output_rows == 0) return 0;
    const size_t input_value_count =
        (size_t) input_rows * (size_t) input_cols;
    if (!immer_f32_values_are_finite(input, input_value_count)) return 1;
    const size_t quantized_blocks = (size_t) input_rows * (size_t) blocks;
    immer_block_q8_0 *quantized_input = (immer_block_q8_0 *) malloc(
        quantized_blocks * sizeof(immer_block_q8_0)
    );
    if (!quantized_input) return 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
        immer_quantize_q8_row(
            input + (size_t) input_row * (size_t) input_cols,
            quantized_input + (size_t) input_row * (size_t) blocks,
            input_cols
        );
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t selected = 0; selected < selected_output_rows; ++selected) {
        const uint8_t *weight_row = weights
            + (size_t) row_ids[selected] * row_bytes;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            const immer_block_q8_0 *active_input = quantized_input
                + (size_t) input_row * (size_t) blocks;
            output[
                (size_t) input_row * (size_t) selected_output_rows
                + (size_t) selected
            ] = immer_dot_packed_q8(
                weight_row, format, active_input, blocks
            );
        }
    }
    free(quantized_input);
    return 0;
}

IMMER_EXPORT int immer_q4_linear_rows_pair_f32(
    const float *input,
    int64_t input_rows,
    int64_t input_cols,
    const uint8_t *weights_a,
    int format_a,
    int64_t output_rows_a,
    const uint8_t *weights_b,
    int format_b,
    int64_t output_rows_b,
    const int64_t *row_ids,
    int64_t selected_rows,
    float *output_a,
    float *output_b,
    int threads
) {
    int64_t blocks_a;
    int64_t blocks_b;
    size_t row_bytes_a;
    size_t row_bytes_b;
    if (
        !input || !weights_a || !weights_b || !row_ids
        || !output_a || !output_b
        || input_rows <= 0 || output_rows_a <= 0 || output_rows_b <= 0
        || selected_rows < 0 || threads <= 0
        || !immer_checked_row_layout(
            format_a, input_cols, &blocks_a, &row_bytes_a
        )
        || !immer_checked_row_layout(
            format_b, input_cols, &blocks_b, &row_bytes_b
        )
        || blocks_a != blocks_b
        || !immer_size_product_fits(
            input_rows, input_cols, sizeof(float)
        )
        || !immer_size_product_fits(
            output_rows_a, 1, row_bytes_a
        )
        || !immer_size_product_fits(
            output_rows_b, 1, row_bytes_b
        )
        || !immer_size_product_fits(
            selected_rows, 1, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, selected_rows, sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, blocks_a, sizeof(immer_block_q8_0)
        )
    ) return 1;
    for (int64_t selected = 0; selected < selected_rows; ++selected) {
        if (
            row_ids[selected] < 0
            || row_ids[selected] >= output_rows_a
            || row_ids[selected] >= output_rows_b
        ) return 2;
    }
    if (selected_rows == 0) return 0;
    const size_t input_value_count =
        (size_t) input_rows * (size_t) input_cols;
    if (!immer_f32_values_are_finite(input, input_value_count)) return 1;
    const size_t quantized_blocks =
        (size_t) input_rows * (size_t) blocks_a;
    immer_block_q8_0 *quantized_input = (immer_block_q8_0 *) malloc(
        quantized_blocks * sizeof(immer_block_q8_0)
    );
    if (!quantized_input) return 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
        immer_quantize_q8_row(
            input + (size_t) input_row * (size_t) input_cols,
            quantized_input + (size_t) input_row * (size_t) blocks_a,
            input_cols
        );
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t selected = 0; selected < selected_rows; ++selected) {
        const size_t weight_row_id = (size_t) row_ids[selected];
        const uint8_t *weight_row_a = weights_a
            + weight_row_id * row_bytes_a;
        const uint8_t *weight_row_b = weights_b
            + weight_row_id * row_bytes_b;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            const immer_block_q8_0 *active_input = quantized_input
                + (size_t) input_row * (size_t) blocks_a;
            const size_t output_offset =
                (size_t) input_row * (size_t) selected_rows
                + (size_t) selected;
            output_a[output_offset] = immer_dot_packed_q8(
                weight_row_a, format_a, active_input, blocks_a
            );
            output_b[output_offset] = immer_dot_packed_q8(
                weight_row_b, format_b, active_input, blocks_a
            );
        }
    }
    free(quantized_input);
    return 0;
}

IMMER_EXPORT int immer_q4_linear_selected_blocks_f32(
    const float *input_blocks,
    int64_t input_rows,
    int64_t selected_blocks,
    const int64_t *block_ids,
    int64_t total_input_cols,
    const uint8_t *weights,
    int format,
    int64_t output_rows,
    float *output,
    int threads
) {
    int64_t total_blocks;
    size_t row_bytes;
    if (
        !input_blocks || !block_ids || !weights || !output
        || input_rows <= 0 || selected_blocks <= 0 || output_rows <= 0
        || threads <= 0
        || !immer_checked_row_layout(
            format, total_input_cols, &total_blocks, &row_bytes
        )
        || selected_blocks > total_blocks
        || !immer_size_product_fits(
            input_rows,
            selected_blocks,
            (size_t) IMMER_QK * sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, selected_blocks, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, selected_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            output_rows, 1, row_bytes
        )
        || !immer_size_product_fits(
            input_rows, output_rows, sizeof(float)
        )
    ) return 1;
    const size_t compact_block_count =
        (size_t) input_rows * (size_t) selected_blocks;
    for (size_t index = 0; index < compact_block_count; ++index) {
        if (block_ids[index] < 0 || block_ids[index] >= total_blocks) return 2;
    }
    const size_t compact_value_count =
        compact_block_count * (size_t) IMMER_QK;
    if (!immer_f32_values_are_finite(input_blocks, compact_value_count)) return 1;
    immer_block_q8_0 *quantized_input = (immer_block_q8_0 *) malloc(
        compact_block_count * sizeof(immer_block_q8_0)
    );
    if (!quantized_input) return 3;
    const int64_t compact_cols = selected_blocks * IMMER_QK;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
        immer_quantize_q8_row(
            input_blocks
                + (size_t) input_row * (size_t) compact_cols,
            quantized_input
                + (size_t) input_row * (size_t) selected_blocks,
            compact_cols
        );
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint8_t *weight_row = weights
            + (size_t) output_row * row_bytes;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            const size_t compact_offset =
                (size_t) input_row * (size_t) selected_blocks;
            output[
                (size_t) input_row * (size_t) output_rows
                + (size_t) output_row
            ] = immer_dot_selected_packed_q8(
                weight_row,
                format,
                quantized_input + compact_offset,
                block_ids + compact_offset,
                selected_blocks
            );
        }
    }
    free(quantized_input);
    return 0;
}

IMMER_EXPORT int immer_q4_linear_routed_f32(
    const float *full_block_values,
    int64_t input_rows,
    int64_t full_blocks,
    const int64_t *full_block_ids,
    const float *sparse_values,
    const int64_t *sparse_coords,
    int64_t sparse_count,
    int64_t total_input_cols,
    const uint8_t *weights,
    int format,
    int64_t output_rows,
    float *output,
    int threads
) {
    int64_t total_blocks;
    size_t row_bytes;
    if (
        !weights || !output
        || input_rows <= 0 || full_blocks < 0 || sparse_count < 0
        || output_rows <= 0 || threads <= 0
        || (full_blocks > 0 && (!full_block_values || !full_block_ids))
        || (sparse_count > 0 && (!sparse_values || !sparse_coords))
        || !immer_checked_row_layout(
            format, total_input_cols, &total_blocks, &row_bytes
        )
        || full_blocks > total_blocks
        || sparse_count > total_input_cols
        || !immer_size_product_fits(
            input_rows,
            full_blocks,
            (size_t) IMMER_QK * sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, full_blocks, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, full_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            input_rows, sparse_count, sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, sparse_count, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            output_rows, 1, row_bytes
        )
        || !immer_size_product_fits(
            input_rows, output_rows, sizeof(float)
        )
    ) return 1;
    const size_t full_block_count =
        (size_t) input_rows * (size_t) full_blocks;
    for (size_t index = 0; index < full_block_count; ++index) {
        if (
            full_block_ids[index] < 0
            || full_block_ids[index] >= total_blocks
        ) return 2;
    }
    const size_t sparse_value_count =
        (size_t) input_rows * (size_t) sparse_count;
    for (size_t index = 0; index < sparse_value_count; ++index) {
        if (sparse_coords[index] < 0 || sparse_coords[index] >= total_input_cols) {
            return 2;
        }
    }
    if (
        full_blocks > 0
        && !immer_f32_values_are_finite(
            full_block_values,
            full_block_count * (size_t) IMMER_QK
        )
    ) return 1;
    if (
        sparse_count > 0
        && !immer_f32_values_are_finite(sparse_values, sparse_value_count)
    ) return 1;
    immer_block_q8_0 *quantized_full = NULL;
    if (full_blocks > 0) {
        quantized_full = (immer_block_q8_0 *) malloc(
            full_block_count * sizeof(immer_block_q8_0)
        );
        if (!quantized_full) return 3;
        const int64_t compact_cols = full_blocks * IMMER_QK;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            immer_quantize_q8_row(
                full_block_values
                    + (size_t) input_row * (size_t) compact_cols,
                quantized_full
                    + (size_t) input_row * (size_t) full_blocks,
                compact_cols
            );
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint8_t *weight_row = weights
            + (size_t) output_row * row_bytes;
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            float sum = 0.0f;
            if (full_blocks > 0) {
                const size_t full_offset =
                    (size_t) input_row * (size_t) full_blocks;
                sum = immer_dot_selected_packed_q8(
                    weight_row,
                    format,
                    quantized_full + full_offset,
                    full_block_ids + full_offset,
                    full_blocks
                );
            }
            if (sparse_count > 0) {
                const size_t sparse_offset =
                    (size_t) input_row * (size_t) sparse_count;
                sum += immer_dot_sparse_packed_f32(
                    weight_row,
                    format,
                    sparse_values + sparse_offset,
                    sparse_coords + sparse_offset,
                    sparse_count
                );
            }
            output[
                (size_t) input_row * (size_t) output_rows
                + (size_t) output_row
            ] = sum;
        }
    }
    free(quantized_full);
    return 0;
}

IMMER_EXPORT int immer_q4_sparse_mlp_f32(
    const float *input,
    int64_t input_rows,
    int64_t hidden_cols,
    const uint8_t *gate_weights,
    int gate_format,
    const uint8_t *up_weights,
    int up_format,
    const uint8_t *down_weights,
    int down_format,
    int64_t intermediate_cols,
    int64_t output_rows,
    const int64_t *pilot_ids,
    int64_t block_count,
    int64_t block_size,
    int64_t pilot_count,
    const double *coefficients,
    int64_t selected_block_count,
    const float *affine_scale,
    const float *affine_bias,
    float *output,
    int64_t *selected_blocks_out,
    int threads
) {
    if (
        !input || !gate_weights || !up_weights || !down_weights
        || !pilot_ids || !coefficients || !affine_scale || !affine_bias
        || !output || !selected_blocks_out
        || input_rows <= 0 || hidden_cols <= 0 || intermediate_cols <= 0
        || output_rows <= 0 || block_count <= 0 || block_size <= 0
        || pilot_count <= 0 || pilot_count > block_size
        || selected_block_count <= 0 || selected_block_count >= block_count
        || block_size % IMMER_QK != 0 || threads <= 0
        || block_count > INT64_MAX / block_size
        || intermediate_cols != block_count * block_size
    ) return 1;

    int64_t hidden_blocks_gate;
    int64_t hidden_blocks_up;
    int64_t down_blocks;
    size_t gate_row_bytes;
    size_t up_row_bytes;
    size_t down_row_bytes;
    if (
        !immer_checked_row_layout(
            gate_format, hidden_cols, &hidden_blocks_gate, &gate_row_bytes
        )
        || !immer_checked_row_layout(
            up_format, hidden_cols, &hidden_blocks_up, &up_row_bytes
        )
        || !immer_checked_row_layout(
            down_format, intermediate_cols, &down_blocks, &down_row_bytes
        )
        || hidden_blocks_gate != hidden_blocks_up
        || down_blocks != intermediate_cols / IMMER_QK
    ) return 1;

    const int64_t packed_blocks_per_logical = block_size / IMMER_QK;
    if (
        selected_block_count > INT64_MAX / packed_blocks_per_logical
    ) return 1;
    const int64_t selected_packed_blocks =
        selected_block_count * packed_blocks_per_logical;
    if (
        !immer_size_product_fits(
            input_rows, hidden_cols, sizeof(float)
        )
        || !immer_size_product_fits(
            intermediate_cols, 1, gate_row_bytes
        )
        || !immer_size_product_fits(
            intermediate_cols, 1, up_row_bytes
        )
        || !immer_size_product_fits(
            output_rows, 1, down_row_bytes
        )
        || !immer_size_product_fits(
            block_count, pilot_count, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            block_count, pilot_count + 1, sizeof(double)
        )
        || !immer_size_product_fits(
            input_rows, hidden_blocks_gate, sizeof(immer_block_q8_0)
        )
        || !immer_size_product3_fits(
            input_rows, block_count, pilot_count, sizeof(float)
        )
        || !immer_size_product3_fits(
            input_rows, block_count, pilot_count, sizeof(double)
        )
        || !immer_size_product_fits(
            input_rows, selected_block_count, sizeof(double)
        )
        || !immer_size_product_fits(
            input_rows, selected_block_count, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, selected_packed_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            input_rows, selected_packed_blocks, sizeof(int64_t)
        )
        || !immer_size_product3_fits(
            input_rows, selected_block_count, block_size, sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, output_rows, sizeof(float)
        )
    ) return 1;

    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t first_neuron = block * block_size;
        const int64_t last_neuron = first_neuron + block_size;
        const int64_t *block_pilots = pilot_ids + block * pilot_count;
        for (int64_t pilot = 0; pilot < pilot_count; ++pilot) {
            const int64_t neuron = block_pilots[pilot];
            if (neuron < first_neuron || neuron >= last_neuron) return 2;
            for (int64_t previous = 0; previous < pilot; ++previous) {
                if (block_pilots[previous] == neuron) return 2;
            }
        }
    }
    const size_t input_value_count =
        (size_t) input_rows * (size_t) hidden_cols;
    if (!immer_f32_values_are_finite(input, input_value_count)) return 1;
    const size_t coefficient_count =
        (size_t) block_count * (size_t) (pilot_count + 1);
    for (size_t index = 0; index < coefficient_count; ++index) {
        if (!isfinite(coefficients[index])) return 1;
    }
    for (int64_t row = 0; row < output_rows; ++row) {
        if (!isfinite(affine_scale[row]) || !isfinite(affine_bias[row])) return 1;
    }

    const size_t quantized_hidden_count =
        (size_t) input_rows * (size_t) hidden_blocks_gate;
    const size_t pilot_activation_count =
        (size_t) input_rows * (size_t) block_count * (size_t) pilot_count;
    const size_t selected_score_count =
        (size_t) input_rows * (size_t) selected_block_count;
    const size_t selected_packed_count =
        (size_t) input_rows * (size_t) selected_packed_blocks;
    const size_t selected_activation_count =
        (size_t) input_rows
        * (size_t) selected_block_count
        * (size_t) block_size;
    immer_block_q8_0 *quantized_hidden = (immer_block_q8_0 *) malloc(
        quantized_hidden_count * sizeof(immer_block_q8_0)
    );
    if (!quantized_hidden) return 3;
    float *pilot_activations = NULL;
    double *pilot_features = NULL;
    double *selected_scores = NULL;
    immer_block_q8_0 *quantized_activations = NULL;
    int64_t *selected_down_blocks = NULL;
    float *selected_activations = NULL;
    int numeric_error = 0;
    int allocation_error = 0;
    /* Implicit worksharing/single barriers protect each compact stage. */
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
            immer_block_q8_0 *target = quantized_hidden
                + (size_t) input_row * (size_t) hidden_blocks_gate;
            immer_quantize_q8_row(
                input + (size_t) input_row * (size_t) hidden_cols,
                target,
                hidden_cols
            );
            for (int64_t block = 0; block < hidden_blocks_gate; ++block) {
                if (!isfinite(immer_half_to_float(target[block].d))) {
                    numeric_error = 1;
                }
            }
        }
#ifdef _OPENMP
#pragma omp single
#endif
        {
            if (!numeric_error) {
                pilot_activations = (float *) malloc(
                    pilot_activation_count * sizeof(float)
                );
                pilot_features = (double *) malloc(
                    pilot_activation_count * sizeof(double)
                );
                selected_scores = (double *) malloc(
                    selected_score_count * sizeof(double)
                );
                if (
                    !pilot_activations
                    || !pilot_features
                    || !selected_scores
                ) allocation_error = 1;
            }
        }

        if (!numeric_error && !allocation_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (
                int64_t task = 0;
                task < (int64_t) pilot_activation_count;
                ++task
            ) {
                const int64_t pilot = task % pilot_count;
                const int64_t row_block = task / pilot_count;
                const int64_t block = row_block % block_count;
                const int64_t input_row = row_block / block_count;
                const immer_block_q8_0 *active_hidden = quantized_hidden
                    + (size_t) input_row * (size_t) hidden_blocks_gate;
                const int64_t neuron = pilot_ids[
                    (size_t) block * (size_t) pilot_count + (size_t) pilot
                ];
                float gate = immer_dot_packed_q8(
                    gate_weights + (size_t) neuron * gate_row_bytes,
                    gate_format,
                    active_hidden,
                    hidden_blocks_gate
                );
                float up = immer_dot_packed_q8(
                    up_weights + (size_t) neuron * up_row_bytes,
                    up_format,
                    active_hidden,
                    hidden_blocks_gate
                );
                if (!isfinite(gate) || !isfinite(up)) {
                    numeric_error = 1;
                    gate = 0.0f;
                    up = 0.0f;
                }
                float activation = immer_silu_f32(gate) * up;
                double feature = immer_route_feature(gate, up);
                if (!isfinite(activation) || !isfinite(feature)) {
                    numeric_error = 1;
                    activation = 0.0f;
                    feature = 0.0;
                }
                pilot_activations[task] = activation;
                pilot_features[task] = feature;
            }
        }

        if (!numeric_error && !allocation_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                const double *active_features = pilot_features
                    + (size_t) input_row
                    * (size_t) block_count
                    * (size_t) pilot_count;
                double *top_scores = selected_scores
                    + (size_t) input_row * (size_t) selected_block_count;
                int64_t *top_blocks = selected_blocks_out
                    + (size_t) input_row * (size_t) selected_block_count;
                int64_t filled = 0;
                for (int64_t block = 0; block < block_count; ++block) {
                    const double *block_coefficients = coefficients
                        + block * (pilot_count + 1);
                    double score = block_coefficients[0];
                    for (int64_t pilot = 0; pilot < pilot_count; ++pilot) {
                        score += block_coefficients[pilot + 1]
                            * active_features[block * pilot_count + pilot];
                    }
                    if (!isfinite(score)) {
                        numeric_error = 1;
                        score = 0.0;
                    } else if (score < 0.0) {
                        score = 0.0;
                    }
                    int64_t insert = 0;
                    while (
                        insert < filled && score <= top_scores[insert]
                    ) ++insert;
                    if (insert < selected_block_count) {
                        const int64_t last = filled < selected_block_count
                            ? filled
                            : selected_block_count - 1;
                        for (
                            int64_t position = last;
                            position > insert;
                            --position
                        ) {
                            top_scores[position] = top_scores[position - 1];
                            top_blocks[position] = top_blocks[position - 1];
                        }
                        top_scores[insert] = score;
                        top_blocks[insert] = block;
                        if (filled < selected_block_count) ++filled;
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp single
#endif
        {
            free(selected_scores);
            selected_scores = NULL;
            free(pilot_features);
            pilot_features = NULL;
            if (!numeric_error && !allocation_error) {
                quantized_activations = (immer_block_q8_0 *) malloc(
                    selected_packed_count * sizeof(immer_block_q8_0)
                );
                selected_down_blocks = (int64_t *) malloc(
                    selected_packed_count * sizeof(int64_t)
                );
                selected_activations = (float *) malloc(
                    selected_activation_count * sizeof(float)
                );
                if (
                    !quantized_activations
                    || !selected_down_blocks
                    || !selected_activations
                ) allocation_error = 1;
            }
        }

        if (!numeric_error && !allocation_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (
                int64_t task = 0;
                task < (int64_t) selected_activation_count;
                ++task
            ) {
                const int64_t selected_neurons_per_row =
                    selected_block_count * block_size;
                const int64_t input_row = task / selected_neurons_per_row;
                const int64_t within_row = task % selected_neurons_per_row;
                const int64_t selected = within_row / block_size;
                const int64_t block_offset = within_row % block_size;
                const int64_t logical_block = selected_blocks_out[
                    (size_t) input_row * (size_t) selected_block_count
                    + (size_t) selected
                ];
                const int64_t neuron =
                    logical_block * block_size + block_offset;
                const immer_block_q8_0 *active_hidden = quantized_hidden
                    + (size_t) input_row * (size_t) hidden_blocks_gate;
                const float *active_pilots = pilot_activations
                    + (size_t) input_row
                    * (size_t) block_count
                    * (size_t) pilot_count;
                const int64_t *block_pilots =
                    pilot_ids + logical_block * pilot_count;
                int64_t cached_pilot = -1;
                for (int64_t pilot = 0; pilot < pilot_count; ++pilot) {
                    if (block_pilots[pilot] == neuron) {
                        cached_pilot = pilot;
                        break;
                    }
                }
                float activation;
                if (cached_pilot >= 0) {
                    activation = active_pilots[
                        logical_block * pilot_count + cached_pilot
                    ];
                } else {
                    float gate = immer_dot_packed_q8(
                        gate_weights + (size_t) neuron * gate_row_bytes,
                        gate_format,
                        active_hidden,
                        hidden_blocks_gate
                    );
                    float up = immer_dot_packed_q8(
                        up_weights + (size_t) neuron * up_row_bytes,
                        up_format,
                        active_hidden,
                        hidden_blocks_gate
                    );
                    if (!isfinite(gate) || !isfinite(up)) {
                        numeric_error = 1;
                        gate = 0.0f;
                        up = 0.0f;
                    }
                    activation = immer_silu_f32(gate) * up;
                }
                if (!isfinite(activation)) {
                    numeric_error = 1;
                    activation = 0.0f;
                }
                selected_activations[task] = activation;
            }
        }

        if (!numeric_error && !allocation_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (
                int64_t task = 0;
                task < (int64_t) selected_packed_count;
                ++task
            ) {
                const int64_t input_row = task / selected_packed_blocks;
                const int64_t compact_block = task % selected_packed_blocks;
                const int64_t selected =
                    compact_block / packed_blocks_per_logical;
                const int64_t packed =
                    compact_block % packed_blocks_per_logical;
                const int64_t logical_block = selected_blocks_out[
                    (size_t) input_row * (size_t) selected_block_count
                    + (size_t) selected
                ];
                immer_block_q8_0 *target =
                    quantized_activations + (size_t) task;
                immer_quantize_q8_row(
                    selected_activations
                        + (size_t) task * (size_t) IMMER_QK,
                    target,
                    IMMER_QK
                );
                if (!isfinite(immer_half_to_float(target->d))) {
                    numeric_error = 1;
                }
                selected_down_blocks[task] =
                    logical_block * packed_blocks_per_logical + packed;
            }
        }

#ifdef _OPENMP
#pragma omp single
#endif
        {
            free(quantized_hidden);
            quantized_hidden = NULL;
            free(pilot_activations);
            pilot_activations = NULL;
            free(selected_activations);
            selected_activations = NULL;
        }

        if (!numeric_error && !allocation_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
                const uint8_t *down_weight_row = down_weights
                    + (size_t) output_row * down_row_bytes;
                for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                    const size_t packed_offset =
                        (size_t) input_row * (size_t) selected_packed_blocks;
                    float value = immer_dot_selected_packed_q8(
                        down_weight_row,
                        down_format,
                        quantized_activations + packed_offset,
                        selected_down_blocks + packed_offset,
                        selected_packed_blocks
                    );
                    value = value * affine_scale[output_row]
                        + affine_bias[output_row];
                    if (!isfinite(value)) {
                        numeric_error = 1;
                        value = 0.0f;
                    }
                    output[
                        (size_t) input_row * (size_t) output_rows
                        + (size_t) output_row
                    ] = value;
                }
            }
        }
    }

    free(quantized_hidden);
    free(pilot_activations);
    free(pilot_features);
    free(selected_scores);
    free(selected_activations);
    free(quantized_activations);
    free(selected_down_blocks);
    if (allocation_error) return 3;
    return numeric_error ? 1 : 0;
}

IMMER_EXPORT int immer_q4_deltanet_step_f32(
    const float *hidden,
    int64_t input_cols,
    const uint8_t *qkv_weights,
    int qkv_format,
    int64_t qkv_rows,
    const uint8_t *z_weights,
    int z_format,
    int64_t z_rows,
    const uint8_t *b_weights,
    int b_format,
    int64_t b_rows,
    const uint8_t *a_weights,
    int a_format,
    int64_t a_rows,
    const float *conv_weight,
    const float *A_log,
    const float *dt_bias,
    const float *norm_weight,
    const float *conv_state,
    const float *recurrent_state,
    int64_t key_heads,
    int64_t value_heads,
    int64_t key_dim,
    int64_t value_dim,
    int64_t kernel_size,
    float rms_eps,
    int pre_recurrence_only,
    float *mixed_output,
    float *qkv_output,
    float *z_output,
    float *b_output,
    float *a_output,
    float *core_output,
    float *next_conv,
    float *next_recurrent,
    int threads
) {
    if (
        !hidden || !qkv_weights || !z_weights || !b_weights || !a_weights
        || !conv_weight || !conv_state
        || !qkv_output || !z_output || !b_output || !a_output || !next_conv
        || (pre_recurrence_only != 0 && pre_recurrence_only != 1)
        || (!pre_recurrence_only && (
            !A_log || !dt_bias || !norm_weight || !recurrent_state
            || !mixed_output || !core_output || !next_recurrent
        ))
        || input_cols <= 0 || qkv_rows <= 0 || z_rows <= 0
        || b_rows <= 0 || a_rows <= 0 || key_heads <= 0
        || value_heads <= 0 || key_dim <= 0 || value_dim <= 0
        || kernel_size <= 0 || !isfinite(rms_eps) || rms_eps <= 0.0f
        || threads <= 0 || value_heads % key_heads != 0
        || key_heads > INT64_MAX / key_dim
        || value_heads > INT64_MAX / value_dim
    ) return 1;
    const int64_t key_features = key_heads * key_dim;
    const int64_t value_features = value_heads * value_dim;
    if (
        key_features > (INT64_MAX - value_features) / 2
        || qkv_rows != 2 * key_features + value_features
        || z_rows != value_features
        || b_rows != value_heads
        || a_rows != value_heads
    ) return 1;

    int64_t qkv_input_blocks;
    int64_t z_input_blocks;
    int64_t b_input_blocks;
    int64_t a_input_blocks;
    size_t qkv_row_bytes;
    size_t z_row_bytes;
    size_t b_row_bytes;
    size_t a_row_bytes;
    if (
        !immer_checked_row_layout(
            qkv_format, input_cols, &qkv_input_blocks, &qkv_row_bytes
        )
        || !immer_checked_row_layout(
            z_format, input_cols, &z_input_blocks, &z_row_bytes
        )
        || !immer_checked_row_layout(
            b_format, input_cols, &b_input_blocks, &b_row_bytes
        )
        || !immer_checked_row_layout(
            a_format, input_cols, &a_input_blocks, &a_row_bytes
        )
        || qkv_input_blocks != z_input_blocks
        || qkv_input_blocks != b_input_blocks
        || qkv_input_blocks != a_input_blocks
    ) return 1;
    if (
        qkv_rows > INT64_MAX - z_rows
        || qkv_rows + z_rows > INT64_MAX - b_rows
        || qkv_rows + z_rows + b_rows > INT64_MAX - a_rows
    ) return 1;
    const int64_t projection_rows = qkv_rows + z_rows + b_rows + a_rows;
    if (
        !immer_size_product_fits(input_cols, 1, sizeof(float))
        || !immer_size_product_fits(qkv_rows, 1, qkv_row_bytes)
        || !immer_size_product_fits(z_rows, 1, z_row_bytes)
        || !immer_size_product_fits(b_rows, 1, b_row_bytes)
        || !immer_size_product_fits(a_rows, 1, a_row_bytes)
        || !immer_size_product_fits(
            1, qkv_input_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(qkv_rows, kernel_size, sizeof(float))
        || !immer_size_product_fits(
            value_heads, key_dim, sizeof(float)
        )
        || !immer_size_product3_fits(
            value_heads, key_dim, value_dim, sizeof(float)
        )
        || !immer_size_product_fits(value_features, 1, sizeof(float))
    ) return 1;

    const size_t hidden_bytes = (size_t) input_cols * sizeof(float);
    const size_t qkv_weight_bytes = (size_t) qkv_rows * qkv_row_bytes;
    const size_t z_weight_bytes = (size_t) z_rows * z_row_bytes;
    const size_t b_weight_bytes = (size_t) b_rows * b_row_bytes;
    const size_t a_weight_bytes = (size_t) a_rows * a_row_bytes;
    const size_t conv_count = (size_t) qkv_rows * (size_t) kernel_size;
    const size_t conv_bytes = conv_count * sizeof(float);
    const size_t control_bytes = (size_t) value_heads * sizeof(float);
    const size_t norm_bytes = (size_t) value_dim * sizeof(float);
    const size_t recurrent_count =
        (size_t) value_heads * (size_t) key_dim * (size_t) value_dim;
    const size_t recurrent_bytes = recurrent_count * sizeof(float);
    const size_t mixed_bytes = (size_t) value_features * sizeof(float);
    const void *output_pointers[8];
    size_t output_bytes[8];
    int output_count = 0;
    output_pointers[output_count] = qkv_output;
    output_bytes[output_count++] = (size_t) qkv_rows * sizeof(float);
    output_pointers[output_count] = z_output;
    output_bytes[output_count++] = mixed_bytes;
    output_pointers[output_count] = b_output;
    output_bytes[output_count++] = control_bytes;
    output_pointers[output_count] = a_output;
    output_bytes[output_count++] = control_bytes;
    output_pointers[output_count] = next_conv;
    output_bytes[output_count++] = conv_bytes;
    if (!pre_recurrence_only) {
        output_pointers[output_count] = mixed_output;
        output_bytes[output_count++] = mixed_bytes;
        output_pointers[output_count] = core_output;
        output_bytes[output_count++] = mixed_bytes;
        output_pointers[output_count] = next_recurrent;
        output_bytes[output_count++] = recurrent_bytes;
    }
    const void *input_pointers[11];
    size_t input_bytes[11];
    int input_count = 0;
    input_pointers[input_count] = hidden;
    input_bytes[input_count++] = hidden_bytes;
    input_pointers[input_count] = qkv_weights;
    input_bytes[input_count++] = qkv_weight_bytes;
    input_pointers[input_count] = z_weights;
    input_bytes[input_count++] = z_weight_bytes;
    input_pointers[input_count] = b_weights;
    input_bytes[input_count++] = b_weight_bytes;
    input_pointers[input_count] = a_weights;
    input_bytes[input_count++] = a_weight_bytes;
    input_pointers[input_count] = conv_weight;
    input_bytes[input_count++] = conv_bytes;
    input_pointers[input_count] = conv_state;
    input_bytes[input_count++] = conv_bytes;
    if (!pre_recurrence_only) {
        input_pointers[input_count] = A_log;
        input_bytes[input_count++] = control_bytes;
        input_pointers[input_count] = dt_bias;
        input_bytes[input_count++] = control_bytes;
        input_pointers[input_count] = norm_weight;
        input_bytes[input_count++] = norm_bytes;
        input_pointers[input_count] = recurrent_state;
        input_bytes[input_count++] = recurrent_bytes;
    }
    for (int left = 0; left < output_count; ++left) {
        for (int right = left + 1; right < output_count; ++right) {
            if (immer_byte_ranges_overlap(
                output_pointers[left],
                output_bytes[left],
                output_pointers[right],
                output_bytes[right]
            )) return 1;
        }
        for (int source = 0; source < input_count; ++source) {
            if (immer_byte_ranges_overlap(
                output_pointers[left],
                output_bytes[left],
                input_pointers[source],
                input_bytes[source]
            )) return 1;
        }
    }

    if (
        !immer_f32_values_are_finite(hidden, (size_t) input_cols)
        || !immer_f32_values_are_finite(conv_weight, conv_count)
        || !immer_f32_values_are_finite(conv_state, conv_count)
    ) return 2;
    if (
        !pre_recurrence_only
        && (
            !immer_f32_values_are_finite(A_log, (size_t) value_heads)
            || !immer_f32_values_are_finite(dt_bias, (size_t) value_heads)
            || !immer_f32_values_are_finite(norm_weight, (size_t) value_dim)
            || !immer_f32_values_are_finite(
                recurrent_state, recurrent_count
            )
        )
    ) return 2;

    immer_block_q8_0 *quantized_hidden = (immer_block_q8_0 *) malloc(
        (size_t) qkv_input_blocks * sizeof(immer_block_q8_0)
    );
    float *projected_qkv = (float *) malloc(
        (size_t) qkv_rows * sizeof(float)
    );
    float *projected_z = (float *) malloc(
        (size_t) z_rows * sizeof(float)
    );
    float *projected_b = (float *) malloc(
        (size_t) b_rows * sizeof(float)
    );
    float *projected_a = (float *) malloc(
        (size_t) a_rows * sizeof(float)
    );
    float *normalized_q = NULL;
    float *normalized_k = NULL;
    float *delta = NULL;
    if (!pre_recurrence_only) {
        normalized_q = (float *) malloc(
            (size_t) key_features * sizeof(float)
        );
        normalized_k = (float *) malloc(
            (size_t) key_features * sizeof(float)
        );
        delta = (float *) malloc(mixed_bytes);
    }
    if (
        !quantized_hidden || !projected_qkv || !projected_z
        || !projected_b || !projected_a
        || (!pre_recurrence_only && (
            !normalized_q || !normalized_k || !delta
        ))
    ) {
        free(quantized_hidden);
        free(projected_qkv);
        free(projected_z);
        free(projected_b);
        free(projected_a);
        free(normalized_q);
        free(normalized_k);
        free(delta);
        return 3;
    }

    int numeric_error = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
#pragma omp single
#endif
        {
            immer_quantize_q8_row(
                hidden, quantized_hidden, input_cols
            );
            for (
                int64_t block = 0;
                block < qkv_input_blocks;
                ++block
            ) {
                if (!isfinite(immer_half_to_float(quantized_hidden[block].d))) {
                    numeric_error = 1;
                }
            }
        }

        if (!numeric_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t row = 0; row < projection_rows; ++row) {
                const uint8_t *weights;
                int format;
                size_t row_bytes;
                int64_t local_row;
                float *target;
                float *external_target = NULL;
                if (row < qkv_rows) {
                    weights = qkv_weights;
                    format = qkv_format;
                    row_bytes = qkv_row_bytes;
                    local_row = row;
                    target = projected_qkv;
                } else if (row < qkv_rows + z_rows) {
                    weights = z_weights;
                    format = z_format;
                    row_bytes = z_row_bytes;
                    local_row = row - qkv_rows;
                    target = projected_z;
                    external_target = z_output;
                } else if (row < qkv_rows + z_rows + b_rows) {
                    weights = b_weights;
                    format = b_format;
                    row_bytes = b_row_bytes;
                    local_row = row - qkv_rows - z_rows;
                    target = projected_b;
                    external_target = b_output;
                } else {
                    weights = a_weights;
                    format = a_format;
                    row_bytes = a_row_bytes;
                    local_row = row - qkv_rows - z_rows - b_rows;
                    target = projected_a;
                    external_target = a_output;
                }
                const float projected = immer_round_bf16(
                    immer_dot_packed_q8(
                        weights + (size_t) local_row * row_bytes,
                        format,
                        quantized_hidden,
                        qkv_input_blocks
                    )
                );
                target[local_row] = projected;
                if (external_target) external_target[local_row] = projected;
                if (!isfinite(projected)) numeric_error = 1;
            }
        }

        if (!numeric_error) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t channel = 0; channel < qkv_rows; ++channel) {
                const size_t offset =
                    (size_t) channel * (size_t) kernel_size;
                for (int64_t index = 0; index + 1 < kernel_size; ++index) {
                    next_conv[offset + (size_t) index] = conv_state[
                        offset + (size_t) index + 1
                    ];
                }
                next_conv[offset + (size_t) kernel_size - 1] =
                    projected_qkv[channel];
                float convolved = 0.0f;
                for (int64_t index = 0; index < kernel_size; ++index) {
                    convolved += next_conv[offset + (size_t) index]
                        * conv_weight[offset + (size_t) index];
                }
                convolved = immer_round_bf16(convolved);
                const float activated = immer_round_bf16(
                    immer_silu_f32(convolved)
                );
                projected_qkv[channel] = activated;
                qkv_output[channel] = activated;
                if (!isfinite(activated)) numeric_error = 1;
            }
        }

        if (!numeric_error && !pre_recurrence_only) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t head = 0; head < value_heads; ++head) {
                projected_b[head] = immer_round_bf16(
                    immer_sigmoid_f32(projected_b[head])
                );
                projected_a[head] = -expf(A_log[head]) * immer_softplus_f32(
                    projected_a[head] + dt_bias[head]
                );
                if (
                    !isfinite(projected_b[head])
                    || !isfinite(projected_a[head])
                ) numeric_error = 1;
            }
        }

        if (!numeric_error && !pre_recurrence_only) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t head = 0; head < key_heads; ++head) {
                const float *raw_q =
                    projected_qkv + (size_t) head * (size_t) key_dim;
                const float *raw_k = projected_qkv
                    + (size_t) key_features
                    + (size_t) head * (size_t) key_dim;
                float q_squared = 0.0f;
                float k_squared = 0.0f;
                for (int64_t index = 0; index < key_dim; ++index) {
                    q_squared += immer_round_bf16(
                        raw_q[index] * raw_q[index]
                    );
                    k_squared += immer_round_bf16(
                        raw_k[index] * raw_k[index]
                    );
                }
                q_squared = immer_round_bf16(q_squared);
                k_squared = immer_round_bf16(k_squared);
                const float q_root = immer_round_bf16(
                    sqrtf(immer_round_bf16(q_squared + 1e-6f))
                );
                const float k_root = immer_round_bf16(
                    sqrtf(immer_round_bf16(k_squared + 1e-6f))
                );
                const float q_inverse = immer_round_bf16(1.0f / q_root);
                const float k_inverse = immer_round_bf16(1.0f / k_root);
                const float q_scale = 1.0f / sqrtf((float) key_dim);
                for (int64_t index = 0; index < key_dim; ++index) {
                    normalized_q[
                        (size_t) head * (size_t) key_dim + (size_t) index
                    ] = immer_round_bf16(raw_q[index] * q_inverse) * q_scale;
                    normalized_k[
                        (size_t) head * (size_t) key_dim + (size_t) index
                    ] = immer_round_bf16(raw_k[index] * k_inverse);
                }
                if (!isfinite(q_inverse) || !isfinite(k_inverse)) {
                    numeric_error = 1;
                }
            }
        }

        if (!numeric_error && !pre_recurrence_only) {
            const int64_t repetitions = value_heads / key_heads;
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t head = 0; head < value_heads; ++head) {
                const int64_t key_head = head / repetitions;
                const float *active_q = normalized_q
                    + (size_t) key_head * (size_t) key_dim;
                const float *active_k = normalized_k
                    + (size_t) key_head * (size_t) key_dim;
                const float *active_v = projected_qkv
                    + (size_t) 2 * (size_t) key_features
                    + (size_t) head * (size_t) value_dim;
                const float beta = projected_b[head];
                const float decay = expf(projected_a[head]);
                const size_t value_offset =
                    (size_t) head * (size_t) value_dim;
                const size_t state_offset =
                    value_offset * (size_t) key_dim;
                const float *old_state = recurrent_state + state_offset;
                float *new_state = next_recurrent + state_offset;
                float *active_output = mixed_output + value_offset;
                float *active_delta = delta + value_offset;
                for (int64_t value = 0; value < value_dim; ++value) {
                    active_output[value] = 0.0f;
                }
                for (int64_t key = 0; key < key_dim; ++key) {
                    const size_t row_offset =
                        (size_t) key * (size_t) value_dim;
                    for (int64_t value = 0; value < value_dim; ++value) {
                        const float remembered =
                            old_state[row_offset + (size_t) value] * decay;
                        new_state[row_offset + (size_t) value] = remembered;
                        active_output[value] += remembered * active_k[key];
                    }
                }
                for (int64_t value = 0; value < value_dim; ++value) {
                    active_delta[value] =
                        (active_v[value] - active_output[value]) * beta;
                    active_output[value] = 0.0f;
                }
                for (int64_t key = 0; key < key_dim; ++key) {
                    const size_t row_offset =
                        (size_t) key * (size_t) value_dim;
                    for (int64_t value = 0; value < value_dim; ++value) {
                        const float updated =
                            new_state[row_offset + (size_t) value]
                            + active_k[key] * active_delta[value];
                        new_state[row_offset + (size_t) value] = updated;
                        active_output[value] += updated * active_q[key];
                        if (!isfinite(updated)) numeric_error = 1;
                    }
                }
                for (int64_t value = 0; value < value_dim; ++value) {
                    active_output[value] = immer_round_bf16(
                        active_output[value]
                    );
                    core_output[value_offset + (size_t) value] =
                        active_output[value];
                    if (!isfinite(active_output[value])) numeric_error = 1;
                }
            }
        }

        if (!numeric_error && !pre_recurrence_only) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
            for (int64_t head = 0; head < value_heads; ++head) {
                const size_t offset =
                    (size_t) head * (size_t) value_dim;
                float squared = 0.0f;
                for (int64_t value = 0; value < value_dim; ++value) {
                    const float active = mixed_output[offset + (size_t) value];
                    squared += active * active;
                }
                const float inverse = 1.0f / sqrtf(
                    squared / (float) value_dim + rms_eps
                );
                for (int64_t value = 0; value < value_dim; ++value) {
                    const size_t index = offset + (size_t) value;
                    const float normalized = immer_round_bf16(
                        mixed_output[index] * inverse
                    );
                    const float scaled = immer_round_bf16(
                        normalized * immer_round_bf16(norm_weight[value])
                    );
                    mixed_output[index] = immer_round_bf16(
                        scaled * immer_silu_f32(projected_z[index])
                    );
                    if (!isfinite(mixed_output[index])) numeric_error = 1;
                }
            }
        }
    }

    free(quantized_hidden);
    free(projected_qkv);
    free(projected_z);
    free(projected_b);
    free(projected_a);
    free(normalized_q);
    free(normalized_k);
    free(delta);
    return numeric_error ? 2 : 0;
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

IMMER_EXPORT int immer_q4_mlp_bf16_f32(
    const float *input,
    int64_t input_rows,
    int64_t hidden_cols,
    const uint8_t *gate_weights,
    int gate_format,
    const uint8_t *up_weights,
    int up_format,
    const uint8_t *down_weights,
    int down_format,
    int64_t intermediate_cols,
    int64_t output_rows,
    const uint16_t *silu_bf16,
    float *output,
    int64_t activation_page_topk,
    int64_t *top_page_ids,
    double *top_page_scores,
    double *total_page_scores,
    int threads
) {
    int64_t hidden_blocks_gate;
    int64_t hidden_blocks_up;
    int64_t activation_blocks;
    size_t gate_row_bytes;
    size_t up_row_bytes;
    size_t down_row_bytes;
    const int64_t activation_page_count = intermediate_cols > 0
        ? (intermediate_cols - 1) / IMMER_MLP_PAGE_NEURONS + 1
        : 0;
    if (
        !input || !gate_weights || !up_weights || !down_weights
        || !silu_bf16 || !output
        || input_rows <= 0 || hidden_cols <= 0 || intermediate_cols <= 0
        || output_rows <= 0 || threads <= 0
        || activation_page_topk < 0
        || activation_page_topk > activation_page_count
        || (
            activation_page_topk > 0
            && (!top_page_ids || !top_page_scores || !total_page_scores)
        )
        || input_rows > INT64_MAX / intermediate_cols
        || input_rows > INT64_MAX / output_rows
        || !immer_checked_row_layout(
            gate_format, hidden_cols, &hidden_blocks_gate, &gate_row_bytes
        )
        || !immer_checked_row_layout(
            up_format, hidden_cols, &hidden_blocks_up, &up_row_bytes
        )
        || !immer_checked_row_layout(
            down_format,
            intermediate_cols,
            &activation_blocks,
            &down_row_bytes
        )
        || hidden_blocks_gate != hidden_blocks_up
        || !immer_size_product_fits(
            input_rows, hidden_blocks_gate, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            input_rows, intermediate_cols, sizeof(float)
        )
        || !immer_size_product_fits(
            input_rows, activation_blocks, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(input_rows, output_rows, sizeof(float))
        || !immer_size_product_fits(
            input_rows, activation_page_topk, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows, activation_page_topk, sizeof(double)
        )
        || (
            activation_page_topk > 0
            && !immer_size_product_fits(input_rows, 1, sizeof(double))
        )
        || (
            activation_page_topk > 0
            && !immer_size_product_fits(
                input_rows,
                activation_page_count,
                sizeof(immer_mlp_page_energy)
            )
        )
    ) return 1;
    const size_t input_count = (size_t) input_rows * (size_t) hidden_cols;
    if (!immer_f32_values_are_finite(input, input_count)) return 2;

    immer_block_q8_0 *quantized_hidden = (immer_block_q8_0 *) malloc(
        (size_t) input_rows
        * (size_t) hidden_blocks_gate
        * sizeof(immer_block_q8_0)
    );
    float *activation = (float *) malloc(
        (size_t) input_rows * (size_t) intermediate_cols * sizeof(float)
    );
    immer_block_q8_0 *quantized_activation = (immer_block_q8_0 *) malloc(
        (size_t) input_rows
        * (size_t) activation_blocks
        * sizeof(immer_block_q8_0)
    );
    immer_mlp_page_energy *page_energies = activation_page_topk > 0
        ? (immer_mlp_page_energy *) malloc(
            (size_t) input_rows
            * (size_t) activation_page_count
            * sizeof(immer_mlp_page_energy)
        )
        : NULL;
    if (
        !quantized_hidden || !activation || !quantized_activation
        || (activation_page_topk > 0 && !page_energies)
    ) {
        free(quantized_hidden);
        free(activation);
        free(quantized_activation);
        free(page_energies);
        return 3;
    }
    int numeric_error = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t row = 0; row < input_rows; ++row) {
            immer_block_q8_0 *target = quantized_hidden
                + (size_t) row * (size_t) hidden_blocks_gate;
            immer_quantize_q8_row(
                input + (size_t) row * (size_t) hidden_cols,
                target,
                hidden_cols
            );
            for (int64_t block = 0; block < hidden_blocks_gate; ++block) {
                if (!isfinite(immer_half_to_float(target[block].d))) {
                    numeric_error = 1;
                }
            }
        }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (
            int64_t task = 0;
            task < input_rows * intermediate_cols;
            ++task
        ) {
            const int64_t row = task / intermediate_cols;
            const int64_t neuron = task % intermediate_cols;
            const immer_block_q8_0 *active_hidden = quantized_hidden
                + (size_t) row * (size_t) hidden_blocks_gate;
            float gate = immer_dot_packed_q8(
                gate_weights + (size_t) neuron * gate_row_bytes,
                gate_format,
                active_hidden,
                hidden_blocks_gate
            );
            float up = immer_dot_packed_q8(
                up_weights + (size_t) neuron * up_row_bytes,
                up_format,
                active_hidden,
                hidden_blocks_gate
            );
            gate = immer_round_bf16(gate);
            up = immer_round_bf16(up);
            uint32_t gate_bits;
            memcpy(&gate_bits, &gate, sizeof(gate_bits));
            const float silu = immer_bf16_bits_to_float(
                silu_bf16[gate_bits >> 16]
            );
            const float value = immer_round_bf16(silu * up);
            if (!isfinite(value)) numeric_error = 1;
            activation[(size_t) row * (size_t) intermediate_cols + neuron] = value;
        }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t row = 0; row < input_rows; ++row) {
            immer_block_q8_0 *target = quantized_activation
                + (size_t) row * (size_t) activation_blocks;
            if (activation_page_topk > 0) {
                if (!immer_quantize_q8_row_with_page_topk(
                    activation + (size_t) row * (size_t) intermediate_cols,
                    target,
                    intermediate_cols,
                    activation_page_topk,
                    top_page_ids
                        + (size_t) row * (size_t) activation_page_topk,
                    top_page_scores
                        + (size_t) row * (size_t) activation_page_topk,
                    total_page_scores + row,
                    page_energies
                        + (size_t) row * (size_t) activation_page_count
                )) numeric_error = 1;
            } else {
                immer_quantize_q8_row(
                    activation + (size_t) row * (size_t) intermediate_cols,
                    target,
                    intermediate_cols
                );
            }
            for (int64_t block = 0; block < activation_blocks; ++block) {
                if (!isfinite(immer_half_to_float(target[block].d))) {
                    numeric_error = 1;
                }
            }
        }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
            const uint8_t *weight_row = down_weights
                + (size_t) output_row * down_row_bytes;
            for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                const float value = immer_dot_packed_q8(
                    weight_row,
                    down_format,
                    quantized_activation
                        + (size_t) input_row * (size_t) activation_blocks,
                    activation_blocks
                );
                if (!isfinite(value)) numeric_error = 1;
                output[(size_t) input_row * (size_t) output_rows + output_row] = value;
            }
        }
    }
    free(quantized_hidden);
    free(activation);
    free(quantized_activation);
    free(page_energies);
    return numeric_error ? 2 : 0;
}

IMMER_EXPORT int immer_q4_mlp_pages_bf16_f32(
    const float *input,
    int64_t input_rows,
    int64_t hidden_cols,
    const uint8_t *gate_weights,
    int gate_format,
    const uint8_t *up_weights,
    int up_format,
    const uint8_t *down_weights,
    int down_format,
    int64_t intermediate_cols,
    int64_t output_rows,
    const uint16_t *silu_bf16,
    const int64_t *page_ids,
    int64_t selected_pages,
    float *output,
    int threads
) {
    int64_t hidden_blocks_gate;
    int64_t hidden_blocks_up;
    int64_t activation_blocks;
    size_t gate_row_bytes;
    size_t up_row_bytes;
    size_t down_row_bytes;
    const int64_t page_count = intermediate_cols > 0
        ? (intermediate_cols - 1) / IMMER_MLP_PAGE_NEURONS + 1
        : 0;
    if (
        !input || !gate_weights || !up_weights || !down_weights
        || !silu_bf16 || !page_ids || !output
        || input_rows <= 0 || hidden_cols <= 0 || intermediate_cols <= 0
        || output_rows <= 0 || selected_pages <= 0
        || selected_pages > page_count || threads <= 0
        || selected_pages > INT64_MAX / 2
        || input_rows > INT64_MAX / selected_pages
        || input_rows > INT64_MAX / output_rows
        || !immer_checked_row_layout(
            gate_format, hidden_cols, &hidden_blocks_gate, &gate_row_bytes
        )
        || !immer_checked_row_layout(
            up_format, hidden_cols, &hidden_blocks_up, &up_row_bytes
        )
        || !immer_checked_row_layout(
            down_format,
            intermediate_cols,
            &activation_blocks,
            &down_row_bytes
        )
        || hidden_blocks_gate != hidden_blocks_up
        || !immer_size_product_fits(
            input_rows, hidden_blocks_gate, sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            input_rows, selected_pages, sizeof(int64_t)
        )
        || !immer_size_product_fits(
            input_rows,
            activation_blocks,
            sizeof(immer_block_q8_0)
        )
        || !immer_size_product_fits(
            input_rows, selected_pages * 2, sizeof(int64_t)
        )
        || !immer_size_product_fits(input_rows, output_rows, sizeof(float))
    ) return 1;
    const size_t input_count = (size_t) input_rows * (size_t) hidden_cols;
    if (!immer_f32_values_are_finite(input, input_count)) return 2;

    const int64_t selected_block_capacity = selected_pages * 2;
    int64_t *sorted_pages = (int64_t *) malloc(
        (size_t) input_rows * (size_t) selected_pages * sizeof(int64_t)
    );
    immer_block_q8_0 *quantized_hidden = (immer_block_q8_0 *) malloc(
        (size_t) input_rows
        * (size_t) hidden_blocks_gate
        * sizeof(immer_block_q8_0)
    );
    immer_block_q8_0 *quantized_activation = (immer_block_q8_0 *) malloc(
        (size_t) input_rows
        * (size_t) activation_blocks
        * sizeof(immer_block_q8_0)
    );
    int64_t *selected_block_ids = (int64_t *) malloc(
        (size_t) input_rows
        * (size_t) selected_block_capacity
        * sizeof(int64_t)
    );
    int64_t *selected_block_counts = (int64_t *) malloc(
        (size_t) input_rows * sizeof(int64_t)
    );
    if (
        !sorted_pages || !quantized_hidden || !quantized_activation
        || !selected_block_ids || !selected_block_counts
    ) {
        free(sorted_pages);
        free(quantized_hidden);
        free(quantized_activation);
        free(selected_block_ids);
        free(selected_block_counts);
        return 3;
    }

    for (int64_t row = 0; row < input_rows; ++row) {
        int64_t *target = sorted_pages
            + (size_t) row * (size_t) selected_pages;
        const int64_t *source = page_ids
            + (size_t) row * (size_t) selected_pages;
        for (int64_t selected = 0; selected < selected_pages; ++selected) {
            const int64_t page = source[selected];
            if (
                page < 0 || page >= page_count
                || (selected > 0 && source[selected - 1] >= page)
            ) {
                free(sorted_pages);
                free(quantized_hidden);
                free(quantized_activation);
                free(selected_block_ids);
                free(selected_block_counts);
                return 4;
            }
            target[selected] = page;
        }
        int64_t block_count = 0;
        int64_t *active_ids = selected_block_ids
            + (size_t) row * (size_t) selected_block_capacity;
        for (int64_t selected = 0; selected < selected_pages; ++selected) {
            const int64_t first_block = target[selected]
                * (IMMER_MLP_PAGE_NEURONS / IMMER_QK);
            const int64_t remaining = activation_blocks - first_block;
            const int64_t blocks = remaining < 2 ? remaining : 2;
            for (int64_t local = 0; local < blocks; ++local) {
                active_ids[block_count++] = first_block + local;
            }
        }
        selected_block_counts[row] = block_count;
    }

    int numeric_error = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t row = 0; row < input_rows; ++row) {
            immer_block_q8_0 *target = quantized_hidden
                + (size_t) row * (size_t) hidden_blocks_gate;
            immer_quantize_q8_row(
                input + (size_t) row * (size_t) hidden_cols,
                target,
                hidden_cols
            );
            for (int64_t block = 0; block < hidden_blocks_gate; ++block) {
                if (!isfinite(immer_half_to_float(target[block].d))) {
                    numeric_error = 1;
                }
            }
            if (selected_block_counts[row] * 2 >= activation_blocks) {
                memset(
                    quantized_activation
                        + (size_t) row * (size_t) activation_blocks,
                    0,
                    (size_t) activation_blocks * sizeof(immer_block_q8_0)
                );
            }
        }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (
            int64_t task = 0;
            task < input_rows * selected_pages;
            ++task
        ) {
            const int64_t row = task / selected_pages;
            const int64_t selected = task % selected_pages;
            const int64_t page = sorted_pages[
                (size_t) row * (size_t) selected_pages + selected
            ];
            const int64_t first_neuron = page * IMMER_MLP_PAGE_NEURONS;
            const int64_t remaining = intermediate_cols - first_neuron;
            const int64_t neuron_count = remaining < IMMER_MLP_PAGE_NEURONS
                ? remaining
                : IMMER_MLP_PAGE_NEURONS;
            const immer_block_q8_0 *active_hidden = quantized_hidden
                + (size_t) row * (size_t) hidden_blocks_gate;
            float page_activation[IMMER_MLP_PAGE_NEURONS];
            for (int64_t offset = 0; offset < neuron_count; ++offset) {
                const int64_t neuron = first_neuron + offset;
                float gate = immer_dot_packed_q8(
                    gate_weights + (size_t) neuron * gate_row_bytes,
                    gate_format,
                    active_hidden,
                    hidden_blocks_gate
                );
                float up = immer_dot_packed_q8(
                    up_weights + (size_t) neuron * up_row_bytes,
                    up_format,
                    active_hidden,
                    hidden_blocks_gate
                );
                gate = immer_round_bf16(gate);
                up = immer_round_bf16(up);
                uint32_t gate_bits;
                memcpy(&gate_bits, &gate, sizeof(gate_bits));
                const float silu = immer_bf16_bits_to_float(
                    silu_bf16[gate_bits >> 16]
                );
                const float value = immer_round_bf16(silu * up);
                if (!isfinite(value)) numeric_error = 1;
                page_activation[offset] = value;
            }
            const int64_t page_blocks = neuron_count / IMMER_QK;
            const int dense_down =
                selected_block_counts[row] * 2 >= activation_blocks;
            immer_block_q8_0 *target = quantized_activation
                + (size_t) row * (size_t) activation_blocks
                + (
                    dense_down
                    ? (size_t) page * (IMMER_MLP_PAGE_NEURONS / IMMER_QK)
                    : (size_t) selected * 2
                );
            for (int64_t local = 0; local < page_blocks; ++local) {
                immer_quantize_q8_block(
                    page_activation + local * IMMER_QK,
                    target + local,
                    NULL
                );
                if (!isfinite(immer_half_to_float(target[local].d))) {
                    numeric_error = 1;
                }
            }
        }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:numeric_error)
#endif
        for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
            const uint8_t *weight_row = down_weights
                + (size_t) output_row * down_row_bytes;
            for (int64_t input_row = 0; input_row < input_rows; ++input_row) {
                const int dense_down =
                    selected_block_counts[input_row] * 2 >= activation_blocks;
                const immer_block_q8_0 *active = quantized_activation
                    + (size_t) input_row * (size_t) activation_blocks;
                const float value = dense_down
                    ? immer_dot_packed_q8(
                        weight_row,
                        down_format,
                        active,
                        activation_blocks
                    )
                    : immer_dot_selected_packed_q8(
                        weight_row,
                        down_format,
                        active,
                        selected_block_ids
                            + (size_t) input_row
                            * (size_t) selected_block_capacity,
                        selected_block_counts[input_row]
                    );
                if (!isfinite(value)) numeric_error = 1;
                output[
                    (size_t) input_row * (size_t) output_rows + output_row
                ] = value;
            }
        }
    }
    free(sorted_pages);
    free(quantized_hidden);
    free(quantized_activation);
    free(selected_block_ids);
    free(selected_block_counts);
    return numeric_error ? 2 : 0;
}
