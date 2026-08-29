/* Float32 sequence-one DeltaNet recurrent step for the Qwen3.8 CPU runtime. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define IMMER_EXPORT __declspec(dllexport)
#else
#define IMMER_EXPORT __attribute__((visibility("default")))
#endif

#define IMMER_DELTANET_ABI 1u

static int immer_checked_multiply(size_t left, size_t right, size_t *result) {
    if (!result || (right != 0 && left > (size_t) -1 / right)) return 0;
    *result = left * right;
    return 1;
}

static int immer_checked_dimension(int64_t value, size_t *result) {
    if (
        !result
        || value <= 0
        || (uint64_t) value > (uint64_t) (size_t) -1
    ) return 0;
    *result = (size_t) value;
    return 1;
}

static int immer_ranges_overlap(
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
    const uintptr_t left_end = left_start + left_bytes;
    const uintptr_t right_end = right_start + right_bytes;
    return left_start < right_end && right_start < left_end;
}

IMMER_EXPORT uint32_t immer_deltanet_abi(void) {
    return IMMER_DELTANET_ABI;
}

IMMER_EXPORT int immer_deltanet_step_f32(
    const float *query,
    const float *key,
    const float *value,
    const float *log_decay,
    const float *beta,
    const float *initial_state,
    int64_t batch,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float *output,
    float *new_state,
    int threads
) {
    if (
        !query || !key || !value || !log_decay || !beta || !initial_state
        || !output || !new_state || threads <= 0
    ) return 1;

    size_t batch_size;
    size_t head_count;
    size_t key_width;
    size_t value_width;
    size_t batch_heads;
    size_t query_count;
    size_t value_count;
    size_t state_count;
    if (
        !immer_checked_dimension(batch, &batch_size)
        || !immer_checked_dimension(heads, &head_count)
        || !immer_checked_dimension(key_dim, &key_width)
        || !immer_checked_dimension(value_dim, &value_width)
        || !immer_checked_multiply(batch_size, head_count, &batch_heads)
        || !immer_checked_multiply(batch_heads, key_width, &query_count)
        || !immer_checked_multiply(batch_heads, value_width, &value_count)
        || !immer_checked_multiply(query_count, value_width, &state_count)
        || query_count > (size_t) -1 / sizeof(float)
        || value_count > (size_t) -1 / sizeof(float)
        || state_count > (size_t) -1 / sizeof(float)
    ) return 1;

    const size_t query_bytes = query_count * sizeof(float);
    const size_t value_bytes = value_count * sizeof(float);
    const size_t scalar_bytes = batch_heads * sizeof(float);
    const size_t state_bytes = state_count * sizeof(float);
    if (
        immer_ranges_overlap(output, value_bytes, new_state, state_bytes)
        || immer_ranges_overlap(output, value_bytes, query, query_bytes)
        || immer_ranges_overlap(output, value_bytes, key, query_bytes)
        || immer_ranges_overlap(output, value_bytes, value, value_bytes)
        || immer_ranges_overlap(output, value_bytes, log_decay, scalar_bytes)
        || immer_ranges_overlap(output, value_bytes, beta, scalar_bytes)
        || immer_ranges_overlap(output, value_bytes, initial_state, state_bytes)
        || immer_ranges_overlap(new_state, state_bytes, query, query_bytes)
        || immer_ranges_overlap(new_state, state_bytes, key, query_bytes)
        || immer_ranges_overlap(new_state, state_bytes, value, value_bytes)
        || immer_ranges_overlap(new_state, state_bytes, log_decay, scalar_bytes)
        || immer_ranges_overlap(new_state, state_bytes, beta, scalar_bytes)
        || immer_ranges_overlap(
            new_state, state_bytes, initial_state, state_bytes
        )
    ) return 1;

    float *delta = (float *) malloc(value_bytes);
    if (!delta) return 3;
    int numeric_error = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(|:numeric_error)
#endif
    for (int64_t batch_head = 0; batch_head < (int64_t) batch_heads; ++batch_head) {
        const size_t query_offset = (size_t) batch_head * key_width;
        const size_t value_offset = (size_t) batch_head * value_width;
        const size_t state_offset = query_offset * value_width;
        const float *active_query = query + query_offset;
        const float *active_key = key + query_offset;
        const float *active_value = value + value_offset;
        const float *active_state = initial_state + state_offset;
        float *active_output = output + value_offset;
        float *active_delta = delta + value_offset;
        float *active_new_state = new_state + state_offset;

        int active_error = 0;
        for (size_t index = 0; index < key_width; ++index) {
            if (!isfinite(active_query[index]) || !isfinite(active_key[index])) {
                active_error = 1;
            }
        }
        for (size_t index = 0; index < value_width; ++index) {
            if (!isfinite(active_value[index])) active_error = 1;
            active_output[index] = 0.0f;
        }
        const float active_log_decay = log_decay[batch_head];
        const float active_beta = beta[batch_head];
        const float decay = expf(active_log_decay);
        if (
            !isfinite(active_log_decay)
            || !isfinite(active_beta)
            || !isfinite(decay)
        ) active_error = 1;

        for (size_t key_index = 0; key_index < key_width; ++key_index) {
            const float active_key_value = active_key[key_index];
            const float *state_row = active_state + key_index * value_width;
            float *new_state_row =
                active_new_state + key_index * value_width;
            for (size_t value_index = 0; value_index < value_width; ++value_index) {
                float state_value = state_row[value_index];
                if (!isfinite(state_value)) {
                    active_error = 1;
                    state_value = 0.0f;
                }
                const float remembered = state_value * decay;
                new_state_row[value_index] = remembered;
                active_output[value_index] += remembered * active_key_value;
            }
        }

        for (size_t value_index = 0; value_index < value_width; ++value_index) {
            const float remembered = active_output[value_index];
            if (!isfinite(remembered)) active_error = 1;
            const float change =
                (active_value[value_index] - remembered) * active_beta;
            active_delta[value_index] = change;
            active_output[value_index] = 0.0f;
            if (!isfinite(change)) active_error = 1;
        }

        for (size_t key_index = 0; key_index < key_width; ++key_index) {
            const float active_key_value = active_key[key_index];
            const float active_query_value = active_query[key_index];
            float *new_state_row =
                active_new_state + key_index * value_width;
            for (size_t value_index = 0; value_index < value_width; ++value_index) {
                const float updated = new_state_row[value_index]
                    + active_key_value * active_delta[value_index];
                new_state_row[value_index] = updated;
                active_output[value_index] += updated * active_query_value;
                if (!isfinite(updated)) active_error = 1;
            }
        }
        for (size_t value_index = 0; value_index < value_width; ++value_index) {
            if (!isfinite(active_output[value_index])) active_error = 1;
        }
        if (active_error) numeric_error = 1;
    }
    free(delta);
    return numeric_error ? 2 : 0;
}
