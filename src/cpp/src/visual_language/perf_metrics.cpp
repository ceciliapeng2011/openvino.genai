// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace ov::genai {
MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair VLMPerfMetrics::get_prepare_embeddings_duration() {
    evaluate_statistics();
    return prepare_embeddings_duration;
}

MeanStdPair VLMPerfMetrics::get_vision_encoder_duration() {
    evaluate_statistics();
    return vision_encoder_duration;
}

MeanStdPair VLMPerfMetrics::get_tokenizer_duration() {
    evaluate_statistics();
    return tokenizer_duration;
}

MeanStdPair VLMPerfMetrics::get_text_embeddings_duration() {
    evaluate_statistics();
    return text_embeddings_duration;
}

MeanStdPair VLMPerfMetrics::get_vision_embeddings_merger_duration() {
    evaluate_statistics();
    return vision_embeddings_merger_duration;
}

MeanStdPair VLMPerfMetrics::get_vision_embeddings_pos_duration() {
    evaluate_statistics();
    return vision_embeddings_pos_duration;
}

MeanStdPair VLMPerfMetrics::get_lm_prefill_duration() {
    evaluate_statistics();
    return lm_prefill_duration;
}

void VLMPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    prepare_embeddings_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.prepare_embeddings_durations);
    vision_encoder_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.vision_encoder_durations);
    tokenizer_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.tokenizer_durations);
    text_embeddings_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.text_embeddings_durations);
    vision_embeddings_merger_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.vision_embeddings_merger_durations);
    vision_embeddings_pos_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.vision_embeddings_pos_durations);
    lm_prefill_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.lm_prefill_durations);
    PerfMetrics::evaluate_statistics(start_time);
};

static void concat_vectors(std::vector<MicroSeconds>& dst, const std::vector<MicroSeconds>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

VLMPerfMetrics VLMPerfMetrics::operator+(const VLMPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    VLMPerfMetrics result{base_result};

    result.vlm_raw_metrics = vlm_raw_metrics;

    concat_vectors(result.vlm_raw_metrics.prepare_embeddings_durations, right.vlm_raw_metrics.prepare_embeddings_durations);
    concat_vectors(result.vlm_raw_metrics.prepare_embeddings_offsets, right.vlm_raw_metrics.prepare_embeddings_offsets);
    concat_vectors(result.vlm_raw_metrics.vision_encoder_durations, right.vlm_raw_metrics.vision_encoder_durations);
    concat_vectors(result.vlm_raw_metrics.tokenizer_durations, right.vlm_raw_metrics.tokenizer_durations);
    concat_vectors(result.vlm_raw_metrics.text_embeddings_durations, right.vlm_raw_metrics.text_embeddings_durations);
    concat_vectors(result.vlm_raw_metrics.vision_embeddings_merger_durations, right.vlm_raw_metrics.vision_embeddings_merger_durations);
    concat_vectors(result.vlm_raw_metrics.vision_embeddings_pos_durations, right.vlm_raw_metrics.vision_embeddings_pos_durations);
    concat_vectors(result.vlm_raw_metrics.lm_prefill_durations, right.vlm_raw_metrics.lm_prefill_durations);
    return result;
}
}
