// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"


namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS VLMRawPerfMetrics {
    /** @brief Duration of preparation of embeddings */
    std::vector<MicroSeconds> prepare_embeddings_durations;
    /** @brief Per-model durations within TTFT */
    std::vector<MicroSeconds> vision_encoder_durations;
    std::vector<MicroSeconds> tokenizer_durations;
    std::vector<MicroSeconds> text_embeddings_durations;
    std::vector<MicroSeconds> vision_embeddings_merger_durations;
    std::vector<MicroSeconds> vision_embeddings_pos_durations;
    std::vector<MicroSeconds> lm_prefill_durations;
};

struct OPENVINO_GENAI_EXPORTS VLMPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of preparation of embeddings in milliseconds */
    MeanStdPair prepare_embeddings_duration;
    MeanStdPair vision_encoder_duration;
    MeanStdPair tokenizer_duration;
    MeanStdPair text_embeddings_duration;
    MeanStdPair vision_embeddings_merger_duration;
    MeanStdPair vision_embeddings_pos_duration;
    MeanStdPair lm_prefill_duration;

    MeanStdPair get_prepare_embeddings_duration();
    MeanStdPair get_vision_encoder_duration();
    MeanStdPair get_tokenizer_duration();
    MeanStdPair get_text_embeddings_duration();
    MeanStdPair get_vision_embeddings_merger_duration();
    MeanStdPair get_vision_embeddings_pos_duration();
    MeanStdPair get_lm_prefill_duration();

    VLMPerfMetrics() = default;

    VLMPerfMetrics(PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics), prepare_embeddings_duration(){};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    VLMPerfMetrics operator+(const VLMPerfMetrics& metrics) const;

    VLMRawPerfMetrics vlm_raw_metrics;
};

}
