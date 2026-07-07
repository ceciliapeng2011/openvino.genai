// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "logger.hpp"

namespace ov::genai {

struct TraceEvent {
    std::string name;
    std::string category;
    char phase;  // 'B' = begin, 'E' = end, 'X' = complete
    int64_t timestamp_us;
    int64_t duration_us;
    int pid;
    int tid;
};

class ChromeTrace {
public:
    static ChromeTrace& get_instance() {
        static ChromeTrace instance;
        return instance;
    }

    bool is_enabled() const {
        return m_enabled;
    }

    void add_complete_event(const std::string& name, const std::string& category,
                            int64_t start_us, int64_t duration_us, int tid = 1) {
        if (!m_enabled) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.push_back({name, category, 'X', start_us, duration_us, 1, tid});
    }

    void add_event_from_timepoints(const std::string& name, const std::string& category,
                                   std::chrono::steady_clock::time_point start,
                                   std::chrono::steady_clock::time_point end, int tid = 1) {
        if (!m_enabled) return;
        int64_t start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start.time_since_epoch()).count();
        int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        add_complete_event(name, category, start_us, dur_us, tid);
    }

    void add_metric_event(const std::string& name, const std::string& category,
                          std::chrono::steady_clock::time_point start,
                          int64_t duration_us, int tid = 1) {
        if (!m_enabled) return;
        int64_t start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start.time_since_epoch()).count();
        add_complete_event(name, category, start_us, duration_us, tid);
    }

    void flush() {
        if (!m_enabled || m_events.empty()) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        write_events();
    }

    ~ChromeTrace() {
        if (!m_enabled || m_events.empty()) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        write_events();
    }

private:
    ChromeTrace() {
        const char* env = std::getenv("OPENVINO_CHROME_TRACE");
        if (env && std::string(env).size() > 0) {
            m_enabled = true;
            m_output_path = env;
        }
    }

    ChromeTrace(const ChromeTrace&) = delete;
    ChromeTrace& operator=(const ChromeTrace&) = delete;

    void write_events() {
        std::ofstream file(m_output_path, std::ios::trunc);
        if (!file.is_open()) return;

        file << "[\n";
        for (size_t i = 0; i < m_events.size(); ++i) {
            const auto& e = m_events[i];
            file << "  {\"name\": \"" << e.name
                 << "\", \"cat\": \"" << e.category
                 << "\", \"ph\": \"" << e.phase
                 << "\", \"ts\": " << e.timestamp_us
                 << ", \"dur\": " << e.duration_us
                 << ", \"pid\": " << e.pid
                 << ", \"tid\": " << e.tid
                 << "}";
            if (i + 1 < m_events.size()) file << ",";
            file << "\n";
        }
        file << "]\n";
    }

    bool m_enabled = false;
    std::string m_output_path;
    std::vector<TraceEvent> m_events;
    std::mutex m_mutex;
};

inline std::string trace_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    struct tm tm_buf;
    localtime_r(&time_t_now, &tm_buf);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms));
    return std::string(buf);
}

struct ModelMetrics {
    float vision_encoder_us = 0;
    float tokenizer_us = 0;
    float text_embeddings_us = 0;
    float vision_embeddings_merger_us = 0;
    float vision_embeddings_pos_us = 0;
    float lm_prefill_us = 0;
    bool collecting = false;

    void reset() {
        vision_encoder_us = 0;
        tokenizer_us = 0;
        text_embeddings_us = 0;
        vision_embeddings_merger_us = 0;
        vision_embeddings_pos_us = 0;
        lm_prefill_us = 0;
    }
};

inline ModelMetrics& get_model_metrics() {
    static thread_local ModelMetrics metrics;
    return metrics;
}

class ScopedTrace {
public:
    ScopedTrace(const std::string& name, const std::string& category = "model")
        : m_name(name), m_category(category) {
        auto& trace = ChromeTrace::get_instance();
        m_trace_enabled = trace.is_enabled();
        m_metrics_enabled = get_model_metrics().collecting;
        if (m_trace_enabled || m_metrics_enabled) {
            m_start = std::chrono::steady_clock::now();
            m_active = true;
        }
    }

    ~ScopedTrace() {
        end();
    }

    void end() {
        if (!m_active) return;
        m_active = false;
        auto end_time = std::chrono::steady_clock::now();
        int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - m_start).count();
        if (m_trace_enabled) {
            auto& trace = ChromeTrace::get_instance();
            int64_t start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                m_start.time_since_epoch()).count();
            trace.add_complete_event(m_name, m_category, start_us, dur_us);
            GENAI_INFO("[TRACE] [%s] %s: %.3f ms", trace_timestamp().c_str(), m_name.c_str(), dur_us / 1000.0);
        }
        if (m_metrics_enabled) {
            auto& mm = get_model_metrics();
            if (m_name.find("VisionEncoder") == 0) mm.vision_encoder_us += dur_us;
            else if (m_name.find("Tokenizer") == 0) mm.tokenizer_us += dur_us;
            else if (m_name == "TextEmbeddings") mm.text_embeddings_us += dur_us;
            else if (m_name.find("VisionEmbeddingsMerger") == 0) mm.vision_embeddings_merger_us += dur_us;
            else if (m_name.find("VisionEmbeddingsPos") == 0) mm.vision_embeddings_pos_us += dur_us;
            else if (m_name.find("LanguageModel") == 0) mm.lm_prefill_us += dur_us;
        }
    }

private:
    std::string m_name;
    std::string m_category;
    std::chrono::steady_clock::time_point m_start;
    bool m_active = false;
    bool m_trace_enabled = false;
    bool m_metrics_enabled = false;
};

}  // namespace ov::genai
