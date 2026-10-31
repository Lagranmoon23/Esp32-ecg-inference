#pragma once

#include "ecg_dsp.hpp"

#include <array>
#include <cstdint>

namespace ecg {

struct SampleTelemetry {
    int64_t sample_index = 0;
    float raw = 0.0f;
    float filtered = 0.0f;
    float normalized = 0.0f;
    bool normalized_valid = false;
    bool lead_off = false;
};

struct BeatFeatures {
    int64_t r_sample = 0;
    float bpm = 0.0f;
    std::array<float, kBeatLength> beat{};
    std::array<float, 6> rr_raw{};
    std::array<float, 6> rr_standardized{};
};

class EcgPipeline {
public:
    EcgPipeline() { reset(); }

    void reset();
    bool push(float raw, bool lead_off,
              SampleTelemetry &telemetry, BeatFeatures &features);

private:
    static constexpr std::size_t kNormalizedRingSize = 8192;
    static constexpr std::size_t kPeakHistorySize = 64;

    void store_normalized(float value, int64_t index);
    bool normalized_at(int64_t index, float &value) const;
    void append_peak(int64_t peak);
    bool build_inputs(std::array<float, kBeatLength> &beat,
                      std::array<float, 6> &rr_raw,
                      int64_t &center) const;
    static float interval_median(const std::array<int64_t, kPeakHistorySize> &peaks,
                                 std::size_t begin_interval,
                                 std::size_t end_interval);

    SosFilter filter_;
    RollingNormalizer normalizer_;
    RPeakDetector detector_;
    int64_t sample_index_ = 0;
    bool disconnected_ = false;

    std::array<float, kNormalizedRingSize> normalized_ring_{};
    std::array<int64_t, kNormalizedRingSize> normalized_indices_{};
    std::array<int64_t, kPeakHistorySize> peaks_{};
    std::size_t peak_count_ = 0;
};

}  // namespace ecg
