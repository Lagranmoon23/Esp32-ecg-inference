#pragma once

#include "ecg_generated_config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ecg {

inline constexpr int kSampleRateHz = generated::kSampleRateHz;
inline constexpr int kBeatHalfWindow = 99;
inline constexpr int kBeatLength = 198;

class SosFilter {
public:
    void reset();
    float push(float sample);

private:
    struct State {
        float z1 = 0.0f;
        float z2 = 0.0f;
    };
    std::array<State, generated::kFilterSections> states_{};
    bool initialized_ = false;
};

class RollingNormalizer {
public:
    static constexpr std::size_t kWindow = 10 * kSampleRateHz;
    static constexpr std::size_t kUpdate = 1 * kSampleRateHz;

    void reset();
    bool push(float filtered, int64_t sample_index, float &normalized);

private:
    void update_statistics();
    static float median(float *values, std::size_t count);

    std::array<float, kWindow> history_{};
    std::array<float, kWindow> scratch_{};
    std::size_t write_ = 0;
    std::size_t count_ = 0;
    float center_ = 0.0f;
    float scale_ = 1.0f;
    bool ready_ = false;
};

class RPeakDetector {
public:
    void reset();
    std::optional<int64_t> push(float filtered, int64_t sample_index);

private:
    struct Candidate {
        int64_t index = -1;
        float strength = 0.0f;
    };
    struct PendingPeak {
        int64_t index = -1;
        float strength = 0.0f;
    };

    void remember_filtered(float value, int64_t index);
    float filtered_at(int64_t index) const;
    void queue_candidate(int64_t index, float strength);
    void process_candidate(const Candidate &candidate);
    void initialize_levels();
    int64_t refine_peak(int64_t candidate_index) const;

    static constexpr int kIntegrationWidth = 54;  // round(0.150 * 360)
    static constexpr int kCandidateDistance = 29; // round(0.080 * 360)
    static constexpr int kStartupSamples = 720;
    static constexpr int kRefractorySamples = 90; // round(0.250 * 360)
    static constexpr int kRefineBefore = 72;
    static constexpr int kRefineAfter = 29;
    static constexpr std::size_t kSignalRing = 512;

    std::array<float, 5> derivative_history_{};
    std::array<float, kIntegrationWidth> energy_history_{};
    std::size_t energy_write_ = 0;
    float energy_sum_ = 0.0f;
    float envelope_prev2_ = 0.0f;
    float envelope_prev1_ = 0.0f;
    bool have_two_envelopes_ = false;
    int envelope_count_ = 0;

    std::array<float, kSignalRing> signal_ring_{};
    std::array<int64_t, kSignalRing> signal_indices_{};

    std::optional<Candidate> queued_candidate_;
    std::optional<PendingPeak> pending_peak_;
    std::optional<int64_t> emitted_peak_;
    std::array<float, 256> startup_strengths_{};
    std::size_t startup_count_ = 0;
    float noise_level_ = 0.0f;
    float signal_level_ = 0.0f;
    bool levels_ready_ = false;
};

}  // namespace ecg
