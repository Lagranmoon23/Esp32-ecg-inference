#include "ecg_pipeline.hpp"

#include <algorithm>
#include <cmath>

namespace ecg {

void EcgPipeline::reset()
{
    filter_.reset();
    normalizer_.reset();
    detector_.reset();
    sample_index_ = 0;
    normalized_ring_.fill(0.0f);
    normalized_indices_.fill(-1);
    peaks_.fill(0);
    peak_count_ = 0;
}

void EcgPipeline::store_normalized(float value, int64_t index)
{
    const auto slot = static_cast<std::size_t>(index) % kNormalizedRingSize;
    normalized_ring_[slot] = value;
    normalized_indices_[slot] = index;
}

bool EcgPipeline::normalized_at(int64_t index, float &value) const
{
    if (index < 0) {
        return false;
    }
    const auto slot = static_cast<std::size_t>(index) % kNormalizedRingSize;
    if (normalized_indices_[slot] != index) {
        return false;
    }
    value = normalized_ring_[slot];
    return std::isfinite(value);
}

void EcgPipeline::append_peak(int64_t peak)
{
    if (peak_count_ < kPeakHistorySize) {
        peaks_[peak_count_++] = peak;
        return;
    }
    std::move(peaks_.begin() + 1, peaks_.end(), peaks_.begin());
    peaks_.back() = peak;
}

float EcgPipeline::interval_median(
    const std::array<int64_t, kPeakHistorySize> &peaks,
    std::size_t begin_interval,
    std::size_t end_interval)
{
    std::array<float, 60> values{};
    std::size_t count = 0;
    for (std::size_t i = begin_interval; i < end_interval; ++i) {
        values[count++] = static_cast<float>(peaks[i + 1] - peaks[i])
            / static_cast<float>(kSampleRateHz);
    }
    if (count == 0) {
        return 1.0f;
    }
    const auto middle = count / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.begin() + count);
    const float upper = values[middle];
    if ((count & 1U) != 0U) {
        return upper;
    }
    const float lower = *std::max_element(values.begin(), values.begin() + middle);
    return 0.5f * (lower + upper);
}

bool EcgPipeline::build_inputs(std::array<float, kBeatLength> &beat,
                               std::array<float, 6> &rr_raw,
                               int64_t &center) const
{
    if (peak_count_ < 3) {
        return false;
    }
    const std::size_t index = peak_count_ - 2; // classify after post-RR arrives
    center = peaks_[index];
    for (int offset = -kBeatHalfWindow; offset < kBeatHalfWindow; ++offset) {
        if (!normalized_at(center + offset, beat[offset + kBeatHalfWindow])) {
            return false;
        }
    }

    const float pre_rr = static_cast<float>(peaks_[index] - peaks_[index - 1])
        / static_cast<float>(kSampleRateHz);
    const float post_rr = static_cast<float>(peaks_[index + 1] - peaks_[index])
        / static_cast<float>(kSampleRateHz);
    const std::size_t local_begin = index > 10 ? index - 10 : 0;
    const std::size_t global_begin = index > 60 ? index - 60 : 0;
    const float local_rr = std::max(
        interval_median(peaks_, local_begin, index), 1.0e-3f);
    const float global_rr = std::max(
        interval_median(peaks_, global_begin, index), 1.0e-3f);

    rr_raw = {
        std::clamp(pre_rr, 0.2f, 2.0f),
        std::clamp(post_rr, 0.2f, 2.0f),
        std::clamp(local_rr, 0.2f, 2.0f),
        std::clamp(global_rr, 0.2f, 2.0f),
        std::clamp(pre_rr / local_rr, 0.3f, 2.0f),
        std::clamp(post_rr / local_rr, 0.3f, 2.0f),
    };
    return true;
}

bool EcgPipeline::push(float raw, bool lead_off,
                       SampleTelemetry &telemetry, BeatFeatures &features)
{
    telemetry = {};
    telemetry.raw = raw;
    telemetry.lead_off = lead_off;
    if (lead_off) {
        if (!disconnected_) {
            reset();
            disconnected_ = true;
        }
        return false;
    }
    if (disconnected_) {
        reset();
        disconnected_ = false;
    }

    telemetry.sample_index = sample_index_;
    telemetry.filtered = filter_.push(raw);
    telemetry.normalized_valid = normalizer_.push(
        telemetry.filtered, sample_index_, telemetry.normalized);
    if (telemetry.normalized_valid) {
        store_normalized(telemetry.normalized, sample_index_);
    }

    const auto peak = detector_.push(telemetry.filtered, sample_index_);
    ++sample_index_;
    if (!peak) {
        return false;
    }
    append_peak(*peak);

    std::array<float, kBeatLength> beat{};
    std::array<float, 6> rr_raw{};
    int64_t center = 0;
    if (!build_inputs(beat, rr_raw, center)) {
        return false;
    }
    std::array<float, 6> rr_standardized{};
    for (std::size_t i = 0; i < rr_standardized.size(); ++i) {
        rr_standardized[i] = (rr_raw[i] - generated::kRrMean[i])
            / generated::kRrStd[i];
    }

    features = {};
    features.r_sample = center;
    features.bpm = 60.0f / rr_raw[1];
    features.beat = beat;
    features.rr_raw = rr_raw;
    features.rr_standardized = rr_standardized;
    return true;
}

}  // namespace ecg
