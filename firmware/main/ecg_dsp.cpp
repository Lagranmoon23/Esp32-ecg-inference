#include "ecg_dsp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ecg {

void SosFilter::reset()
{
    states_ = {};
    initialized_ = false;
}

float SosFilter::push(float sample)
{
    if (!initialized_) {
        for (std::size_t i = 0; i < generated::kFilterSections; ++i) {
            states_[i].z1 = generated::kFilterZi[i][0] * sample;
            states_[i].z2 = generated::kFilterZi[i][1] * sample;
        }
        initialized_ = true;
    }
    float value = sample;
    for (std::size_t i = 0; i < generated::kFilterSections; ++i) {
        const auto &sos = generated::kFilterSos[i];
        auto &state = states_[i];
        // Direct Form II transposed. Exporter guarantees a0 == 1.
        const float output = sos[0] * value + state.z1;
        state.z1 = sos[1] * value - sos[4] * output + state.z2;
        state.z2 = sos[2] * value - sos[5] * output;
        value = output;
    }
    return value;
}

void RollingNormalizer::reset()
{
    history_.fill(0.0f);
    scratch_.fill(0.0f);
    write_ = 0;
    count_ = 0;
    center_ = 0.0f;
    scale_ = 1.0f;
    ready_ = false;
}

float RollingNormalizer::median(float *values, std::size_t count)
{
    if (count == 0) {
        return 0.0f;
    }
    const std::size_t middle = count / 2;
    std::nth_element(values, values + middle, values + count);
    const float upper = values[middle];
    if ((count & 1U) != 0U) {
        return upper;
    }
    const float lower = *std::max_element(values, values + middle);
    return 0.5f * (lower + upper);
}

void RollingNormalizer::update_statistics()
{
    for (std::size_t i = 0; i < kWindow; ++i) {
        scratch_[i] = history_[i];
    }
    center_ = median(scratch_.data(), kWindow);
    for (std::size_t i = 0; i < kWindow; ++i) {
        scratch_[i] = std::fabs(history_[i] - center_);
    }
    scale_ = 1.4826f * median(scratch_.data(), kWindow);
    if (!std::isfinite(scale_) || scale_ < 1.0e-6f) {
        float sum = 0.0f;
        float square_sum = 0.0f;
        for (float value : history_) {
            sum += value;
            square_sum += value * value;
        }
        const float mean = sum / static_cast<float>(kWindow);
        scale_ = std::sqrt(std::max(0.0f, square_sum / kWindow - mean * mean)) + 1.0e-8f;
    }
    ready_ = true;
}

bool RollingNormalizer::push(float filtered, int64_t sample_index, float &normalized)
{
    // Match Python: statistics for [block_start, block_start+360) use only
    // the preceding 3600 samples.
    if (count_ == kWindow && sample_index % static_cast<int64_t>(kUpdate) == 0) {
        update_statistics();
    }

    bool valid = false;
    if (ready_) {
        normalized = std::clamp((filtered - center_) / scale_, -5.0f, 5.0f);
        valid = std::isfinite(normalized);
    }

    history_[write_] = filtered;
    write_ = (write_ + 1) % kWindow;
    count_ = std::min(count_ + 1, kWindow);
    return valid;
}

void RPeakDetector::reset()
{
    derivative_history_.fill(0.0f);
    energy_history_.fill(0.0f);
    energy_write_ = 0;
    energy_sum_ = 0.0f;
    envelope_prev2_ = 0.0f;
    envelope_prev1_ = 0.0f;
    have_two_envelopes_ = false;
    envelope_count_ = 0;
    signal_ring_.fill(0.0f);
    signal_indices_.fill(-1);
    queued_candidate_.reset();
    pending_peak_.reset();
    emitted_peak_.reset();
    startup_strengths_.fill(0.0f);
    startup_count_ = 0;
    noise_level_ = 0.0f;
    signal_level_ = 0.0f;
    levels_ready_ = false;
}

void RPeakDetector::remember_filtered(float value, int64_t index)
{
    const auto slot = static_cast<std::size_t>(index) % kSignalRing;
    signal_ring_[slot] = value;
    signal_indices_[slot] = index;
}

float RPeakDetector::filtered_at(int64_t index) const
{
    if (index < 0) {
        return 0.0f;
    }
    const auto slot = static_cast<std::size_t>(index) % kSignalRing;
    return signal_indices_[slot] == index ? signal_ring_[slot] : 0.0f;
}

void RPeakDetector::initialize_levels()
{
    if (levels_ready_) {
        return;
    }
    if (startup_count_ == 0) {
        noise_level_ = 0.0f;
        signal_level_ = 1.0e-12f;
    } else {
        std::sort(startup_strengths_.begin(), startup_strengths_.begin() + startup_count_);
        const auto percentile = [this](float q) {
            const float position = q * static_cast<float>(startup_count_ - 1);
            const auto low = static_cast<std::size_t>(position);
            const auto high = std::min(low + 1, startup_count_ - 1);
            const float fraction = position - static_cast<float>(low);
            return startup_strengths_[low] * (1.0f - fraction)
                + startup_strengths_[high] * fraction;
        };
        noise_level_ = percentile(0.50f);
        signal_level_ = percentile(0.95f);
        if (signal_level_ <= noise_level_) {
            signal_level_ = startup_strengths_[startup_count_ - 1] + 1.0e-12f;
        }
    }
    levels_ready_ = true;
}

int64_t RPeakDetector::refine_peak(int64_t candidate_index) const
{
    int64_t best_index = candidate_index;
    float best_value = -1.0f;
    for (int64_t index = candidate_index - kRefineBefore;
         index <= candidate_index + kRefineAfter; ++index) {
        const float value = std::fabs(filtered_at(index));
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

void RPeakDetector::process_candidate(const Candidate &candidate)
{
    if (candidate.index < kStartupSamples) {
        if (startup_count_ < startup_strengths_.size()) {
            startup_strengths_[startup_count_++] = candidate.strength;
        }
        return;
    }
    initialize_levels();
    const float threshold = noise_level_ + 0.25f * (signal_level_ - noise_level_);
    if (candidate.strength >= threshold) {
        const int64_t refined = refine_peak(candidate.index);
        if (!pending_peak_) {
            pending_peak_ = PendingPeak{refined, candidate.strength};
        } else if (refined - pending_peak_->index >= kRefractorySamples) {
            emitted_peak_ = pending_peak_->index;
            pending_peak_ = PendingPeak{refined, candidate.strength};
        } else if (candidate.strength > pending_peak_->strength) {
            pending_peak_ = PendingPeak{refined, candidate.strength};
        }
        signal_level_ = 0.125f * candidate.strength + 0.875f * signal_level_;
    } else {
        noise_level_ = 0.125f * candidate.strength + 0.875f * noise_level_;
    }
}

void RPeakDetector::queue_candidate(int64_t index, float strength)
{
    Candidate incoming{index, strength};
    if (!queued_candidate_) {
        queued_candidate_ = incoming;
        return;
    }
    if (index - queued_candidate_->index < kCandidateDistance) {
        if (strength > queued_candidate_->strength) {
            queued_candidate_ = incoming;
        }
        return;
    }
    process_candidate(*queued_candidate_);
    queued_candidate_ = incoming;
}

std::optional<int64_t> RPeakDetector::push(float filtered, int64_t sample_index)
{
    emitted_peak_.reset();
    remember_filtered(filtered, sample_index);

    for (std::size_t i = derivative_history_.size() - 1; i > 0; --i) {
        derivative_history_[i] = derivative_history_[i - 1];
    }
    derivative_history_[0] = filtered;
    const float derivative = (static_cast<float>(kSampleRateHz) / 8.0f)
        * (derivative_history_[0] + 2.0f * derivative_history_[1]
           - 2.0f * derivative_history_[3] - derivative_history_[4]);
    const float energy = derivative * derivative;
    energy_sum_ += energy - energy_history_[energy_write_];
    energy_history_[energy_write_] = energy;
    energy_write_ = (energy_write_ + 1) % kIntegrationWidth;
    const float envelope = energy_sum_ / static_cast<float>(kIntegrationWidth);

    if (envelope_count_ >= 2 && envelope_prev1_ > envelope_prev2_
        && envelope_prev1_ >= envelope) {
        queue_candidate(sample_index - 1, envelope_prev1_);
    }
    envelope_prev2_ = envelope_prev1_;
    envelope_prev1_ = envelope;
    ++envelope_count_;

    if (queued_candidate_
        && sample_index - queued_candidate_->index >= kCandidateDistance) {
        process_candidate(*queued_candidate_);
        queued_candidate_.reset();
    }

    // Finalize only after the replacement window has closed.
    if (!emitted_peak_ && pending_peak_
        && sample_index - pending_peak_->index >= kRefractorySamples + kRefineAfter) {
        emitted_peak_ = pending_peak_->index;
        pending_peak_.reset();
    }
    return emitted_peak_;
}

}  // namespace ecg
