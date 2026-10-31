#include "ecg_model.hpp"

#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>

extern const uint8_t ecg_espdl[] asm("_binary_ecg_espdl_start");

#ifdef ECG_BOARD_EVAL_EMBEDDED
extern const uint8_t board_eval_bin_start[] asm("_binary_board_eval_bin_start");
extern const uint8_t board_eval_bin_end[] asm("_binary_board_eval_bin_end");
#endif

namespace ecg {
namespace {
constexpr const char *kTag = "ECG_MODEL";
constexpr const char *kClassNames[] = {"N", "S", "VF"};

using ConfusionMatrix = std::array<std::array<unsigned, 3>, 3>;

float top_two_margin(const std::array<float, 3> &logits)
{
    float best = -std::numeric_limits<float>::infinity();
    float second = -std::numeric_limits<float>::infinity();
    for (const float value : logits) {
        if (value > best) {
            second = best;
            best = value;
        } else if (value > second) {
            second = value;
        }
    }
    return best - second;
}

void log_classification_metrics(const char *name, const ConfusionMatrix &matrix)
{
    unsigned total = 0;
    unsigned correct = 0;
    float macro_f1 = 0.0f;

    ESP_LOGI(kTag, "%s confusion matrix (rows=true, columns=predicted)", name);
    for (std::size_t row = 0; row < 3; ++row) {
        ESP_LOGI(kTag, "%s %s: [%u,%u,%u]", name, kClassNames[row],
                 matrix[row][0], matrix[row][1], matrix[row][2]);
        for (std::size_t column = 0; column < 3; ++column) {
            total += matrix[row][column];
        }
        correct += matrix[row][row];
    }

    for (std::size_t class_id = 0; class_id < 3; ++class_id) {
        unsigned actual = 0;
        unsigned predicted = 0;
        for (std::size_t other = 0; other < 3; ++other) {
            actual += matrix[class_id][other];
            predicted += matrix[other][class_id];
        }
        const float precision = predicted == 0
            ? 0.0f
            : static_cast<float>(matrix[class_id][class_id]) / predicted;
        const float recall = actual == 0
            ? 0.0f
            : static_cast<float>(matrix[class_id][class_id]) / actual;
        const float f1 = (precision + recall) == 0.0f
            ? 0.0f
            : 2.0f * precision * recall / (precision + recall);
        macro_f1 += f1;
        ESP_LOGI(kTag, "%s %s: precision=%.4f recall=%.4f f1=%.4f",
                 name, kClassNames[class_id], precision, recall, f1);
    }

    const float accuracy = total == 0
        ? 0.0f
        : static_cast<float>(correct) / total;
    ESP_LOGI(kTag, "%s metrics: samples=%u accuracy=%.4f macro_f1=%.4f",
             name, total, accuracy, macro_f1 / 3.0f);
}
}  // namespace

EcgModel::EcgModel()
{
    if (!generated::kReady) {
        ESP_LOGE(kTag, "ecg_generated_config.hpp is still the placeholder");
        return;
    }

    model_ = new dl::Model(
        reinterpret_cast<const char *>(ecg_espdl),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    model_beat_ = find_input("x1", kBeatLength);
    model_rr_ = find_input("x2", 6);

    auto outputs = model_->get_outputs();
    if (!outputs.empty()) {
        model_output_ = outputs.begin()->second;
    }
    if (!model_beat_ || !model_rr_ || !model_output_
        || model_output_->get_size() != 3) {
        ESP_LOGE(kTag, "Unexpected model inputs/outputs");
        return;
    }

    float_beat_ = new dl::TensorBase(
        {1, kBeatLength, 1, 1}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    float_rr_ = new dl::TensorBase({1, 6}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    float_output_ = new dl::TensorBase({1, 3}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    ready_ = float_beat_->data && float_rr_->data && float_output_->data;
    ESP_LOGI(kTag, "Model %s", ready_ ? "ready" : "tensor allocation failed");
}

EcgModel::~EcgModel()
{
    delete float_output_;
    delete float_rr_;
    delete float_beat_;
    delete model_;
}

dl::TensorBase *EcgModel::find_input(const char *preferred_name, std::size_t expected_size)
{
    auto inputs = model_->get_inputs();
    const auto named = inputs.find(preferred_name);
    if (named != inputs.end() && named->second->get_size() == expected_size) {
        return named->second;
    }
    for (const auto &[name, tensor] : inputs) {
        if (tensor && tensor->get_size() == expected_size) {
            ESP_LOGW(kTag, "Using input '%s' as '%s'", name.c_str(), preferred_name);
            return tensor;
        }
    }
    return nullptr;
}

bool EcgModel::infer(const std::array<float, kBeatLength> &beat,
                     const std::array<float, 6> &rr,
                     InferenceResult &result)
{
    if (!ready_) {
        return false;
    }
    std::memcpy(float_beat_->data, beat.data(), sizeof(float) * beat.size());
    std::memcpy(float_rr_->data, rr.data(), sizeof(float) * rr.size());
    if (!model_beat_->assign(float_beat_) || !model_rr_->assign(float_rr_)) {
        ESP_LOGE(kTag, "Input quantization/assignment failed");
        return false;
    }
    model_->run();
    if (!float_output_->assign(model_output_)) {
        ESP_LOGE(kTag, "Output dequantization failed");
        return false;
    }
    std::memcpy(result.logits.data(), float_output_->data,
                sizeof(float) * result.logits.size());
    result.class_id = static_cast<int>(
        std::max_element(result.logits.begin(), result.logits.end())
        - result.logits.begin());
    return true;
}

bool EcgModel::run_golden_test()
{
    if (!ready_ || generated::kGoldenCount == 0) {
        ESP_LOGE(kTag, "Golden vectors are unavailable");
        return false;
    }
#ifndef ECG_GENERATED_HAS_REFERENCE_LOGITS
    ESP_LOGE(kTag,
             "Golden reference logits are unavailable; regenerate "
             "ecg_generated_config.hpp with the updated export script");
    return false;
#else
    std::size_t matched = 0;
    std::size_t failed_inference = 0;
    float largest_logit_error = 0.0f;
    ConfusionMatrix host_confusion{};
    ConfusionMatrix device_confusion{};
    for (std::size_t i = 0; i < generated::kGoldenCount; ++i) {
        InferenceResult result;
        const bool ok = infer(generated::kGoldenBeats[i], generated::kGoldenRr[i], result);
        float max_abs_error = 0.0f;
        if (ok) {
            for (std::size_t j = 0; j < result.logits.size(); ++j) {
                max_abs_error = std::max(
                    max_abs_error,
                    std::fabs(result.logits[j]
                              - generated::kGoldenReferenceLogits[i][j]));
            }
        }
        largest_logit_error = std::max(largest_logit_error, max_abs_error);
        const int truth = generated::kGoldenLabels[i];
        const int host_prediction = generated::kGoldenReferencePredictions[i];
        if (truth >= 0 && truth < 3 && host_prediction >= 0 && host_prediction < 3) {
            ++host_confusion[truth][host_prediction];
        }
        if (ok && truth >= 0 && truth < 3
            && result.class_id >= 0 && result.class_id < 3) {
            ++device_confusion[truth][result.class_id];
        }
        const bool class_match = ok && result.class_id == host_prediction;
        if (class_match) {
            ++matched;
        } else {
            failed_inference += ok ? 0 : 1;
            ESP_LOGE(kTag,
                     "golden[%u] mismatch truth=%s host=%s device=%s "
                     "host_logits=[%.4f,%.4f,%.4f] host_margin=%.4f "
                     "device_logits=[%.4f,%.4f,%.4f] device_margin=%.4f",
                     static_cast<unsigned>(i),
                     truth >= 0 && truth < 3 ? kClassNames[truth] : "ERROR",
                     host_prediction >= 0 && host_prediction < 3
                         ? kClassNames[host_prediction] : "ERROR",
                     ok && result.class_id >= 0 && result.class_id < 3
                         ? kClassNames[result.class_id] : "ERROR",
                     generated::kGoldenReferenceLogits[i][0],
                     generated::kGoldenReferenceLogits[i][1],
                     generated::kGoldenReferenceLogits[i][2],
                     top_two_margin(generated::kGoldenReferenceLogits[i]),
                     result.logits[0], result.logits[1], result.logits[2],
                     top_two_margin(result.logits));
        }

        if ((i + 1) % 16 == 0 || i + 1 == generated::kGoldenCount) {
            ESP_LOGI(kTag, "golden progress: %u/%u, prediction matches=%u",
                     static_cast<unsigned>(i + 1),
                     static_cast<unsigned>(generated::kGoldenCount),
                     static_cast<unsigned>(matched));
        }

        // Let the CPU0 idle task run so the watchdog is not starved during the
        // complete golden-suite inference loop.
        vTaskDelay(1);
    }
    const bool passed = matched == generated::kGoldenCount;
    log_classification_metrics("host", host_confusion);
    log_classification_metrics("device", device_confusion);
    ESP_LOGI(kTag,
             "golden summary: matches=%u/%u agreement=%.2f%% "
             "inference_failures=%u largest_logit_abs_error=%.5f result=%s",
             static_cast<unsigned>(matched),
             static_cast<unsigned>(generated::kGoldenCount),
             100.0f * static_cast<float>(matched)
                 / static_cast<float>(generated::kGoldenCount),
             static_cast<unsigned>(failed_inference),
             largest_logit_error,
             passed ? "PASS" : "FAIL");
    return passed;
#endif
}

bool EcgModel::run_board_evaluation()
{
#ifndef ECG_BOARD_EVAL_EMBEDDED
    ESP_LOGE(kTag, "Board evaluation dataset is not embedded");
    return false;
#else
    constexpr char kExpectedMagic[8] = {'E', 'C', 'G', 'E', 'V', '0', '0', '1'};
    constexpr std::size_t kHeaderBytes = 8 + 5 * sizeof(std::uint32_t);
    constexpr std::size_t kRecordBytes =
        2 * sizeof(std::int32_t)
        + (kBeatLength + 6 + 3) * sizeof(float);

    const std::uint8_t *cursor = board_eval_bin_start;
    const std::uint8_t *const end = board_eval_bin_end;
    const std::size_t data_size = static_cast<std::size_t>(end - cursor);
    if (data_size < kHeaderBytes
        || std::memcmp(cursor, kExpectedMagic, sizeof(kExpectedMagic)) != 0) {
        ESP_LOGE(kTag, "Invalid board evaluation package header");
        return false;
    }
    cursor += sizeof(kExpectedMagic);

    const auto read_u32 = [&cursor](std::uint32_t &value) {
        std::memcpy(&value, cursor, sizeof(value));
        cursor += sizeof(value);
    };
    std::uint32_t version = 0;
    std::uint32_t sample_count = 0;
    std::uint32_t beat_length = 0;
    std::uint32_t rr_length = 0;
    std::uint32_t class_count = 0;
    read_u32(version);
    read_u32(sample_count);
    read_u32(beat_length);
    read_u32(rr_length);
    read_u32(class_count);

    const std::size_t expected_size = kHeaderBytes
        + static_cast<std::size_t>(sample_count) * kRecordBytes;
    if (version != 1 || beat_length != kBeatLength || rr_length != 6
        || class_count != 3 || data_size != expected_size) {
        ESP_LOGE(kTag,
                 "Unsupported board evaluation package: version=%u samples=%u "
                 "beat=%u rr=%u classes=%u bytes=%u expected=%u",
                 version, sample_count, beat_length, rr_length, class_count,
                 static_cast<unsigned>(data_size),
                 static_cast<unsigned>(expected_size));
        return false;
    }

    ESP_LOGI(kTag, "board evaluation started: %u balanced DS2 samples",
             sample_count);
    ConfusionMatrix host_confusion{};
    ConfusionMatrix device_confusion{};
    unsigned host_device_matches = 0;
    unsigned inference_failures = 0;
    std::array<float, kBeatLength> beat{};
    std::array<float, 6> rr{};
    const std::int64_t start_us = esp_timer_get_time();

    for (std::uint32_t i = 0; i < sample_count; ++i) {
        std::int32_t truth = -1;
        std::int32_t host_prediction = -1;
        std::memcpy(&truth, cursor, sizeof(truth));
        cursor += sizeof(truth);
        std::memcpy(&host_prediction, cursor, sizeof(host_prediction));
        cursor += sizeof(host_prediction);
        std::memcpy(beat.data(), cursor, sizeof(float) * beat.size());
        cursor += sizeof(float) * beat.size();
        std::memcpy(rr.data(), cursor, sizeof(float) * rr.size());
        cursor += sizeof(float) * rr.size();
        cursor += 3 * sizeof(float);  // Host logits are retained in the package for audit.

        if (truth >= 0 && truth < 3
            && host_prediction >= 0 && host_prediction < 3) {
            ++host_confusion[truth][host_prediction];
        }

        InferenceResult result;
        const bool ok = infer(beat, rr, result);
        if (!ok || result.class_id < 0 || result.class_id >= 3) {
            ++inference_failures;
        } else {
            if (truth >= 0 && truth < 3) {
                ++device_confusion[truth][result.class_id];
            }
            if (result.class_id == host_prediction) {
                ++host_device_matches;
            }
        }

        if ((i + 1) % 100 == 0 || i + 1 == sample_count) {
            ESP_LOGI(kTag, "board evaluation progress: %u/%u", i + 1,
                     sample_count);
        }
        vTaskDelay(1);
    }

    const std::int64_t elapsed_us = esp_timer_get_time() - start_us;
    log_classification_metrics("host-900", host_confusion);
    log_classification_metrics("device-900", device_confusion);
    ESP_LOGI(kTag,
             "board evaluation summary: samples=%u failures=%u "
             "host_device_agreement=%.2f%% elapsed=%.3fs average=%.3fms/sample",
             sample_count, inference_failures,
             sample_count == 0
                 ? 0.0f
                 : 100.0f * static_cast<float>(host_device_matches)
                     / static_cast<float>(sample_count),
             static_cast<double>(elapsed_us) / 1000000.0,
             sample_count == 0
                 ? 0.0
                 : static_cast<double>(elapsed_us) / 1000.0 / sample_count);
    return inference_failures == 0;
#endif
}

}  // namespace ecg
