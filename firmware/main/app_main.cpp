#include "ecg_generated_config.hpp"
#include "ecg_model.hpp"
#include "ecg_pipeline.hpp"

#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

namespace {

constexpr const char *kTag = "ECG_LIVE";
constexpr std::uint64_t kSamplePeriodUs =
    1000000ULL / ecg::kSampleRateHz;
constexpr std::size_t kSampleStackBytes = 16 * 1024;
constexpr std::size_t kInferenceStackBytes = 48 * 1024;
constexpr std::size_t kBeatQueueLength = 4;

DRAM_ATTR static StackType_t g_sample_stack[kSampleStackBytes];
DRAM_ATTR static StackType_t g_inference_stack[kInferenceStackBytes];
static StaticTask_t g_sample_tcb;
static StaticTask_t g_inference_tcb;
static TaskHandle_t g_sample_task = nullptr;
static TaskHandle_t g_inference_task = nullptr;

DRAM_ATTR static std::uint8_t
    g_beat_queue_storage[kBeatQueueLength * sizeof(ecg::BeatFeatures)];
static StaticQueue_t g_beat_queue_control;
static QueueHandle_t g_beat_queue = nullptr;

static adc_oneshot_unit_handle_t g_adc = nullptr;
static esp_timer_handle_t g_sample_timer = nullptr;

void wait_forever()
{
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

bool configure_lead_pin(int gpio_number)
{
    if (gpio_number < 0) {
        return true;
    }
    gpio_config_t config{};
    config.pin_bit_mask = 1ULL << gpio_number;
    config.mode = GPIO_MODE_INPUT;
    config.pull_up_en = GPIO_PULLUP_DISABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;
    return gpio_config(&config) == ESP_OK;
}

bool lead_is_off()
{
    bool off = false;
#if CONFIG_ECG_LO_PLUS_GPIO >= 0
    off = off || gpio_get_level(
        static_cast<gpio_num_t>(CONFIG_ECG_LO_PLUS_GPIO));
#endif
#if CONFIG_ECG_LO_MINUS_GPIO >= 0
    off = off || gpio_get_level(
        static_cast<gpio_num_t>(CONFIG_ECG_LO_MINUS_GPIO));
#endif
    return off;
}

void sample_timer_callback(void *)
{
    if (g_sample_task != nullptr) {
        xTaskNotifyGive(g_sample_task);
    }
}

void sample_task(void *)
{
    void *pipeline_memory = heap_caps_malloc(
        sizeof(ecg::EcgPipeline), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (pipeline_memory == nullptr) {
        ESP_LOGE(kTag, "PSRAM allocation failed for ECG pipeline");
        wait_forever();
    }
    auto *pipeline = new (pipeline_memory) ecg::EcgPipeline();
    ESP_LOGI(kTag, "ECG pipeline allocated in PSRAM (%u bytes)",
             static_cast<unsigned>(sizeof(ecg::EcgPipeline)));
    std::uint64_t received_ticks = 0;
    std::uint64_t dropped_ticks = 0;
    std::uint64_t dropped_beats = 0;
    bool previous_lead_off = true;

    ESP_LOGI(kTag, "sampling task ready on CPU%d", xPortGetCoreID());
    while (true) {
        const std::uint32_t notifications =
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (notifications > 1) {
            dropped_ticks += notifications - 1;
        }
        ++received_ticks;

        int raw = 0;
        const esp_err_t read_result = adc_oneshot_read(
            g_adc, static_cast<adc_channel_t>(CONFIG_ECG_ADC_CHANNEL), &raw);
        if (read_result != ESP_OK) {
            ESP_LOGE(kTag, "ADC read failed: %s",
                     esp_err_to_name(read_result));
            continue;
        }

        const bool lead_off = lead_is_off();
        if (lead_off != previous_lead_off) {
            ESP_LOGI(kTag, "electrodes: %s",
                     lead_off ? "DETACHED" : "CONNECTED; warming up for 10 s");
            previous_lead_off = lead_off;
        }

        ecg::SampleTelemetry telemetry;
        ecg::BeatFeatures features;
        if (pipeline->push(static_cast<float>(raw), lead_off,
                           telemetry, features)) {
            if (xQueueSend(g_beat_queue, &features, 0) != pdTRUE) {
                ++dropped_beats;
                ESP_LOGW(kTag, "inference queue full; dropped beat=%" PRIu64,
                         dropped_beats);
            }
        }

#if CONFIG_ECG_STREAM_EVERY_N > 0
        if (received_ticks % CONFIG_ECG_STREAM_EVERY_N == 0) {
            std::printf("ECG,%" PRId64 ",%d,%.6f,%.6f,%d\n",
                        telemetry.sample_index, raw, telemetry.filtered,
                        telemetry.normalized_valid ? telemetry.normalized : 0.0f,
                        telemetry.lead_off ? 1 : 0);
        }
#endif

        if (received_ticks % (30 * ecg::kSampleRateHz) == 0) {
            ESP_LOGI(kTag,
                     "sampling health: ticks=%" PRIu64
                     " dropped_ticks=%" PRIu64 " dropped_beats=%" PRIu64,
                     received_ticks, dropped_ticks, dropped_beats);
        }
    }
}

void inference_task(void *)
{
    ESP_LOGI(kTag, "loading ESP-DL model on CPU%d", xPortGetCoreID());
    auto *model = new (std::nothrow) ecg::EcgModel();
    if (model == nullptr || !model->ready()) {
        ESP_LOGE(kTag, "model initialization failed; sampling not started");
        wait_forever();
    }

    ESP_LOGI(kTag, "model ready; starting %d Hz sampling",
             ecg::kSampleRateHz);
    ESP_ERROR_CHECK(esp_timer_start_periodic(g_sample_timer, kSamplePeriodUs));

    static constexpr const char *kNames[] = {"N", "S", "VF"};
    std::uint64_t inference_count = 0;
    std::int64_t inference_time_us = 0;
    while (true) {
        ecg::BeatFeatures features;
        if (xQueueReceive(g_beat_queue, &features, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        ecg::InferenceResult result;
        const std::int64_t start_us = esp_timer_get_time();
        const bool ok = model->infer(features.beat, features.rr_standardized,
                                     result);
        inference_time_us += esp_timer_get_time() - start_us;
        ++inference_count;
        if (!ok || result.class_id < 0 || result.class_id >= 3) {
            ESP_LOGE(kTag, "live inference failed at R sample=%" PRId64,
                     features.r_sample);
            continue;
        }

        std::printf(
            "PRED,%" PRId64 ",%s,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            features.r_sample, kNames[result.class_id], features.bpm,
            result.logits[0], result.logits[1], result.logits[2],
            features.rr_raw[0], features.rr_raw[1], features.rr_raw[2]);

        if (inference_count % 32 == 0) {
            ESP_LOGI(kTag, "inference health: beats=%" PRIu64
                     " average=%.3f ms/beat queue=%u",
                     inference_count,
                     static_cast<double>(inference_time_us)
                         / 1000.0 / inference_count,
                     static_cast<unsigned>(uxQueueMessagesWaiting(g_beat_queue)));
        }
    }
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(kTag, "starting real-time AD8232 ECG pipeline");
    if (!ecg::generated::kReady) {
        ESP_LOGE(kTag, "generated ECG configuration is unavailable");
        return;
    }

    g_beat_queue = xQueueCreateStatic(
        kBeatQueueLength, sizeof(ecg::BeatFeatures), g_beat_queue_storage,
        &g_beat_queue_control);
    if (g_beat_queue == nullptr) {
        ESP_LOGE(kTag, "beat queue creation failed");
        return;
    }

    adc_oneshot_unit_init_cfg_t unit_config{};
    unit_config.unit_id = ADC_UNIT_1;
    unit_config.ulp_mode = ADC_ULP_MODE_DISABLE;
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_config, &g_adc));

    adc_oneshot_chan_cfg_t channel_config{};
    channel_config.atten = ADC_ATTEN_DB_12;
    channel_config.bitwidth = ADC_BITWIDTH_12;
    ESP_ERROR_CHECK(adc_oneshot_config_channel(
        g_adc, static_cast<adc_channel_t>(CONFIG_ECG_ADC_CHANNEL),
        &channel_config));

    ESP_ERROR_CHECK(configure_lead_pin(CONFIG_ECG_LO_PLUS_GPIO)
                        ? ESP_OK : ESP_FAIL);
    ESP_ERROR_CHECK(configure_lead_pin(CONFIG_ECG_LO_MINUS_GPIO)
                        ? ESP_OK : ESP_FAIL);

    esp_timer_create_args_t timer_args{};
    timer_args.callback = sample_timer_callback;
    timer_args.name = "ecg_360hz";
    timer_args.dispatch_method = ESP_TIMER_TASK;
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &g_sample_timer));

    g_sample_task = xTaskCreateStaticPinnedToCore(
        sample_task, "ecg_sample", kSampleStackBytes, nullptr, 15,
        g_sample_stack, &g_sample_tcb, 0);
    g_inference_task = xTaskCreateStaticPinnedToCore(
        inference_task, "ecg_infer", kInferenceStackBytes, nullptr, 10,
        g_inference_stack, &g_inference_tcb, 1);
    if (g_sample_task == nullptr || g_inference_task == nullptr) {
        ESP_LOGE(kTag, "static task creation failed");
        return;
    }

    ESP_LOGI(kTag,
             "tasks ready: sampling=CPU0, inference=CPU1, ADC1 channel=%d "
             "(GPIO%d), LO+=GPIO%d, LO-=GPIO%d",
             CONFIG_ECG_ADC_CHANNEL, CONFIG_ECG_ADC_CHANNEL + 1,
             CONFIG_ECG_LO_PLUS_GPIO, CONFIG_ECG_LO_MINUS_GPIO);
}
