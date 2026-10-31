#include "ecg_generated_config.hpp"
#include "ecg_model.hpp"

#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"

#include <cstddef>
#include <new>

namespace {

constexpr const char *kTag = "ECG_SMOKE";
constexpr std::size_t kSmokeStackBytes = 48 * 1024;

// The stack is fixed in internal DRAM. This avoids the heap/PSRAM allocator
// completely while ESP-DL constructs and executes the model.
DRAM_ATTR static StackType_t g_smoke_stack[kSmokeStackBytes];
static StaticTask_t g_smoke_tcb;
static TaskHandle_t g_smoke_task = nullptr;

void wait_forever()
{
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void smoke_task(void *)
{
    ESP_LOGI(kTag, "static task entered on CPU%d", xPortGetCoreID());
    ESP_LOGI(kTag, "internal free=%u largest=%u, PSRAM free=%u",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI(kTag, "constructing ESP-DL model");

    auto *model = new (std::nothrow) ecg::EcgModel();
    if (model == nullptr) {
        ESP_LOGE(kTag, "EcgModel allocation failed");
        wait_forever();
    }

    ESP_LOGI(kTag, "model constructor returned; internal free=%u, PSRAM free=%u",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

    if (!model->ready()) {
        ESP_LOGE(kTag, "model is not ready");
        wait_forever();
    }

    ESP_LOGI(kTag, "running embedded 900-sample board evaluation");
    const bool completed = model->run_board_evaluation();
    ESP_LOGI(kTag, "board evaluation execution: %s",
             completed ? "COMPLETED" : "FAILED");
    ESP_LOGI(kTag, "evaluation finished; realtime ADC sampling is intentionally disabled");

    wait_forever();
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(kTag, "minimal model smoke-test firmware");

    if (!ecg::generated::kReady) {
        ESP_LOGE(kTag, "generated ECG configuration is not ready");
        return;
    }

    g_smoke_task = xTaskCreateStaticPinnedToCore(
        smoke_task,
        "ecg_smoke",
        kSmokeStackBytes,
        nullptr,
        5,
        g_smoke_stack,
        &g_smoke_tcb,
        0);

    if (g_smoke_task == nullptr) {
        ESP_LOGE(kTag, "static task creation failed");
        return;
    }

    ESP_LOGI(kTag, "static smoke-test task created");
}
