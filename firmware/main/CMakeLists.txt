# Register first: ESP-IDF's dependency scan stops at this call. Model checks
# and ESP-DL utility loading must only run during the actual configure stage.
idf_component_register(
    SRCS
        "app_main.cpp"
        "ecg_dsp.cpp"
        "ecg_model.cpp"
        "ecg_pipeline.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_adc esp_driver_gpio esp_timer
)

set(model_file "${COMPONENT_DIR}/models/ecg.espdl")
if(NOT EXISTS "${model_file}")
    message(FATAL_ERROR
        "Missing main/models/ecg.espdl. Copy stage1_n_s_vf_w8a16.espdl here before building.")
endif()

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
else()
    message(FATAL_ERROR "ESP-DL component was not found")
endif()
# ESP-DL's helper references cmake_dir from its caller's scope.
set(cmake_dir "${espdl_dir}/fbs_loader/cmake")
include("${cmake_dir}/utilities.cmake")

target_add_aligned_binary_data(${COMPONENT_LIB} "${model_file}" BINARY)

set(board_eval_file "${COMPONENT_DIR}/data/board_eval.bin")
option(ECG_EMBED_BOARD_EVAL "Embed the offline board-evaluation dataset" OFF)
if(ECG_EMBED_BOARD_EVAL AND EXISTS "${board_eval_file}")
    target_add_aligned_binary_data(${COMPONENT_LIB} "${board_eval_file}" BINARY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ECG_BOARD_EVAL_EMBEDDED=1)
    message(STATUS "Embedding board evaluation dataset: ${board_eval_file}")
elseif(ECG_EMBED_BOARD_EVAL)
    message(STATUS
        "Board evaluation dataset is absent; copy it to main/data/board_eval.bin when ready")
else()
    message(STATUS "Offline board evaluation dataset embedding is disabled")
endif()
