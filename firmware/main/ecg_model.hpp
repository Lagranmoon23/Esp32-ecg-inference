#pragma once

#include "ecg_dsp.hpp"

#include <array>
#include <memory>

namespace dl {
class Model;
class TensorBase;
}  // namespace dl

namespace ecg {

struct InferenceResult {
    int class_id = -1;
    std::array<float, 3> logits{};
};

class EcgModel {
public:
    EcgModel();
    ~EcgModel();

    EcgModel(const EcgModel &) = delete;
    EcgModel &operator=(const EcgModel &) = delete;

    bool ready() const { return ready_; }
    bool infer(const std::array<float, kBeatLength> &beat,
               const std::array<float, 6> &rr,
               InferenceResult &result);
    bool run_golden_test();
    bool run_board_evaluation();

private:
    dl::TensorBase *find_input(const char *preferred_name, std::size_t expected_size);

    dl::Model *model_ = nullptr;
    dl::TensorBase *model_beat_ = nullptr;
    dl::TensorBase *model_rr_ = nullptr;
    dl::TensorBase *model_output_ = nullptr;
    dl::TensorBase *float_beat_ = nullptr;
    dl::TensorBase *float_rr_ = nullptr;
    dl::TensorBase *float_output_ = nullptr;
    bool ready_ = false;
};

}  // namespace ecg
