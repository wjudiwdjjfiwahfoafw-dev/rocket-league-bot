// Build against the same libtorch as GigaLearnBot. Operates on COPIES only.
// Usage: calibrate_policy INPUT_POLICY.lt OUTPUT_POLICY.lt TEMPERATURE
// Scales final linear weight and bias; mask and sampling remain in the native evaluator.
#include <torch/script.h>
#include <iostream>
#include <cmath>
int main(int argc, char** argv) {
    try {
        if (argc != 4 || std::string(argv[1]) == argv[2])
            throw std::runtime_error("Usage: calibrate_policy INPUT OUTPUT TEMPERATURE (distinct paths)");
        double temperature = std::stod(argv[3]);
        if (!std::isfinite(temperature) || temperature < 0.1 || temperature > 10)
            throw std::runtime_error("Temperature must be finite and in [0.1, 10]");
        auto module = torch::jit::load(argv[1], torch::kCPU);
        std::string output;
        for (auto p : module.named_parameters(true)) {
            if (p.value.dim() == 2) output = p.name;
        }
        if (output.empty() || output.size() < 6)
            throw std::runtime_error("No linear output weight found");
        std::string bias = output.substr(0, output.size() - 6) + "bias";
        torch::NoGradGuard guard;
        int changed = 0;
        for (auto p : module.named_parameters(true)) {
            if (p.name == output || p.name == bias) {
                p.value.div_(temperature);
                std::cout << p.name << " " << p.value.sizes() << std::endl;
                changed++;
            }
        }
        if (changed != 2) throw std::runtime_error("Expected final linear weight and bias");
        module.save(argv[2]);
        std::cout << "Saved calibrated policy; temperature " << temperature << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
