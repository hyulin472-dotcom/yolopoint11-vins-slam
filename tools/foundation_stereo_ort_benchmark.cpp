#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

struct Args {
    std::string model = "onxx/fast_foundationstereo/20_30_48_iters_4_res_320x736.onnx";
    std::string trt_cache = "onxx/fast_foundationstereo/trt_cache";
    int height = 320;
    int width = 736;
    int warmup = 20;
    int runs = 200;
    int trt_min_subgraph_size = 1;
    std::string provider = "cuda";
};

void ensureDirectory(const std::string &path) {
    if (path.empty()) {
        return;
    }
    std::string current;
    if (path[0] == '/') {
        current = "/";
    }
    size_t start = path[0] == '/' ? 1 : 0;
    while (start <= path.size()) {
        size_t end = path.find('/', start);
        std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/') {
                current += "/";
            }
            current += part;
            mkdir(current.c_str(), 0755);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

Args parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            return argv[++i];
        };
        if (key == "--model") {
            args.model = next();
        } else if (key == "--trt-cache") {
            args.trt_cache = next();
        } else if (key == "--height") {
            args.height = std::stoi(next());
        } else if (key == "--width") {
            args.width = std::stoi(next());
        } else if (key == "--warmup") {
            args.warmup = std::stoi(next());
        } else if (key == "--runs") {
            args.runs = std::stoi(next());
        } else if (key == "--trt-min-subgraph-size") {
            args.trt_min_subgraph_size = std::stoi(next());
        } else if (key == "--provider") {
            args.provider = next();
        } else if (key == "--cpu") {
            args.provider = "cpu";
        } else if (key == "--cuda") {
            args.provider = "cuda";
        } else if (key == "--tensorrt" || key == "--trt") {
            args.provider = "tensorrt";
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

double averageMs(const std::vector<double> &times) {
    if (times.empty()) {
        return 0.0;
    }
    return std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
}

double percentileMs(std::vector<double> times, double p) {
    if (times.empty()) {
        return 0.0;
    }
    std::sort(times.begin(), times.end());
    size_t idx = static_cast<size_t>(p * static_cast<double>(times.size() - 1));
    return times[idx];
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parseArgs(argc, argv);
        std::cout << "model: " << args.model << "\n";
        std::cout << "input: 2 x 1x3x" << args.height << "x" << args.width << "\n";
        std::cout << "requested provider: " << args.provider << "\n";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "foundation_stereo_ort_benchmark");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        if (args.provider == "tensorrt") {
            ensureDirectory(args.trt_cache);
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = 0;
            trt_options.trt_max_partition_iterations = 1000;
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_min_subgraph_size = args.trt_min_subgraph_size;
            trt_options.trt_max_workspace_size = 1ULL << 32;
            trt_options.trt_engine_cache_enable = 1;
            trt_options.trt_engine_cache_path = args.trt_cache.c_str();
            session_options.AppendExecutionProvider_TensorRT(trt_options);

            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } else if (args.provider == "cuda") {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } else if (args.provider != "cpu") {
            throw std::runtime_error("unknown provider: " + args.provider + " (use cpu, cuda, or tensorrt)");
        }

        auto session_build_start = std::chrono::steady_clock::now();
        Ort::Session session(env, args.model.c_str(), session_options);
        auto session_build_end = std::chrono::steady_clock::now();
        std::cout << "session build: "
                  << std::chrono::duration<double, std::milli>(session_build_end - session_build_start).count()
                  << " ms\n";

        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> input_names_str;
        std::vector<const char *> input_names;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names_str.emplace_back(name.get());
        }
        for (const auto &name : input_names_str) {
            input_names.push_back(name.c_str());
        }

        std::vector<std::string> output_names_str;
        std::vector<const char *> output_names;
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names_str.emplace_back(name.get());
        }
        for (const auto &name : output_names_str) {
            output_names.push_back(name.c_str());
        }

        if (input_names.size() < 2) {
            throw std::runtime_error("FoundationStereo model expects at least two inputs");
        }

        std::cout << "inputs:";
        for (const auto &name : input_names_str) {
            std::cout << " " << name;
        }
        std::cout << "\noutputs:";
        for (const auto &name : output_names_str) {
            std::cout << " " << name;
        }
        std::cout << "\n";

        std::vector<int64_t> input_shape = {1, 3, args.height, args.width};
        size_t input_size = 1ULL * 3ULL * static_cast<size_t>(args.height) * static_cast<size_t>(args.width);
        std::vector<float> left(input_size);
        std::vector<float> right(input_size);
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);
        for (auto &v : left) {
            v = dist(rng);
        }
        for (auto &v : right) {
            v = dist(rng);
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value left_tensor = Ort::Value::CreateTensor<float>(
            memory_info, left.data(), left.size(), input_shape.data(), input_shape.size());
        Ort::Value right_tensor = Ort::Value::CreateTensor<float>(
            memory_info, right.data(), right.size(), input_shape.data(), input_shape.size());
        std::array<Ort::Value, 2> tensors = {std::move(left_tensor), std::move(right_tensor)};

        for (int i = 0; i < args.warmup; ++i) {
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), 2,
                                       output_names.data(), output_names.size());
        }

        std::vector<double> times;
        times.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), 2,
                                       output_names.data(), output_names.size());
            auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::cout << "avg: " << averageMs(times) << " ms\n";
        std::cout << "min: " << *std::min_element(times.begin(), times.end()) << " ms\n";
        std::cout << "p50: " << percentileMs(times, 0.50) << " ms\n";
        std::cout << "p95: " << percentileMs(times, 0.95) << " ms\n";
        std::cout << "max: " << *std::max_element(times.begin(), times.end()) << " ms\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "benchmark failed: " << e.what() << "\n";
        return 1;
    }
}
