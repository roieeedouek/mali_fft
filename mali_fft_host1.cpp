
/*
 * Mali GPU FFT Host Code
 * 
 * This file demonstrates how to set up and execute the Mali-optimized FFT
 * from the host side using OpenCL C++ API.
 */

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>

// FFT Configuration
struct FFTConfig {
    size_t N;                    // FFT size
    std::vector<int> radix;      // Radix sequence (e.g., {4, 4, 4} for N=64)
    size_t work_group_size;      // 128 or 256 for Mali
    bool use_fp16;               // Use FP16 for 2x performance
};

class MaliFFT {
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    // Kernels for different radix stages
    cl::Kernel kernel_digit_reverse;
    cl::Kernel kernel_radix_2;
    cl::Kernel kernel_radix_4;
    cl::Kernel kernel_radix_8;
    cl::Kernel kernel_radix_2_first;
    cl::Kernel kernel_radix_4_first;

    FFTConfig config;
    std::vector<unsigned int> digit_reverse_indices;

public:
    MaliFFT(const FFTConfig& cfg) : config(cfg) {
        initializeOpenCL();
        loadKernels();
        computeDigitReverseIndices();
    }

    void initializeOpenCL() {
        // Get Mali GPU platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        cl::Platform mali_platform;
        for (auto& platform : platforms) {
            std::string vendor = platform.getInfo<CL_PLATFORM_VENDOR>();
            if (vendor.find("ARM") != std::string::npos) {
                mali_platform = platform;
                std::cout << "Found ARM Mali platform: " 
                         << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
                break;
            }
        }

        // Get GPU device
        std::vector<cl::Device> devices;
        mali_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        if (devices.empty()) {
            throw std::runtime_error("No Mali GPU found!");
        }

        cl::Device mali_gpu = devices[0];
        std::cout << "Using device: " << mali_gpu.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Max compute units: " << mali_gpu.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "Global memory: " << mali_gpu.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << " MB" << std::endl;

        // Create context and command queue
        context = cl::Context(mali_gpu);
        queue = cl::CommandQueue(context, mali_gpu, CL_QUEUE_PROFILING_ENABLE);
    }

    void loadKernels() {
        // Load kernel source from file
        std::ifstream kernel_file("mali_fft_optimized.cl");
        std::string kernel_source(
            (std::istreambuf_iterator<char>(kernel_file)),
            std::istreambuf_iterator<char>()
        );

        // Build program
        program = cl::Program(context, kernel_source);

        std::string build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math";
        if (config.use_fp16) {
            build_options += " -DMALI_FP16_SUPPORT -cl-fp16-enable";
        }

        try {
            program.build(build_options.c_str());
        } catch (const cl::Error& err) {
            std::cerr << "Build error: " << err.what() << std::endl;
            // Print build log
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                context.getInfo<CL_CONTEXT_DEVICES>()[0]
            );
            std::cerr << "Build log:\n" << log << std::endl;
            throw;
        }

        // Create kernel objects
        kernel_digit_reverse = cl::Kernel(program, "digit_reverse");
        kernel_radix_2 = cl::Kernel(program, "radix_2");
        kernel_radix_4 = cl::Kernel(program, "radix_4");
        kernel_radix_8 = cl::Kernel(program, "radix_8");
        kernel_radix_2_first = cl::Kernel(program, "radix_2_first_stage");
        kernel_radix_4_first = cl::Kernel(program, "radix_4_first_stage");

        std::cout << "Kernels loaded successfully" << std::endl;
    }

    void computeDigitReverseIndices() {
        digit_reverse_indices.resize(config.N);

        for (size_t n = 0; n < config.N; ++n) {
            size_t k = n;
            size_t Nx = config.radix[0];

            // Apply digit reversal through all stages
            for (size_t s = 1; s < config.radix.size(); ++s) {
                size_t Ny = config.radix[s];
                size_t Ni = Ny * Nx;
                k = (k * Ny) % Ni + (k / Nx) % Ny + Ni * (k / Ni);
                Nx *= Ny;
            }

            digit_reverse_indices[n] = k;
        }

        std::cout << "Computed digit-reverse indices for N=" << config.N << std::endl;
    }

    void executeFFT(const std::vector<std::complex<float>>& input,
                    std::vector<std::complex<float>>& output) {

        if (input.size() != config.N) {
            throw std::runtime_error("Input size doesn't match FFT configuration");
        }

        // Create buffers
        cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, 
                               2 * config.N * sizeof(float));
        cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, 
                                2 * config.N * sizeof(float));
        cl::Buffer indices_buffer(context, CL_MEM_READ_ONLY, 
                                 config.N * sizeof(unsigned int));

        // Convert complex to float array [real, imag, real, imag, ...]
        std::vector<float> input_float(2 * config.N);
        for (size_t i = 0; i < config.N; ++i) {
            input_float[2*i] = input[i].real();
            input_float[2*i + 1] = input[i].imag();
        }

        // Upload data to GPU
        queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, 
                                2 * config.N * sizeof(float), 
                                input_float.data());
        queue.enqueueWriteBuffer(indices_buffer, CL_TRUE, 0, 
                                config.N * sizeof(unsigned int), 
                                digit_reverse_indices.data());

        // Stage 0: Digit reversal
        kernel_digit_reverse.setArg(0, input_buffer);
        kernel_digit_reverse.setArg(1, output_buffer);
        kernel_digit_reverse.setArg(2, indices_buffer);

        cl::NDRange global_size_digit(config.N);
        cl::NDRange local_size_digit(config.work_group_size);
        queue.enqueueNDRangeKernel(kernel_digit_reverse, cl::NullRange, 
                                   global_size_digit, local_size_digit);

        // Swap buffers (output becomes input for next stage)
        cl::Buffer temp_buffer = input_buffer;
        input_buffer = output_buffer;
        output_buffer = temp_buffer;

        // Execute radix stages
        size_t Nx = 1;
        for (size_t stage = 0; stage < config.radix.size(); ++stage) {
            size_t Ny = config.radix[stage];
            size_t Ni = Nx * Ny;

            // Pre-compute constant for twiddle factors
            float exp_const = -2.0f * M_PI / (float)Ni;

            // Select appropriate kernel
            cl::Kernel* radix_kernel = nullptr;
            size_t num_butterflies = config.N / Ny;

            if (stage == 0) {
                // First stage uses linear memory access
                if (Ny == 2) radix_kernel = &kernel_radix_2_first;
                else if (Ny == 4) radix_kernel = &kernel_radix_4_first;
            } else {
                // Subsequent stages with twiddle factor fusion
                if (Ny == 2) radix_kernel = &kernel_radix_2;
                else if (Ny == 4) radix_kernel = &kernel_radix_4;
                else if (Ny == 8) radix_kernel = &kernel_radix_8;
            }

            if (radix_kernel == nullptr) {
                throw std::runtime_error("Unsupported radix: " + std::to_string(Ny));
            }

            // Set kernel arguments
            if (stage == 0) {
                radix_kernel->setArg(0, input_buffer);
            } else {
                radix_kernel->setArg(0, input_buffer);
                radix_kernel->setArg(1, (unsigned int)Nx);
                radix_kernel->setArg(2, (unsigned int)Ni);
                radix_kernel->setArg(3, exp_const);
            }

            // Launch kernel with optimal work group size
            cl::NDRange global_size(num_butterflies);
            cl::NDRange local_size(config.work_group_size);

            cl::Event event;
            queue.enqueueNDRangeKernel(*radix_kernel, cl::NullRange, 
                                      global_size, local_size, nullptr, &event);
            event.wait();

            // Print timing information
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            double time_ms = (end - start) / 1e6;
            std::cout << "Stage " << stage << " (radix-" << Ny 
                     << "): " << time_ms << " ms" << std::endl;

            // Update span for next stage
            Nx = Ni;
        }

        // Read results back from GPU
        std::vector<float> output_float(2 * config.N);
        queue.enqueueReadBuffer(input_buffer, CL_TRUE, 0, 
                               2 * config.N * sizeof(float), 
                               output_float.data());

        // Convert back to complex
        output.resize(config.N);
        for (size_t i = 0; i < config.N; ++i) {
            output[i] = std::complex<float>(output_float[2*i], 
                                           output_float[2*i + 1]);
        }

        std::cout << "FFT completed successfully" << std::endl;
    }
};

// Helper function to create optimal FFT configuration
FFTConfig createOptimalConfig(size_t N, bool use_fp16 = false) {
    FFTConfig config;
    config.N = N;
    config.use_fp16 = use_fp16;
    config.work_group_size = 256;  // Optimal for most Mali GPUs

    // Determine optimal radix sequence
    size_t n = N;

    // Prefer radix-8 for power-of-8 sizes (best performance)
    while (n % 8 == 0) {
        config.radix.push_back(8);
        n /= 8;
    }

    // Then radix-4 for remaining power-of-2 (1.7x faster than radix-2)
    while (n % 4 == 0) {
        config.radix.push_back(4);
        n /= 4;
    }

    // Then radix-2
    while (n % 2 == 0) {
        config.radix.push_back(2);
        n /= 2;
    }

    // Handle remaining prime factors
    while (n % 3 == 0) {
        config.radix.push_back(3);
        n /= 3;
    }
    while (n % 5 == 0) {
        config.radix.push_back(5);
        n /= 5;
    }
    while (n % 7 == 0) {
        config.radix.push_back(7);
        n /= 7;
    }

    if (n != 1) {
        throw std::runtime_error("N must be factorizable into 2,3,4,5,7,8");
    }

    return config;
}

// Example usage
int main() {
    try {
        // Create FFT configuration for N=1024
        // Radix sequence: {4, 4, 4, 4, 4} since 1024 = 4^5
        FFTConfig config = createOptimalConfig(1024, false);

        std::cout << "FFT Configuration:" << std::endl;
        std::cout << "  Size: " << config.N << std::endl;
        std::cout << "  Radix sequence: [";
        for (size_t i = 0; i < config.radix.size(); ++i) {
            std::cout << config.radix[i];
            if (i < config.radix.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Work group size: " << config.work_group_size << std::endl;
        std::cout << "  FP16 mode: " << (config.use_fp16 ? "enabled" : "disabled") << std::endl;

        // Initialize FFT
        MaliFFT fft(config);

        // Create test input (impulse signal)
        std::vector<std::complex<float>> input(config.N, {0.0f, 0.0f});
        input[0] = {1.0f, 0.0f};  // Impulse at t=0

        // Execute FFT
        std::vector<std::complex<float>> output;
        fft.executeFFT(input, output);

        // Verify result (impulse should give flat spectrum)
        std::cout << "\nFirst 8 output values:" << std::endl;
        for (size_t i = 0; i < 8 && i < output.size(); ++i) {
            std::cout << "  [" << i << "] = " 
                     << output[i].real() << " + " 
                     << output[i].imag() << "i" << std::endl;
        }

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error: " << err.what() 
                 << " (" << err.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}

/*
 * COMPILATION:
 * 
 * g++ -std=c++11 mali_fft_host.cpp -o mali_fft \
 *     -lOpenCL -I/path/to/opencl/headers
 * 
 * MALI-SPECIFIC OPTIMIZATIONS USED:
 * 
 * 1. Work group size: 256 threads (optimal for Mali)
 * 2. Mixed-radix: Prefers radix-8 > radix-4 > radix-2
 * 3. Kernel fusion: Twiddle factors computed in radix kernels
 * 4. Memory coalescing: Adjacent threads access consecutive memory
 * 5. FP16 support: Optional 2x performance boost
 * 6. Profiling: Measures per-stage execution time
 */
