/*
 * Universal GPU FFT Host Code (OpenCL C API)
 * 
 * This version works with ANY OpenCL-compatible GPU:
 * - ARM Mali (for Renesas RZ/G2L target)
 * - Intel HD Graphics / Iris (for Ubuntu/PC)
 * - NVIDIA GeForce / Quadro
 * - AMD Radeon
 * 
 * The code automatically detects and uses the first available GPU.
 */

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdio>

// Error checking macro
#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << msg << " (code: " << err << ")" << std::endl; \
        return 1; \
    }

// FFT Configuration
struct FFTConfig {
    size_t N;                    // FFT size
    std::vector<int> radix;      // Radix sequence (e.g., {4, 4, 4} for N=64)
    size_t work_group_size;      // Work group size
    bool use_fp16;               // Use FP16 for 2x performance
};

// Helper function to create optimal FFT configuration
FFTConfig createOptimalConfig(size_t N, bool use_fp16 = false) {
    FFTConfig config;
    config.N = N;
    config.use_fp16 = use_fp16;
    config.work_group_size = 64;  // Default, will be adjusted based on device

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

// Compute digit-reverse indices
std::vector<unsigned int> computeDigitReverseIndices(const FFTConfig& config) {
    std::vector<unsigned int> indices(config.N);

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

        indices[n] = k;
    }

    return indices;
}

// Load kernel source from file
std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Try alternate location
        std::string alt_path = std::string("/usr/local/share/opencl/kernels/") + filename;
        file.open(alt_path);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open kernel file: " << filename << std::endl;
            std::cerr << "Tried locations:" << std::endl;
            std::cerr << "  - ./" << filename << std::endl;
            std::cerr << "  - " << alt_path << std::endl;
            return "";
        }
    }

    std::string source(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    return source;
}

int main() {
    cl_int err;

    std::cout << "Universal GPU FFT using OpenCL" << std::endl;
    std::cout << "==============================" << std::endl;

    // Create FFT configuration for N=1024
    FFTConfig config = createOptimalConfig(1024, false);

    std::cout << "\nFFT Configuration:" << std::endl;
    std::cout << "  Size: " << config.N << std::endl;
    std::cout << "  Radix sequence: [";
    for (size_t i = 0; i < config.radix.size(); ++i) {
        std::cout << config.radix[i];
        if (i < config.radix.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Initial work group size: " << config.work_group_size << std::endl;
    std::cout << "  FP16 mode: " << (config.use_fp16 ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    // Get platform
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_CL_ERROR(err, "Failed to get number of platforms");

    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        std::cerr << "\nTroubleshooting:" << std::endl;
        std::cerr << "  1. Check if OpenCL ICD is installed:" << std::endl;
        std::cerr << "     Ubuntu: sudo apt install intel-opencl-icd (or nvidia/mesa)" << std::endl;
        std::cerr << "     RZ/G2L: Ensure Mali GPU drivers are loaded (lsmod | grep mali)" << std::endl;
        std::cerr << "  2. Check with: clinfo" << std::endl;
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    CHECK_CL_ERROR(err, "Failed to get platforms");

    std::cout << "Available OpenCL Platforms:" << std::endl;
    for (cl_uint i = 0; i < num_platforms; i++) {
        char name[128], vendor[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        std::cout << "  [" << i << "] " << name << " (" << vendor << ")" << std::endl;
    }
    std::cout << std::endl;

    // Try to find GPU device on any platform
    cl_platform_id selected_platform = NULL;
    cl_device_id device = NULL;

    for (cl_uint i = 0; i < num_platforms && device == NULL; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err == CL_SUCCESS) {
                selected_platform = platforms[i];
                break;
            }
        }
    }

    // If no GPU found, try to use CPU as fallback
    if (device == NULL) {
        std::cout << "⚠ No GPU devices found, trying CPU..." << std::endl;
        for (cl_uint i = 0; i < num_platforms && device == NULL; i++) {
            cl_uint num_devices;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                if (err == CL_SUCCESS) {
                    selected_platform = platforms[i];
                    break;
                }
            }
        }
    }

    if (device == NULL) {
        std::cerr << "ERROR: No OpenCL devices (GPU or CPU) found!" << std::endl;
        return 1;
    }

    // Print selected platform and device info
    char platform_name[128], device_name[128], device_vendor[128];
    clGetPlatformInfo(selected_platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);

    std::cout << "Selected Platform: " << platform_name << std::endl;
    std::cout << "Selected Device:   " << device_name << " (" << device_vendor << ")" << std::endl;

    cl_device_type device_type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    std::cout << "Device Type:       " << (device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << std::endl;

    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    std::cout << "Compute Units:     " << compute_units << std::endl;

    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    std::cout << "Max Work Group:    " << max_work_group_size << std::endl;

    cl_ulong global_mem;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    std::cout << "Global Memory:     " << (global_mem / (1024*1024)) << " MB" << std::endl;

    // Adjust work group size based on device capabilities
    if (config.work_group_size > max_work_group_size) {
        config.work_group_size = max_work_group_size;
        std::cout << "\n⚠ Adjusted work group size to: " << config.work_group_size << std::endl;
    }

    // Detect Mali GPU and provide specific info
    if (strstr(device_name, "Mali") != NULL) {
        std::cout << "\n✓ Mali GPU Detected - Using Mali-specific optimizations" << std::endl;
        if (strstr(device_name, "G31") != NULL) {
            std::cout << "  Target: Renesas RZ/G2L with Mali-G31" << std::endl;
            config.work_group_size = 64;  // Optimal for Mali-G31
        }
    } else if (strstr(device_vendor, "Intel") != NULL) {
        std::cout << "\n✓ Intel GPU Detected" << std::endl;
        config.work_group_size = (max_work_group_size >= 256) ? 256 : max_work_group_size;
    } else if (strstr(device_vendor, "NVIDIA") != NULL) {
        std::cout << "\n✓ NVIDIA GPU Detected" << std::endl;
        config.work_group_size = (max_work_group_size >= 256) ? 256 : max_work_group_size;
    } else if (strstr(device_vendor, "AMD") != NULL || strstr(device_vendor, "Advanced Micro Devices") != NULL) {
        std::cout << "\n✓ AMD GPU Detected" << std::endl;
        config.work_group_size = (max_work_group_size >= 256) ? 256 : max_work_group_size;
    }

    std::cout << "Final Work Group:  " << config.work_group_size << std::endl;
    std::cout << std::endl;

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create context");

    // Create command queue with profiling
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL_ERROR(err, "Failed to create command queue");

    // Load kernel source
    std::string kernel_source = loadKernelSource("mali_fft_optimized.cl");
    if (kernel_source.empty()) {
        return 1;
    }

    // Create program
    const char* source_ptr = kernel_source.c_str();
    size_t source_size = kernel_source.length();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    CHECK_CL_ERROR(err, "Failed to create program");

    // Build program
    const char* build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program!" << std::endl;
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return 1;
    }

    std::cout << "✓ Kernels compiled successfully" << std::endl;

    // Create kernels
    cl_kernel kernel_digit_reverse = clCreateKernel(program, "digit_reverse", &err);
    CHECK_CL_ERROR(err, "Failed to create digit_reverse kernel");

    cl_kernel kernel_radix_4 = clCreateKernel(program, "radix_4", &err);
    CHECK_CL_ERROR(err, "Failed to create radix_4 kernel");

    cl_kernel kernel_radix_4_first = clCreateKernel(program, "radix_4_first_stage", &err);
    CHECK_CL_ERROR(err, "Failed to create radix_4_first_stage kernel");

    std::cout << "✓ Kernels created successfully" << std::endl;
    std::cout << std::endl;

    // Compute digit-reverse indices
    std::vector<unsigned int> digit_reverse_indices = computeDigitReverseIndices(config);

    // Create test input (impulse signal)
    std::vector<float> input_data(2 * config.N, 0.0f);
    input_data[0] = 1.0f;  // Impulse at t=0 (real part)
    input_data[1] = 0.0f;  // Imaginary part

    // Create buffers
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                        2 * config.N * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create input buffer");

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                         2 * config.N * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create output buffer");

    cl_mem indices_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                          config.N * sizeof(unsigned int), NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create indices buffer");

    // Upload data
    err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, 
                              2 * config.N * sizeof(float), input_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err, "Failed to write input buffer");

    err = clEnqueueWriteBuffer(queue, indices_buffer, CL_TRUE, 0, 
                              config.N * sizeof(unsigned int), digit_reverse_indices.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err, "Failed to write indices buffer");

    // Execute FFT stages
    std::cout << "Executing FFT stages..." << std::endl;

    // Stage 0: Digit reversal
    clSetKernelArg(kernel_digit_reverse, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel_digit_reverse, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel_digit_reverse, 2, sizeof(cl_mem), &indices_buffer);

    size_t global_size = config.N;
    size_t local_size = config.work_group_size;

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel_digit_reverse, 1, NULL, 
                                &global_size, &local_size, 0, NULL, &event);
    CHECK_CL_ERROR(err, "Failed to execute digit_reverse kernel");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);

    // Swap buffers
    cl_mem temp = input_buffer;
    input_buffer = output_buffer;
    output_buffer = temp;

    // Execute radix stages
    size_t Nx = 1;
    for (size_t stage = 0; stage < config.radix.size(); ++stage) {
        size_t Ny = config.radix[stage];
        size_t Ni = Nx * Ny;

        float exp_const = -2.0f * M_PI / (float)Ni;

        cl_kernel kernel = (stage == 0) ? kernel_radix_4_first : kernel_radix_4;
        size_t num_butterflies = config.N / Ny;

        if (stage == 0) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        } else {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
            cl_uint nx_arg = (cl_uint)Nx;
            cl_uint ni_arg = (cl_uint)Ni;
            clSetKernelArg(kernel, 1, sizeof(cl_uint), &nx_arg);
            clSetKernelArg(kernel, 2, sizeof(cl_uint), &ni_arg);
            clSetKernelArg(kernel, 3, sizeof(float), &exp_const);
        }

        // Launch kernel
        size_t global = num_butterflies;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
                                    &global, &local_size, 0, NULL, &event);
        CHECK_CL_ERROR(err, "Failed to execute radix kernel");

        clWaitForEvents(1, &event);

        // Get profiling info
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        double time_ms = (end - start) / 1e6;

        std::cout << "  Stage " << stage << " (radix-" << Ny << "): " 
                 << time_ms << " ms" << std::endl;

        clReleaseEvent(event);

        Nx = Ni;
    }

    // Read results
    std::vector<float> output_data(2 * config.N);
    err = clEnqueueReadBuffer(queue, input_buffer, CL_TRUE, 0, 
                             2 * config.N * sizeof(float), output_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err, "Failed to read output buffer");

    std::cout << "\n✓ FFT completed successfully!" << std::endl;
    std::cout << "\nFirst 8 output values:" << std::endl;
    for (size_t i = 0; i < 8 && i < config.N; ++i) {
        std::cout << "  [" << i << "] = " 
                 << output_data[2*i] << " + " 
                 << output_data[2*i+1] << "i" << std::endl;
    }

    // Cleanup
    clReleaseKernel(kernel_digit_reverse);
    clReleaseKernel(kernel_radix_4);
    clReleaseKernel(kernel_radix_4_first);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(indices_buffer);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}