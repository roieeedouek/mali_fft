# Makefile for Mali GPU Optimized FFT
# Target: Renesas RZ/G2L with Mali-G31
# Toolchain: Yocto SDK (rz-vlp-5.0.8)
#
# Prerequisites:
#   1. Source the Yocto environment first:
#      $ source /opt/rz-vlp/5.0.8/environment-setup-cortexa55-poky-linux
#   2. Ensure Mali GPU drivers and OpenCL runtime are installed in sysroot
#   3. Run 'make' to build, 'make install' to deploy

# ============================================================================
# Configuration
# ============================================================================

# Project name
PROJECT = fft_test_arm

# Source files
HOST_SRC = fft_test.cpp
KERNEL_SRC = mali_fft_optimized.cl

# Output binary
TARGET = $(PROJECT)

# Test binary
TEST_SRC = mali_fft_test.cpp
TEST_TARGET = $(PROJECT)_test


# Installation directory on target (can override with DESTDIR)
INSTALL_DIR = /usr/local/bin
KERNEL_INSTALL_DIR = /usr/local/share/opencl/kernels

# ============================================================================
# Toolchain Configuration (from Yocto SDK environment)
# ============================================================================

# These should be set by the environment-setup script
# If not set, provide defaults (though build will likely fail)
SDKTARGETSYSROOT ?= /opt/rz-vlp/5.0.8/sysroots/cortexa55-poky-linux
CXX ?= aarch64-poky-linux-g++
STRIP ?= aarch64-poky-linux-strip

# ============================================================================
# Compiler Flags
# ============================================================================

# C++ standard
CXXFLAGS += -std=c++11

# Optimization flags
CXXFLAGS += -O3 -march=armv8.2-a+crypto

# Mali-G31 specific optimizations
CXXFLAGS += -mcpu=cortex-a55+crypto
CXXFLAGS += -mtune=cortex-a55
CXXFLAGS += -ftree-vectorize
CXXFLAGS += -ffast-math

# Debug symbols (strip them out later for production)
CXXFLAGS += -g

# Warnings
CXXFLAGS += -Wall -Wextra -Wpedantic

# Include paths
CXXFLAGS += -I$(SDKTARGETSYSROOT)/usr/include
CXXFLAGS += -I$(SDKTARGETSYSROOT)/usr/include/CL

# Preprocessor defines
CPPFLAGS += -DCL_TARGET_OPENCL_VERSION=200
CPPFLAGS += -DMALI_G31_TARGET
CPPFLAGS += -D_USE_MATH_DEFINES

# ============================================================================
# Linker Flags
# ============================================================================

LDFLAGS += -L$(SDKTARGETSYSROOT)/usr/lib
LDFLAGS += -L$(SDKTARGETSYSROOT)/usr/lib/aarch64-linux-gnu

# OpenCL library
LIBS += -lOpenCL

# Math library
LIBS += -lm

# Threading library (if needed)
LIBS += -lpthread

# Ensure correct rpath for runtime linking on target
LDFLAGS += -Wl,-rpath-link,$(SDKTARGETSYSROOT)/usr/lib
LDFLAGS += -Wl,-rpath,/usr/lib

# ============================================================================
# Build Rules
# ============================================================================

.PHONY: all clean install install-strip check-env help

all: check-env $(TARGET)

# Main target
$(TARGET): $(HOST_SRC) $(KERNEL_SRC)
	@echo "==================================================================="
	@echo "Building $(TARGET) for Renesas RZ/G2L (Mali-G31)"
	@echo "==================================================================="
	@echo "CXX      : $(CXX)"
	@echo "CXXFLAGS : $(CXXFLAGS)"
	@echo "LDFLAGS  : $(LDFLAGS)"
	@echo "LIBS     : $(LIBS)"
	@echo "==================================================================="
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(HOST_SRC) -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo ""
	@echo "Build successful! Binary: $(TARGET)"
	@echo "Size information:"
	@ls -lh $(TARGET)
	@echo ""
	@echo "To deploy to target device:"
	@echo "  1. Copy $(TARGET) to /usr/local/bin on target"
	@echo "  2. Copy $(KERNEL_SRC) to /usr/local/share/opencl/kernels on target"
	@echo "  3. Or run: make install DESTDIR=/path/to/target/rootfs"

# Stripped version for production
$(TARGET).stripped: $(TARGET)
	@echo "Creating stripped binary for production..."
	@cp $(TARGET) $(TARGET).stripped
	$(STRIP) $(TARGET).stripped
	@echo "Stripped binary size:"
	@ls -lh $(TARGET).stripped

# Check environment setup
check-env:
	@echo "Checking Yocto SDK environment..."
	@if [ -z "$(OECORE_NATIVE_SYSROOT)" ]; then \
		echo "ERROR: Yocto SDK environment not set up!"; \
		echo "Please source the environment setup script first:"; \
		echo "  source /opt/rz-vlp/5.0.8/environment-setup-cortexa55-poky-linux"; \
		echo "Or use the alias: rvenv"; \
		exit 1; \
	fi
	@echo "✓ Yocto SDK environment detected"
	@echo "  SDK Version    : $(OECORE_SDK_VERSION)"
	@echo "  Target Sysroot : $(SDKTARGETSYSROOT)"
	@echo ""
	@if [ ! -f "$(SDKTARGETSYSROOT)/usr/include/CL/cl.h" ]; then \
		echo "WARNING: OpenCL headers not found in sysroot!"; \
		echo "  Expected location: $(SDKTARGETSYSROOT)/usr/include/CL/cl.h"; \
		echo "  You may need to install opencl-headers in your Yocto build"; \
		echo ""; \
	fi
	@if [ ! -f "$(SDKTARGETSYSROOT)/usr/lib/libOpenCL.so" ] && \
	   [ ! -f "$(SDKTARGETSYSROOT)/usr/lib/aarch64-linux-gnu/libOpenCL.so" ]; then \
		echo "WARNING: OpenCL library not found in sysroot!"; \
		echo "  Expected location: $(SDKTARGETSYSROOT)/usr/lib/libOpenCL.so"; \
		echo "  You may need to install mali-gpu packages in your Yocto build"; \
		echo ""; \
	fi

# Install to target filesystem
install: $(TARGET) $(TARGET).stripped
	@echo "Installing to target filesystem..."
	@if [ -n "$(DESTDIR)" ]; then \
		install -d $(DESTDIR)$(INSTALL_DIR); \
		install -d $(DESTDIR)$(KERNEL_INSTALL_DIR); \
		install -m 0755 $(TARGET).stripped $(DESTDIR)$(INSTALL_DIR)/$(TARGET); \
		install -m 0644 $(KERNEL_SRC) $(DESTDIR)$(KERNEL_INSTALL_DIR)/; \
		install -m 0644 mali_g31_optimization_guide.txt $(DESTDIR)$(KERNEL_INSTALL_DIR)/ 2>/dev/null || true; \
		echo "Installed to $(DESTDIR)"; \
	else \
		echo "ERROR: Please specify DESTDIR for installation"; \
		echo "Example: make install DESTDIR=/path/to/target/rootfs"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TARGET) $(TARGET).stripped $(TEST_TARGET) *.o
	@echo "Clean complete"

# Help target
help:
	@echo "Mali GPU Optimized FFT - Makefile Help"
	@echo "======================================"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Source Yocto SDK environment:"
	@echo "     $$ source /opt/rz-vlp/5.0.8/environment-setup-cortexa55-poky-linux"
	@echo "     or use alias: $$ rvenv"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build the FFT application (default)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install to target filesystem (requires DESTDIR)"
	@echo "  make check-env    - Verify Yocto SDK environment setup"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  $$ make"
	@echo "  $$ make install DESTDIR=/path/to/nfs/rootfs"
	@echo ""
	@echo "Configuration:"
	@echo "  Current SDK      : $(OECORE_SDK_VERSION)"
	@echo "  Target sysroot   : $(SDKTARGETSYSROOT)"
	@echo "  Compiler         : $(CXX)"
	@echo "  Target CPU       : Cortex-A55 + Mali-G31"
	@echo ""
	@echo "Deployment:"
	@echo "  On target device, ensure:"
	@echo "    - Mali GPU kernel modules loaded"
	@echo "    - OpenCL runtime installed"
	@echo "    - Check with: clinfo"

# ============================================================================
# Additional Targets for Development
# ============================================================================

# Show compiler and linker commands (verbose build)
verbose: CXXFLAGS += -v
verbose: $(TARGET)

# Build with debug symbols and no optimization (for debugging)
debug: CXXFLAGS := $(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -O0 -DDEBUG
debug: $(TARGET)
	@echo "Debug build complete"

# Generate assembly output (for analyzing generated code)
asm: $(HOST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -S -fverbose-asm $(HOST_SRC) -o $(TARGET).s
	@echo "Assembly output: $(TARGET).s"

# Check library dependencies
ldd-check: $(TARGET)
	@echo "Library dependencies for $(TARGET):"
	@$(READELF) -d $(TARGET) | grep NEEDED || true
	@echo ""
	@echo "Note: This shows dependencies. Run ldd on target device to verify."

# Size analysis
size: $(TARGET)
	@echo "Size breakdown:"
	@$(NM) --size-sort --print-size $(TARGET) | tail -20

# Create deployment package
package: $(TARGET).stripped $(KERNEL_SRC)
	@echo "Creating deployment package..."
	@mkdir -p $(PROJECT)_deploy/bin
	@mkdir -p $(PROJECT)_deploy/share/opencl/kernels
	@mkdir -p $(PROJECT)_deploy/doc
	@cp $(TARGET).stripped $(PROJECT)_deploy/bin/$(TARGET)
	@cp $(KERNEL_SRC) $(PROJECT)_deploy/share/opencl/kernels/
	@cp mali_g31_optimization_guide.txt $(PROJECT)_deploy/doc/ 2>/dev/null || true
	@echo "README for Mali FFT Deployment" > $(PROJECT)_deploy/README.txt
	@echo "" >> $(PROJECT)_deploy/README.txt
	@echo "1. Copy bin/$(TARGET) to /usr/local/bin on target" >> $(PROJECT)_deploy/README.txt
	@echo "2. Copy share/opencl/kernels/* to /usr/local/share/opencl/kernels" >> $(PROJECT)_deploy/README.txt
	@echo "3. Ensure Mali GPU drivers are loaded: lsmod | grep mali" >> $(PROJECT)_deploy/README.txt
	@echo "4. Verify OpenCL: clinfo" >> $(PROJECT)_deploy/README.txt
	@echo "5. Run: $(TARGET)" >> $(PROJECT)_deploy/README.txt
	@tar czf $(PROJECT)_deploy.tar.gz $(PROJECT)_deploy
	@rm -rf $(PROJECT)_deploy
	@echo "Deployment package created: $(PROJECT)_deploy.tar.gz"
	@ls -lh $(PROJECT)_deploy.tar.gz


# Test and benchmark binary
$(TEST_TARGET): $(TEST_SRC) $(KERNEL_SRC)
	@echo "==================================================================="
	@echo "Building $(TEST_TARGET) for testing and benchmarking"
	@echo "==================================================================="
	@if [ ! -f "$(TEST_SRC)" ]; then \
		echo "ERROR: Test source file $(TEST_SRC) not found!"; \
		exit 1; \
	fi
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET) $(LDFLAGS) $(LIBS)
	@echo ""
	@echo "✓ Test suite built: $(TEST_TARGET)"
	@ls -lh $(TEST_TARGET)

# Run all tests
test: $(TEST_TARGET)
	@echo "Running test suite..."
	@if [ ! -f "$(KERNEL_SRC)" ]; then \
		echo "ERROR: Kernel file $(KERNEL_SRC) not found!"; \
		exit 1; \
	fi
	./$(TEST_TARGET)

# Run only correctness tests
test-correctness: $(TEST_TARGET)
	@echo "Running correctness tests..."
	./$(TEST_TARGET) --correctness-only

# Run only performance benchmarks
test-performance: $(TEST_TARGET)
	@echo "Running performance benchmarks..."
	./$(TEST_TARGET) --performance-only

# Run stress test
test-stress: $(TEST_TARGET)
	@echo "Running stress test (60 seconds)..."
	./$(TEST_TARGET) --stress


.PHONY: verbose debug asm ldd-check size package test test-correctness test-performance test-stress
