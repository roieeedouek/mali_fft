# Mali GPU FFT - Main Makefile
# Unified build system with organized directory structure

.PHONY: all clean build-native build-arm help

all: help

# ============================================================================
# Directory Structure
# ============================================================================

SRC_DIR = src
BUILD_DIR = build
DOCS_DIR = docs

# ============================================================================
# Native Build (Ubuntu/Linux x86_64)
# ============================================================================

build-native: $(BUILD_DIR)
	@echo "Building FFT for native Ubuntu/Linux..."
	@cd $(BUILD_DIR) && make -f Makefile.ubuntu all

test-native: $(BUILD_DIR)
	@echo "Running native FFT tests..."
	@cd $(BUILD_DIR) && make -f Makefile.ubuntu test-correctness

# ============================================================================
# ARM Build (Mali GPU - Requires Yocto SDK)
# ============================================================================

build-arm: $(BUILD_DIR)
	@echo "Building FFT for ARM Mali GPU (requires Yocto SDK)..."
	@cd $(BUILD_DIR) && make -f Makefile.arm all

test-arm: $(BUILD_DIR)
	@echo "Building ARM test suite..."
	@cd $(BUILD_DIR) && make -f Makefile.arm test

# ============================================================================
# Validation
# ============================================================================

validate: build-native
	@echo "Running comprehensive FFT validation..."
	@if [ -f $(BUILD_DIR)/fft_validation ]; then \
		$(BUILD_DIR)/fft_validation; \
	else \
		echo "ERROR: fft_validation not found. Run 'make build-native' first"; \
		exit 1; \
	fi

verify: build-native
	@echo "Running GPU correctness verification..."
	@if [ -f $(BUILD_DIR)/fft_test_new ]; then \
		$(BUILD_DIR)/fft_test_new --correctness-only; \
	else \
		echo "ERROR: fft_test_new not found. Run 'make build-native' first"; \
		exit 1; \
	fi

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Documentation:"
	@echo "  $(DOCS_DIR)/VERIFICATION_SUMMARY.md - Quick reference guide"
	@echo "  $(DOCS_DIR)/FFT_VALIDATION_REPORT.md - Comprehensive analysis"
	@echo "  $(DOCS_DIR)/README.md - Documentation index"

show-docs:
	@cat $(DOCS_DIR)/README.md

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	@cd $(BUILD_DIR) && make -f Makefile.ubuntu clean 2>/dev/null || true
	@cd $(BUILD_DIR) && make -f Makefile.arm clean 2>/dev/null || true
	@rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.s 2>/dev/null || true
	@echo "Clean complete"

distclean: clean
	@echo "Removing all generated files..."
	@rm -rf build bin *.o *.s fft_test_new fft_validation
	@echo "Distclean complete"

# ============================================================================
# Project Organization
# ============================================================================

info:
	@echo "=================================================="
	@echo "Mali FFT Project Structure"
	@echo "=================================================="
	@echo ""
	@echo "Directories:"
	@echo "  src/          - Source code (.cpp, .cl, .h)"
	@echo "  build/        - Build system and compiled binaries"
	@echo "  docs/         - Documentation and reports"
	@echo "  include/      - Header files (if needed)"
	@echo "  bin/          - Installed binaries"
	@echo ""
	@echo "Source Files (src/):"
	@find src -type f -name "*.cpp" -o -name "*.cl" -o -name "*.h" 2>/dev/null | sed 's/^/  /'
	@echo ""
	@echo "Documentation (docs/):"
	@find docs -type f -name "*.md" 2>/dev/null | sed 's/^/  /'
	@echo ""

# ============================================================================
# Help
# ============================================================================

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║   Mali GPU FFT - Build System                                 ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Quick Start:"
	@echo "  make build-native         Build for Ubuntu/Linux (x86_64)"
	@echo "  make test-native          Run correctness tests"
	@echo "  make validate             Run validation test suite (10 tests)"
	@echo "  make verify               Run GPU correctness tests"
	@echo ""
	@echo "ARM Build (requires Yocto SDK):"
	@echo "  make build-arm            Build for ARM Mali GPU"
	@echo "  make test-arm             Build ARM test suite"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs                 Show documentation files"
	@echo "  make show-docs            Display documentation index"
	@echo ""
	@echo "Maintenance:"
	@echo "  make info                 Show project structure"
	@echo "  make clean                Remove build artifacts"
	@echo "  make distclean            Clean all generated files"
	@echo ""
	@echo "Directory Structure:"
	@echo "  ./"
	@echo "  ├── src/                  Source code"
	@echo "  ├── build/                Build system & binaries"
	@echo "  ├── docs/                 Documentation"
	@echo "  ├── include/              Headers"
	@echo "  └── Makefile              This file"
	@echo ""
	@echo "For detailed build options, see:"
	@echo "  build/Makefile.ubuntu     Ubuntu/Linux build"
	@echo "  build/Makefile.arm        ARM Mali GPU build"
	@echo ""
