# Mali GPU Optimized FFT - Makefile Wrapper
# This file redirects to the appropriate build system

.PHONY: help all test clean

# Default target
all:
	@echo "Please use one of the specific Makefiles:"
	@echo ""
	@echo "  make -f Makefile.ubuntu    - Build for Ubuntu/Linux x86_64"
	@echo "  make -f Makefile.arm       - Build for ARM Mali GPU (requires Yocto SDK)"
	@echo ""
	@echo "Or from the root directory:"
	@echo "  make build-native          - Build for Ubuntu/Linux"
	@echo "  make build-arm             - Build for ARM"
	@echo "  make help                  - Show all commands"

help:
	@echo "Mali GPU FFT - Build System"
	@echo "============================"
	@echo ""
	@echo "Available build systems:"
	@echo ""
	@echo "1. Ubuntu/Linux (Native Build):"
	@echo "   $$ make -f Makefile.ubuntu"
	@echo "   $$ make -f Makefile.ubuntu test"
	@echo "   $$ make -f Makefile.ubuntu validate"
	@echo ""
	@echo "2. ARM Mali GPU (Cross-compile):"
	@echo "   $$ source /opt/rz-vlp/5.0.8/environment-setup-cortexa55-poky-linux"
	@echo "   $$ make -f Makefile.arm"
	@echo "   $$ make -f Makefile.arm install DESTDIR=/path/to/rootfs"
	@echo ""
	@echo "From root directory:"
	@echo "   $$ make build-native       - Build for your system"
	@echo "   $$ make build-arm          - Build for ARM (needs Yocto)"
	@echo "   $$ make test-native        - Run tests"
	@echo "   $$ make validate           - Run validation suite"
	@echo "   $$ make help               - Show all commands"

ubuntu:
	@make -f Makefile.ubuntu

arm:
	@make -f Makefile.arm

.PHONY: ubuntu arm
