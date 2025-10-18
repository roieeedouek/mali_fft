# Update both Makefiles to include the test target

# Update the main Makefile for RZ/G2L
makefile_with_test = open('Makefile', 'r').read()

# Find the PROJECT line and add test target
test_addition = '''
# Test binary
TEST_SRC = mali_fft_test.cpp
TEST_TARGET = $(PROJECT)_test
'''

# Insert after PROJECT definition
makefile_with_test = makefile_with_test.replace(
    'TARGET = $(PROJECT)',
    'TARGET = $(PROJECT)\n' + test_addition
)

# Add test build rule before the .PHONY line at the end
test_rules = '''
# Test and benchmark binary
$(TEST_TARGET): $(TEST_SRC) $(KERNEL_SRC)
\t@echo "==================================================================="
\t@echo "Building $(TEST_TARGET) for testing and benchmarking"
\t@echo "==================================================================="
\t@if [ ! -f "$(TEST_SRC)" ]; then \\
\t\techo "ERROR: Test source file $(TEST_SRC) not found!"; \\
\t\texit 1; \\
\tfi
\t$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET) $(LDFLAGS) $(LIBS)
\t@echo ""
\t@echo "✓ Test suite built: $(TEST_TARGET)"
\t@ls -lh $(TEST_TARGET)

# Run all tests
test: $(TEST_TARGET)
\t@echo "Running test suite..."
\t@if [ ! -f "$(KERNEL_SRC)" ]; then \\
\t\techo "ERROR: Kernel file $(KERNEL_SRC) not found!"; \\
\t\texit 1; \\
\tfi
\t./$(TEST_TARGET)

# Run only correctness tests
test-correctness: $(TEST_TARGET)
\t@echo "Running correctness tests..."
\t./$(TEST_TARGET) --correctness-only

# Run only performance benchmarks
test-performance: $(TEST_TARGET)
\t@echo "Running performance benchmarks..."
\t./$(TEST_TARGET) --performance-only

# Run stress test
test-stress: $(TEST_TARGET)
\t@echo "Running stress test (60 seconds)..."
\t./$(TEST_TARGET) --stress

'''

# Add before the last .PHONY line
makefile_with_test = makefile_with_test.replace(
    '.PHONY: verbose debug asm ldd-check size package',
    test_rules + '\n.PHONY: verbose debug asm ldd-check size package test test-correctness test-performance test-stress'
)

# Update clean target
makefile_with_test = makefile_with_test.replace(
    '@rm -f $(TARGET) $(TARGET).stripped *.o',
    '@rm -f $(TARGET) $(TARGET).stripped $(TEST_TARGET) *.o'
)

with open('Makefile', 'w') as f:
    f.write(makefile_with_test)

# Update Ubuntu Makefile similarly
makefile_ubuntu_with_test = open('Makefile.ubuntu', 'r').read()

makefile_ubuntu_with_test = makefile_ubuntu_with_test.replace(
    'TARGET = $(PROJECT)',
    'TARGET = $(PROJECT)\n\n# Test binary\nTEST_SRC = mali_fft_test.cpp\nTEST_TARGET = $(PROJECT)_test'
)

ubuntu_test_rules = '''
# Test and benchmark binary
$(TEST_TARGET): $(TEST_SRC)
\t@echo "==================================================================="
\t@echo "Building $(TEST_TARGET) for testing and benchmarking"
\t@echo "==================================================================="
\t@if [ ! -f "$(TEST_SRC)" ]; then \\
\t\techo "ERROR: Test source file $(TEST_SRC) not found!"; \\
\t\texit 1; \\
\tfi
\t$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET) $(LIBS)
\t@echo ""
\t@echo "✓ Test suite built: $(TEST_TARGET)"
\t@ls -lh $(TEST_TARGET)

# Run all tests (override the simple test target)
test: $(TEST_TARGET)
\t@echo "Running comprehensive test suite..."
\t@if [ ! -f "$(KERNEL_SRC)" ]; then \\
\t\techo "ERROR: Kernel file $(KERNEL_SRC) not found!"; \\
\t\texit 1; \\
\tfi
\t./$(TEST_TARGET)

# Run only correctness tests
test-correctness: $(TEST_TARGET)
\t@echo "Running correctness tests..."
\t./$(TEST_TARGET) --correctness-only

# Run only performance benchmarks
test-performance: $(TEST_TARGET)
\t@echo "Running performance benchmarks..."
\t./$(TEST_TARGET) --performance-only

# Run stress test
test-stress: $(TEST_TARGET)
\t@echo "Running stress test (60 seconds)..."
\t./$(TEST_TARGET) --stress

# Quick test with simple program
test-simple: $(TARGET)
\t@echo "Running simple FFT test..."
\t@if [ ! -f "$(KERNEL_SRC)" ]; then \\
\t\techo "ERROR: Kernel file $(KERNEL_SRC) not found!"; \\
\t\texit 1; \\
\tfi
\t./$(TARGET)

'''

# Remove old test target and add new ones
makefile_ubuntu_with_test = makefile_ubuntu_with_test.replace(
    '# Quick test run\ntest: $(TARGET)\n\t@echo "Running FFT test..."\n\t@if [ ! -f "$(KERNEL_SRC)" ]; then \\\n\t\techo "ERROR: Kernel file $(KERNEL_SRC) not found!"; \\\n\t\texit 1; \\\n\tfi\n\t./$(TARGET)',
    ubuntu_test_rules
)

makefile_ubuntu_with_test = makefile_ubuntu_with_test.replace(
    '.PHONY: debug verbose asm uninstall',
    '.PHONY: debug verbose asm uninstall test test-correctness test-performance test-stress test-simple'
)

makefile_ubuntu_with_test = makefile_ubuntu_with_test.replace(
    '@rm -f $(TARGET) $(TARGET).stripped *.o',
    '@rm -f $(TARGET) $(TARGET).stripped $(TEST_TARGET) *.o'
)

with open('Makefile.ubuntu', 'w') as f:
    f.write(makefile_ubuntu_with_test)

print("✓ Updated both Makefiles with test targets")
print("\n" + "="*70)
print("NEW MAKEFILE TARGETS")
print("="*70)
print("\nBuild:")
print("  make                      - Build main FFT program")
print("  make mali_fft_test        - Build test suite")
print("")
print("Run Tests:")
print("  make test                 - Run full test suite (correctness + performance)")
print("  make test-correctness     - Run only correctness verification")
print("  make test-performance     - Run only performance benchmarks")
print("  make test-stress          - Run 60-second stress test")
print("  make test-simple          - Run simple FFT demo (Ubuntu only)")
print("")
print("="*70)
print("USAGE EXAMPLES")
print("="*70)
print("\nOn Ubuntu:")
print("  $ make -f Makefile.ubuntu mali_fft_test")
print("  $ make -f Makefile.ubuntu test")
print("")
print("On RZ/G2L (cross-compile):")
print("  $ rvenv")
print("  $ make mali_fft_test")
print("  $ make install DESTDIR=/path/to/rootfs")
print("  # Then on target:")
print("  target$ mali_fft_test")
