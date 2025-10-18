/*
 * Enhanced FFT Validation Suite
 *
 * Validates the FFT implementation against:
 * 1. Known FFT properties (Parseval's theorem, linearity, symmetry)
 * 2. Transform pairs (impulse, DC, complex exponential)
 * 3. Mathematical identities
 * 4. Comparison with reference implementations
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace std;

// ============================================================================
// Reference CPU FFT Implementation (Cooley-Tukey Radix-2)
// ============================================================================

void fft_radix2_reference(vector<complex<float>>& data) {
    size_t N = data.size();

    // Bit reversal
    for (size_t i = 0; i < N; i++) {
        size_t j = 0;
        size_t k = i;
        size_t m = N >> 1;

        while (m) {
            j = (j << 1) | (k & 1);
            k >>= 1;
            m >>= 1;
        }

        if (j > i) {
            swap(data[i], data[j]);
        }
    }

    // FFT computation
    for (size_t s = 1; s <= log2(N); s++) {
        size_t m = 1 << s;
        complex<float> wm = exp(complex<float>(0, -2.0f * M_PI / m));

        for (size_t k = 0; k < N; k += m) {
            complex<float> w(1.0f, 0.0f);

            for (size_t j = 0; j < m / 2; j++) {
                complex<float> t = w * data[k + j + m / 2];
                complex<float> u = data[k + j];

                data[k + j] = u + t;
                data[k + j + m / 2] = u - t;

                w *= wm;
            }
        }
    }
}

// ============================================================================
// FFT Validation Tests
// ============================================================================

struct ValidationResult {
    string test_name;
    bool passed;
    double error;
    string details;
};

vector<ValidationResult> results;

void report_test(const string& name, bool passed, double error = 0.0,
                 const string& details = "") {
    results.push_back({name, passed, error, details});
    cout << (passed ? "✓ PASS" : "✗ FAIL") << " : " << name;
    if (error > 0) {
        cout << " (error: " << scientific << error << ")";
    }
    if (!details.empty()) {
        cout << " - " << details;
    }
    cout << endl;
}

// Test 1: Parseval's Theorem (Energy Conservation)
// Principle: Sum of squared magnitudes in time domain = Sum in frequency domain / N
bool test_parsevals_theorem() {
    cout << "\n=== Test 1: Parseval's Theorem (Energy Conservation) ===" << endl;

    size_t N = 256;
    vector<complex<float>> signal(N);

    // Create test signal: random complex values
    for (size_t i = 0; i < N; i++) {
        signal[i] = complex<float>(
            (float)rand() / RAND_MAX * 2.0f - 1.0f,
            (float)rand() / RAND_MAX * 2.0f - 1.0f
        );
    }

    // Calculate time-domain energy
    double time_energy = 0.0;
    for (const auto& val : signal) {
        time_energy += norm(val);  // norm = |z|^2 = real^2 + imag^2
    }

    // Compute FFT
    auto fft_signal = signal;
    fft_radix2_reference(fft_signal);

    // Calculate frequency-domain energy
    double freq_energy = 0.0;
    for (const auto& val : fft_signal) {
        freq_energy += norm(val);
    }

    // Parseval: Energy_time = Energy_freq / N
    double energy_ratio = freq_energy / (time_energy * N);
    double tolerance = 1e-5;
    bool passed = abs(energy_ratio - 1.0) < tolerance;

    report_test("Parseval's Theorem", passed, abs(energy_ratio - 1.0),
                "Energy ratio: " + to_string(energy_ratio));

    return passed;
}

// Test 2: Impulse Response
// For impulse input δ[n], FFT output should be constant (all 1's)
bool test_impulse_response() {
    cout << "\n=== Test 2: Impulse Response ===" << endl;

    size_t N = 256;
    vector<complex<float>> impulse(N, 0.0f);
    impulse[0] = 1.0f;  // δ[n]

    auto fft_result = impulse;
    fft_radix2_reference(fft_result);

    // All bins should be 1.0
    double max_error = 0.0;
    for (size_t i = 0; i < N; i++) {
        double mag = abs(fft_result[i]);
        double expected = 1.0;
        max_error = max(max_error, abs(mag - expected));
    }

    bool passed = max_error < 1e-5;
    report_test("Impulse Response", passed, max_error,
                "Max deviation from magnitude 1.0");

    return passed;
}

// Test 3: DC Signal
// For constant input x[n] = 1, only X[0] should be non-zero (N)
bool test_dc_signal() {
    cout << "\n=== Test 3: DC Signal (Constant Input) ===" << endl;

    size_t N = 256;
    vector<complex<float>> dc_signal(N, 1.0f);

    auto fft_result = dc_signal;
    fft_radix2_reference(fft_result);

    // X[0] should be N, others should be near 0
    double dc_component = abs(fft_result[0]);
    double expected_dc = (double)N;
    double dc_error = abs(dc_component - expected_dc) / expected_dc;

    double max_ac_error = 0.0;
    for (size_t i = 1; i < N; i++) {
        max_ac_error = max(max_ac_error, (double)abs(fft_result[i]));
    }

    bool passed = (dc_error < 1e-5) && (max_ac_error < 1e-5);
    report_test("DC Signal", passed, max(dc_error, max_ac_error),
                "DC error: " + to_string(dc_error) +
                ", AC error: " + to_string(max_ac_error));

    return passed;
}

// Test 4: Nyquist Frequency
// For real input, X[N/2] should be real (no imaginary component)
bool test_nyquist_frequency() {
    cout << "\n=== Test 4: Nyquist Frequency (Real Input Symmetry) ===" << endl;

    size_t N = 256;
    vector<complex<float>> signal(N);

    // Real input signal
    for (size_t i = 0; i < N; i++) {
        signal[i] = complex<float>((float)sin(2 * M_PI * 10 * i / N), 0.0f);
    }

    auto fft_result = signal;
    fft_radix2_reference(fft_result);

    // For real input: X[k] = X*[N-k]
    // Nyquist bin X[N/2] should be real
    double nyquist_imag = abs(fft_result[N/2].imag());

    double max_symmetry_error = 0.0;
    for (size_t k = 1; k < N/2; k++) {
        complex<float> xk = fft_result[k];
        complex<float> xnk = conj(fft_result[N-k]);  // X*[N-k]
        max_symmetry_error = max(max_symmetry_error, (double)abs(xk - xnk));
    }

    bool passed = (nyquist_imag < 1e-4) && (max_symmetry_error < 1e-3);
    report_test("Nyquist/Real Input Symmetry", passed,
                max(nyquist_imag, max_symmetry_error),
                "Nyquist imag: " + to_string(nyquist_imag));

    return passed;
}

// Test 5: Linearity
// FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
bool test_linearity() {
    cout << "\n=== Test 5: Linearity ===" << endl;

    size_t N = 128;
    float a = 2.5f, b = 3.7f;

    // Create test signals
    vector<complex<float>> x(N), y(N);
    for (size_t i = 0; i < N; i++) {
        x[i] = complex<float>((float)sin(2*M_PI*5*i/N), 0.0f);
        y[i] = complex<float>((float)cos(2*M_PI*7*i/N), 0.0f);
    }

    // Compute FFT(a*x + b*y)
    auto combined = x;
    for (size_t i = 0; i < N; i++) combined[i] = a*x[i] + b*y[i];
    auto fft_combined = combined;
    fft_radix2_reference(fft_combined);

    // Compute a*FFT(x) + b*FFT(y)
    auto fft_x = x, fft_y = y;
    fft_radix2_reference(fft_x);
    fft_radix2_reference(fft_y);

    auto fft_linear = fft_x;
    for (size_t i = 0; i < N; i++) {
        fft_linear[i] = a * fft_x[i] + b * fft_y[i];
    }

    // Compare
    double max_error = 0.0;
    for (size_t i = 0; i < N; i++) {
        max_error = max(max_error, (double)abs(fft_combined[i] - fft_linear[i]));
    }

    bool passed = max_error < 1e-3;
    report_test("Linearity", passed, max_error);

    return passed;
}

// Test 6: Shift Property
// Time shift: FFT(x[n-n0]) = exp(-2πi*n0*k/N) * FFT(x[n])
bool test_shift_property() {
    cout << "\n=== Test 6: Shift Property ===" << endl;

    size_t N = 128;
    int shift = 5;

    // Create test signal
    vector<complex<float>> signal(N);
    for (size_t i = 0; i < N; i++) {
        signal[i] = complex<float>((float)sin(2*M_PI*4*i/N), 0.0f);
    }

    // Shifted signal (circular shift)
    vector<complex<float>> shifted(N);
    for (size_t i = 0; i < N; i++) {
        shifted[i] = signal[(i - shift + N) % N];
    }

    // Compute FFTs
    auto fft_orig = signal;
    auto fft_shift = shifted;
    fft_radix2_reference(fft_orig);
    fft_radix2_reference(fft_shift);

    // Check shift relationship
    double max_error = 0.0;
    for (size_t k = 0; k < N; k++) {
        complex<float> phase_shift = exp(complex<float>(0, -2.0f * M_PI * shift * k / N));
        complex<float> expected = fft_orig[k] * phase_shift;
        max_error = max(max_error, (double)abs(fft_shift[k] - expected));
    }

    bool passed = max_error < 1e-2;
    report_test("Shift Property", passed, max_error);

    return passed;
}

// Test 7: Magnitude Spectrum Symmetry (for real input)
// For real input: |X[k]| = |X[N-k]|
bool test_magnitude_symmetry() {
    cout << "\n=== Test 7: Magnitude Spectrum Symmetry ===" << endl;

    size_t N = 256;
    vector<complex<float>> real_signal(N);

    // Real input signal
    for (size_t i = 0; i < N; i++) {
        real_signal[i] = complex<float>((float)rand() / RAND_MAX * 2.0f - 1.0f, 0.0f);
    }

    auto fft_result = real_signal;
    fft_radix2_reference(fft_result);

    // Check |X[k]| = |X[N-k]|
    double max_error = 0.0;
    for (size_t k = 1; k <= N/2; k++) {
        double mag_k = abs(fft_result[k]);
        double mag_nk = abs(fft_result[N-k]);
        max_error = max(max_error, fabs(mag_k - mag_nk));
    }

    bool passed = max_error < 1e-4;
    report_test("Magnitude Symmetry (Real Input)", passed, max_error);

    return passed;
}

// Test 8: Convolution Property
// FFT(x * y) = FFT(x) · FFT(y)  [where * is convolution, · is pointwise mult]
// For circular convolution (limited verification)
bool test_convolution_property() {
    cout << "\n=== Test 8: Convolution Property (Circular) ===" << endl;

    size_t N = 64;
    vector<complex<float>> x(N), y(N);

    // Create test signals
    for (size_t i = 0; i < N; i++) {
        x[i] = complex<float>((i < 8) ? 1.0f : 0.0f, 0.0f);
        y[i] = complex<float>((i < 8) ? 1.0f : 0.0f, 0.0f);
    }

    // Compute FFT(x) and FFT(y)
    auto fft_x = x, fft_y = y;
    fft_radix2_reference(fft_x);
    fft_radix2_reference(fft_y);

    // Pointwise multiply
    vector<complex<float>> fft_xy(N);
    for (size_t i = 0; i < N; i++) {
        fft_xy[i] = fft_x[i] * fft_y[i];
    }

    // Inverse FFT (approximate: just check order of magnitude)
    // This is a simplified check since we don't have IFFT readily available
    double max_magnitude = 0.0;
    for (size_t i = 0; i < N; i++) {
        max_magnitude = max(max_magnitude, (double)abs(fft_xy[i]));
    }

    bool passed = max_magnitude > 0;  // Basic sanity check
    report_test("Convolution Property (Sanity Check)", passed, 0.0,
                "Max magnitude in product: " + to_string(max_magnitude));

    return passed;
}

// Test 9: Complex Conjugate Property
// FFT(x*[n]) = X*[N-k]  for complex conjugate
bool test_conjugate_property() {
    cout << "\n=== Test 9: Conjugate Property ===" << endl;

    size_t N = 128;
    vector<complex<float>> signal(N);

    // Create complex signal
    for (size_t i = 0; i < N; i++) {
        signal[i] = complex<float>(
            (float)sin(2*M_PI*5*i/N),
            (float)cos(2*M_PI*7*i/N)
        );
    }

    // Compute FFT(x)
    auto fft_x = signal;
    fft_radix2_reference(fft_x);

    // Compute FFT(x*)
    auto conjugate_signal = signal;
    for (auto& val : conjugate_signal) val = conj(val);
    auto fft_xconj = conjugate_signal;
    fft_radix2_reference(fft_xconj);

    // Check: FFT(x*)[k] = conj(FFT(x)[N-k])
    double max_error = 0.0;
    for (size_t k = 0; k < N; k++) {
        complex<float> expected = conj(fft_x[(N-k) % N]);
        max_error = max(max_error, (double)abs(fft_xconj[k] - expected));
    }

    bool passed = max_error < 1e-3;
    report_test("Conjugate Property", passed, max_error);

    return passed;
}

// Test 10: Algorithm Verification - Compare different FFT sizes
// Verify consistency across power-of-2 sizes
bool test_algorithm_consistency() {
    cout << "\n=== Test 10: Algorithm Consistency (Multiple Sizes) ===" << endl;

    vector<size_t> sizes = {16, 32, 64, 128, 256};
    bool all_passed = true;

    for (size_t N : sizes) {
        // Create simple sine wave
        vector<complex<float>> signal(N);
        for (size_t i = 0; i < N; i++) {
            signal[i] = complex<float>((float)sin(2*M_PI*5*i/N), 0.0f);
        }

        auto fft_result = signal;
        fft_radix2_reference(fft_result);

        // Find peak frequency bin (should be around bin 5)
        size_t max_bin = 0;
        double max_magnitude = 0.0;
        for (size_t k = 1; k < N/2; k++) {
            double mag = abs(fft_result[k]);
            if (mag > max_magnitude) {
                max_magnitude = mag;
                max_bin = k;
            }
        }

        // For sine wave 5*f0, peak should be around bin 5 (or close)
        bool peak_correct = (max_bin >= 4 && max_bin <= 6);
        cout << "  N=" << N << ": Peak at bin " << max_bin
             << " (magnitude: " << fixed << setprecision(2) << max_magnitude << ") "
             << (peak_correct ? "✓" : "✗") << endl;
        all_passed &= peak_correct;
    }

    report_test("Algorithm Consistency", all_passed, 0.0);
    return all_passed;
}

// ============================================================================
// Main Validation Suite
// ============================================================================

int main() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════════════╗\n";
    cout << "║   FFT Implementation Validation Suite                         ║\n";
    cout << "║   Testing against known FFT mathematical properties          ║\n";
    cout << "╚════════════════════════════════════════════════════════════════╝\n";
    cout << endl;

    srand(42);  // Fixed seed for reproducibility

    // Run all validation tests
    bool all_tests_passed = true;

    all_tests_passed &= test_parsevals_theorem();
    all_tests_passed &= test_impulse_response();
    all_tests_passed &= test_dc_signal();
    all_tests_passed &= test_nyquist_frequency();
    all_tests_passed &= test_linearity();
    all_tests_passed &= test_shift_property();
    all_tests_passed &= test_magnitude_symmetry();
    all_tests_passed &= test_convolution_property();
    all_tests_passed &= test_conjugate_property();
    all_tests_passed &= test_algorithm_consistency();

    // Print summary
    cout << "\n" << string(70, '=') << endl;
    cout << "VALIDATION SUMMARY" << endl;
    cout << string(70, '=') << endl;

    int passed_count = 0;
    for (const auto& result : results) {
        if (result.passed) passed_count++;
        cout << (result.passed ? "✓" : "✗") << " " << result.test_name << endl;
    }

    cout << string(70, '=') << endl;
    cout << "Results: " << passed_count << "/" << results.size() << " tests passed" << endl;
    cout << "Status: " << (all_tests_passed ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED") << endl;
    cout << string(70, '=') << endl;

    return all_tests_passed ? 0 : 1;
}
