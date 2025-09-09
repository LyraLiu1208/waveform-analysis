#!/usr/bin/env python3
"""
Waveform Generator for SenseGlove Haptic Feedback
Analyzes material .wav files to generate C++ waveform arrays for LRA actuators.

Usage:
    python3 waveform_generator.py Material40
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Configuration
PICO_SAMPLE_RATE = 6400  # Hz - Should match your Pico I2S configuration
MAX_AMPLITUDE = 32767    # 16-bit signed integer max value
NUM_DOMINANT_FREQS = 3   # Number of dominant frequencies to extract per material

def load_and_analyze_wav(wav_file_path):
    """
    Load a .wav file and perform frequency analysis.
    
    Returns:
        tuple: (sample_rate, data, dominant_frequencies, frequency_amplitudes)
    """
    print(f"  Loading: {wav_file_path}")
    
    # Load the wav file
    sample_rate, data = wavfile.read(wav_file_path)
    
    # If stereo, take the first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize data to [-1, 1] range
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    else:
        data = data / np.max(np.abs(data))
    
    duration = len(data) / sample_rate
    print(f"    Duration: {duration:.2f}s, Sample Rate: {sample_rate}Hz")
    
    # Perform FFT
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / sample_rate)[:N//2]  # Only positive frequencies
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    
    # Find peaks in the frequency spectrum
    # Only look at frequencies between 20Hz and 1000Hz (reasonable for haptic feedback)
    freq_mask = (xf >= 20) & (xf <= 1000)
    valid_freqs = xf[freq_mask]
    valid_magnitudes = magnitude[freq_mask]
    
    # Find peaks with minimum prominence
    peaks, properties = find_peaks(valid_magnitudes, 
                                 prominence=np.max(valid_magnitudes) * 0.1,  # 10% of max
                                 distance=sample_rate // 50)  # Min 20Hz spacing
    
    if len(peaks) == 0:
        print("    Warning: No significant peaks found, using broadband analysis")
        # Fallback: find the frequency with maximum energy
        max_idx = np.argmax(valid_magnitudes)
        dominant_frequencies = [valid_freqs[max_idx]]
        frequency_amplitudes = [valid_magnitudes[max_idx]]
    else:
        # Sort peaks by magnitude and take the top N
        peak_magnitudes = valid_magnitudes[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # Sort descending
        
        # Take top NUM_DOMINANT_FREQS peaks
        n_peaks = min(NUM_DOMINANT_FREQS, len(sorted_indices))
        dominant_peak_indices = sorted_indices[:n_peaks]
        
        dominant_frequencies = [valid_freqs[peaks[i]] for i in dominant_peak_indices]
        frequency_amplitudes = [valid_magnitudes[peaks[i]] for i in dominant_peak_indices]
    
    # Round frequencies to nearest integer
    dominant_frequencies = [round(freq) for freq in dominant_frequencies]
    
    print(f"    Dominant frequencies: {dominant_frequencies} Hz")
    print(f"    Amplitudes: {[f'{amp:.4f}' for amp in frequency_amplitudes]}")
    
    return sample_rate, data, dominant_frequencies, frequency_amplitudes

def generate_sine_wave_array(frequency, pico_sample_rate=PICO_SAMPLE_RATE, max_amplitude=MAX_AMPLITUDE):
    """
    Generate a sine wave array for the given frequency.
    
    Args:
        frequency (float): Target frequency in Hz
        pico_sample_rate (int): Sample rate for the Pico
        max_amplitude (int): Maximum amplitude (16-bit signed)
    
    Returns:
        numpy.array: Quantized sine wave samples
    """
    if frequency <= 0:
        # Return empty array for invalid frequency
        return np.zeros(32, dtype=int)
    
    # Calculate number of samples for one complete period
    samples_per_period = int(pico_sample_rate / frequency)
    
    # Ensure we have at least a reasonable number of samples
    if samples_per_period < 8:
        samples_per_period = 8
    elif samples_per_period > 320:  # Limit array size
        samples_per_period = 320
    
    # Generate time points for one period
    t = np.linspace(0, 1/frequency, samples_per_period, endpoint=False)
    
    # Generate sine wave
    sine_wave = max_amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Quantize to integers
    quantized_wave = np.round(sine_wave).astype(int)
    
    # Ensure values are within 16-bit signed integer range
    quantized_wave = np.clip(quantized_wave, -32767, 32767)
    
    return quantized_wave

def generate_composite_waveform(frequencies, amplitudes, duration_ms=50, pico_sample_rate=PICO_SAMPLE_RATE, max_amplitude=MAX_AMPLITUDE):
    """
    Generate a composite waveform by adding multiple frequencies.
    
    Args:
        frequencies (list): List of frequencies in Hz
        amplitudes (list): List of relative amplitudes for each frequency
        duration_ms (float): Duration of the waveform in milliseconds
        pico_sample_rate (int): Sample rate for the Pico
        max_amplitude (int): Maximum amplitude (16-bit signed)
    
    Returns:
        numpy.array: Quantized composite waveform samples
    """
    if not frequencies or len(frequencies) == 0:
        return np.zeros(32, dtype=int)
    
    # Calculate number of samples
    num_samples = int(pico_sample_rate * duration_ms / 1000)
    if num_samples > 320:  # Limit array size
        num_samples = 320
    elif num_samples < 32:
        num_samples = 32
    
    # Generate time points
    t = np.linspace(0, duration_ms/1000, num_samples, endpoint=False)
    
    # Normalize amplitudes
    amplitudes = np.array(amplitudes)
    amplitudes = amplitudes / np.sum(amplitudes)
    
    # Generate composite waveform
    composite = np.zeros(num_samples)
    for freq, amp in zip(frequencies, amplitudes):
        if freq > 0:
            sine_wave = amp * np.sin(2 * np.pi * freq * t)
            composite += sine_wave
    
    # Scale to desired amplitude
    if np.max(np.abs(composite)) > 0:
        composite = composite / np.max(np.abs(composite)) * max_amplitude
    
    # Quantize to integers
    quantized_wave = np.round(composite).astype(int)
    
    # Ensure values are within 16-bit signed integer range
    quantized_wave = np.clip(quantized_wave, -32767, 32767)
    
    return quantized_wave

def format_cpp_array(array, array_name):
    """
    Format a numpy array as a C++ constexpr array declaration.
    
    Args:
        array (numpy.array): The array to format
        array_name (str): Name for the C++ array
    
    Returns:
        str: Formatted C++ array declaration
    """
    # Convert array to comma-separated string
    array_str = ", ".join(map(str, array))
    
    # Format as C++ array with proper indentation
    cpp_code = f"        inline constexpr pico::sample_type {array_name}[] = {{{array_str}}};"
    
    return cpp_code

def analyze_material(material_name, base_dir):
    """
    Analyze all .wav files for a given material and generate waveforms.
    
    Args:
        material_name (str): Name of the material (e.g., 'Material40')
        base_dir (str): Base directory of the waveform-analysis project
    
    Returns:
        dict: Dictionary containing generated waveforms and metadata for each participant
    """
    data_dir = os.path.join(base_dir, 'data', material_name)
    sliding_dir = os.path.join(data_dir, 'Sliding_Friction-roughness')
    
    if not os.path.isdir(sliding_dir):
        print(f"Error: Sliding directory not found at '{sliding_dir}'")
        return None
    
    print(f"Analyzing material: {material_name}")
    print(f"Looking in: {sliding_dir}")
    
    participant_results = {}
    
    # Process each participant separately
    for participant_dir in os.listdir(sliding_dir):
        participant_path = os.path.join(sliding_dir, participant_dir)
        
        if not os.path.isdir(participant_path):
            continue
            
        print(f"\n--- Processing {participant_dir} ---")
        
        wav_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"  Warning: No .wav files found for {participant_dir}")
            continue
        
        participant_dominant_frequencies = []
        participant_frequency_amplitudes = []
        participant_data = {}
        participant_original_data = []
        
        # Process each wav file for this participant
        for wav_file in wav_files:
            wav_path = os.path.join(participant_path, wav_file)
            
            # Analyze the wav file
            try:
                sample_rate, data, dom_freqs, freq_amps = load_and_analyze_wav(wav_path)
                
                participant_data[f"{participant_dir}_{wav_file}"] = {
                    'dominant_frequencies': dom_freqs,
                    'frequency_amplitudes': freq_amps,
                    'sample_rate': sample_rate,
                    'raw_data': data
                }
                
                participant_dominant_frequencies.extend(dom_freqs)
                participant_frequency_amplitudes.extend(freq_amps)
                
                # Store original data for this participant (take a representative segment)
                if len(data) > 0:
                    # Take middle section to avoid start/end artifacts
                    start_idx = len(data) // 4
                    end_idx = 3 * len(data) // 4
                    segment = data[start_idx:end_idx]
                    participant_original_data.extend(segment[:1000])  # Limit length
                
            except Exception as e:
                print(f"    Error processing {wav_path}: {e}")
                continue
        
        if not participant_dominant_frequencies:
            print(f"  Warning: No dominant frequencies found for {participant_dir}")
            continue
        
        # Create downsampled original waveform for this participant
        original_waveform = np.array([])
        if participant_original_data:
            # Downsample and quantize original data
            original_combined = np.array(participant_original_data)
            # Take every nth sample to fit reasonable array size
            downsample_factor = max(1, len(original_combined) // 160)  # Target ~160 samples
            downsampled = original_combined[::downsample_factor]
            # Quantize to 16-bit integers
            original_waveform = np.round(downsampled * MAX_AMPLITUDE).astype(int)
            original_waveform = np.clip(original_waveform, -32767, 32767)
            
            print(f"  {participant_dir} original waveform: {len(original_waveform)} samples")
        
        # Find the most significant frequencies for this participant
        unique_freqs, freq_counts = np.unique(np.round(participant_dominant_frequencies), return_counts=True)
        
        # Sort by frequency of occurrence and amplitude
        freq_importance = []
        for freq in unique_freqs:
            # Find all amplitudes for this frequency
            freq_mask = np.abs(np.array(participant_dominant_frequencies) - freq) < 2  # 2Hz tolerance
            avg_amplitude = np.mean([participant_frequency_amplitudes[i] for i, mask in enumerate(freq_mask) if mask])
            freq_count = np.sum(freq_mask)
            
            # Importance score: combination of occurrence and amplitude
            importance = freq_count * avg_amplitude
            freq_importance.append((freq, importance, avg_amplitude))
        
        # Sort by importance and take top frequencies
        freq_importance.sort(key=lambda x: x[1], reverse=True)
        selected_frequencies = [int(freq) for freq, _, _ in freq_importance[:NUM_DOMINANT_FREQS]]
        
        print(f"  {participant_dir} selected frequencies: {selected_frequencies} Hz")
        
        # Store results for this participant
        participant_results[participant_dir] = {
            'material_name': material_name,
            'participant_name': participant_dir,
            'selected_frequencies': selected_frequencies,
            'participant_data': participant_data,
            'frequency_importance': freq_importance,
            'original_waveform': original_waveform,
            'original_data': participant_original_data
        }
    
    if not participant_results:
        print("Error: No participant data found for this material")
        return None
    
    return participant_results

def generate_cpp_output(participant_result, output_path):
    """
    Generate C++ header code with waveform arrays for a single participant.
    
    Args:
        participant_result (dict): Result from analyze_material() for one participant
        output_path (str): Path to save the output file
    """
    if not participant_result:
        print("Error: No participant result to generate output from")
        return
    
    material_name = participant_result['material_name']
    participant_name = participant_result['participant_name']
    selected_frequencies = participant_result['selected_frequencies']
    original_waveform = participant_result.get('original_waveform', np.array([]))
    
    print(f"\nGenerating C++ arrays for {material_name} - {participant_name}...")
    
    # Start building the C++ code
    cpp_lines = []
    cpp_lines.append("// Auto-generated waveform arrays for haptic feedback")
    cpp_lines.append(f"// Material: {material_name}")
    cpp_lines.append(f"// Participant: {participant_name}")
    cpp_lines.append(f"// Generated on: {np.datetime64('now')}")
    cpp_lines.append("// Frequencies based on FFT analysis of sliding friction audio")
    cpp_lines.append("")
    cpp_lines.append("namespace _vibro_lut_arrays")
    cpp_lines.append("{")
    
    # Always include an empty array
    empty_array = np.zeros(32, dtype=int)
    cpp_lines.append(format_cpp_array(empty_array, "EMPTY_ARRAY"))
    
    # Add original waveform array (downsampled)
    if len(original_waveform) > 0:
        array_name = f"ORIGINAL_{material_name.upper()}_{participant_name.upper()}_ARRAY"
        cpp_lines.append(format_cpp_array(original_waveform, array_name))
        print(f"  Generated: {array_name} ({len(original_waveform)} samples)")
    
    # Generate composite waveform (sum of dominant frequencies)
    if len(selected_frequencies) > 0:
        # Use equal amplitudes for simplicity
        amplitudes = [1.0] * len(selected_frequencies)
        composite_array = generate_composite_waveform(selected_frequencies, amplitudes)
        array_name = f"COMPOSITE_{material_name.upper()}_{participant_name.upper()}_ARRAY"
        cpp_lines.append(format_cpp_array(composite_array, array_name))
        print(f"  Generated: {array_name} ({len(composite_array)} samples)")
    
    # Generate arrays for each individual frequency (for switching method)
    generated_arrays = []
    for freq in selected_frequencies:
        if freq > 0:  # Skip invalid frequencies
            array_name = f"FREQ_{freq}HZ_{material_name.upper()}_{participant_name.upper()}_ARRAY"
            sine_array = generate_sine_wave_array(freq)
            cpp_lines.append(format_cpp_array(sine_array, array_name))
            generated_arrays.append((freq, array_name))
            
            print(f"  Generated: {array_name} ({len(sine_array)} samples)")
    
    cpp_lines.append("}")
    cpp_lines.append("")
    
    # Add comment with metadata
    cpp_lines.append("/*")
    cpp_lines.append(f"Material Analysis Summary for {material_name} - {participant_name}:")
    cpp_lines.append(f"Selected Frequencies: {selected_frequencies} Hz")
    cpp_lines.append("")
    cpp_lines.append("Usage Options:")
    cpp_lines.append("1. Use COMPOSITE array for complex texture (sum of frequencies)")
    cpp_lines.append("2. Use individual FREQ arrays for switching between frequencies")
    cpp_lines.append("3. Use ORIGINAL array for raw material sound reproduction")
    cpp_lines.append("")
    
    # Add participant data summary
    participant_data = participant_result['participant_data']
    for participant, data in participant_data.items():
        cpp_lines.append(f"{participant}:")
        cpp_lines.append(f"  Dominant frequencies: {data['dominant_frequencies']} Hz")
        cpp_lines.append(f"  Sample rate: {data['sample_rate']} Hz")
    
    cpp_lines.append("*/")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(cpp_lines))
    
    print(f"\nC++ arrays saved to: {output_path}")
    print(f"Generated {len(generated_arrays) + 2} waveform arrays total")

def create_analysis_plots(participant_result, output_dir):
    """
    Create visualization plots for the frequency analysis of a single participant.
    
    Args:
        participant_result (dict): Result from analyze_material() for one participant
        output_dir (str): Directory to save plots
    """
    if not participant_result:
        return
    
    material_name = participant_result['material_name']
    participant_name = participant_result['participant_name']
    selected_frequencies = participant_result['selected_frequencies']
    original_data = participant_result.get('original_data', [])
    
    # Create plots in the same directory as the waveform output
    plots_dir = os.path.join(output_dir, 'waveform', material_name, participant_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original waveform (time domain)
    plt.subplot(3, 1, 1)
    if original_data:
        original_array = np.array(original_data[:2000])  # Show first 2000 samples
        time_axis = np.arange(len(original_array)) / 44100 * 1000  # Assume 44.1kHz, convert to ms
        plt.plot(time_axis, original_array)
        plt.title(f'{material_name} - {participant_name}: Original Waveform (Time Domain)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No original data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{material_name} - {participant_name}: Original Waveform (Time Domain)')
    
    # Plot 2: FFT Result (frequency domain)
    plt.subplot(3, 1, 2)
    if original_data:
        # Perform FFT on original data
        original_array = np.array(original_data[:8192])  # Use 8192 samples for better frequency resolution
        if len(original_array) > 0:
            N = len(original_array)
            yf = fft(original_array)
            xf = fftfreq(N, 1 / 44100)[:N//2]  # Assume 44.1kHz sample rate
            magnitude = 2.0/N * np.abs(yf[0:N//2])
            
            # Only show frequencies up to 1000Hz
            freq_mask = xf <= 1000
            plt.plot(xf[freq_mask], magnitude[freq_mask])
            
            # Mark the selected dominant frequencies
            for freq in selected_frequencies:
                if freq <= 1000:
                    plt.axvline(x=freq, color='red', linestyle='--', alpha=0.7, label=f'{freq}Hz')
            
            plt.title(f'{material_name} - {participant_name}: FFT Result (Frequency Domain)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No FFT data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{material_name} - {participant_name}: FFT Result (Frequency Domain)')
    
    # Plot 3: Processed waveform (composite of dominant frequencies)
    plt.subplot(3, 1, 3)
    if selected_frequencies:
        # Generate composite waveform
        amplitudes = [1.0] * len(selected_frequencies)  # Equal amplitudes
        composite_array = generate_composite_waveform(selected_frequencies, amplitudes, duration_ms=100)
        time_axis = np.arange(len(composite_array)) / PICO_SAMPLE_RATE * 1000  # Convert to ms
        
        plt.plot(time_axis, composite_array, 'b-', linewidth=2, label='Composite')
        
        # Also plot individual frequency components
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, freq in enumerate(selected_frequencies[:5]):  # Show first 5 frequencies
            if freq > 0:
                single_freq_array = generate_sine_wave_array(freq)
                # Extend or trim to match composite length
                if len(single_freq_array) < len(composite_array):
                    # Repeat the single frequency array to match length
                    repeats = len(composite_array) // len(single_freq_array) + 1
                    single_extended = np.tile(single_freq_array, repeats)[:len(composite_array)]
                else:
                    single_extended = single_freq_array[:len(composite_array)]
                
                time_axis_single = np.arange(len(single_extended)) / PICO_SAMPLE_RATE * 1000
                plt.plot(time_axis_single, single_extended * 0.3, '--', 
                        color=colors[i % len(colors)], alpha=0.7, label=f'{freq}Hz')
        
        plt.title(f'{material_name} - {participant_name}: Processed Waveform (Composite of Dominant Frequencies)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No processed waveform available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{material_name} - {participant_name}: Processed Waveform (Composite of Dominant Frequencies)')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'{material_name}_{participant_name}_waveform_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plots saved to: {plot_path}")

def main():
    """Main function to run waveform generation."""
    parser = argparse.ArgumentParser(description="Generate haptic waveforms from material audio data.")
    parser.add_argument('material', type=str, 
                       help="Name of the material folder (e.g., 'Material40')")
    parser.add_argument('--output-dir', type=str, default=None,
                       help="Output directory (default: ../output/)")
    args = parser.parse_args()
    
    # Determine paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = args.output_dir or os.path.join(base_dir, 'output')
    
    print("=" * 60)
    print("SenseGlove Haptic Waveform Generator")
    print("=" * 60)
    
    # Analyze the material (now returns results for each participant)
    participant_results = analyze_material(args.material, base_dir)
    
    if not participant_results:
        print("Analysis failed. Exiting.")
        return 1
    
    # Process each participant separately
    for participant_name, participant_result in participant_results.items():
        print(f"\n{'='*30} {participant_name} {'='*30}")
        
        # Generate C++ output for this participant
        waveform_output_dir = os.path.join(output_dir, 'waveform', args.material, participant_name)
        cpp_output_path = os.path.join(waveform_output_dir, f'{args.material}_{participant_name}_waveforms.txt')
        os.makedirs(waveform_output_dir, exist_ok=True)
        generate_cpp_output(participant_result, cpp_output_path)
        
        # Create analysis plots for this participant
        create_analysis_plots(participant_result, output_dir)
    
    print("\n" + "=" * 60)
    print("Waveform generation completed successfully!")
    print(f"Generated results for {len(participant_results)} participants")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    exit(main())
