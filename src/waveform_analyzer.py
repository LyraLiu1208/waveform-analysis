#!/usr/bin/env python3
"""
Waveform Analysis Script for Vibration Data

This script analyzes .wav vibration data files from different materials,
performs FFT analysis to extract dominant frequencies, reconstructs waveforms,
and generates outputs compatible with Pico microcontroller requirements.

Usage:
    python waveform_analyzer.py [MaterialName]
    
    If MaterialName is not provided, the script will process all materials
    found in the data directory.

Example:
    python waveform_analyzer.py Material40
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample
import re
from pathlib import Path


class WaveformAnalyzer:
    """
    A class to analyze vibration waveform data from .wav files.
    """
    
    def __init__(self, base_dir=".", original_fs=44100, target_fs=8000):
        """
        Initialize the WaveformAnalyzer.
        
        Args:
            base_dir (str): Base directory containing the waveform-analysis folder
            original_fs (int): Original sampling frequency of the .wav files
            target_fs (int): Target sampling frequency for Pico processing
        """
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output" / "waveform"
        self.original_fs = original_fs
        self.target_fs = target_fs
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_materials(self):
        """
        Find all available materials in the data directory.
        
        Returns:
            list: List of material names found in the data directory
        """
        materials = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir() and (item / "Sliding_Friction-roughness").exists():
                    materials.append(item.name)
        return sorted(materials)
    
    def extract_material_number(self, material_name):
        """
        Extract the numeric part from material name for file matching.
        
        Args:
            material_name (str): Material name (e.g., "Material40")
            
        Returns:
            str: Numeric part (e.g., "40")
        """
        match = re.search(r'\d+', material_name)
        return match.group() if match else material_name.lower()
    
    def load_wav_file(self, file_path):
        """
        Load a .wav file and return the audio data and sample rate.
        Loads full data for FFT analysis.
        
        Args:
            file_path (Path): Path to the .wav file
            
        Returns:
            tuple: (sample_rate, audio_data)
        """
        try:
            sample_rate, data = wavfile.read(file_path)
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            print(f"    Loaded full audio: {len(data)} samples ({len(data)/sample_rate:.2f} seconds)")
            
            return sample_rate, data.astype(np.float64)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def analyze_frequencies(self, data, fs):
        """
        Analyze frequencies using FFT to find dominant frequencies for LRA vibration reconstruction.
        
        This method extracts the main frequency components from noisy vibration data
        and prepares them for clean periodic signal reconstruction.
        
        Args:
            data (np.array): Audio data (noisy vibration signal)
            fs (int): Sampling frequency
            
        Returns:
            tuple: (frequencies, amplitudes, phases, dominant_freqs_info)
        """
        # Step 1: Preprocessing - Remove DC component
        data_centered = data - np.mean(data)
        
        # Step 2: Apply window to reduce spectral leakage
        windowed_data = data_centered * np.hanning(len(data_centered))
        
        # Step 3: Compute FFT for frequency domain analysis
        fft_result = np.fft.fft(windowed_data)
        frequencies = np.fft.fftfreq(len(windowed_data), 1/fs)
        
        # Take only positive frequencies (single-sided spectrum)
        positive_mask = frequencies > 0
        frequencies = frequencies[positive_mask]
        fft_result = fft_result[positive_mask]
        
        # Step 4: Compute amplitude and phase
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Step 5: Filter frequency range suitable for LRA (typically 50-300 Hz, but allowing wider range)
        valid_mask = (frequencies >= 5) & (frequencies <= 200)
        valid_frequencies = frequencies[valid_mask]
        valid_amplitudes = amplitudes[valid_mask]
        valid_phases = phases[valid_mask]
        
        # Step 6: Find dominant frequency components (main resonant frequencies)
        from scipy.signal import find_peaks
        
        # Find peaks with significant amplitude
        amplitude_threshold = np.max(valid_amplitudes) * 0.1  # At least 10% of max amplitude
        peaks, properties = find_peaks(
            valid_amplitudes, 
            height=amplitude_threshold,
            distance=int(len(valid_amplitudes) * 0.02)  # Minimum 2% of spectrum apart
        )
        
        if len(peaks) < 3:
            # If not enough significant peaks, use largest amplitudes
            dominant_indices = np.argsort(valid_amplitudes)[-3:][::-1]
        else:
            # Sort peaks by amplitude and take top 3 (representing main harmonics)
            peak_amplitudes = valid_amplitudes[peaks]
            sorted_peak_indices = np.argsort(peak_amplitudes)[-3:][::-1]
            dominant_indices = peaks[sorted_peak_indices]
        
        # Step 7: Extract frequency components for reconstruction
        dominant_freqs_info = []
        for idx in dominant_indices:
            freq = valid_frequencies[idx]
            amp = valid_amplitudes[idx]
            phase = valid_phases[idx]
            
            # Store the actual FFT magnitude (this represents the true energy at this frequency)
            dominant_freqs_info.append({
                'frequency': freq,
                'amplitude': amp,  # This is the true spectral amplitude
                'phase': phase
            })
        
        # Sort by frequency for consistent output
        dominant_freqs_info.sort(key=lambda x: x['frequency'])
        
        return frequencies, amplitudes, phases, dominant_freqs_info
    
    def generate_sine_waves(self, dominant_freqs_info, duration, fs):
        """
        Generate sine waves for the dominant frequencies.
        
        Args:
            dominant_freqs_info (list): List of dominant frequency information
            duration (float): Duration of the signal in seconds
            fs (int): Sampling frequency
            
        Returns:
            tuple: (time_array, list_of_sine_waves)
        """
        t = np.linspace(0, duration, int(duration * fs), endpoint=False)
        sine_waves = []
        
        for freq_info in dominant_freqs_info:
            freq = freq_info['frequency']
            phase = freq_info['phase']
            amplitude = freq_info['amplitude']
            
            # Use the actual amplitude from FFT analysis
            # Normalize to reasonable range for audio processing
            normalized_amplitude = amplitude / np.max([f['amplitude'] for f in dominant_freqs_info])
            sine_wave = normalized_amplitude * np.sin(2 * np.pi * freq * t + phase)
            sine_waves.append(sine_wave)
        
        return t, sine_waves
    
    def reconstruct_waveform_from_frequencies(self, dominant_freqs_info, duration, fs, preserve_original_amplitude=True, original_data_rms=None):
        """
        Reconstruct a clean periodic waveform from dominant frequency components.
        
        This implements the frequency domain reconstruction approach:
        1. Use the dominant frequencies extracted from FFT
        2. Reconstruct as sum of sine waves with proper amplitude scaling
        3. Preserve original amplitude range for realistic LRA driving
        
        Args:
            dominant_freqs_info (list): List of dominant frequency information
            duration (float): Duration of the signal in seconds
            fs (int): Sampling frequency
            preserve_original_amplitude (bool): Whether to preserve original data amplitude range
            original_data_rms (float): RMS of original data for amplitude scaling
            
        Returns:
            tuple: (time_array, reconstructed_waveform)
        """
        t = np.linspace(0, duration, int(duration * fs), endpoint=False)
        
        # Initialize reconstructed signal
        reconstructed = np.zeros(len(t))
        
        # For realistic LRA driving, we need to preserve the original amplitude characteristics
        # Calculate the relative contribution of each frequency based on FFT magnitudes
        total_magnitude = sum(freq_info['amplitude'] for freq_info in dominant_freqs_info)
        
        for freq_info in dominant_freqs_info:
            freq = freq_info['frequency']
            amplitude = freq_info['amplitude']
            phase = freq_info['phase']
            
            # Use the relative magnitude contribution (not energy-based)
            # This better preserves the original signal characteristics
            relative_contribution = amplitude / total_magnitude if total_magnitude > 0 else 1/len(dominant_freqs_info)
            
            # Add this frequency component to the reconstruction
            component = relative_contribution * np.sin(2 * np.pi * freq * t + phase)
            reconstructed += component
        
        if preserve_original_amplitude and original_data_rms is not None:
            # Scale to match the original data's RMS amplitude
            current_rms = np.sqrt(np.mean(reconstructed**2))
            if current_rms > 0:
                reconstructed = reconstructed * (original_data_rms / current_rms)
        else:
            # Default normalization (keep for backward compatibility)
            max_amplitude = np.max(np.abs(reconstructed))
            if max_amplitude > 0:
                reconstructed = reconstructed / max_amplitude * 0.8
        
        return t, reconstructed
    
    def downsample_signal(self, signal, original_fs, target_fs):
        """
        Downsample a signal from original_fs to target_fs.
        
        Args:
            signal (np.array): Input signal
            original_fs (int): Original sampling frequency
            target_fs (int): Target sampling frequency
            
        Returns:
            np.array: Downsampled signal
        """
        if original_fs == target_fs:
            return signal
        
        # Calculate the number of samples in the downsampled signal
        num_samples = int(len(signal) * target_fs / original_fs)
        return resample(signal, num_samples)
    
    def generate_one_cycle_waveforms(self, dominant_freqs_info, fs, original_data_rms=None):
        """
        Generate one-cycle waveforms for each dominant frequency for LRA driving.
        
        This creates clean sine wave cycles that preserve the amplitude characteristics
        of the original vibration data for realistic LRA reproduction.
        
        Args:
            dominant_freqs_info (list): List of dominant frequency information
            fs (int): Target sampling frequency (8000 Hz)
            original_data_rms (float): RMS of original data for amplitude scaling
            
        Returns:
            list: List of one-cycle waveforms as int16 arrays
        """
        # Calculate total magnitude for proper amplitude scaling
        total_magnitude = sum(freq_info['amplitude'] for freq_info in dominant_freqs_info)
        
        one_cycle_waveforms = []
        
        for freq_info in dominant_freqs_info:
            freq = freq_info['frequency']
            phase = freq_info['phase']
            amplitude = freq_info['amplitude']
            
            # Calculate number of samples for one complete cycle
            cycle_length = int(fs / freq)
            
            # Generate time array for one cycle
            t = np.linspace(0, 1/freq, cycle_length, endpoint=False)
            
            # Use relative magnitude contribution to preserve original signal characteristics
            relative_contribution = amplitude / total_magnitude if total_magnitude > 0 else 1.0 / len(dominant_freqs_info)
            sine_wave = relative_contribution * np.sin(2 * np.pi * freq * t + phase)
            
            # If we have original RMS data, scale to match original amplitude range
            if original_data_rms is not None:
                # Scale the sine wave to have appropriate amplitude relative to original data
                current_rms = np.sqrt(np.mean(sine_wave**2))
                if current_rms > 0:
                    # Scale to preserve the relationship with original data
                    scale_factor = (original_data_rms / len(dominant_freqs_info)) / current_rms
                    sine_wave = sine_wave * scale_factor
            
            # Convert to int16 format, preserving the physical amplitude scale
            # Don't scale up - keep the values in the original physical range
            int16_wave = np.clip(sine_wave, -32767, 32767).astype(np.int16)
            
            one_cycle_waveforms.append(int16_wave)
        
        return one_cycle_waveforms
    
    def convert_to_int16_preserving_amplitude(self, signal):
        """
        Convert signal to 16-bit signed integer format while preserving the physical amplitude.
        
        This function directly converts the signal values to int16 without normalization,
        maintaining the original physical amplitude relationships.
        
        Args:
            signal (np.array): Input signal with physical amplitude values
            
        Returns:
            np.array: Signal as int16, preserving original amplitude scaling
        """
        # Clamp values to int16 range and convert
        clamped_signal = np.clip(signal, -32767, 32767)
        return clamped_signal.astype(np.int16)
    
    def normalize_to_int16(self, signal, target_max_amplitude=0.8):
        """
        Normalize signal and convert to 16-bit signed integer format.
        
        Args:
            signal (np.array): Input signal
            target_max_amplitude (float): Target maximum amplitude as fraction of full scale (0.8 = 80%)
            
        Returns:
            np.array: Normalized signal as int16
        """
        # Normalize to target amplitude range
        if np.max(np.abs(signal)) > 0:
            normalized = signal / np.max(np.abs(signal)) * target_max_amplitude
        else:
            normalized = signal
        
        # Convert to int16 range [-32767, 32767]
        return (normalized * 32767).astype(np.int16)
    
    def save_cpp_arrays(self, one_cycle_waveforms, reconstructed_downsampled, 
                       dominant_freqs_info, output_file):
        """
        Save the waveform arrays in C++ header format.
        Only saves one cycle for each frequency, matching the vibro_sample_lut.hpp format.
        
        Args:
            one_cycle_waveforms (list): One-cycle waveforms as int16 arrays
            reconstructed_downsampled (np.array): Downsampled reconstructed waveform
            dominant_freqs_info (list): Dominant frequency information
            output_file (Path): Output file path
        """
        with open(output_file, 'w') as f:
            f.write("namespace _vibro_lut_arrays\n{\n")
            
            # Write one-cycle arrays for each frequency
            for i, (one_cycle_wave, freq_info) in enumerate(zip(one_cycle_waveforms, dominant_freqs_info)):
                freq_hz = int(round(freq_info['frequency']))
                f.write(f"    inline constexpr pico::sample_type FREQ_{freq_hz}HZ_ARRAY[] = {{")
                f.write(", ".join(map(str, one_cycle_wave)))
                f.write("};\n")
            
            # Write reconstructed array (one cycle of reconstructed waveform)
            # Calculate the LCM of all frequencies to get one complete reconstructed cycle
            frequencies = [freq_info['frequency'] for freq_info in dominant_freqs_info]
            
            # For simplicity, use one cycle of the lowest frequency as reconstructed cycle length
            min_freq = min(frequencies)
            reconstructed_cycle_length = int(self.target_fs / min_freq)
            
            # Take one cycle of reconstructed waveform
            if len(reconstructed_downsampled) >= reconstructed_cycle_length:
                reconstructed_cycle = reconstructed_downsampled[:reconstructed_cycle_length]
            else:
                reconstructed_cycle = reconstructed_downsampled
            
            # Convert reconstructed waveform to int16 while preserving physical amplitude
            # Keep the values in the original physical range (same as displayed in plots)
            int16_reconstructed = np.clip(reconstructed_cycle, -32767, 32767).astype(np.int16)
                
            f.write(f"    inline constexpr pico::sample_type RECONSTRUCTED_ARRAY[] = {{")
            f.write(", ".join(map(str, int16_reconstructed)))
            f.write("};\n")
            
            f.write("}\n")
    
    def create_analysis_plot(self, original_data, original_fs, frequencies, amplitudes,
                           sine_waves, reconstructed, dominant_freqs_info, 
                           material_name, participant, output_file):
        """
        Create and save the analysis plot with multiple subplots.
        
        Args:
            original_data (np.array): Original waveform data
            original_fs (int): Original sampling frequency
            frequencies (np.array): FFT frequencies
            amplitudes (np.array): FFT amplitudes
            sine_waves (list): Generated sine waves
            reconstructed (np.array): Reconstructed waveform
            dominant_freqs_info (list): Dominant frequency information
            material_name (str): Material name
            participant (str): Participant identifier
            output_file (Path): Output file path for the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{material_name} - {participant} - Waveform Analysis', fontsize=16)
        
        # Time arrays
        t_original = np.linspace(0, len(original_data) / original_fs, len(original_data))
        t_sine = np.linspace(0, len(sine_waves[0]) / original_fs, len(sine_waves[0]))
        
        # Plot 1: Original waveform
        axes[0, 0].plot(t_original, original_data)
        axes[0, 0].set_title('Original Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Plot 2: FFT Spectrum
        axes[0, 1].semilogy(frequencies, amplitudes)
        axes[0, 1].set_title('FFT Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, 1000)  # Focus on lower frequencies
        
        # Mark dominant frequencies
        for freq_info in dominant_freqs_info:
            axes[0, 1].axvline(x=freq_info['frequency'], color='red', linestyle='--', alpha=0.7)
            axes[0, 1].text(freq_info['frequency'], freq_info['amplitude'], 
                          f'{freq_info["frequency"]:.1f}Hz', rotation=90)
        
        # Plot 3: First dominant frequency
        axes[1, 0].plot(t_sine, sine_waves[0])
        freq1 = dominant_freqs_info[0]['frequency']
        axes[1, 0].set_title(f'Dominant Frequency 1: {freq1:.1f} Hz')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        # Plot 4: Second dominant frequency
        axes[1, 1].plot(t_sine, sine_waves[1])
        freq2 = dominant_freqs_info[1]['frequency']
        axes[1, 1].set_title(f'Dominant Frequency 2: {freq2:.1f} Hz')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)
        
        # Plot 5: Third dominant frequency
        axes[2, 0].plot(t_sine, sine_waves[2])
        freq3 = dominant_freqs_info[2]['frequency']
        axes[2, 0].set_title(f'Dominant Frequency 3: {freq3:.1f} Hz')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].grid(True)
        
        # Plot 6: Reconstructed waveform
        axes[2, 1].plot(t_sine, reconstructed)
        axes[2, 1].set_title('Reconstructed Waveform')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Amplitude')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_participant(self, material_name, participant):
        """
        Analyze data for a specific material and participant.
        
        Args:
            material_name (str): Name of the material to analyze
            participant (str): Participant identifier (e.g., "Participant1")
        """
        print(f"Analyzing {material_name} - {participant}")
        
        # Construct file paths
        material_number = self.extract_material_number(material_name)
        wav_file = (self.data_dir / material_name / "Sliding_Friction-roughness" / 
                   participant / f"Material_{material_number}.wav")
        
        if not wav_file.exists():
            print(f"Warning: {wav_file} not found, skipping...")
            return
        
        # Create output directory for this material
        material_output_dir = self.output_dir / material_name
        material_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the wav file
        fs, data = self.load_wav_file(wav_file)
        if data is None:
            return
        
        print(f"  Loaded {wav_file.name}: {len(data)} samples at {fs} Hz")
        
        # Perform FFT analysis
        frequencies, amplitudes, phases, dominant_freqs_info = self.analyze_frequencies(data, fs)
        
        # Calculate original data RMS for amplitude preservation
        original_data_rms = np.sqrt(np.mean(data**2))
        print(f"  Original data RMS: {original_data_rms:.2f}")
        
        # Calculate energy weights for displaying meaningful amplitude values
        total_energy = sum(freq_info['amplitude']**2 for freq_info in dominant_freqs_info)
        
        print("  Dominant frequencies found:")
        for i, freq_info in enumerate(dominant_freqs_info):
            # Calculate the physical amplitude that this frequency contributes to the reconstructed signal
            energy_weight = (freq_info['amplitude']**2) / total_energy if total_energy > 0 else 1/len(dominant_freqs_info)
            scaled_amplitude = np.sqrt(energy_weight)
            physical_amplitude = scaled_amplitude * original_data_rms
            
            print(f"    {i+1}. {freq_info['frequency']:.2f} Hz "
                  f"(physical amplitude: {physical_amplitude:.2f})")
        
        # Generate sine waves for different purposes
        # Full duration for FFT analysis was already used
        
        # For plotting: generate reconstructed waveform (0.2s duration for visualization)
        plot_duration = 0.2
        t_plot, reconstructed_plot = self.reconstruct_waveform_from_frequencies(
            dominant_freqs_info, plot_duration, fs, preserve_original_amplitude=True, 
            original_data_rms=original_data_rms
        )
        
        # Generate individual sine waves for plotting comparison
        t_plot_sine, sine_waves_plot = self.generate_sine_waves(dominant_freqs_info, plot_duration, fs)
        
        # For saving to txt: generate one cycle for each frequency at target sampling rate
        one_cycle_waveforms = self.generate_one_cycle_waveforms(
            dominant_freqs_info, self.target_fs, original_data_rms=original_data_rms
        )
        
        # Also generate a longer reconstructed waveform for reference (and downsample it)
        save_duration = 0.5  # Shorter duration for reconstructed reference
        t_save, reconstructed_save = self.reconstruct_waveform_from_frequencies(
            dominant_freqs_info, save_duration, fs, preserve_original_amplitude=True,
            original_data_rms=original_data_rms
        )
        reconstructed_downsampled = self.downsample_signal(reconstructed_save, fs, self.target_fs)
        
        # Generate output file names
        plot_file = material_output_dir / f"{material_name}_{participant}_waveform_analysis.png"
        txt_file = material_output_dir / f"{material_name}_{participant}_waveforms.txt"
        
        # Create and save the analysis plot (using 0.2s data for plotting)
        # Get 0.2s of original data for plotting
        plot_samples = int(0.2 * fs)
        data_plot = data[:plot_samples] if len(data) > plot_samples else data
        
        self.create_analysis_plot(
            data_plot, fs, frequencies, amplitudes, sine_waves_plot, reconstructed_plot,
            dominant_freqs_info, material_name, participant, plot_file
        )
        
        # Save C++ arrays (one cycle per frequency)
        self.save_cpp_arrays(
            one_cycle_waveforms, reconstructed_downsampled,
            dominant_freqs_info, txt_file
        )
        
        print(f"  Results saved:")
        print(f"    Plot: {plot_file}")
        print(f"    Data: {txt_file}")
        print()
    
    def analyze_material(self, material_name):
        """
        Analyze all participants for a given material.
        
        Args:
            material_name (str): Name of the material to analyze
        """
        material_path = self.data_dir / material_name / "Sliding_Friction-roughness"
        
        if not material_path.exists():
            print(f"Error: {material_path} not found")
            return
        
        # Find all participant directories
        participants = []
        for item in material_path.iterdir():
            if item.is_dir() and item.name.startswith("Participant"):
                participants.append(item.name)
        
        participants.sort()
        
        if not participants:
            print(f"No participant directories found in {material_path}")
            return
        
        print(f"Found participants for {material_name}: {participants}")
        
        # Analyze each participant
        for participant in participants:
            self.analyze_participant(material_name, participant)
    
    def analyze_all_materials(self):
        """
        Analyze all materials found in the data directory.
        """
        materials = self.find_materials()
        
        if not materials:
            print("No materials found in the data directory")
            return
        
        print(f"Found materials: {materials}")
        print()
        
        for material in materials:
            self.analyze_material(material)


def main():
    """
    Main function to handle command line arguments and execute analysis.
    """
    parser = argparse.ArgumentParser(
        description='Analyze vibration waveform data from .wav files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python waveform_analyzer.py Material40    # Analyze only Material40
  python waveform_analyzer.py               # Analyze all materials
        """
    )
    
    parser.add_argument(
        'material', 
        nargs='?', 
        help='Material name to analyze (e.g., Material40). If not provided, all materials will be processed.'
    )
    
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory containing the waveform-analysis project (default: current directory)'
    )
    
    parser.add_argument(
        '--original-fs',
        type=int,
        default=44100,
        help='Original sampling frequency of the .wav files (default: 44100 Hz)'
    )
    
    parser.add_argument(
        '--target-fs',
        type=int,
        default=8000,
        help='Target sampling frequency for Pico processing (default: 8000 Hz)'
    )
    
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = WaveformAnalyzer(
        base_dir=args.base_dir,
        original_fs=args.original_fs,
        target_fs=args.target_fs
    )
    
    # Check if the data directory exists
    if not analyzer.data_dir.exists():
        print(f"Error: Data directory {analyzer.data_dir} not found")
        sys.exit(1)
    
    # Analyze the specified material or all materials
    if args.material:
        if args.material in analyzer.find_materials():
            analyzer.analyze_material(args.material)
        else:
            print(f"Error: Material '{args.material}' not found in {analyzer.data_dir}")
            available_materials = analyzer.find_materials()
            if available_materials:
                print(f"Available materials: {available_materials}")
            sys.exit(1)
    else:
        analyzer.analyze_all_materials()


if __name__ == "__main__":
    main()
