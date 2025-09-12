#!/usr/bin/env python3
"""
FFT Reconstruction Script for Vibration Data

This script analyzes .wav vibration data files from different materials,
performs segmented FFT analysis to preserve all frequency components in the 20-400Hz range,
and reconstructs waveforms with and without phase information for Pico microcontroller.

Usage:
    python fft_reconstructor.py [MaterialName]
    
    If MaterialName is not provided, the script will process all materials
    found in the data directory.

Example:
    python fft_reconstructor.py Material40
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample, welch
import re
from pathlib import Path


class WaveformReconstructor:
    """
    A class to reconstruct vibration waveform data from .wav files using FFT.
    """
    
    def __init__(self, base_dir=".", original_fs=44100, target_fs=8000, 
                 analysis_duration=60.0, segment_duration=0.2):
        """
        Initialize the WaveformReconstructor.
        
        Args:
            base_dir (str): Base directory containing the waveform-analysis folder
            original_fs (int): Original sampling frequency of the .wav files
            target_fs (int): Target sampling frequency for Pico processing
            analysis_duration (float): Duration of audio segment to analyze (seconds)
            segment_duration (float): Duration of each FFT segment (seconds) - for 5Hz resolution
        """
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output" / "waveform2"
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.analysis_duration = analysis_duration
        self.segment_duration = segment_duration
        
        # Frequency range for reconstruction
        self.freq_min = 20.0  # Hz
        self.freq_max = 400.0  # Hz
        
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
            
            print(f"    Loaded audio: {len(data)} samples ({len(data)/sample_rate:.2f} seconds)")
            
            return sample_rate, data.astype(np.float64)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_middle_segment(self, data, fs):
        """
        Extract the middle segment of data for analysis.
        
        Args:
            data (np.array): Full audio data
            fs (int): Sampling frequency
            
        Returns:
            tuple: (segment_data, start_time_offset)
        """
        total_duration = len(data) / fs
        
        # Calculate start time to center the analysis segment
        if total_duration <= self.analysis_duration:
            # Use all data if shorter than analysis duration
            return data, 0.0
        
        start_time = (total_duration - self.analysis_duration) / 2
        start_sample = int(start_time * fs)
        end_sample = start_sample + int(self.analysis_duration * fs)
        
        segment = data[start_sample:end_sample]
        return segment, start_time
    
    def perform_segmented_fft(self, data, fs):
        """
        Perform FFT analysis using hybrid approach: Welch for amplitudes, direct FFT for phases.
        Uses the first 1-second segment for both amplitude and phase to avoid interpolation errors.
        
        Args:
            data (np.array): Audio data segment for analysis (should be longer, e.g., 60s)
            fs (int): Sampling frequency
            
        Returns:
            tuple: (frequencies, amplitudes, phases)
        """
        # Use configured segment duration
        segment_samples = int(self.segment_duration * fs)
        
        print(f"    Using direct FFT from first {self.segment_duration}s segment for 5Hz resolution")
        
        # Use ONLY the first segment to avoid phase interpolation errors
        first_segment = data[:segment_samples]
        
        # Preprocessing (same as in Welch's method)
        first_segment_centered = first_segment - np.mean(first_segment)
        windowed_first_segment = first_segment_centered * np.hanning(len(first_segment_centered))
        
        # FFT of first segment for both amplitude and phase
        fft_result = np.fft.fft(windowed_first_segment)
        fft_frequencies = np.fft.fftfreq(len(windowed_first_segment), 1/fs)
        
        # Take only positive frequencies
        positive_mask = fft_frequencies > 0
        fft_frequencies_pos = fft_frequencies[positive_mask]
        fft_result_pos = fft_result[positive_mask]
        fft_amplitudes_pos = np.abs(fft_result_pos)
        fft_phases_pos = np.angle(fft_result_pos)
        
        print(f"    Direct FFT analysis: {len(fft_frequencies_pos)} frequency bins")
        print(f"    Frequency resolution: {fft_frequencies_pos[1] - fft_frequencies_pos[0]:.4f} Hz")
        print(f"    Using single segment: {len(first_segment)/fs:.1f} seconds")
        
        return fft_frequencies_pos, fft_amplitudes_pos, fft_phases_pos
    
    def filter_frequency_range(self, frequencies, amplitudes, phases):
        """
        Filter frequency components to the target range (20-400 Hz).
        
        Args:
            frequencies (np.array): Frequency array
            amplitudes (np.array): Amplitude array
            phases (np.array): Phase array
            
        Returns:
            tuple: (filtered_frequencies, filtered_amplitudes, filtered_phases)
        """
        freq_mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        
        filtered_frequencies = frequencies[freq_mask]
        filtered_amplitudes = amplitudes[freq_mask]
        filtered_phases = phases[freq_mask]
        
        # 调试信息
        print(f"    Filtered to {len(filtered_frequencies)} frequency components "
              f"in range {self.freq_min}-{self.freq_max} Hz")
        
        # 找出最强的几个频率分量
        if len(filtered_amplitudes) > 0:
            strongest_indices = np.argsort(filtered_amplitudes)[-5:][::-1]  # 前5个最强
            print(f"    Top 5 strongest frequencies:")
            for i, idx in enumerate(strongest_indices):
                if idx < len(filtered_frequencies):
                    freq = filtered_frequencies[idx]
                    amp = filtered_amplitudes[idx]
                    print(f"      {i+1}. {freq:.1f} Hz (amplitude: {amp:.2e})")
        
        return filtered_frequencies, filtered_amplitudes, filtered_phases
    
    def reconstruct_waveform(self, frequencies, amplitudes, phases, duration, fs, 
                           preserve_phase=True, original_data_rms=None):
        """
        Reconstruct waveform from frequency components with proper period alignment.
        
        Args:
            frequencies (np.array): Frequency components
            amplitudes (np.array): Amplitude components
            phases (np.array): Phase components
            duration (float): Desired duration of reconstructed signal
            fs (int): Sampling frequency
            preserve_phase (bool): Whether to preserve phase information
            original_data_rms (float): RMS of original data for amplitude scaling
            
        Returns:
            tuple: (time_array, reconstructed_waveform)
        """
        # 找出主导频率
        strongest_idx = np.argmax(amplitudes)
        dominant_freq = frequencies[strongest_idx]
        
        # 计算主导频率的周期
        period = 1.0 / dominant_freq
        
        # 调整重建时长为周期的整数倍，确保边界连续
        num_periods = int(duration / period)
        if num_periods == 0:
            num_periods = 1
        
        adjusted_duration = num_periods * period
        
        print(f"    Dominant frequency: {dominant_freq:.1f} Hz (period: {period:.3f}s)")
        print(f"    Adjusted duration: {adjusted_duration:.3f}s ({num_periods} periods)")
        
        t = np.linspace(0, adjusted_duration, int(adjusted_duration * fs), endpoint=False)
        reconstructed = np.zeros(len(t))
        
        # 使用20-400Hz范围内的所有频率分量
        print(f"    Reconstructing with {len(frequencies)} frequency components")
        print(f"    Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
        
        # 使用相对幅值重建，5Hz分辨率应该能减少拍频
        max_amplitude = np.max(amplitudes)
        
        # 使用所有频率分量，5Hz分辨率应该避免过多相近频率
        print(f"    Using all {len(frequencies)} frequency components with 5Hz resolution")
        
        # 确保重建的波形在端点连续（消除边界效应）
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            # 保持相对幅值关系
            relative_amp = amp / max_amplitude
            
            # Use original phase or set to zero
            actual_phase = phase if preserve_phase else 0.0
            
            # Add frequency component to reconstruction
            # 使用余弦而不是正弦，通常相位更稳定
            component = relative_amp * np.cos(2 * np.pi * freq * t + actual_phase)
            reconstructed += component
        
        # 调试：检查重建前的DC分量
        dc_before = np.mean(reconstructed)
        print(f"    DC component before scaling: {dc_before:.6f}")
        
        # 移除可能的DC偏移
        reconstructed = reconstructed - np.mean(reconstructed)
        
        # Scale to int16 full range [-32767, 32767]
        max_val = np.max(np.abs(reconstructed))
        if max_val > 0:
            # Scale to use full int16 range
            reconstructed = reconstructed / max_val * 32767.0
        
        return t, reconstructed
    
    def find_fundamental_frequency(self, frequencies, amplitudes):
        """
        Find the fundamental frequency from the frequency spectrum.
        
        Args:
            frequencies (np.array): Frequency components
            amplitudes (np.array): Amplitude components
            
        Returns:
            float: Fundamental frequency in Hz
        """
        # Find frequency with maximum amplitude
        max_idx = np.argmax(amplitudes)
        fundamental_freq = frequencies[max_idx]
        
        return fundamental_freq
    
    def extract_one_cycle(self, signal, fundamental_freq, fs):
        """
        Extract one cycle of the signal starting from near-zero crossing.
        
        Args:
            signal (np.array): Input signal
            fundamental_freq (float): Fundamental frequency
            fs (int): Sampling frequency
            
        Returns:
            np.array: One cycle of the signal
        """
        cycle_length = int(fs / fundamental_freq)
        
        if len(signal) < cycle_length:
            return signal
        
        # Find zero crossings (or near-zero values)
        # Look for points where signal crosses zero with positive slope
        zero_crossings = []
        threshold = 0.1 * np.std(signal)  # Small threshold for near-zero detection
        
        for i in range(1, min(len(signal), cycle_length * 2)):
            if (abs(signal[i]) < threshold and 
                signal[i] >= signal[i-1] and  # Positive slope
                i > cycle_length // 4):  # Skip initial samples
                zero_crossings.append(i)
        
        # Start from first suitable zero crossing
        if zero_crossings:
            start_idx = zero_crossings[0]
        else:
            # Fallback: start from beginning
            start_idx = 0
        
        end_idx = start_idx + cycle_length
        if end_idx > len(signal):
            end_idx = len(signal)
            start_idx = max(0, end_idx - cycle_length)
        
        return signal[start_idx:end_idx]
    
    def save_cpp_arrays(self, waveform_with_phase, waveform_without_phase, 
                       material_name, participant, output_file):
        """
        Save the reconstructed waveforms in C++ header format.
        
        Args:
            waveform_with_phase (np.array): Reconstructed waveform with phase
            waveform_without_phase (np.array): Reconstructed waveform without phase
            material_name (str): Material name
            participant (str): Participant identifier
            output_file (Path): Output file path
        """
        # Convert to int16
        def to_int16(signal):
            return np.clip(signal, -32767, 32767).astype(np.int16)
        
        int16_with_phase = to_int16(waveform_with_phase)
        int16_without_phase = to_int16(waveform_without_phase)
        
        with open(output_file, 'w') as f:
            f.write(f"// Reconstructed waveforms for {material_name} - {participant}\n")
            f.write("namespace _vibro_lut_arrays\n{\n")
            
            # Array with phase preservation
            f.write(f"    inline constexpr pico::sample_type {material_name.upper()}_{participant.upper()}_WITH_PHASE[] = {{")
            f.write(", ".join(map(str, int16_with_phase)))
            f.write("};\n\n")
            
            # Array without phase (magnitude only)
            f.write(f"    inline constexpr pico::sample_type {material_name.upper()}_{participant.upper()}_WITHOUT_PHASE[] = {{")
            f.write(", ".join(map(str, int16_without_phase)))
            f.write("};\n")
            
            f.write("}\n")
    
    def create_reconstruction_plot(self, original_segment, original_fs, segment_start_time,
                                 frequencies, amplitudes, waveform_with_phase, 
                                 waveform_without_phase, material_name, participant, 
                                 output_file):
        """
        Create and save the reconstruction plot.
        
        Args:
            original_segment (np.array): Original audio segment used for FFT
            original_fs (int): Original sampling frequency
            segment_start_time (float): Start time of the segment in original file
            frequencies (np.array): Frequency components
            amplitudes (np.array): Amplitude components
            waveform_with_phase (np.array): Reconstructed waveform with phase
            waveform_without_phase (np.array): Reconstructed waveform without phase
            material_name (str): Material name
            participant (str): Participant identifier
            output_file (Path): Output file path for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{material_name} - {participant} - FFT Reconstruction', fontsize=16)
        
        # Plot 1: Original waveform segment (first 2s)
        plot_duration = 2.0  # 改为2秒
        plot_samples = int(plot_duration * original_fs)
        plot_samples = min(plot_samples, len(original_segment))
        
        original_plot = original_segment[:plot_samples]
        t_original = np.linspace(segment_start_time, segment_start_time + len(original_plot)/original_fs, len(original_plot))
        
        axes[0, 0].plot(t_original, original_plot)
        axes[0, 0].set_title(f'Original Waveform (first {plot_duration}s of analysis segment)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Plot 2: Average FFT spectrum
        axes[0, 1].semilogy(frequencies, amplitudes)
        axes[0, 1].set_title(f'Average FFT Spectrum ({self.freq_min}-{self.freq_max} Hz)')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(self.freq_min, self.freq_max)
        
        # Plot 3: Reconstructed waveform with phase (first 2s)
        plot_samples_recon = min(int(plot_duration * self.target_fs), len(waveform_with_phase))
        waveform_with_phase_plot = waveform_with_phase[:plot_samples_recon]
        t_recon = np.linspace(0, len(waveform_with_phase_plot)/self.target_fs, len(waveform_with_phase_plot))
        
        axes[1, 0].plot(t_recon, waveform_with_phase_plot)
        axes[1, 0].set_title(f'Reconstructed with Phase (first {plot_duration}s)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        # Plot 4: Reconstructed waveform without phase (first 2s)
        waveform_without_phase_plot = waveform_without_phase[:plot_samples_recon]
        
        axes[1, 1].plot(t_recon, waveform_without_phase_plot)
        axes[1, 1].set_title(f'Reconstructed without Phase (first {plot_duration}s)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)
        
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
        
        # Extract middle segment for analysis
        analysis_segment, segment_start_time = self.extract_middle_segment(data, fs)
        print(f"  Using segment from {segment_start_time:.2f}s to {segment_start_time + len(analysis_segment)/fs:.2f}s")
        
        # Perform segmented FFT analysis
        frequencies, amplitudes, phases = self.perform_segmented_fft(analysis_segment, fs)
        
        # Filter to target frequency range
        filtered_frequencies, filtered_amplitudes, filtered_phases = self.filter_frequency_range(
            frequencies, amplitudes, phases
        )
        
        # Calculate original data RMS for amplitude preservation
        original_data_rms = np.sqrt(np.mean(analysis_segment**2))
        print(f"  Original segment RMS: {original_data_rms:.2f}")
        
        # Reconstruct waveforms (with and without phase)
        reconstruction_duration = 2.0  # Reconstruct 2 seconds for visualization
        
        # With phase preservation
        t_recon, waveform_with_phase = self.reconstruct_waveform(
            filtered_frequencies, filtered_amplitudes, filtered_phases,
            reconstruction_duration, self.target_fs, preserve_phase=True,
            original_data_rms=original_data_rms
        )
        
        # Without phase (magnitude only)
        t_recon, waveform_without_phase = self.reconstruct_waveform(
            filtered_frequencies, filtered_amplitudes, filtered_phases,
            reconstruction_duration, self.target_fs, preserve_phase=False,
            original_data_rms=original_data_rms
        )
        
        # Find fundamental frequency and extract one cycle for each reconstruction
        fundamental_freq = self.find_fundamental_frequency(filtered_frequencies, filtered_amplitudes)
        print(f"  Fundamental frequency: {fundamental_freq:.2f} Hz")
        
        # Extract one cycle from each reconstruction
        one_cycle_with_phase = self.extract_one_cycle(waveform_with_phase, fundamental_freq, self.target_fs)
        one_cycle_without_phase = self.extract_one_cycle(waveform_without_phase, fundamental_freq, self.target_fs)
        
        print(f"  One cycle length: {len(one_cycle_with_phase)} samples "
              f"({len(one_cycle_with_phase)/self.target_fs*1000:.1f} ms)")
        
        # Generate output file names
        plot_file = material_output_dir / f"{material_name}_{participant}_reconstruction.png"
        txt_file = material_output_dir / f"{material_name}_{participant}_waveforms.txt"
        
        # Create and save the reconstruction plot
        self.create_reconstruction_plot(
            analysis_segment, fs, segment_start_time, filtered_frequencies, filtered_amplitudes,
            waveform_with_phase, waveform_without_phase, material_name, participant, plot_file
        )
        
        # Save C++ arrays (one cycle for each reconstruction mode)
        self.save_cpp_arrays(
            one_cycle_with_phase, one_cycle_without_phase,
            material_name, participant, txt_file
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
        description='Reconstruct vibration waveform data using FFT analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fft_reconstructor.py Material40    # Analyze only Material40
  python fft_reconstructor.py               # Analyze all materials
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
    
    parser.add_argument(
        '--analysis-duration',
        type=float,
        default=60.0,
        help='Duration of audio segment to analyze in seconds (default: 60.0)'
    )
    
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=0.2,
        help='Duration of each FFT segment in seconds for 5Hz resolution (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Initialize the reconstructor
    reconstructor = WaveformReconstructor(
        base_dir=args.base_dir,
        original_fs=args.original_fs,
        target_fs=args.target_fs,
        analysis_duration=args.analysis_duration,
        segment_duration=args.segment_duration
    )
    
    # Check if the data directory exists
    if not reconstructor.data_dir.exists():
        print(f"Error: Data directory {reconstructor.data_dir} not found")
        sys.exit(1)
    
    # Analyze the specified material or all materials
    if args.material:
        if args.material in reconstructor.find_materials():
            reconstructor.analyze_material(args.material)
        else:
            print(f"Error: Material '{args.material}' not found in {reconstructor.data_dir}")
            available_materials = reconstructor.find_materials()
            if available_materials:
                print(f"Available materials: {available_materials}")
            sys.exit(1)
    else:
        reconstructor.analyze_all_materials()


if __name__ == "__main__":
    main()
