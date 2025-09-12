#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# 生成25Hz正弦波，1秒时长，8000Hz采样率
duration = 1.0
fs = 8000
freq = 25.0

t = np.linspace(0, duration, int(duration * fs), endpoint=False)
signal = np.sin(2 * np.pi * freq * t)

print(f"Time array length: {len(t)}")
print(f"Time range: {t[0]:.3f} to {t[-1]:.3f} seconds")
print(f"Signal range: {np.min(signal):.3f} to {np.max(signal):.3f}")
print(f"Expected cycles: {duration * freq}")

# 绘制前0.2秒的信号
plt.figure(figsize=(12, 6))
samples_to_plot = int(0.2 * fs)  # 0.2秒的样本
plt.plot(t[:samples_to_plot], signal[:samples_to_plot])
plt.title(f'25Hz Sine Wave - First 0.2 seconds (should show {0.2 * freq} cycles)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('/home/lyra/SenseGlove/waveform-analysis/test_25hz_sine.png', dpi=150)
print("Test plot saved to test_25hz_sine.png")

# 计算应该看到的周期数
cycles_in_02s = 0.2 * freq
print(f"In 0.2 seconds, we should see {cycles_in_02s} complete cycles")
