# Audio
sample_rate = 22050  # Hz
frame_length_ms = 50  # ms
frame_shift_ms = 12.5  # ms
num_mels = 80  # filters
min_freq = 0  # Hz
max_freq = 8000  # Hz
ref_level_db = 20
power = 1.5
fft_size = 1024
hop_size = 256
floor_freq = 0.01  # clip value, prior to log compression
max_abs_value = 4
preemphasis = 0.97
min_level_db = -100

# Eval:
griffin_lim_iters = 60
