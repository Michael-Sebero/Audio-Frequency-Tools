import numpy as np
import sounddevice as sd
import threading
import signal
import sys
import queue

stop_event = threading.Event()

def generate_sinusoidal_block(frequencies, sampling_rate, block_size, t0, binaural=False):
    """Generate a block of audio samples starting from time t0."""
    t = (np.arange(block_size) + t0) / sampling_rate
    
    if not binaural:
        # Vectorized summation
        waves = np.sin(2 * np.pi * np.outer(t, frequencies))
        signal = waves.sum(axis=1) / len(frequencies)
        return signal.astype(np.float32)
    else:
        left_channel = np.zeros(block_size)
        right_channel = np.zeros(block_size)
        
        for frequency in frequencies:
            if frequency < 40:
                diff = 4.0
            elif frequency < 100:
                diff = 6.0
            else:
                diff = 8.0
            
            left_channel += np.sin(2 * np.pi * frequency * t)
            right_channel += np.sin(2 * np.pi * (frequency + diff) * t)
        
        left_channel /= len(frequencies)
        right_channel /= len(frequencies)
        
        stereo_signal = np.vstack((left_channel, right_channel)).T
        return stereo_signal.astype(np.float32)


def play_continuous(frequencies, sampling_rate, binaural):
    """Stream audio continuously until stopped."""
    block_size = 1024
    t0 = 0
    
    def callback(outdata, frames, time, status):
        nonlocal t0
        if stop_event.is_set():
            raise sd.CallbackStop()
        block = generate_sinusoidal_block(frequencies, sampling_rate, frames, t0, binaural)
        outdata[:] = block.reshape(outdata.shape)
        t0 += frames
    
    channels = 2 if binaural else 1
    with sd.OutputStream(samplerate=sampling_rate, channels=channels,
                         blocksize=block_size, dtype='float32', callback=callback):
        print("\nPlaying continuously... Press Ctrl+C to stop.\n")
        stop_event.wait()  # Wait until stop_event is set


def play_finite(frequencies, duration, sampling_rate, binaural):
    """Play a finite duration sound (blocking)."""
    total_samples = int(duration * sampling_rate)
    block_size = 1024
    t0 = 0
    
    def callback(outdata, frames, time, status):
        nonlocal t0
        remaining = total_samples - t0
        if remaining <= 0:
            raise sd.CallbackStop()
        frames = min(frames, remaining)
        block = generate_sinusoidal_block(frequencies, sampling_rate, frames, t0, binaural)
        outdata[:frames] = block.reshape(outdata[:frames].shape)
        if frames < len(outdata):
            outdata[frames:] = 0
        t0 += frames
    
    channels = 2 if binaural else 1
    with sd.OutputStream(samplerate=sampling_rate, channels=channels,
                         blocksize=block_size, dtype='float32', callback=callback):
        print("\nPlaying for {} seconds...\n".format(duration))
        sd.sleep(int(duration * 1000))


def convert_to_seconds(hours, minutes):
    return hours * 3600 + minutes * 60


def parse_input(input_string):
    parts = input_string.strip().split()
    binaural = False
    continuous = False
    frequency_string = input_string
    
    # Handle flags
    if parts[-1].lower() == 'b':
        binaural = True
        frequency_string = ' '.join(parts[:-1])
    elif parts[-1].lower() == 'c':
        continuous = True
        frequency_string = ' '.join(parts[:-1])
    elif parts[-1].lower() in ('b', 'c'):  # both flags
        binaural = 'b' in parts
        continuous = 'c' in parts
        frequency_string = ' '.join(p for p in parts if p.lower() not in ('b','c'))
    
    frequencies = [float(f.strip()) for f in frequency_string.split(',')]
    return frequencies, binaural, continuous


def signal_handler(sig, frame):
    print("\nStopping playback...")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    try:
        frequency_input = input("Enter frequencies (Hz, comma-separated). Add 'b' for binaural, 'c' for continuous: ")
        frequencies, binaural, continuous = parse_input(frequency_input)
        
        if binaural:
            print("\n\033[1mBinaural Mode\033[0m")
        else:
            print("\n\033[1mMono Mode\033[0m")
        
        sampling_rate = 44100
        
        if continuous:
            play_continuous(frequencies, sampling_rate, binaural)
        else:
            hours = int(input("Enter duration (hours): "))
            minutes = int(input("Enter duration (minutes): "))
            duration = convert_to_seconds(hours, minutes)
            play_finite(frequencies, duration, sampling_rate, binaural)
            
    except ValueError as e:
        print(f"Error: Invalid input - {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
