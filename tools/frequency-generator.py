import numpy as np
import sounddevice as sd
import threading

def generate_sinusoidal_signal(frequencies, duration, sampling_rate, binaural=False):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    
    if not binaural:
        # Original mono behavior
        signal = np.zeros_like(t)
        for frequency in frequencies:
            signal += np.sin(2 * np.pi * frequency * t)
        # Normalize
        signal = signal / len(frequencies)
        return signal
    else:
        # Enhanced binaural mode
        left_channel = np.zeros_like(t)
        right_channel = np.zeros_like(t)
        
        for frequency in frequencies:
            # Create binaural beats with a 4-10 Hz difference
            # The difference depends on the base frequency to maintain proportionality
            if frequency < 40:
                # For lower frequencies, use smaller differences
                diff = 4.0  # 4 Hz difference
            elif frequency < 100:
                diff = 6.0  # 6 Hz difference
            else:
                diff = 8.0  # 8 Hz difference
                
            # Left channel gets base frequency
            left_freq = frequency
            # Right channel gets base frequency plus difference
            right_freq = frequency + diff
            
            print(f"Generating binaural beat: Left={left_freq:.1f}Hz, Right={right_freq:.1f}Hz, "
                  f"Beat frequency={diff:.1f}Hz")
            
            # Generate pure sine waves for each channel
            left_channel += np.sin(2 * np.pi * left_freq * t)
            right_channel += np.sin(2 * np.pi * right_freq * t)
        
        # Normalize both channels
        left_channel = left_channel / len(frequencies)
        right_channel = right_channel / len(frequencies)
        
        # Apply slight panning to enhance stereo separation
        left_channel *= 1.0   # Full volume on left
        right_channel *= 1.0  # Full volume on right
        
        # Stack channels for stereo output
        stereo_signal = np.vstack((left_channel, right_channel)).T
        
        # Ensure the output is in float32 format for better audio quality
        return stereo_signal.astype(np.float32)

def play_sound(signal, sampling_rate):
    # Convert to float32 if not already
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    
    sd.play(signal, samplerate=sampling_rate)
    sd.wait()

def convert_to_seconds(hours, minutes):
    return hours * 3600 + minutes * 60

def parse_input(input_string):
    # Split input and check for binaural flag
    parts = input_string.strip().split()
    
    # Default to non-binaural if no 'b' flag
    binaural = False
    frequency_string = input_string
    
    # Check for binaural flag
    if parts[-1].lower() == 'b':
        binaural = True
        frequency_string = ' '.join(parts[:-1])  # Remove the 'b' flag for frequency parsing
    
    # Parse frequencies
    frequencies = [float(f.strip()) for f in frequency_string.split(',')]
    
    return frequencies, binaural

def main():
    try:
        frequency_input = input("Enter the frequencies you want to play (in Hz, separated by commas, add 'b' for binaural): ")
        frequencies, binaural = parse_input(frequency_input)
        
        # Add newline before mode announcement
        if binaural:
            print("\n\033[1m" + "Binaural Mode" + "\033[0m\n")
        else:
            print("\n\033[1m" + "Mono Mode" + "\033[0m\n")
        
        hours = int(input("Enter the duration (hours): "))
        minutes = int(input("Enter the duration (minutes): "))
        duration = convert_to_seconds(hours, minutes)
        sampling_rate = 44100  # Standard audio sampling rate

        signal = generate_sinusoidal_signal(frequencies, duration, sampling_rate, binaural)
        
        # Creating a thread to play the sound
        play_thread = threading.Thread(target=play_sound, args=(signal, sampling_rate))
        play_thread.start()
        
    except ValueError as e:
        print(f"Error: Invalid input - {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
