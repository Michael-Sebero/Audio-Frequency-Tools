import numpy as np
import librosa
import soundfile as sf

def detect_frequency(file_path, duration=60):
    """
    Detect the dominant frequency in an audio file.

    Parameters:
        file_path (str): Path to the audio file.
        duration (int, optional): Duration in seconds to analyze. Defaults to 60.

    Returns:
        float: The dominant frequency in Hz.
    """
    try:
        # Read audio file with soundfile (supports many formats)
        audio_data, sample_rate = sf.read(file_path, always_2d=True)
        
        # Extract the first channel if the audio is stereo
        audio_data = audio_data[:, 0]

        # Truncate or pad the audio to the desired duration
        max_samples = int(sample_rate * duration)
        audio_data = audio_data[:max_samples] if len(audio_data) > max_samples else np.pad(
            audio_data, (0, max_samples - len(audio_data)), 'constant'
        )

        if len(audio_data) == 0:
            raise ValueError("Audio file contains no data or is too short for analysis.")

        # Perform FFT on the audio data
        fft_result = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1.0 / sample_rate)

        # Find the dominant frequency
        dominant_freq = freqs[np.argmax(np.abs(fft_result))]
        print(f"The dominant frequency is {dominant_freq:.2f} Hz.")
        return dominant_freq

    except FileNotFoundError:
        print("Error: The specified audio file was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ").strip()
    detect_frequency(file_path)
