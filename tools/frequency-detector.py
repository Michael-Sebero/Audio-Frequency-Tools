import numpy as np
import librosa
import pyaudio

def detect_frequency(file_path):
    # Load the first minute of the audio file with librosa
    audio_data, sample_rate = librosa.load(file_path, sr=None, duration=60)
    
    # Assuming the audio is mono, we take the first channel
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    
    # Perform FFT and find the dominant frequency
    fft_result = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), 1.0 / sample_rate)
    dominant_freq = freqs[np.argmax(np.abs(fft_result))]
    
    print(f"The dominant frequency is {dominant_freq} Hz.")

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ")
    detect_frequency(file_path)
