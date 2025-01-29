from pathlib import Path
from pydub import AudioSegment
import os


def get_wav_duration(file_path):
    """Get the duration of a WAV file in seconds."""
    audio = AudioSegment.from_wav(file_path)
    return len(audio) / 1000.0  # Duration in seconds

def calculate_average_duration(directory):
    """Calculate the average duration of all WAV files in the given directory."""
    wav_files = list(Path(directory).glob('*.wav'))
    
    if not wav_files:
        print("No WAV files found.")
        return None
    
    total_duration = 0.0
    max_duration = 0.0
    min_duration = 100.0
    for wav_file in wav_files:
        try:
            duration = get_wav_duration(wav_file)
            total_duration += duration
            max_duration = max(max_duration, duration)
            min_duration = min(min_duration, duration)
            print(f"Processed {wav_file.name}: {duration:.2f} seconds")
        except Exception as e:
            print(f"Failed to process {wav_file.name}: {e}")
    
    average_duration = total_duration / len(wav_files)
    print(f"\nAverage duration: {average_duration:.2f} seconds")
    print(f"Max duration: {max_duration:.2f} seconds")
    print(f"Min duration: {min_duration:.2f} seconds")
    return average_duration

if __name__ == "__main__":
    audio_directory = 'audio_2'  # 修改为你的音频文件夹路径
    if not os.path.exists(audio_directory):
        print(f"The directory '{audio_directory}' does not exist.")
    else:
        calculate_average_duration(audio_directory)