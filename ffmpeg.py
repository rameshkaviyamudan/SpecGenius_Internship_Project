import subprocess
import os

# Ensure ffmpeg is installed
def check_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("ffmpeg is not installed. Please install ffmpeg and try again.")
        raise
    except FileNotFoundError:
        print("ffmpeg is not installed or not found in PATH. Please install ffmpeg and ensure it is in your PATH.")
        raise

# Create a list of video files to merge
video_files = ["output/specifications/2024-06-14 16-55-23.mkv", "output/specifications/2024-06-14 17-04-06.mkv"]

# Verify all video files exist
for file in video_files:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"The file {file} does not exist. Please check the file path.")

# Create the file list for ffmpeg
file_list = "file_list.txt"

with open(file_list, 'w') as f:
    for file in video_files:
        f.write(f"file '{file}'\n")

# Merge the videos using ffmpeg
output_file = r'output.mp4'

def merge_videos(file_list, output_file):
    try:
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', file_list, '-c', 'copy', output_file], check=True)
        print(f"Merging complete. Output file: {output_file}")
    except subprocess.CalledProcessError:
        print("An error occurred while merging the videos.")
        raise

# Check if ffmpeg is installed
check_ffmpeg_installed()

# Merge videos
merge_videos(file_list, output_file)

# Clean up
os.remove(file_list)
