import os
import whisper
import subprocess

def check_audio_stream(video_file_path):
    """Check if the video file has an audio stream using ffmpeg."""
    try:
        # Run ffmpeg to check if there is an audio stream
        cmd = [
            "ffmpeg", "-i", video_file_path, "-map", "0:a", "-c", "copy", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # If 'Stream #0' contains an audio codec, we have an audio stream
        if "Stream #0" in result.stderr and "Audio" in result.stderr:
            return True
        return False
    except Exception as e:
        print(f"Error checking audio stream: {e}")
        return False

def transcribe_video(video_file_path):
    """Transcribe the audio from the video file using Whisper."""
    # Load the Whisper model
    model = whisper.load_model("base")

    # Check if the video contains an audio stream
    if not check_audio_stream(video_file_path):
        print(f"No audio stream found in {video_file_path}. Please provide a video with an audio track.")
        return ""

    # Transcribe the video
    try:
        result = model.transcribe(video_file_path)
        transcribed_text = result['text']
        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

# Example usage
video_path = "/Users/sunho/Desktop/agent_workflow/venv/convovid.mp4"

if os.path.exists(video_path):
    transcribed_text = transcribe_video(video_path)
    if transcribed_text:
        print(f"Transcribed Video Text:\n{transcribed_text}")
else:
    print(f"File not found: {video_path}")
