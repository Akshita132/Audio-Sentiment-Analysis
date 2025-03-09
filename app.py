

import streamlit as st
import os
import subprocess
import textwrap
import time
import glob
from transformers import pipeline



# Title
st.title("🎬 Video Sentiment Analysis")
st.write("Upload a video, extract audio, transcribe speech, and analyze sentiment!")

# File Upload
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save uploaded file
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # Display video
    st.video(video_path)

    # Extract audio
    audio_path = "audio.wav"

    # Delete old files to prevent conflicts
    for file in glob.glob("*.txt") + [audio_path]:
        if os.path.exists(file):
            os.remove(file)

    st.subheader("🔊 Extracting Audio...")
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        st.error("❌ Audio extraction failed! Please check FFmpeg installation.")
        st.text(result.stderr)
    elif not os.path.exists(audio_path):
        st.error("❌ Audio file was not created. Something went wrong with FFmpeg.")
    else:
        st.success("✅ Audio extraction successful!")
        st.audio(audio_path)  # Play extracted audio

        # Transcribe using Whisper
        st.subheader("📝 Transcribing Speech...")
        whisper_command = ["whisper", audio_path, "--model", "small", "--output_dir", ".", "--output_format", "txt"]

        whisper_result = subprocess.run(whisper_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Debug logs
        #st.text_area("Whisper Debug Logs", whisper_result.stderr, height=100)

        # Wait for Whisper to complete
        #time.sleep(2)

        # Find the latest transcript file
        transcript_files = glob.glob("*.txt")  # Find all .txt files
        transcript_path = None

        for file in transcript_files:
            if file != "cookies.txt":  # Ignore unwanted files
                transcript_path = file
                break

        if not transcript_path:
            st.error("❌ Transcript file not found. Whisper may have failed.")
        else:
            # Read the latest transcript
            with open(transcript_path, "r", encoding="utf-8") as file:
                transcript = file.read().strip()
                st.text_area("Transcription:", transcript, height=200)

            # Perform Sentiment Analysis only if transcript is valid
            if transcript:
                st.subheader("📊 Sentiment Analysis:")
                sentiment_model = pipeline("sentiment-analysis")

                # Token limit per DistilBERT model
                MAX_TOKENS = 512
                transcript_chunks = textwrap.wrap(transcript, width=MAX_TOKENS)

                sentiment_results = []
                for i, chunk in enumerate(transcript_chunks):
                    sentiment_result = sentiment_model(chunk)
                    sentiment_results.append(sentiment_result[0])

                # Display sentence-wise sentiment
                for i, result in enumerate(sentiment_results):
                    st.write(f"**Part {i+1}:** {result['label']} (Score: {result['score']:.2f})")

    st.success("🚀 App is running successfully!") """)

