import streamlit as st
import boto3
import os
import json
import time
import io
import requests
import ffmpeg
import base64
import pandas as pd

# AWS Configurations
AWS_REGION = "us-east-1"
S3_BUCKET = "medicaltranscriptionbacke-medicaltranscriptionback-dpapkt0xat7l"
DYNAMODB_TABLE = "SentimentAnalysisResults"

# AWS Clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
comprehend_client = boto3.client("comprehendmedical", region_name=AWS_REGION)
dynamodb_client = boto3.resource("dynamodb", region_name=AWS_REGION)

# Streamlit UI
st.title("🎥 Video Sentiment Analysis (AWS-Powered)")

# 📂 File Upload
uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    video_name = uploaded_file.name
    s3_video_uri = f"s3://{S3_BUCKET}/{video_name}"

    # Upload video to S3
    s3_client.upload_fileobj(uploaded_file, S3_BUCKET, video_name)
    st.success(f"✅ Video uploaded to S3: {s3_video_uri}")

    # Extract Audio from Video
    audio_path = f"temp_audio_{video_name}.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())  # Save uploaded file as a temp audio file

    s3_audio_uri = f"s3://{S3_BUCKET}/{audio_path}"
    s3_client.upload_file(audio_path, S3_BUCKET, audio_path)
    st.success(f"✅ Audio extracted and uploaded to S3: {s3_audio_uri}")

    # Start Medical Transcription
    transcribe_job_name = f"medical_{video_name.replace('.', '_')}"
    transcribe_client.start_medical_transcription_job(
        MedicalTranscriptionJobName=transcribe_job_name,
        LanguageCode="en-US",
        Media={"MediaFileUri": s3_audio_uri},
        MediaFormat="mp3",
        Specialty="PRIMARYCARE",
        Type="DICTATION",
    )

    st.info("📝 Transcribing audio using Amazon Transcribe Medical...")

    # Wait for Transcription to complete
    while True:
        status = transcribe_client.get_medical_transcription_job(MedicalTranscriptionJobName=transcribe_job_name)
        if status["MedicalTranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)

    if status["MedicalTranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
        transcript_uri = status["MedicalTranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        response = requests.get(transcript_uri)
        transcript_json = response.json()
        transcript_text = transcript_json["results"]["transcripts"][0]["transcript"]
        st.text_area("📝 Transcription:", transcript_text)

        # Medical Sentiment Analysis
        sentiment_response = comprehend_client.detect_entities(Text=transcript_text)
        entities = sentiment_response["Entities"]
        st.write("🔍 **Detected Medical Entities:**", entities)

        # Store Results in DynamoDB
        table = dynamodb_client.Table(DYNAMODB_TABLE)
        table.put_item(
            Item={
                "VideoName": video_name,
                "Transcript": transcript_text,
                "Analysis": json.dumps(entities)
            }
        )
        st.success("✅ Results stored in DynamoDB!")

# 📊 View Past Analyses
if st.button("📂 View Past Analyses"):
    table = dynamodb_client.Table(DYNAMODB_TABLE)
    response = table.scan()
    for item in response["Items"]:
        st.write(f"📂 **Video:** {item['VideoName']}")
        st.write(f"📝 **Transcript:** {item['Transcript']}")
        st.write(f"🔍 **Analysis:** {item['Analysis']}")

