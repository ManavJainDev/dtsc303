from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# --- CONFIGURATION ---
S3_BUCKET_NAME = "my-flame-emr-bucket"
VIDEO_PREFIX = "tf-idf-manav-heetej/"
OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-results/"

def get_video_keys():
    """Fetches all video file paths from your S3 bucket."""
    import boto3
    s3_client = boto3.client('s3')
    keys = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=VIDEO_PREFIX)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].lower().endswith('.mp4'):
                    keys.append(obj['Key'])
    return keys

def transcribe_video(s3_key):
    """Self-heals the environment, then transcribes the video."""
    import os
    import sys
    import tempfile
    import subprocess
    
    BASE_DIR = "/tmp/whisper_env_v2"
    
    # ==========================================
    # 1. SELF-HEALING BOOTSTRAPPER
    # ==========================================
    # If this specific node doesn't have the tools, it installs them itself right now.
    if not os.path.exists(f"{BASE_DIR}/bin/ffmpeg") or not os.path.exists(f"{BASE_DIR}/whisper"):
        os.makedirs(f"{BASE_DIR}/bin", exist_ok=True)
        
        # 1a. Install Whisper locally to /tmp
        subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper", "boto3", 
                        "--target", BASE_DIR, "--no-cache-dir", "--ignore-installed"], check=False)
        
        # 1b. Detect the CPU architecture to fix the 'Exec format error'
        arch_check = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        arch = arch_check.stdout.strip()
        
        if arch == "aarch64":
            # Graviton / ARM64 instances
            ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
        else:
            # Intel / AMD instances
            ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            
        # Download and extract the correct FFmpeg
        os.system(f"wget -q {ffmpeg_url} -O /tmp/ffmpeg.tar.xz")
        os.system(f"cd /tmp && tar -xf ffmpeg.tar.xz && cp ffmpeg-*-static/ffmpeg {BASE_DIR}/bin/ && chmod 755 {BASE_DIR}/bin/ffmpeg")

    # ==========================================
    # 2. PATH INJECTION
    # ==========================================
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    if f"{BASE_DIR}/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + f"{BASE_DIR}/bin"
        
    # ==========================================
    # 3. TRANSCRIBE
    # ==========================================
    try:
        import boto3
        import whisper
        
        s3_client = boto3.client('s3')
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
            local_video_path = tmp_file.name
            s3_client.download_file(S3_BUCKET_NAME, s3_key, local_video_path)
            
            # Load Whisper model
            model = whisper.load_model("base", download_root="/tmp/")
            
            # Transcribe
            result = model.transcribe(local_video_path)
            return (s3_key, result["text"].strip())
            
    except Exception as e:
        return (s3_key, f"ERROR: {str(e)}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Whisper_Transcription").getOrCreate()
    
    print(f"Scanning S3 for videos in s3://{S3_BUCKET_NAME}/{VIDEO_PREFIX}...")
    video_keys = get_video_keys()
    
    if len(video_keys) > 0:
        rdd = spark.sparkContext.parallelize(video_keys, numSlices=len(video_keys))
        transcribed_rdd = rdd.map(transcribe_video)
        
        schema = StructType([
            StructField("file_key", StringType(), True),
            StructField("transcription", StringType(), True)
        ])
        df = spark.createDataFrame(transcribed_rdd, schema)
        
        print(f"Saving transcriptions to {OUTPUT_PATH}...")
        df.write.mode("overwrite").parquet(OUTPUT_PATH)
        
        print("SUCCESS! Transcription phase complete.")
        df.show(truncate=80)
    else:
        print("No videos found. Please check your S3 bucket and prefix path.")
        
    spark.stop()
