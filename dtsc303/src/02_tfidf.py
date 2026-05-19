import os
import sys
import subprocess
from pyspark.sql import SparkSession

# ==========================================
# 1. THE BULLETPROOF ENVIRONMENT FIX
# ==========================================
# Add the custom Whisper path locally on the driver
sys.path.insert(0, "/tmp/whisper_env")

# Force install on the driver using the EXACT Python binary PySpark is using
subprocess.run(["sudo", sys.executable, "-m", "pip", "install", "numpy"], capture_output=True)

# Inject the environment variables directly into Spark configuration.
# This completely bypasses YARN terminal stripping!
spark = SparkSession.builder \
    .appName("TFIDF_Top15_Extraction") \
    .config("spark.executorEnv.PYTHONPATH", "/tmp/whisper_env") \
    .config("spark.yarn.appMasterEnv.PYTHONPATH", "/tmp/whisper_env") \
    .getOrCreate()

# Force every Worker to secure NumPy before we do any math
def secure_worker_env(task_id):
    import sys
    import subprocess
    # 1. Try to use the Whisper environment first
    sys.path.insert(0, "/tmp/whisper_env")
    try:
        import numpy
        return f"Worker {task_id} OK: NumPy found in whisper_env"
    except ImportError:
        try:
            # 2. If that fails, force-install using the EXACT PySpark python binary
            subprocess.run(["sudo", sys.executable, "-m", "pip", "install", "numpy"], check=True)
            return f"Worker {task_id} OK: Force-installed on {sys.executable}"
        except Exception as e:
            return f"Worker {task_id} FAILED: {str(e)}"

print("Locking down NumPy on all cluster nodes...")
statuses = spark.sparkContext.parallelize(range(20), 20).map(secure_worker_env).collect()
for s in set(statuses):
    print(s)
print("Bootstrap complete. Proceeding with TF-IDF...")

# ==========================================
# 2. THE TF-IDF SCRIPT
# ==========================================
# NOW we import the Machine Learning libraries!
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF

# --- CONFIGURATION ---
S3_BUCKET_NAME = "my-flame-emr-bucket" 
INPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-results/" 
OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-keywords/"

if __name__ == "__main__":
    print(f"Reading transcriptions from {INPUT_PATH}...")

    try:
        df = spark.read.parquet(INPUT_PATH)
        df = df.filter(F.col("transcription").isNotNull() & (F.length(F.trim(F.col("transcription"))) > 0))

        print("Tokenizing and removing stop words...")
        tokenizer = RegexTokenizer(inputCol="transcription", outputCol="words", pattern="\\W+")
        wordsData = tokenizer.transform(df)

        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        filteredData = remover.transform(wordsData)

        print("Calculating Term Frequency (TF)...")
        cv = CountVectorizer(inputCol="filtered_words", outputCol="rawFeatures", vocabSize=5000)
        cvModel = cv.fit(filteredData)
        featurizedData = cvModel.transform(filteredData)

        print("Calculating Inverse Document Frequency (IDF)...")
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(featurizedData)
        rescaledData = idfModel.transform(featurizedData)

        print("Extracting Top 15 keywords per video...")
        vocab = cvModel.vocabulary
        vocab_b = spark.sparkContext.broadcast(vocab)

        @F.udf(returnType=ArrayType(StringType()))
        def get_top_15_keywords(features):
            indices = features.indices
            values = features.values
            word_scores = sorted(zip(indices, values), key=lambda x: x[1], reverse=True)
            top_15_indices = [idx for idx, score in word_scores[:15]]
            return [vocab_b.value[idx] for idx in top_15_indices]

        final_df = rescaledData.withColumn("top_keywords", get_top_15_keywords(F.col("features")))
        
        output_df = final_df.select("file_key", "top_keywords")

        print(f"Saving final keywords to {OUTPUT_PATH}...")
        output_df.write.mode("overwrite").json(OUTPUT_PATH)

        print("SUCCESS! TF-IDF Keyword Extraction complete.")
        output_df.show(truncate=False)

    except Exception as e:
        print(f"ERROR processing TF-IDF: {str(e)}")

    finally:
        spark.stop()
