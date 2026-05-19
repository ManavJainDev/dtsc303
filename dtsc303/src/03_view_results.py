from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws

# --- CONFIGURATION ---
S3_BUCKET_NAME = "my-flame-emr-bucket"
OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-keywords/"
CSV_OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-csv-export/"

if __name__ == "__main__":
    spark = SparkSession.builder.appName("View_Results").getOrCreate()
    
    try:
        # Read the JSON files
        df = spark.read.json(OUTPUT_PATH)
        
        # 1. Print the beautiful table to the terminal
        print("=== FINAL TF-IDF KEYWORDS ===")
        df.show(truncate=False)
        
        # 2. Fix the CSV Error: Convert the Array of strings into a single string
        # This is the crucial line that prevents the crash!
        df_for_csv = df.withColumn("top_keywords", concat_ws(", ", df["top_keywords"]))
        
        # 3. Save as CSV
        print(f"Saving a flat CSV copy to {CSV_OUTPUT_PATH}...")
        df_for_csv.coalesce(1).write.mode("overwrite").csv(CSV_OUTPUT_PATH, header=True)
        print("CSV saved successfully!")
        
    except Exception as e:
        print(f"Error reading results: {str(e)}")
        
    spark.stop()
