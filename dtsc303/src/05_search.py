import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_contains

# --- CONFIGURATION ---
S3_BUCKET_NAME = "my-flame-emr-bucket"
OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/tf-idf-manav-heetej-keywords/"

if __name__ == "__main__":
    # Check if the user actually typed a word to search for
    if len(sys.argv) < 2:
        print("\n❌ ERROR: You forgot the search term!")
        print("Usage: spark-submit 05_search.py <word_to_search>\n")
        sys.exit(1)
        
    # Grab the search term from the terminal and make it lowercase
    search_term = sys.argv[1].lower()
    
    # Start Spark
    spark = SparkSession.builder.appName("Video_Search_Engine").getOrCreate()
    
    # Suppress all the noisy Spark INFO logs so our search results are easy to read
    spark.sparkContext.setLogLevel("WARN")
    
    print("="*60)
    print(f"🔍 SEARCHING CLUSTER FOR VIDEOS ABOUT: '{search_term.upper()}'...")
    print("="*60)
    
    try:
        # Load the keyword arrays from our JSON data
        df = spark.read.json(OUTPUT_PATH)
        
        # Filter the Big Data! Look inside the 'top_keywords' array for the exact search term
        matches = df.filter(array_contains(col("top_keywords"), search_term))
        
        match_count = matches.count()
        
        if match_count == 0:
            print(f"\n❌ No videos found matching the keyword '{search_term}'. Try another word!\n")
        else:
            print(f"\n✅ SUCCESS! Found {match_count} matching video(s):\n")
            # Print only the file names of the matches
            matches.select("file_key").show(truncate=False)
            
    except Exception as e:
        print(f"Error searching results: {str(e)}")
        
    spark.stop()
