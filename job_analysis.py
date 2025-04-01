from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, count, desc, size, split
from pyspark.ml.feature import Tokenizer
from pyspark.sql import functions as F

# start spark
spark = SparkSession.builder.appName("Job Analysis Phase2").getOrCreate()

# read data
try:
    input_path = "/opt/bitnami/spark/final_cleaned_job_data.csv"
    df = spark.read.option("header", True).csv(input_path)
    print("Data read success")
except Exception as e:
    print("Data read failed:", e)
    spark.stop()
    exit()

# check schema
print("Schema:")
df.printSchema()

# view some data
print("Sample data:")
df.show(5, truncate=False)

# remove missing
print("Drop missing values")
essential_columns = ["job_link", "job_title", "company", "job_location"]
df = df.dropna(subset=essential_columns)

# remove outliers
print("Remove very long job titles")
df = df.withColumn("title_length", length(col("job_title")))
df = df.filter(col("title_length") <= 150)

# fix date type
print("\nFix date column")
df = df.withColumn("first_seen", col("first_seen").cast("timestamp"))

# tokenize titles
print("\nTokenize job titles")
tokenizer = Tokenizer(inputCol="job_title", outputCol="title_words")
df = tokenizer.transform(df)
df = df.withColumn("title_word_count", size(col("title_words")))

# show word count
df.select("job_title", "title_word_count").show(5, truncate=False)

# skills analysis if present
if "job_skills" in df.columns:
    print("\nSkills analysis")
    skills_df = df.select(F.explode(split(col("job_skills"), ",")).alias("skill"))
    skills_df.groupBy("skill").count().orderBy(desc("count")).show(10, truncate=False)

# jobs count
print("\nTop jobs posted")
df.groupBy("job_title").count().orderBy(desc("count")).show(10, truncate=False)

# duplicates
print("\nFind duplicate job posts")
duplicate_jobs = df.groupBy("company", "job_title").count().filter(col("count") > 1)
duplicate_jobs.show(10, truncate=False)

# top companies
print("\nTop hiring companies")
df.groupBy("company").count().orderBy(desc("count")).show(5, truncate=False)

# top locations
print("\nTop locations")
df.groupBy("job_location").count().orderBy(desc("count")).show(5, truncate=False)

# save output
print("\nSave data to HDFS")
try:
    output_path = "hdfs://namenode:9000/user/spark/output/final_processed_jobs"
    df.write.mode("overwrite").parquet(output_path)
    print("Data saved successfully")
except Exception as e:
    print("Save failed:", e)

# stop spark
spark.stop()
print("\nFinished")

# Hadoop Commands

# make user folder
# hdfs dfs -mkdir -p /user/spark

# give full permission
# hdfs dfs -chmod -R 777 /user/spark

# hdfs dfs -ls /user/spark

# hdfs dfs -ls /user/spark/output/final_processed_jobs

# hdfs dfs -cat /user/spark/output/final_processed_jobs/part-*

