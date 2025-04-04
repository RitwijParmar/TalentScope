
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pandas as pd
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder.appName("EDA using PySpark").getOrCreate()

# UDF to clean job titles
def clean_link(link):
    link = str(link)
    index = len(link)-1
    job = ""
    prohibited_chars = ["%", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    start_transcribing = False
    while index != 0:
        if not start_transcribing and link[index] == '-':
            start_transcribing = True
        elif start_transcribing and link[index] == '/':
            job = job[::-1]
            break
        elif start_transcribing:
            if link[index] not in prohibited_chars:
                job = job + str(link[index])
            elif link[index] == '-':
                job = job + ' '
        index -= 1
    truncate = False
    for i in range(len(job)):
        if i + 3 < len(job):
            if job[i] == ' ' and job[i+1:i+4] == 'at ' and i > 0:
                truncate = i
                break
    if truncate:
        job = job[0:truncate]
    return job.strip()

clean_link_udf = udf(clean_link, StringType())

# Read CSV
df = spark.read.option("header", True).csv("hdfs://namenode:9000/input/job_skills.csv")
df = df.withColumn("job_name", clean_link_udf(df["job_link"]))
df = df.select("job_name", "job_skills").na.drop()

# Save cleaned data to HDFS
df.coalesce(1).write.mode("overwrite").option("header", True).csv("hdfs://namenode:9000/output/cleaned_job_skills")

# Summary statistics using Pandas
sample_df = df.limit(10000).toPandas()
summary = sample_df.describe(include='all')
summary.to_csv("/tmp/job_summary.csv")

# Skill frequency analysis
skills_count = {}
for line in df.select("job_skills").rdd.flatMap(lambda row: row[0].split(',')).collect():
    skill = line.strip().lower()
    if skill == "problemsolving":
        skill = "problem solving"
    skills_count[skill] = skills_count.get(skill, 0) + 1

# Convert to Pandas for plotting
skills_df = pd.DataFrame(sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Skill", "Count"])

# Plotting top skills
plt.figure(figsize=(12,6))
plt.bar(skills_df["Skill"], skills_df["Count"])
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Job Skills")
plt.tight_layout()
plt.savefig("/opt/bitnami/spark/top_skills.png")

print("✅ EDA complete — results written to HDFS and visualization saved at /opt/bitnami/spark/top_skills.png")
spark.stop()
