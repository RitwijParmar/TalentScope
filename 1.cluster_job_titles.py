from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col, count

# Create Spark session
spark = SparkSession.builder.appName("Job Title Clustering").getOrCreate()

# Load data
df = spark.read.csv("cleaned_job_skills.csv", header=True, inferSchema=True)

# Drop nulls in job_name
df = df.dropna(subset=["job_name"])

# Our goal is to 'cluster' the jobs based on their tokenized values.
# For unsupervised learning, we don't label the data. We just use the tokenized values to see how many
# distinct clusters/categories/groups there are.

# Text processing pipeline
tokenizer = Tokenizer(inputCol="job_name", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Begin by tokenizing the inputs, then removing stop words, then the hasher
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
model = pipeline.fit(df)
processed_df = model.transform(df)

# Apply KMeans clustering
kmeans = KMeans(k=10, seed=42, featuresCol="features", predictionCol="cluster")
kmeans_model = kmeans.fit(processed_df)
clustered = kmeans_model.transform(processed_df)

# Show the results
clustered.select("job_name", "cluster").show(25, truncate=False)

# kmeans_model.save("kmeans_model_job_titles")
# clustered.select("job_name", "cluster").write.mode("overwrite").csv("job_title_clusters")

# Convert to Pandas for visualization (limit to 1000 to keep things snappy)
sample_pd = clustered.select("job_name", "cluster").limit(1000).toPandas()

# Count job titles per cluster
cluster_counts = sample_pd["cluster"].value_counts().sort_index()

# Plotting
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title("Job Title Distribution per Cluster (Sample)")
plt.xlabel("Cluster")
plt.ylabel("Number of Job Titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Stop Spark session
spark.stop()

