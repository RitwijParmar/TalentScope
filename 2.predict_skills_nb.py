from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# This script does supervised learning on the jobs and skills, mapping skills to jobs and vice versa.

spark = SparkSession.builder.appName("Predict Skills from Job Title using Naive Bayes").config("spark.driver.memory", "6g").config("spark.kryoserializer.buffer.max", "1024m").getOrCreate()

# load the data, remove NA values.
df = spark.read.csv("cleaned_job_skills.csv", header=True, inferSchema=True)
df = df.sample(False, 0.1, seed=42)
df = df.dropna(subset=["job_name", "job_skills"])

# Tokenize the data yet again
tokenizer = Tokenizer(inputCol="job_name", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=500)
idf = IDF(inputCol="rawFeatures", outputCol="features")
indexer = StringIndexer(inputCol="job_skills", outputCol="label", handleInvalid="keep")

# use naive bayes to classify
# naive works best
nb = NaiveBayes(featuresCol="features", labelCol="label")

#
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, indexer, nb])
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [1.0]).build()
tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)

# Train-Test Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = tvs.fit(train)

predictions = model.transform(test)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")


#Visualization
pdf = predictions.select("label", "prediction").sample(False, 0.01, seed=1).toPandas()
plt.figure(figsize=(10, 6))
# Reverse index mapping
# Replace indices with real names
sns.countplot(x="prediction", data=pdf, order=pdf["prediction"].value_counts().index[:10])
plt.title("Top 10 Predicted Skill Labels (Sampled)")
plt.xlabel("Predicted Label Index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

spark.stop()

