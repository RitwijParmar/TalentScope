from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, IndexToString
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

#start Spark session
spark = SparkSession.builder.appName("Predict Job Title from Skills - Filtered NB").config("spark.driver.memory", "6g").config("spark.executor.memory", "6g").config("spark.kryoserializer.buffer.max", "1024m").getOrCreate()

# Do the raw data processing a second time for this file

# Load dataset
path = "/Users/ritwijaryanparmar/Documents/DIC_Phase2/cleaned_job_skills.csv" # THIS IS HARD CODED, change this based on the user
df = spark.read.csv(path, header=True, inferSchema=True)

# Drop rows with missing values
df = df.dropna(subset=["job_skills", "job_name"])

# Keep only top 100 frequent job titles
top_jobs = df.groupBy("job_name").count().orderBy("count", ascending=False).limit(100)
top_jobs_list = [row["job_name"] for row in top_jobs.collect()]
df = df.filter(df.job_name.isin(top_jobs_list))

# Feature Engineering
tokenizer = Tokenizer(inputCol="job_skills", outputCol="skills_tokens")
hashingTF = HashingTF(inputCol="skills_tokens", outputCol="skills_tf", numFeatures=500)
idf = IDF(inputCol="skills_tf", outputCol="skills_features")

# Label encoding
label_indexer = StringIndexer(inputCol="job_name", outputCol="label")
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_job_name", labels=label_indexer.fit(df).labels)

# Classifier- use a naive bayes estimator.
nb = NaiveBayes(featuresCol="skills_features", labelCol="label", smoothing=1.0, modelType="multinomial")

# Pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_indexer, nb, label_converter])

# Train-test split
train, test = df.randomSplit([0.8, 0.2], seed=42)


model = pipeline.fit(train)
predictions = model.transform(test)

#evaluate
# Because there are many discrete classes, us multiclassclassificationevaluation.
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy") # we are aiming for overall accuracy
accuracy = evaluator.evaluate(predictions)

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

print("Test Accuracy:",str(accuracy))
print("Test F1 Score:",str(f1_score))

#Visualization
sample = predictions.select("job_skills", "job_name", "predicted_job_name").sample(False, 0.001, seed=42).limit(100).toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(sample)), [1]*len(sample), label="True vs Predicted")
plt.xticks(range(len(sample)), sample['predicted_job_name'], rotation=90)
plt.title("True vs Predicted Job Titles (Top 100 Classes Only)")
plt.tight_layout()
plt.show()

top_pred = sample['predicted_job_name'].value_counts().nlargest(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_pred.values, y=top_pred.index)
plt.title("Top 10 Predicted Job Titles (Sampled)")
plt.xlabel("Frequency")
plt.ylabel("Predicted Job Titles")
plt.tight_layout()
plt.show()

# model.save("/Users/ritwijaryanparmar/Documents/DIC_Phase2/predict_job_from_skills_nb_filtered_model")
# predictions.select("job_skills", "job_name", "predicted_job_name").write.mode("overwrite").csv("/Users/ritwijaryanparmar/Documents/DIC_Phase2/predicted_job_from_skills_output")
