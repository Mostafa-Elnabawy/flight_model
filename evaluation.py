from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import SparkContext

sc = SparkContext("local", "Flight_model")

spark = SparkSession.builder.master("local")\
                            .appName('Flight_model')\
                            .getOrCreate()

pickleRdd = sc.pickleFile('test_pickled').collect()
test = spark.createDataFrame(pickleRdd)

model = LogisticRegressionModel.load('bestmodel')
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Use the model to predict the test set
test_results = model.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))
print("Done")
