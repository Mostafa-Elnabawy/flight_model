from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator


spark = SparkSession.builder.master("local")\
                            .appName('Flight_model')\
                            .getOrCreate()
planes = spark.read.csv('planes.csv', header=True)
flights = spark.read.csv('flights.csv',header=True)
# Rename year column
planes = planes.withColumnRenamed('year','plane_year')

# Join the DataFrames
model_data = flights.join(planes, on='tailnum', how="leftouter")

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol = "carrier" , outputCol="carrier_index")
dest_indexer = StringIndexer(inputCol = "dest" , outputCol="dest_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index" , outputCol="carrier_fact")
dest_encoder = OneHotEncoder(inputCol="dest_index" , outputCol="dest_fact")

features = ["month", "air_time", "carrier_fact", "dest_fact", "plane_age"]

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([0.6,0.4])

#modeling
#LogisticRegression
# Create a LogisticRegression Estimator
lr = LogisticRegression()


#cross_validation using area under the curve AUC
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the parameter grid
grid = ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0,1])

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )
# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel
best_lr.save("bestmodel")
print(best_lr)