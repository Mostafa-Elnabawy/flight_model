from pyspark.sql import SparkSession

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
model_data.show(5)