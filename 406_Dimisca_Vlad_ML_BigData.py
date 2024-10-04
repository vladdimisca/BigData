from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import corr
from pyspark.sql.functions import col


stroke_data_file = "Data/stroke-data.csv"

# create a spark session and read the data from the csv file
spark = SparkSession.builder.appName('Heart Disease Prediction').getOrCreate()

print("READING THE INPUT FILE")

df = spark.read.csv(stroke_data_file, inferSchema=True, header=True)

# display the schema and the first rows from the dataframe
df.printSchema()

print("DISPLAY THE FIRST ROWS FROM THE DATAFRAME\n")

df.show(10)

print("DISPLAY STATISTICS\n")

# display statistics about the dataset
df.describe().show()

print("DISPLAY CATEGORICAL VARIABLES COUNT\n")

df.groupBy('gender').count().show()
df.groupBy('ever_married').count().show()
df.groupBy('work_type').count().show()
df.groupBy('Residence_type').count().show()
df.groupBy('smoking_status').count().show()

print("CLEANING THE DATA")

# cast columns containing numeric values (but provided as string) to double ('bmi')
df = df.withColumn("bmi", df["bmi"].cast("double"))

# remove the rows where bmi is null and smoking_status is unknown
df = df.filter((col("bmi").isNotNull()) | (col("smoking_status") != "Unknown"))

# fill the unknown bmi values with the mean value
df = df.na.fill(df.select(mean(df["bmi"])).collect()[0][0], ["bmi"])

# drop irrelevant columns ('id')
df = df.drop("id")

# transform relevant columns containing string values into categorical variables using StringIndexer
columns_to_transform = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

indexers = []
for column in columns_to_transform:
    indexers.append(StringIndexer(inputCol=column, outputCol=column + "_cat"))

print("\n----CLASSIFICATION PART----\n")

# assemble the relevant columns in a vector of features for classification
assembler = VectorAssembler(
    inputCols=[
        "gender_cat",
        "age",
        "hypertension",
        "stroke",
        "ever_married_cat",
        "work_type_cat",
        "Residence_type_cat",
        "avg_glucose_level",
        "bmi",
        "smoking_status_cat"
    ],
    outputCol="features")

print("CREATE A PIPELINE WITH THE INDEXERS AND ASSEMBLER\n")

# create a Pipeline with the previously defined indexers and assembler
pipeline = Pipeline(stages=[*indexers, assembler])

# train the pipeline
output = pipeline.fit(df).transform(df)

# display 'features' and 'heart_disease' columns
print("DISPLAY 'features' AND 'heart_disease' COLUMNS")
output.select("features", "heart_disease").show()

final_data = output.select("features", "heart_disease")

# divide the dataset into train and test subsets
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# Use decision tree classifier
dt = DecisionTreeClassifier(labelCol="heart_disease", featuresCol="features")
dt_model = dt.fit(train_data)

predictions = dt_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="heart_disease", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("DECISION TREE CLASSIFIER ACCURACY: ", accuracy)

# Use random forest classifier
rf = RandomForestClassifier(labelCol="heart_disease", featuresCol="features", numTrees=10)
rf_model = rf.fit(train_data)

predictions = rf_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="heart_disease", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("RANDOM FOREST CLASSIFIER ACCURACY: ", accuracy)

print("\n----REGRESSION PART----\n")

# assemble the relevant columns in a vector of features for regression
assembler = VectorAssembler(
    inputCols=[
        "gender_cat",
        "hypertension",
        "heart_disease",
        "ever_married_cat",
        "work_type_cat",
        "Residence_type_cat",
        "avg_glucose_level",
        "bmi",
        "smoking_status_cat",
        "stroke"
    ],
    outputCol="features")

print("CREATE A PIPELINE WITH THE INDEXERS AND ASSEMBLER\n")

pipeline = Pipeline(stages=[*indexers, assembler])
output = pipeline.fit(df).transform(df)

# display 'features' and 'age' columns
print("DISPLAY 'features' AND 'age' COLUMNS")
output.select("features", "age").show()

final_data = output.select("features", "age")

# divide the dataset into train and test subsets
train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(labelCol="age", predictionCol="prediction", regParam=0.1)
lr_model = lr.fit(train_data)

# evaluate the model
test_results = lr_model.evaluate(test_data)

# display evaluation results
print("REGRESSION RESULTS\n")
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))

# Display the correlation between age and other columns
print("\nDISPLAY CORRELATIONS BETWEEN 'age' AND OTHER COLUMNS")
df.select(corr('age', 'stroke')).show()
df.select(corr('age', 'heart_disease')).show()
df.select(corr('age', 'hypertension')).show()

# apply the model on unlabeled data
print("APPLY THE MODEL ON UNLABELED DATA AND DISPLAY PREDICTIONS\n")
unlabeled_data = test_data.select("features")
predictions = lr_model.transform(unlabeled_data)
predictions.show()

print("DISPLAY REAL VALUES")
test_data.show()
