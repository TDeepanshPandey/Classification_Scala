// Importing Spark, Logistic Regression and log4j
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creating Spark Session
val spark = SparkSession.builder().appName("LogisticRegressionExample").getOrCreate()

// Reading the Data
val data = spark.read.option("header","true").option("inferschema","true").format("csv").load("titanic.csv")
data.printSchema()
data.head(1)

// Selecting relevant columns and dropping null values
val logregdatall = (data.select(data("Survived").as("label"),$"Pclass",
$"Sex", $"Age", $"SibSp", $"Parch",$"Fare", $"Embarked"))
val logregdata = logregdatall.na.drop()

// Categorical Variables
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

// Assembler to add columns into one feature column
val assembler = (new VectorAssembler().setInputCols(Array("Pclass", "SexVec", "Age",
  "SibSp", "Parch","Fare", "EmbarkVec")).setOutputCol("features"))

// Dividing the data into test and training set
val Array(training,test) = logregdata.randomSplit(Array(0.7,0.3),seed=12345)

// Importing the pipeline
import org.apache.spark.ml.Pipeline

// Creating the pipeline and fitting model to data
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,assembler,lr))
val model = pipeline.fit(training)
val results = model.transform(test)

//Model Evaluation
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val predictionAndLabels = results.select($"prediction",$"label").as[(Double,Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

//Print Dataset Scahema
data.printSchema()

//Confusion Metrix
println("Confusion Matrix")
println(metrics.confusionMatrix)
