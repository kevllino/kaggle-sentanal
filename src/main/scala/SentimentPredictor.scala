import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.col
import Utils._
import Processing._
import Evaluation._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SaveMode

object SentimentPredictor extends App {
  val spark = createSparkSession("sentimentPredictor")

  import spark.implicits._

  val trainingDataPath = getClass.getResource("train-data.txt").toString

  // read and process data
  val inputData = spark
    .read
    .option("sep", "\t")
    .csv(trainingDataPath)
    .toDF("sentiment", "text")
    .withColumn("cleanedText", removePunctuation(col("text")))

  val dataCV = inputData.randomSplit(Array(0.7, 0.3), 1000)
  val trainingDataCV = dataCV(0).cache()
  val testDataCV = dataCV(1)

  val tokenizer = new Tokenizer()
    .setInputCol("cleanedText")
    .setOutputCol("tokens")
  val stopwordsremover = new StopWordsRemover()
    .setCaseSensitive(false)
    .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
    .setInputCol("tokens")
    .setOutputCol("filtered")
  val hashingTF = new HashingTF()
    .setInputCol("filtered")
    .setOutputCol("rawFeatures")
  val idf = new IDF()
    .setInputCol("rawFeatures")
    .setOutputCol("idf")
  val assembler = new VectorAssembler()
    .setInputCols(Array("idf"))
    .setOutputCol("features")
  val labeler = new StringIndexer()
    .setInputCol("sentiment")
    .setOutputCol("label")
  val lsvc_optimizable = new LinearSVC()
    .setFeaturesCol("features")
    .setLabelCol("label")

  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, stopwordsremover, hashingTF, idf, assembler, labeler, lsvc_optimizable))

  // Hyper-parameters tuning
  val paramGrid = new ParamGridBuilder()
    .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
    .addGrid(lsvc_optimizable.maxIter, Array(10, 20))
    .addGrid(lsvc_optimizable.regParam, Array(0.01, 0.1, 1))
    .build()

  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new BinaryClassificationEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)

  val cvModel: CrossValidatorModel = cv.fit(trainingDataCV)
  val bestModel = cvModel.bestModel
  val optimizedPredictions = bestModel.transform(testDataCV)
  val metricsReport = createMetricsReport(optimizedPredictions)

  metricsReport
    .toDF("metricName", "threshold", "metricValue")
    .coalesce(1)
    .write
    .mode(SaveMode.Overwrite)
    .option("header", true)
    .csv("/Users/kevineid/Projects/data-science/kaggle-sentanal/src/main/resources/metrics.csv")
}
