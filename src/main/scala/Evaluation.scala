import java.io.PrintWriter

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object Evaluation {

  implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel) {
    def bestEstimatorParamMap: (ParamMap, Double) = {
      cvModel.getEstimatorParamMaps
        .zip(cvModel.avgMetrics)
        .maxBy(_._2)
    }

    def estimatorMetricsMap: Array[(ParamMap, Double)] = {
      cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)
    }
  }

  def createMetricsReport(df: DataFrame): RDD[(String, Double, Double)] = {
    val predictionAndLabels = df.rdd.map( row =>
      (row.getAs[Double]("label"), row.getAs[Double]("prediction"))
    )
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val precisions = metrics.precisionByThreshold.map { case (t, p) =>
      ("precision", t, p)
    }

    val recalls = metrics.recallByThreshold().map { case (t, p) =>
      ("recall", t, p)
    }

    val f1Measures = metrics.fMeasureByThreshold().map { case (t, p) =>
      ("f1measure", t, p)
    }

    precisions.union(recalls).union(f1Measures)
  }

}
