import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object Utils {

  def createSparkSession(appName: String) =
    SparkSession
      .builder()
      .master("local[*]")
      .appName(appName)
      .getOrCreate()


}
