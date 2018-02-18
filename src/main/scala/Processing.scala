import org.apache.spark.sql.functions.udf

object Processing {
  val removePunctuation = udf { (text: String) => {
    val textWithoutPunctuations = text.replaceAll("""[\p{Punct}]""", "")
    textWithoutPunctuations
  }
  }


}
