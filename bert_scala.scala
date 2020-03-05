
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
object ner_dl_pipeline extends App{
    val spark: SparkSession = SparkSession
        .builder()
        .appName("test")
        .master("local[*]")
        .config("spark.driver.memory", "12G")
        .config("spark.kryoserializer.buffer.max","200M")
        .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()

    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    val document = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
        .setInputCols("document")
        .setOutputCol("sentence")

    val token = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")

    val normalizer = new Normalizer()
        .setInputCols("token")
        .setOutputCol("normal")

    val wordEmbeddings = BertEmbeddings.pretrained("bert_base_cased")
        .setInputCols("document", "token")
        .setOutputCol("word_embeddings")

    val ner = NerDLModel.pretrained("ner_dl_bert")
        .setInputCols("normal", "document","word_embeddings")
        .setOutputCol("ner")


    val nerConverter = new NerConverter()
        .setInputCols("document", "normal", "ner")
        .setOutputCol("ner_converter")

    val finisher = new Finisher()
        .setInputCols("ner", "ner_converter")
        .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, normalizer, wordEmbeddings, ner, nerConverter, finisher))
    val inputPath = "/Users/saicharan/Documents/codex/ner/old_articles_tagged.json"
    val inputDF = spark.read.json(inputPath)

    def concatenate:((String,String) => String) = (title:String,content:String)=>{
        (title+" . "+content)
    }
    val concatenateUDF = udf(concatenate)
    val inputDFConcat = inputDF.withColumn("text",concatenateUDF($"title",$"content"))


    val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(inputDFConcat)
    result.select("ner_converter.metadata","ner_converter.result")
        .coalesce(1)
        .write.format("json")
        .option("header", "true")
        .save("output available @ https://github.com/saicharannivarthi/spark_nlp_issue_resources/blob/master/scala_output.json")

}
