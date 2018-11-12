package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation
import org.apache.spark.ml.tuning
import org.apache.spark.ml.Pipeline

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** Lire les fichiers parquets **/
    val data = spark.read.parquet("/home/david/TP_ParisTech_2018_2019_starter/TP_ParisTech_2017_2018_starter/prepared_trainingset")

    /** Random Split Training & Test Datesets **/
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    /** Stage 1 **/
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /** Stage 2 **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("StopRemove")

    /** Stage 3 **/
    val cvModel = new CountVectorizer()
      .setInputCol("StopRemove")
      .setOutputCol("Counting")

    /** Stage 4 **/
    val idf = new IDF()
      .setInputCol("Counting")
      .setOutputCol("TFidf")

    /** Stage 5 **/
    val indexer_country = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    /** Stage 6 **/
    val indexer_currency = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** Stage 7 **/
    val encoder_country = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_encoded")

    /** Stage 8 **/
    val encoder_currency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_encoded")

    /** Stage 9 **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("TFidf", "days_campaign", "hours_prepa", "goal", "country_encoded", "currency_encoded"))
      .setOutputCol("features")

    /** Model **/
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** Evaluator **/
    val evaluator = new evaluation.MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    /** Grid-search : Cross-Validation **/
    val paramGrid = new tuning.ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .build()

    /** Stage 10 **/
    val trainValidationSplit = new tuning.TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /** Pipeline **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexer_country, indexer_currency, encoder_country, encoder_currency, assembler, trainValidationSplit))

    /** On entraîne le modèle **/
    val model = pipeline.fit(training)

    /** Prediction & Score **/
    val df_WithPredictions = model.transform(test)
    val holdout = df_WithPredictions.select("predictions", "final_status")
    val F1_score = evaluator.evaluate(holdout)
    println("Test set accuracy = " + F1_score)

    /** Résultats obtenus du modèle **/
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    /** Sauvegarder le modèle Trainer **/
    model.write.overwrite().save("./src/main/resources/gridsearch-model")

  }
}