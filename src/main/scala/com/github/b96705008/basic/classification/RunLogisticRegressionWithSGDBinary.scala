package com.github.b96705008.basic.classification

import com.github.b96705008.context.Env
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.joda.time.{DateTime, Duration}
/**
  * Created by roger19890107 on 5/8/16.
  */
object RunLogisticRegressionWithSGDBinary {
  def main(args: Array[String]) {
    Env.setLogger
    val sc = Env.setupContext("LogisticRegression")

    println("Prepare data ...")
    val (trainData, validateData, testData, categoriesMap) = prepareData(sc)
    println(categoriesMap)
    trainData.persist(); validateData.persist(); testData.persist()

    println("Train and Evaluate data ...")
    val model = trainEvaluate(trainData, validateData)
    //val model = parameterTuning(trainData, validateData)

    println("Test data ...")
    val auc = evaluateModel(model, testData)
    println("test data and AUC is " + auc)

    println("Predict data ...")
    predictData(sc, model, categoriesMap)

    trainData.unpersist(); validateData.unpersist(); testData.unpersist()
  }

  def prepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint],
    RDD[LabeledPoint], Map[String, Int]) = {
    // 1. import data
    val rawDataWithHeader = sc.textFile("data/evergreen/train.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    val lines = rawData.map(_.split("\t"))
    println("total lines: " + lines.count())

    // 2. build RDD[LabeledPoint]
    val categoriesMap = lines.map(_(3)).distinct.collect.zipWithIndex.toMap
    val labelPointRDD = lines.map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val label = trFields(fields.size - 1).toInt
      LabeledPoint(label,
        Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
    }

    // 3. normalize data
    val featureData = labelPointRDD.map(_.features)
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featureData)
    val scaledRDD = labelPointRDD.map(d => LabeledPoint(d.label, stdScaler.transform(d.features)))

    // 4. split data
    val Array(trainData, validateData, testData) =
      scaledRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("Split data to" +
      " trainData:"+ trainData.count() +
      " validateData:" + validateData.count() +
      " testData:" + testData.count())
    (trainData, validateData, testData, categoriesMap)
  }

  def trainModel(trainData: RDD[LabeledPoint], numIterations: Int,
                 stepSize: Double, miniBatchFraction: Double): (LogisticRegressionModel, Long) = {
    val startTime = new DateTime()
    val model = LogisticRegressionWithSGD.train(trainData,
      numIterations, stepSize, miniBatchFraction)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis)
  }

  def evaluateModel(model: LogisticRegressionModel, validateData: RDD[LabeledPoint]): Double = {
    val scoreAndLabels = validateData.map { data =>
      val predict = model.predict(data.features)
      (predict, data.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    metrics.areaUnderROC
  }

  def trainEvaluate(trainData: RDD[LabeledPoint],
                    validateData: RDD[LabeledPoint]): LogisticRegressionModel = {
    println("Start training ...")
    val (model, time) = trainModel(trainData, 5, 50, 0.5)
    println("Training time(ms):" + time)
    val auc = evaluateModel(model, validateData)
    println("AUC:" + auc)
    model
  }

  def predictData(sc: SparkContext, model: LogisticRegressionModel, categoriesMap: Map[String, Int]): Unit ={
    // 1. import data
    val rawDataWithHeader = sc.textFile("data/evergreen/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    val lines = rawData.map(_.split("\t"))
    println("total lines: " + lines.count())

    // 2. build data
    val testData = lines.map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)
      (trFields(0), features)
    }

    // 3. normalize data
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(testData.map(_._2))
    val scaledTestRDD = testData.map(d => (d._1, stdScaler.transform(d._2)))

    // 4. predict
    scaledTestRDD.take(20).foreach { d =>
      val url = d._1
      val features = d._2
      val predict = model.predict(features)
      val predictDesc = { predict match {
        case 0 => "ephemeral"
        case 1 => "evergreen"
      }}
      println("URL: " + url + " ==> predict: " + predictDesc)
    }
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validateData: RDD[LabeledPoint],
                           iterArray: Array[Int],
                           stepArray: Array[Double],
                           miniFracArray: Array[Double]): LogisticRegressionModel = {
    val evaluationsArray =
      for (numIterations <- iterArray;
           stepSize <- stepArray;
           miniBatchFraction <- miniFracArray) yield {
        val (model, time) = trainModel(trainData, numIterations, stepSize, miniBatchFraction)
        val auc = evaluateModel(model, validateData)
        println("params => numIterations:" + numIterations +
          ", stepSize:" + stepSize + ", miniBatchFraction:" + miniBatchFraction + ", AUC:" + auc)
        (numIterations, stepSize, miniBatchFraction, auc)
      }

    val bestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("Best params => numIterations:" + bestEval._1 +
      ", stepSize:" + bestEval._2 + ", miniBatchFraction:" + bestEval._3 + ", AUC:" + bestEval._4)

    val (bestModel, time) = trainModel(trainData.union(validateData),
      bestEval._1, bestEval._2, bestEval._3)
    bestModel
  }

  def parameterTuning(trainData: RDD[LabeledPoint],
                      validateData: RDD[LabeledPoint]): LogisticRegressionModel = {
    val iterArray = Array(5, 15, 20, 60, 100)
    val stepArray = Array(10.0, 50.0, 100.0, 200.0)
    val miniBatchFraction = Array(0.5, 0.8, 1)
    evaluateAllParameter(trainData, validateData, iterArray, stepArray, miniBatchFraction)
  }
}
