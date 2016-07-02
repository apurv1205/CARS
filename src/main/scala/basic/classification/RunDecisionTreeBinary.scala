package basic.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.joda.time.{DateTime, Duration}

/**
  * Created by roger19890107 on 5/8/16.
  */
object RunDecisionTreeBinary {
  def main(args: Array[String]) {
    context.Config.setLogger
    val sc = context.Config.setupContext("DecisionTreeBinary")

    println("Prepare data ...")
    val (trainData, validateData, testData, categoriesMap) = prepareData(sc)
    println(categoriesMap)
    trainData.persist(); validateData.persist(); testData.persist()

    println("Train and Evaluate data ...")
    //val model = trainEvaluate(trainData, validateData)
    val model = parameterTuning(trainData, validateData)

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

      // 3. split data
      val Array(trainData, validateData, testData) =
        labelPointRDD.randomSplit(Array(0.8, 0.1, 0.1))
      println("Split data to" +
        " trainData:"+ trainData.count() +
        " validateData:" + validateData.count() +
        " testData:" + testData.count())
      (trainData, validateData, testData, categoriesMap)
    }

    def trainModel(trainData: RDD[LabeledPoint], impurity: String,
                   maxDepth: Int, maxBins: Int): (DecisionTreeModel, Long) = {
      val startTime = new DateTime()
      val model = DecisionTree.trainClassifier(trainData, 2, Map[Int, Int](),
        impurity, maxDepth, maxBins)
      val endTime = new DateTime()
      val duration = new Duration(startTime, endTime)
      (model, duration.getMillis)
    }

    def evaluateModel(model: DecisionTreeModel, validateData: RDD[LabeledPoint]): Double = {
      val scoreAndLabels = validateData.map { data =>
        val predict = model.predict(data.features)
        (predict, data.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      metrics.areaUnderROC
    }

  def trainEvaluate(trainData: RDD[LabeledPoint],
                    validateData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("Start training ...")
    val (model, time) = trainModel(trainData, "entropy", 10, 10)
    println("Training time(ms):" + time)
    val auc = evaluateModel(model, validateData)
    println("AUC:" + auc)
    model
  }

  def predictData(sc: SparkContext, model: DecisionTreeModel, categoriesMap: Map[String, Int]): Unit ={
    // 1. import data
    val rawDataWithHeader = sc.textFile("data/evergreen/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    val lines = rawData.map(_.split("\t"))
    println("total lines: " + lines.count())

    // 2. build data
    lines.take(20).map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)

      // 3. predict
      val url = trFields(0)
      val features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)
      val predict = model.predict(features)
      val predictDesc = { predict match {
        case 0 => "ephemeral"
        case 1 => "evergreen"
      }}
      println("URL: " + url + " ==> predict: " + predictDesc)
    }
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validateData: RDD[LabeledPoint],
                           impurityArray: Array[String],
                           maxDepthArray: Array[Int],
                           maxBinsArray: Array[Int]): DecisionTreeModel = {
    val evaluationsArray =
      for (impurity <- impurityArray;
           maxDepth <- maxDepthArray;
           maxBins <- maxBinsArray) yield {
        val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
        val auc = evaluateModel(model, validateData)
        println("params => impurity:" + impurity +
          ", maxDepth:" + maxDepth + ", maxBins:" + maxBins + ", AUC:" + auc)
        (impurity, maxDepth, maxBins, auc)
      }

    val bestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("Best params => impurity:" + bestEval._1 +
      ", maxDepth:" + bestEval._2 + ", maxBins:" + bestEval._3 + ", AUC:" + bestEval._4)

    val (bestModel, time) = trainModel(trainData.union(validateData),
      bestEval._1, bestEval._2, bestEval._3)
    bestModel
  }

  def parameterTuning(trainData: RDD[LabeledPoint],
                      validateData: RDD[LabeledPoint]): DecisionTreeModel = {
    val impurityArray = Array("gini", "entropy")
    val maxDepthArray = Array(3, 5, 10, 30)
    val maxBinsArray = Array(3, 5, 10, 50, 100)
    evaluateAllParameter(trainData, validateData, impurityArray, maxDepthArray, maxBinsArray)
  }
}
