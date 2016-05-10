package Classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.joda.time.{DateTime, Duration}

/**
  * Created by roger19890107 on 5/10/16.
  */
object RunDecisionTreeMulti {

  def main(args: Array[String]) {
    Utils.Config.setLogger
    val sc = Utils.Config.setupContext("DecisionTreeMulti")

    println("Prepare data ...")
    val (trainData, validateData, testData) = prepareData(sc)
    trainData.persist(); validateData.persist(); testData.persist()

    println("Train and Evaluate data ...")
    val model = parameterTuning(trainData, validateData)

    println("Test data ...")
    val auc = evaluateModel(model, testData)
    println("test data and Precision is " + auc)

    println("Predict data ...")
    predictData(sc, model)

    trainData.unpersist(); validateData.unpersist(); testData.unpersist()
  }

  def prepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    // 1. import data
    println("Start import data ...")
    val rawData = sc.textFile("data/covtype.data")
    println("Data count: " + rawData.count)

    // 2. build labeledPoints
    val labelPointRDD = rawData.map { record =>
      val fields = record.split(",").map(_.toDouble)
      val label = fields.last - 1
      LabeledPoint(label, Vectors.dense(fields))
    }

    // 3. split data
    val Array(trainData, validateData, testData) =
      labelPointRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("Split data to" +
      " trainData:"+ trainData.count() +
      " validateData:" + validateData.count() +
      " testData:" + testData.count())
    (trainData, validateData, testData)
  }

  def trainModel(trainData: RDD[LabeledPoint], impurity: String,
                 maxDepth: Int, maxBins: Int): (DecisionTreeModel, Long) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](),
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
    val metrics = new MulticlassMetrics(scoreAndLabels)
    metrics.precision
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
        val precision = evaluateModel(model, validateData)
        println("params => impurity:" + impurity +
          ", maxDepth:" + maxDepth + ", maxBins:" + maxBins + ", precision:" + precision)
        (impurity, maxDepth, maxBins, precision)
      }

    val bestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("Best params => impurity:" + bestEval._1 +
      ", maxDepth:" + bestEval._2 + ", maxBins:" + bestEval._3 + ", precision:" + bestEval._4)

    val (bestModel, time) = trainModel(trainData.union(validateData),
      bestEval._1, bestEval._2, bestEval._3)
    bestModel
  }

  def predictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    // 1. import data
    println("Start import data ...")
    val rawData = sc.textFile("data/covtype.data")

    // 2. prepare testing data
    println("Prepare testing data ...")
    val Array(pData, oData) = rawData.randomSplit(Array(0.1, 0.9))
    val data = pData.take(20).map { record =>
      val fields = record.split(",").map(_.toDouble)
      val features = Vectors.dense(fields)
      val label = fields.last - 1
      val predict = model.predict(features)
      val result = if (label == predict) "Correct" else "Fail"
      println("Height:" + features(0) +
        ", Direction:" + features(1) +
        ", Slop:" + features(2) +
        ", Source vertical dist:" + features(3) +
        ", Source horizonal dist:" + features(4) +
        " => predict:" + predict +
        ", label:" + label +
        ", result:" + result)
    }
  }

  def parameterTuning(trainData: RDD[LabeledPoint],
                      validateData: RDD[LabeledPoint]): DecisionTreeModel = {
    val impurityArray = Array("gini", "entropy")
    val maxDepthArray = Array(5, 10, 15, 25, 30)
    val maxBinsArray = Array(5, 10, 50, 100, 150)
    evaluateAllParameter(trainData, validateData, impurityArray, maxDepthArray, maxBinsArray)
  }
}
