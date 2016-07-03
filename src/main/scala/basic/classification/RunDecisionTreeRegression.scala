package basic.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.joda.time.{DateTime, Duration}

/**
  * Created by roger19890107 on 5/10/16.
  */
object RunDecisionTreeRegression {
  def main(args: Array[String]) {
    context.Env.setLogger
    val sc = context.Env.setupContext("DecisionTreeBinary")

    println("Prepare data ...")
    val (trainData, validateData, testData) = prepareData(sc)
    trainData.persist(); validateData.persist(); testData.persist()

    println("Train and Evaluate data ...")
    val model = parameterTuning(trainData, validateData)

    println("Test data ...")
    val rmse = evaluateModel(model, testData)
    println("test data and RMSE is " + rmse)

    println("Predict data ...")
    predictData(sc, model)

    trainData.unpersist(); validateData.unpersist(); testData.unpersist()
  }

  def prepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    // 1. import data
    val rawDataWithHeader = sc.textFile("data/bike/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) =>
      if (idx == 0) iter.drop(1) else iter
    }
    println("Data count:" + rawData.count().toString)

    // 2. build labeled points
    println("prepare data ...")
    val records = rawData.map(_.split(","))
    val labelPointRDD = records.map { fields =>
      val label = fields.last.toInt
      val season = fields(2).toDouble
      val features = fields.slice(4, fields.length - 3).map(_.toDouble)
      LabeledPoint(label, Vectors.dense(season +: features))
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

  def trainModel(trainData: RDD[LabeledPoint],
                 maxDepth: Int, maxBins: Int): (DecisionTreeModel, Long) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainRegressor(trainData, Map[Int, Int](),
      "variance", maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis)
  }

  def evaluateModel(model: DecisionTreeModel, validateData: RDD[LabeledPoint]): Double = {
    val scoreAndLabels = validateData.map { data =>
      val predict = model.predict(data.features)
      (predict, data.label)
    }
    val metrics = new RegressionMetrics(scoreAndLabels)
    metrics.rootMeanSquaredError
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validateData: RDD[LabeledPoint],
                           maxDepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel = {
    val evaluationsArray =
      for (maxDepth <- maxDepthArray;
           maxBins <- maxBinsArray) yield {
        val (model, time) = trainModel(trainData, maxDepth, maxBins)
        val rmse = evaluateModel(model, validateData)
        println("params => maxDepth:" + maxDepth + ", maxBins:" + maxBins + ", RMSE:" + rmse)
        (maxDepth, maxBins, rmse)
      }
    val evalWithASC = evaluationsArray.sortBy(_._3)
    val bestEval = evalWithASC(0)
    println("Best params => maxDepth:" + bestEval._1 + ", maxBins:" + bestEval._2 + ", RMSE:" + bestEval._3)

    val (bestModel, time) = trainModel(trainData.union(validateData), bestEval._1, bestEval._2)
    bestModel
  }

  def parameterTuning(trainData: RDD[LabeledPoint],
                      validateData: RDD[LabeledPoint]): DecisionTreeModel = {
    val maxDepthArray = Array(5, 10, 15, 25, 30)
    val maxBinsArray = Array(5, 10, 50, 100, 150)
    evaluateAllParameter(trainData, validateData, maxDepthArray, maxBinsArray)
  }

  def predictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    // 1. import data
    val rawDataWithHeader = sc.textFile("data/bike/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) =>
      if (idx == 0) iter.drop(1) else iter
    }

    // 2. prepare testing data
    println("Prepare testing data ...")
    val Array(pData, oData) = rawData.randomSplit(Array(0.1, 0.9))
    val records = pData.take(20).map(_.split(","))
    records.map { fields =>
      val label = fields.last.toDouble
      val season = fields(2).toDouble
      val features = fields.slice(4, fields.length - 3).map(_.toDouble)
      val featuresVectors = Vectors.dense(season +: features)
      val predict = model.predict(featuresVectors)
      val error = math.abs(label - predict)
      println(predict, label, error)
    }
  }
}
