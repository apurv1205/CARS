package basic.recommender

import context.Chart

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.DateTime
import org.joda.time.Duration

/**
  * Created by roger19890107 on 4/5/16.
  */
object AlsEvaluation {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/hadoop-spark/ml-100k/"

  def main(args: Array[String]) {
    //Log only warn
    context.Config.setLogger

    // Prepare data
    println("prepare data...")
    val (trainData, validationData, testData) = prepareDate()
    trainData.persist()
    validationData.persist()
    testData.persist()

    // Train & Validate data
    println("Train & Validate data...")
    val bestModel = trainValiation(trainData, validationData)

    // Test phase
    println("testing the data...")
    val testRMSE = computeRMSE(bestModel, testData)
    println("Use best model to test data, and the RMSE is " + testRMSE)

    //cancel cache
    trainData.unpersist()
    validationData.unpersist()
    testData.unpersist()
  }

  def prepareDate(): (RDD[Rating], RDD[Rating], RDD[Rating]) = {
    // 1. build user rating data
    val sc = context.Config.setupContext("Recommend")
    val rawUserData = sc.textFile(base + "u.data")
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingsRDD = rawRatings.map {
      case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    println("total: " + ratingsRDD.count().toString + "ratings.")

    // 3. show data
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("total ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)

    // 4. split to training, validate and testing data
    val Array(trainData, validationData, testData) =
      ratingsRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("trainData: " + trainData.count() +
            " validationData: " + validationData.count() +
            " testData: " + testData.count())

    return (trainData, validationData, testData)
  }

  def computeRMSE(model: MatrixFactorizationModel, ratingRDD: RDD[Rating]): Double = {
    val predictedRDD = model.predict(ratingRDD.map(r => (r.user, r.product)))
    val predictedAndRatings = predictedRDD.map(p => ((p.user, p.product), p.rating))
        .join(ratingRDD.map(r => ((r.user, r.product), r.rating)))
        .values
    val num = ratingRDD.count()
    math.sqrt(predictedAndRatings
      .map(x => (x._1 - x._2) * (x._1 - x._2))
      .reduce(_+_) / num)
  }

  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating],
                  rank: Int, iterations: Int, lambda: Double): (Double, Double) = {
    val startTime = new DateTime()
    val model = ALS.train(trainData, rank, iterations, lambda)
    val endTime = new DateTime()
    val rmse = computeRMSE(model, validationData)
    val duration = new Duration(startTime, endTime)
    println(f"train parameters => rank:$rank%3d, iterations:$iterations%.2f, lambda:$lambda%.2f Reslult => RMSE:$rmse%.2f" +
      " training time:" + duration.getMillis + " million seconds.")
    (rmse, duration.getStandardSeconds)
  }

  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                        evaluateParameter: String, rankArray: Array[Int],
                        numIterationsArray: Array[Int], lambdaArray: Array[Double]) = {
    var dataBarChart = new DefaultCategoryDataset
    var dataLineChart = new DefaultCategoryDataset
    for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) {
      val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)
      val parameterData = evaluateParameter match {
        case "rank" => rank
        case "numIterations" => numIterations
        case "lambda" => lambda
      }
      dataBarChart.addValue(rmse, evaluateParameter, parameterData.toString)
      dataLineChart.addValue(time, "Time", parameterData.toString)
    }
    Chart.plotBarLineChart("ALS Evaluation " + evaluateParameter,
      evaluateParameter, "RMSE", 0, 5, "Time", dataBarChart, dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                           rankArray: Array[Int], numIterationsArray: Array[Int],
                           lambdaArray: Array[Double]): MatrixFactorizationModel = {
    val evaluations = for (rank <- rankArray; numIterations <- numIterationsArray;
                           lambda <- lambdaArray) yield {
      val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)
      (rank, numIterations, lambda, rmse)
    }

    // find the eval with the smallest rmse
    val eval = evaluations.sortBy(_._4)
    val bestEval = eval(0)
    println("The best group of parameters for model => rank:" + bestEval._1 +
      ", iterations:" + bestEval._2 + ", lambda:" + bestEval._3)
    val bestModel = ALS.train(trainData, bestEval._1, bestEval._2, bestEval._3)
    return bestModel
  }

  def trainValiation(trainData: RDD[Rating], validationData: RDD[Rating]):
    MatrixFactorizationModel = {
    println("--evaluate the rank params--")
    evaluateParameter(trainData, validationData, "rank",
      Array(5, 10, 15, 20, 50, 100), Array(10), Array(0.1))

    println("--evaluate the numIterations params--")
    evaluateParameter(trainData, validationData, "numIterations",
      Array(10), Array(5, 10, 15, 20, 25), Array(0.1))

    println("--evaluate the lambda params--")
    evaluateParameter(trainData, validationData, "lambda",
      Array(10), Array(10), Array(0.05, 0.1, 1.0, 5.0, 10.0))

    println("--evaluate all the combination of params--")
    val bestModel = evaluateAllParameter(trainData, validationData,
      Array(5, 10, 15, 20, 25),
      Array(5, 10, 15, 20, 25),
      Array(0.05, 0.1, 1.0, 5.0, 10.0))

    return bestModel
  }
}
