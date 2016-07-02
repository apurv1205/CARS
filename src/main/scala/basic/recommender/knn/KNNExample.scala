package basic.recommender.knn

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

/**
  * Created by roger19890107 on 7/2/16.
  */
object KNNExample {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/hadoop-spark/ml-100k/"

  def main(args: Array[String]) {
    context.Config.setLogger
    val sc = context.Config.setupContext("KNN Recommend")

    // dataset
    val (trainData, validationData, testData) = prepareDate(sc)

    val model = KNN.trainItemBasedRecommender(trainData, 500)

    println("user 196 with product 242, rating: 3 and predict: " + model.predict(196, 242))
    println("user 22 with product 377, rating: 1 and predict: " + model.predict(22, 377))

    val userProducts = sc.parallelize(List((196, 242), (22, 377)))
    model.predict(userProducts).foreach(println)

    println("recommend products for user 10:")
    model.recommendProducts(196, 15).foreach(println)

    println("recommend user for Product 242:")
    model.recommendUsers(242, 15).foreach(println)

    println("similar products for product 242:")
    model.similarProducts(242, 10).foreach(println)
  }

  def prepareDate(sc: SparkContext): (RDD[Rating], RDD[Rating], RDD[Rating]) = {
    // 1. build user rating data
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

    (trainData, validationData, testData)
  }
}