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
    context.Env.setLogger
    val sc = context.Env.setupContext("KNN Recommend")

    // dataset
    val (trainData, validateData, testData) = prepareDate(sc)
    trainData.cache()
    validateData.cache()
    testData.cache()

    // recommend example
    evaluateAllParameter(trainData, validateData, Array(200, 400, 600, 800, 1000))

    trainData.unpersist()
    validateData.unpersist()
    testData.unpersist()
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
    val Array(trainData, validateData, testData) =
      ratingsRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("trainData: " + trainData.count() +
      " validationData: " + validateData.count() +
      " testData: " + testData.count())

    (trainData, validateData, testData)
  }

  def recommendExample(ratingRDD: RDD[Rating]): Unit = {
    //val model = KNN.trainItemBasedRecommender(trainData, 500)
    val model = KNN.trainUserBasedRecommender(ratingRDD, 500)
    //val modelToSave = KNN.trainUserBasedRecommender(trainData, 500)
    //modelToSave.save("data/models/user-based-knn")
    //val model = UserBasedKNNModel.load(sc, "data/models/user-based-knn")

    println("user 196 with product 242, rating: 3 and predict: " + model.predict(196, 242))
    println("user 22 with product 377, rating: 1 and predict: " + model.predict(22, 377))

    //    val userProducts = sc.parallelize(List((196, 242), (22, 377)))
    //    model.predict(userProducts).foreach(println)

    println("recommend products for user 10:")
    model.recommendProducts(196, 15).foreach(println)

    println("recommend user for Product 242:")
    model.recommendUsers(242, 15).foreach(println)

    //println("similar products for product 242:")
    //model.getSimilarProducts(242, 10).foreach(println)
  }

  def computeRMSE(model: KNNRecommendModel, ratingRDD: RDD[Rating]): Double = {
    val predictedRDD = model.predict(ratingRDD.map(r => (r.user, r.product)))
    val predictedAndRatings = predictedRDD.map(p => ((p.user, p.product), p.rating))
      .join(ratingRDD.map(r => ((r.user, r.product), r.rating)))
      .values
    val num = ratingRDD.count()
    math.sqrt(predictedAndRatings
      .map(x => (x._1 - x._2) * (x._1 - x._2))
      .reduce(_+_) / num)
  }

  def evaluateAllParameter(trainData: RDD[Rating], validateData: RDD[Rating],
                           topKArray: Array[Int]): Unit = {
    val evaluations = for (topK <- topKArray) yield {
      val userModel = KNN.trainUserBasedRecommender(trainData, topK)
      val rmseInUser = computeRMSE(userModel, validateData)

      val itemModel = KNN.trainItemBasedRecommender(trainData, topK)
      val rmseInItem = computeRMSE(itemModel, validateData)

      println(f"topK:$topK%3d => User-RMSE:$rmseInUser%.2f, Item-RMSE:$rmseInItem%.2f")
    }
  }


  def compareModel(trainData: RDD[Rating], testData: RDD[Rating]): Unit = {


    val userModel = KNN.trainUserBasedRecommender(trainData, 500)
    val rmseInUser = computeRMSE(userModel, testData)
    println("user rmse: " + rmseInUser)

    val itemModel = KNN.trainItemBasedRecommender(trainData, 500)
    val rmseInItem = computeRMSE(itemModel, testData)
    println("item rmse: " + rmseInItem)
  }
}