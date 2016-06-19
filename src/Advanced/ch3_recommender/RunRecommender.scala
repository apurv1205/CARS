package advanced.ch3_recommender

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

import scala.collection.Map
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by roger19890107 on 6/10/16.
  */
object RunRecommender {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/aas/ch3-music/"

  def main(args: Array[String]) {
    context.Config.setLogger
    val sc = context.Config.setupContext("Recommender")
    val rawUserArtistData = sc.textFile(base + "user_artist_data.txt")
    val rawArtistData = sc.textFile(base + "artist_data.txt")
    val rawArtistAlias = sc.textFile(base + "artist_alias.txt")

    //preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
    //model(sc, rawUserArtistData, rawArtistData, rawArtistAlias)
    //evaluate(sc, rawUserArtistData, rawArtistAlias)
    recommend(sc, rawUserArtistData, rawArtistData, rawArtistAlias)
  }

  /* preparation */
  def buildArtistByID(rawArtistData: RDD[String]) =
    rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some(id.toInt, name.trim)
        } catch {
          case e: NumberFormatException => None
        }
      }
    }

  def buildArtistAlias(rawArtistAlias: RDD[String]): Map[Int, Int] =
    rawArtistAlias.flatMap { line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()

  def preparation(rawUserArtistData: RDD[String],
                  rawArtistData: RDD[String],
                  rawArtistAlias: RDD[String]): Unit = {
    val userIDStats = rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
    val itemIDStats = rawUserArtistData.map(_.split(' ')(1).toDouble).stats()
    println(userIDStats)
    println(itemIDStats)

    val artistByID = buildArtistByID(rawArtistData)
    val artistAlias = buildArtistAlias(rawArtistAlias)

    val (badID, goodID) = artistAlias.head
    println(artistByID.lookup(badID).head + " -> " + artistByID.lookup(goodID).head)
  }

  /* model */
  def buildRatings(rawUserArtistData: RDD[String],
                   bArtistAlias: Broadcast[Map[Int, Int]]) =
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val fintalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, fintalArtistID, count)
    }

  def model(sc: SparkContext,
            rawUserArtistData: RDD[String],
            rawArtistData: RDD[String],
            rawArtistAlias: RDD[String]): Unit = {
    val bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))

    val trainData = buildRatings(rawUserArtistData, bArtistAlias).cache()

    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    trainData.unpersist()

    println(model.userFeatures.mapValues(_.mkString(", ")).first())

    val userID = 2093760
    val recommendations = model.recommendProducts(userID, 5)
    recommendations.foreach(println)
    val recommendProductIDs = recommendations.map(_.product).toSet

    val rawArtistsForUser = rawUserArtistData.map(_.split(' '))
      .filter { case Array(user, _, _) => user.toInt == userID }

    val existingProducts = rawArtistsForUser.map {case Array(_, artist, _) => artist.toInt}
      .collect().toSet

    val artistByID = buildArtistByID(rawArtistData)

    println("User 2093760 has listened ...")
    artistByID.filter { case (id, name) => existingProducts.contains(id) }
      .values.collect().foreach(println)

    println("Recommend for user 2093760 ...")
    artistByID.filter { case (id, name) => recommendProductIDs.contains(id) }
      .values.collect().foreach(println)

    unpersist(model)
  }

  def areaUnderCurve(positiveData: RDD[Rating],
                     bAllItemIDs: Broadcast[Array[Int]],
                     predictFunction: RDD[(Int, Int)] => RDD[Rating]) = {

    val postiveUserProducts = positiveData.map(r => (r.user, r.product))

    val postivePredictions = predictFunction(postiveUserProducts).groupBy(_.user)

    val negativeUserProducts = postiveUserProducts.groupByKey().mapPartitions {

      userIDAndPosItemIDs => {
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0

          while (i < allItemIDs.length && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.length))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          negative.map(itemID => (userID, itemID))
        }
      }
    }.flatMap(t => t)

    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    postivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        var correct = 0L
        var total = 0L

        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }

        correct.toDouble / total
    }.mean()
  }

  def predictMostListened(sc: SparkContext,
                          train: RDD[Rating])(allData: RDD[(Int, Int)]) = {
   val bListenCount =
     sc.broadcast(train.map(r => (r.product, r.rating))
       .reduceByKey(_ + _).collectAsMap())

   allData.map { case (user, product) =>
     Rating(user, product, bListenCount.value.getOrElse(product, 0.0))
   }
  }

  def evaluate(sc: SparkContext,
               rawUserArtistData: RDD[String],
               rawArtistAlias: RDD[String]): Unit = {

    val bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))

    val allData = buildRatings(rawUserArtistData, bArtistAlias)
    val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    val allItemIDs = allData.map(_.product).distinct().collect()
    val bAllItemIDs = sc.broadcast(allItemIDs)

    val mostListenedAUC = areaUnderCurve(cvData, bAllItemIDs,
      predictMostListened(sc, trainData))
    println("mostListenedAUC: " + mostListenedAUC)

    val evaluations =
      for (rank <- Array(10, 50);
           lambda <- Array(1.0, 0.0001);
           alpha <- Array(1.0, 40.0))
        yield {
          val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
          val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
          unpersist(model)
          println(rank, lambda, alpha, auc)
          ((rank, lambda, alpha), auc)
        }

    evaluations.sortBy(_._2).reverse.foreach(println)
    trainData.unpersist()
    cvData.unpersist()
  }

  def recommend(sc: SparkContext,
                rawUserArtistData: RDD[String],
                rawArtistData: RDD[String],
                rawArtistAlias: RDD[String]): Unit = {
    val bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    val allData = buildRatings(rawUserArtistData, bArtistAlias)
    val model = ALS.trainImplicit(allData, 50, 10, 1.0, 40.0)
    allData.unpersist()

    println("recommend for some users ...")
    val someUsers = allData.map(_.user).distinct().take(100)
    val someRecommendations = someUsers.map(userID => model.recommendProducts(userID, 5))
    someRecommendations.map { recs =>
      recs.head.user + " -> " + recs.map(_.product).mkString(", ")
    }.foreach(println)

    unpersist(model)
  }

  def unpersist(model: MatrixFactorizationModel): Unit = {
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()
  }
}
