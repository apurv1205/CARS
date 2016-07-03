package basic.recommender

import java.io.File
import scala.io.Source
import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
import scala.collection.immutable.Map

/**
  * Created by roger19890107 on 4/4/16.
  */
object Recommend {
  def main(args: Array[String]) {
    //Log only warn
    context.Env.setLogger

    //Prepare data
    println("prepare data...")
    val (ratings, movieTitle) = prepareData()

    //Train model
    println("start train model using data count: " + ratings.count())
    val model = ALS.train(ratings, 5, 20, 0.1)
    println("finish training model...")

    //Recommend phase
    println("start recommend...")
    recommend(model, movieTitle)
  }

  def prepareData(): (RDD[Rating], Map[Int, String]) = {
    // 1. build user rating data
    val sc = context.Env.setupContext("Recommend")
    val rawUserData = sc.textFile("data/ml-100k/u.data")
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingsRDD = rawRatings.map {
      case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    println("total: " + ratingsRDD.count().toString + "ratings.")

    // 2. build movie id map to name
    println("load the items data...")
    val itemRDD = sc.textFile("data/ml-100k/u.item")
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2))
      .map(array => (array(0).toInt, array(1))).collect().toMap

    // 3. show data
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("total ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    return (ratingsRDD, movieTitle)
  }

  def recommendMovies(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputUserID: Int) = {
    val recommendMovie = model.recommendProducts(inputUserID, 10)
    println("For user: " + inputUserID + ", we recommend the movies:")
    var i = 0
    recommendMovie.foreach(r => {
      println(i.toString + "." + movieTitle(r.product) + " ratings: " + r.rating.toString)
      i += 1
    })
  }

  def recommendUsers(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputMovieID: Int) = {
    val recommendUser = model.recommendUsers(inputMovieID, 10)
    println("For movie: " + inputMovieID + " name: "+ movieTitle(inputMovieID) + ", we recommend the users:")
    var i = 0
    recommendUser.foreach(r => {
      println(i.toString + " user id: " + r.user + " ratings: " + r.rating.toString)
      i += 1
    })
  }

  def recommend(model: MatrixFactorizationModel, movieTitle: Map[Int, String]) = {
    var choose = ""
    while (choose != "3") {
      print("recommend type? 1.movies for user 2.users for movie 3.exit: ")
      choose = readLine()

      if (choose == "1") {
        print("user id?")
        val inputUserID = readLine()
        recommendMovies(model, movieTitle, inputUserID.toInt)
      } else if (choose == "2") {
        print("movie id?")
        val inputMovieID = readLine()
        recommendUsers(model, movieTitle, inputMovieID.toInt)
      }
    }
  }
}
