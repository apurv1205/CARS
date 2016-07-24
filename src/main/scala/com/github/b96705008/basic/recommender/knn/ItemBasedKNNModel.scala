package com.github.b96705008.basic.recommender.knn

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import KNNRecommendModel._
import org.apache.spark.SparkContext

/**
  * Item-based KNN recommender (Product-based)
  *
  * @param usersRatings - product ratings for every user
  * @param productsSims - similarities for every product
  */
class ItemBasedKNNModel(val usersRatings: RDD[(Int, Array[(Int, Double)])],
                        val productsSims: RDD[(Int, Array[(Int, Double)])]
                       ) extends KNNRecommendModel {

  def predict(user: Int, product: Int): Double = {
    val userRatings = usersRatings.lookup(user).head
    val productSims = productsSims.lookup(product).head

    getWeightScores(userRatings, productSims)
  }

  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] = {
    val users = usersRatings.join(userProducts).map {
      case (user, (ratings, product)) => (product, (user, ratings))
    }
    users.join(productsSims).map {
      case (product, ((user, ratings), sims)) =>
        Rating(user, product, getWeightScores(ratings, sims))
    }
  }

  def recommendProducts(user: Int, num: Int): Array[Rating] = {
    recommend(
      usersRatings.lookup(user).head,
      productsSims,
      num).map(t => Rating(user, t._1, t._2))
  }

  def recommendUsers(product: Int, num: Int): Array[Rating] = {
    recommend(
      usersRatings,
      productsSims.lookup(product).head,
      num).map(t => Rating(t._1, product, t._2))
  }

  def getSimilarProducts(product: Int, num: Int): Array[(Int, Double)] = {
    productsSims.lookup(product).head
      .sortBy(-_._2).take(num)
  }

  def save(path: String): Unit = {
    saveRelationRDD(ratingsPath(path), usersRatings)
    saveRelationRDD(similaritiesPath(path), productsSims)
  }
}

object ItemBasedKNNModel {
  def load(sc: SparkContext, path: String): UserBasedKNNModel = {
    val usersRatings = loadRelationRDD(sc, ratingsPath(path))
    val productsSims = loadRelationRDD(sc, similaritiesPath(path))
    new UserBasedKNNModel(usersRatings, productsSims)
  }
}

