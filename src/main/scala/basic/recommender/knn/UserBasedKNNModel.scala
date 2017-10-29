package recommender.knn

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import KNNRecommendModel._

/**
  * User-based KNN recommender
  *
  * @param productsRatings Ratings group by products
  * @param usersSims Cosine similarities to nearest neighbors by every user
  */
class UserBasedKNNModel(val productsRatings: RDD[(Int, Array[(Int, Double)])],
                        val usersSims: RDD[(Int, Array[(Int, Double)])]
                       ) extends KNNRecommendModel {


  def predict(user: Int, product: Int): Double = {
    val prodRatings = productsRatings.lookup(product).head
    val userSims = usersSims.lookup(user).head

    getWeightScores(prodRatings, userSims)
  }

  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] = {
    val products = productsRatings.join(userProducts.map(_.swap)).map {
      case (product, (ratings, user)) => (user, (product, ratings))
    }
    products.join(usersSims).map {
      case (user, ((product, ratings), sims)) =>
        Rating(user, product, getWeightScores(ratings, sims))
    }
  }

  def recommendProducts(user: Int, num: Int): Array[Rating] = {
    recommend(
      productsRatings,
      usersSims.lookup(user).head,
      num).map(t => Rating(user, t._1, t._2))
  }

  def recommendUsers(product: Int, num: Int): Array[Rating] = {
    recommend(
      productsRatings.lookup(product).head,
      usersSims,
      num).map(t => Rating(t._1, product, t._2))
  }

  def getSimilarUsers(user: Int, num: Int): Array[(Int, Double)] = {
    usersSims.lookup(user).head
      .sortBy(_._2).take(num)
  }

  def save(path: String): Unit = {
    saveRelationRDD(ratingsPath(path), productsRatings)
    saveRelationRDD(similaritiesPath(path), usersSims)
  }
}

object UserBasedKNNModel {

  def load(sc: SparkContext, path: String): UserBasedKNNModel = {
    val productsRatings = loadRelationRDD(sc, ratingsPath(path))
    val usersSims = loadRelationRDD(sc, similaritiesPath(path))
    new UserBasedKNNModel(productsRatings, usersSims)
  }
}


