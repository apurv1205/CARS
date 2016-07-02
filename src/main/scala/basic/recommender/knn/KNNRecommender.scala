package basic.recommender.knn

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD

/**
  * KNN recommend model includes user-based and item-based
  */
trait KNNRecommendModel {
  def predict(user: Int, product: Int): Double
  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating]
  def recommendProducts(user: Int, num: Int): Array[Rating]
  def recommendUsers(product: Int, num: Int): Array[Rating]
}

object KNNRecommendModel {
  /**
    * Compute the predict scores
    * Reference:
    *   http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf
    * @param ratings a set of ratings for user or product
    * @param similarities
    * @return
    */
  def getWeightScores(ratings: Array[(Int, Double)],
                      similarities: Array[(Int, Double)]): Double = {
    // join buy key
    val joinRatsAndSims = (ratings ++ similarities)
      .groupBy(_._1)
      .filter(_._2.length == 2)
      .mapValues(arr => (arr(0)._2, arr(1)._2))
      .values

    if (joinRatsAndSims.isEmpty) return 0

    // weight predict scores
    val weightSum = joinRatsAndSims.map(_._2).sum

    joinRatsAndSims.map {
      case (rating, similarity) => rating * similarity
    }.sum / weightSum
  }

  def recommend(ratings: Array[(Int, Double)],
                similaritiesRDD: RDD[(Int, Array[(Int, Double)])],
                num: Int): Array[(Int, Double)] = {
    similaritiesRDD.map {
      case (id, similarities) =>
        val scores = getWeightScores(ratings, similarities)
        (id, scores)
    }.sortBy(-_._2).take(num)
  }

  def recommend(ratingRDD: RDD[(Int, Array[(Int, Double)])],
                similarities: Array[(Int, Double)],
                num: Int): Array[(Int, Double)] = {
    ratingRDD.map {
      case (id, ratings) =>
        val scores = getWeightScores(ratings, similarities)
        (id, scores)
    }.sortBy(-_._2).take(num)
  }
}

/**
  * Item-based KNN recommender
  * @param usersRatings - product ratings for every user
  * @param productsSims - similarities for every product
  */
class ItemBasedKNNModel(val usersRatings: RDD[(Int, Array[(Int, Double)])],
                        val productsSims: RDD[(Int, Array[(Int, Double)])]
                       ) extends KNNRecommendModel with Serializable {

  def predict(user: Int, product: Int): Double = {
    val userRatings = usersRatings.lookup(user).head
    val productSims = productsSims.lookup(product).head

    KNNRecommendModel.getWeightScores(userRatings, productSims)
  }

  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] = {
    val users = usersRatings.join(userProducts).map {
      case (user, (ratings, product)) => (product, (user, ratings))
    }
    users.join(productsSims).map {
      case (product, ((user, ratings), sims)) =>
        Rating(user, product, KNNRecommendModel.getWeightScores(ratings, sims))
    }
  }

  def recommendProducts(user: Int, num: Int): Array[Rating] = {
    KNNRecommendModel.recommend(
      usersRatings.lookup(user).head,
      productsSims,
      num).map(t => Rating(user, t._1, t._2))
  }

  def recommendUsers(product: Int, num: Int): Array[Rating] = {
    KNNRecommendModel.recommend(
      usersRatings,
      productsSims.lookup(product).head,
      num).map(t => Rating(t._1, product, t._2))
  }

  def similarProducts(product: Int, num: Int): Array[(Int, Double)] = {
    productsSims.lookup(product).head
      .sortBy(-_._2).take(num)
  }
}

/**
  * User-based KNN recommender
  * @param productsRatings
  * @param usersSims
  */
class UserBasedKNNModel(val productsRatings: RDD[(Int, Array[(Int, Double)])],
                        val usersSims: RDD[(Int, Array[(Int, Double)])]
                       ) extends KNNRecommendModel with Serializable {

  def predict(user: Int, product: Int): Double = {
    val prodRatings = productsRatings.lookup(product).head
    val userSims = usersSims.lookup(user).head

    KNNRecommendModel.getWeightScores(prodRatings, userSims)
  }

  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating] = {
    val products = productsRatings.join(userProducts.map(_.swap)).map {
      case (product, (ratings, user)) => (user, (product, ratings))
    }
    products.join(usersSims).map {
      case (user, ((product, ratings), sims)) =>
        Rating(user, product, KNNRecommendModel.getWeightScores(ratings, sims))
    }
  }

  def recommendProducts(user: Int, num: Int): Array[Rating] = {
    KNNRecommendModel.recommend(
      productsRatings,
      usersSims.lookup(user).head,
      num).map(t => Rating(user, t._1, t._2))
  }

  def recommendUsers(product: Int, num: Int): Array[Rating] = {
    KNNRecommendModel.recommend(
      productsRatings.lookup(product).head,
      usersSims,
      num).map(t => Rating(t._1, product, t._2))
  }

  def getSimilarUsers(user: Int, num: Int): Array[(Int, Double)] = {
    usersSims.lookup(user).head
      .sortBy(_._2).take(num)
  }
}
