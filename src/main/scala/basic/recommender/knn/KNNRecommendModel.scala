package basic.recommender.knn


import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Logging, SparkContext}

/**
  * KNN recommend model includes user-based and item-based
  */
trait KNNRecommendModel extends Serializable with Logging {
  def predict(user: Int, product: Int): Double
  def predict(userProducts: RDD[(Int, Int)]): RDD[Rating]
  def recommendProducts(user: Int, num: Int): Array[Rating]
  def recommendUsers(product: Int, num: Int): Array[Rating]

  def save(path: String): Unit
}

object KNNRecommendModel {
  /**
    * Compute the predict scores
    * Reference:
    *   http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf
    * @param ratings A set of ratings for user or product
    * @param similarities A set of similarities to each entity
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

  private[knn] def ratingsPath(path: String) = path + "/ratings"

  private[knn] def similaritiesPath(path: String) = path + "/similarities"

  private[knn] def loadRelationRDD(sc: SparkContext, rddPath: String): RDD[(Int, Array[(Int, Double)])] = {
    sc.objectFile[(Int, Array[(Int, Double)])](rddPath)
  }

  private[knn] def saveRelationRDD(rddPath: String,
                      relationRDD: RDD[(Int, Array[(Int, Double)])]): Unit = {
    relationRDD.saveAsObjectFile(rddPath)
  }
}
