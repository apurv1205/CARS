package recommender.knn


import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

/**
  * KNN recommend model includes user-based and item-based
  */
trait KNNRecommendModel extends Serializable {
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
    var scores: Double = 0.0
    var total: Double = 0.0
    var i = 0
    var j = 0
    while (i < ratings.length && j < similarities.length) {
      val rat = ratings(i)
      val sim = similarities(j)
      if (rat._1 == sim._1) {
        scores += (rat._2 * sim._2)
        total += sim._2
        i += 1
        j += 1
      } else if (rat._1 < sim._1) {
        i += 1
      } else {
        j += 1
      }
    }

    if (total == 0.0) 0.0 else scores / total
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
