package basic.recommender.knn

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._

/**
  * Created by roger19890107 on 7/2/16.
  */
object KNN {

  private def getTopKSims(ratingMatrix: CoordinateMatrix, topK: Int): RDD[(Int, Array[(Int, Double)])] = {
    ratingMatrix.toRowMatrix()
      .columnSimilarities().entries
      .flatMap {
        case m @ MatrixEntry(i, j, value) =>
          List(m, MatrixEntry(j, i, value))
      }
      .map(m => (m.i.toInt, m))
      .topByKey(topK)(Ordering.by(_.value))
      .mapValues(iters => {
        iters.map(m => (m.j.toInt, m.value))
      })
  }

  def trainItemBasedRecommender(ratings: RDD[Rating], topK: Int): ItemBasedKNNModel = {
    // col is item
    val ratingMatrix = new CoordinateMatrix(ratings
      .map(r => MatrixEntry(r.user, r.product, r.rating)))

    val topKProductsSims = getTopKSims(ratingMatrix, topK)

    val usersRatings = ratings.groupBy(_.user).mapValues(rats => {
      rats.map(r => (r.product, r.rating)).toArray
    })

    new ItemBasedKNNModel(usersRatings, topKProductsSims)
  }

  def trainUserBasedRecommender(ratings: RDD[Rating], topK: Int): UserBasedKNNModel = {
    // col is user
    val ratingMatrix = new CoordinateMatrix(ratings
      .map(r => MatrixEntry(r.product, r.user, r.rating)))

    val topKUsersSims = getTopKSims(ratingMatrix, topK)

    val productsRatings = ratings.groupBy(_.product).mapValues(rats => {
      rats.map(r => (r.user, r.rating)).toArray
    })

    new UserBasedKNNModel(productsRatings, topKUsersSims)
  }
}

