package recommender.knn

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.mllib.stat.Statistics

object KNN {

  private def getSims(ratingMatrix: RowMatrix, topK: Int): Unit = {
    Statistics.corr(ratingMatrix.rows, "pearson")(1, 1)


  }

  private def getTopKSims(ratingMatrix: RowMatrix, topK: Int): RDD[(Int, Array[(Int, Double)])] = {
    ratingMatrix
      .columnSimilarities().entries
      .flatMap {
        case m @ MatrixEntry(i, j, value) =>
          List(m, MatrixEntry(j, i, value))
      }
      .map(m => (m.i.toInt, m))
      .topByKey(topK)(Ordering.by(_.value))
      .mapValues(iters => {
        iters.map(m => (m.j.toInt, m.value)).sortBy(_._1)
      })
  }

  def trainItemBasedRecommender(ratings: RDD[Rating], topK: Int): ItemBasedKNNModel = {
    // col is item
    val ratingMatrix = new CoordinateMatrix(ratings
      .map(r => MatrixEntry(r.user, r.product, r.rating)))
      .toRowMatrix()

    val topKProductsSims = getTopKSims(ratingMatrix, topK)

    val usersRatings = ratings.groupBy(_.user).mapValues(rats => {
      rats.map(r => (r.product, r.rating)).toArray.sortBy(_._1)
    })

    new ItemBasedKNNModel(usersRatings, topKProductsSims)
  }

  def trainUserBasedRecommender(ratings: RDD[Rating], topK: Int): UserBasedKNNModel = {
    // col is user
    val ratingMatrix = new CoordinateMatrix(ratings
      .map(r => MatrixEntry(r.product, r.user, r.rating)))
      .toRowMatrix()

    val topKUsersSims = getTopKSims(ratingMatrix, topK)

    val productsRatings = ratings.groupBy(_.product).mapValues(rats => {
      rats.map(r => (r.user, r.rating)).toArray.sortBy(_._1)
    })

    new UserBasedKNNModel(productsRatings, topKUsersSims)
  }
}

