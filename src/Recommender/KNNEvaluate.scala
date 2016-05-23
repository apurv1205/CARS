package Recommender

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD



/**
  * Created by roger19890107 on 5/21/16.
  */
object KNNEvaluate {
  def main(args: Array[String]): Unit = {
    Utils.Config.setLogger
    val sc = Utils.Config.setupContext("KNN Evaluate")
    val k = 25

    println("load installations data ...")
    val rawDataWithHeader = sc.textFile("data/installations.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    val lines = rawData.map(_.split(","))
    val pkgToIndexMap = lines.map(_(0)).distinct.zipWithIndex().collectAsMap()
    val scPkgToIndexMap = sc.broadcast(pkgToIndexMap)

    val ratings = lines.map { data =>
      val productID = scPkgToIndexMap.value(data(0)).toInt
      val userID = data(1).toInt - 1
      val rating = data(2).toDouble
      Rating(userID, productID, rating)
    }.filter(_.rating != 0)

    val entries = ratings.map(r => MatrixEntry(r.user, r.product, r.rating))
    val ratingMat = new CoordinateMatrix(entries)
    println("ratingMat: " + ratingMat.numRows() + "*" + ratingMat.numCols())


    println("Calculate similarity matrix ...")
    val simMat = ratingMat.toIndexedRowMatrix().columnSimilarities()
    val nRows = simMat.numRows()
    val nCols = simMat.numCols()

    println("Extract top K entries ...")
    val topKMatEntries = simMat.entries
      .filter(m => m.i != m.j)
      .groupBy(_.i).mapValues { iters =>
        iters.toList.sortBy(_.value).reverse.take(k)
      }.flatMap(_._2)
    val topKMatMat = new CoordinateMatrix(topKMatEntries, nRows, nCols)

    println("Multiple rating matrix to topK matrix (transpose) ...")
    val predictMat = ratingMat.toBlockMatrix.multiply(topKMatMat.toBlockMatrix().transpose)

    val reverseMap = Map() ++ pkgToIndexMap.map(_.swap)
    predictMat.toCoordinateMatrix().entries.filter(_.i == 0).sortBy(_.value * -1).take(10).foreach {
      m => println(reverseMap(m.j))
    }
  }
}
