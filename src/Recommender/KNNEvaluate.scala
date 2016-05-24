package Recommender

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._


/**
  * Created by roger19890107 on 5/21/16.
  */
object KNNEvaluate extends App {
  Utils.Config.setLogger
  val sc = Utils.Config.setupContext("KNN Evaluate")
  val K = 25

  //==== init data ====
  println("load installations data ...")
  val rawDataWithHeader = sc.textFile("data/installations.csv")
  val rawData = rawDataWithHeader.mapPartitionsWithIndex {
    (idx, iter) => if (idx == 0) iter.drop(1) else iter
  }.map(_.split(","))

  // product map
  val pNameToIndex = rawData.map(_(0)).distinct.zipWithIndex().collectAsMap()
  val pIndexToName = Map() ++ pNameToIndex.map(_.swap)
  val pNameToIndexVar = sc.broadcast(pNameToIndex)

  // user map
  val uNameToIndex = rawData.map(_(1)).distinct.zipWithIndex().collectAsMap()
  val uIndexToName = Map() ++ uNameToIndex.map(_.swap)
  val uNameToIndexVar = sc.broadcast(uNameToIndex)

  //==== build matrix ====
  val entries = rawData.map { line =>
      val product = pNameToIndexVar.value(line(0))
      val user = uNameToIndexVar.value(line(1))
      val rating = line(2).toDouble
      MatrixEntry(user, product, rating)
  }.filter(_.value != 0)

  // R: rating matrix
  val ratingMat = new CoordinateMatrix(entries)
  val itemNum = ratingMat.numCols()
  val userNum = ratingMat.numRows()
  println("Rating Matrix: " + userNum + "*" + itemNum)

  // S: item similarity matrix
  val itemSimMat = ratingMat.toIndexedRowMatrix().columnSimilarities()
  println("Item Similarity Matrix: " + itemSimMat.numRows() + "*" + itemSimMat.numCols())

  // SK: top k similar item matrix
  println("Extract top K entries ...")
  val topKItemEntries = itemSimMat.entries
    .filter(m => m.i != m.j)
    .map(m => (m.i, m))
    .topByKey(K)(Ordering.by(_.value))
    .flatMap(_._2)
  val topKItemMat = new CoordinateMatrix(topKItemEntries, itemNum, itemNum)
  println("top K item matrix: " + topKItemMat.numRows() + "*" + topKItemMat.numCols())

  // SS: Sum of top K similarity matrix
  val sumSimEntries = topKItemEntries.map(m => (m.i, m.value)).reduceByKey(_ + _).map {
    case (i, sumValue) => MatrixEntry(i, 0, sumValue)
  }
  val sumSimMat = new CoordinateMatrix(sumSimEntries, itemNum, 1)
  println("Sum Similarity matrix: " + sumSimMat.numRows() + "*" + sumSimMat.numCols())

  // P: predict matrix
  println("Start predicting the item-based recommender ...")
  val predictMat = ratingMat.toBlockMatrix().multiply(topKItemMat.transpose().toBlockMatrix())
  println("predict matrix: " + predictMat.numRows() + "*" + predictMat.numCols())

  // NP: normalized predict
  println("Start normalize matrix ...")
  val normPredictEntries = predictMat.toCoordinateMatrix().entries.map(m => (m.j, m))
    .join(sumSimEntries.map(m => (m.i, m.value)))
    .mapValues {
      case (m, sumVal) =>
        val normVal = if (sumVal == 0) 0 else m.value / sumVal
        MatrixEntry(m.i, m.j, normVal)
    }.map(_._2)
//  val normalPredictMat = new CoordinateMatrix(normPredictEntries, userNum, itemNum)
//  println("normal predict matrix: " + normalPredictMat.numRows() + "*" + normalPredictMat.numCols())

  println("recommend for User ...")
  val topN = 10
  val userName = "1"
  val userIndex = uNameToIndex(userName)

  normPredictEntries
    .filter(_.i == userIndex)
    .map(m => (m.j, m.value))
    .top(topN)(Ordering.by(_._2))
    .foreach {
      case (index, value) => println(pIndexToName(index) + "" + value)
    }

//    .map(_._1).foreach {
//      index => println(pIndexToName(index) + "")
//    }
}
