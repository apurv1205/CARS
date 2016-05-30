package Recommender

import Context.Context
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._


/**
  * Created by roger19890107 on 5/21/16.
  */
object KNNEvaluate extends App {
  Context.setLogger
  val (sc, _) = Context.initSpark("KNN Evaluate")
  val K = 2
  val uIndex = 0
  val pIndex = 1

  //==== init data ====
  println("load installations data ...")
  //val rawDataWithHeader = sc.textFile("data/installations.csv")
  val rawDataWithHeader = sc.textFile("data/behavior.csv")
  val rawData = rawDataWithHeader.mapPartitionsWithIndex {
    (idx, iter) => if (idx == 0) iter.drop(1) else iter
  }.map(_.split(","))

  // product map
  val pNameToIndex = rawData.map(_(pIndex)).distinct.zipWithIndex().collectAsMap()
  val pIndexToName = Map() ++ pNameToIndex.map(_.swap)
  val pNameToIndexVar = sc.broadcast(pNameToIndex)

  // user map
  val uNameToIndex = rawData.map(_(uIndex)).distinct.zipWithIndex().collectAsMap()
  val uIndexToName = Map() ++ uNameToIndex.map(_.swap)
  val uNameToIndexVar = sc.broadcast(uNameToIndex)

  //==== build matrix ====
  val entries = rawData.map { line =>
    val product = pNameToIndexVar.value(line(pIndex))
    val user = uNameToIndexVar.value(line(uIndex))
    val rating = line(2).toDouble
    MatrixEntry(user, product, rating)
  }.filter(_.value != 0)

  // R: rating matrix
  val ratingMat = new CoordinateMatrix(entries)
  val itemNum = ratingMat.numCols()
  val userNum = ratingMat.numRows()
  println("Rating Matrix: " + userNum + "*" + itemNum)

  // S: item similarity matrix
  println("Calculate item similarities ...")
  val partialSimMat = ratingMat.toIndexedRowMatrix().columnSimilarities()
  val itemSimEntries = partialSimMat.entries.flatMap {
    case m @ MatrixEntry(i, j, value) if i != j =>
      List(m, MatrixEntry(j, i, value))
  }

  // SK: top k similar item matrix
  println("Extract top K entries ...")
  val topKItemEntries = itemSimEntries
    .filter(m => m.i != m.j)
    .map(m => (m.i, m))
    .topByKey(K)(Ordering.by(_.value))
    .flatMap(_._2)
  val topKItemMat = new CoordinateMatrix(topKItemEntries, itemNum, itemNum)
  println("top K item matrix: " + topKItemMat.numRows() + "*" + topKItemMat.numCols())

  // SS: Sum of top K similarity matrix
  println("Sum Similarity matrix ...")
  val sumSimEntries = topKItemEntries.map(m => (m.i, m.value)).reduceByKey(_ + _).map {
    case (i, sumValue) => MatrixEntry(i, 0, sumValue)
  }

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
        //val normVal = m.value
        MatrixEntry(m.i, m.j, normVal)
    }.map(_._2)

  println("recommend for User ...")
  val topN = 5
  val userName = "u1"
  val userIndex = uNameToIndex(userName)

  normPredictEntries
    .filter(_.i == userIndex)
    .map(m => (m.j, m.value))
    .top(topN)(Ordering.by(_._2))
    .foreach {
      case (index, value) => println(pIndexToName(index) + " - " + value)
    }
}
