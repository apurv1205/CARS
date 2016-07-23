package aas.ch6_lsa

import org.apache.spark.{SparkConf, SparkContext}
import ParseWikipedia._
import org.apache.spark.mllib.linalg.{SingularValueDecomposition, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._

import breeze.linalg.{
  DenseMatrix => BDenseMatrix,
  DenseVector => BDenseVector,
  SparseVector => BSparseVector
}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer


/**
  * Created by roger19890107 on 6/26/16.
  */
object RunLSA {
  //val wikiDumpPath = "/Volumes/TOS_Mac/WikiDump/enwiki-20160501-pages-articles-multistream.xml"

  def main(args: Array[String]) {
    // default args
    val k = 10
    val numTerms = 5000
    val sampleSize = 0.00001

    // spark config
    context.Env.setLogger
    val conf = new SparkConf()
      .setAppName("Wiki LSA")
      .setMaster("local[*]")
      .setExecutorEnv("driver-memory", "6g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint/")

    // term vectors
    val wikiVSM = new WikiVSM
    wikiVSM.loadAndParse(sc, sampleSize, numTerms)
    //val wikiVSM = WikiVSM.load(sc)

    val termVecs = wikiVSM.termVecs.get.cache()
    val termIds = wikiVSM.termIds
    val docIds = wikiVSM.docIds
    val idfs = wikiVSM.idfs

    // SVD
    val mat = new RowMatrix(termVecs)
    val svd = mat.computeSVD(k, computeU = true)

    // Top concepts
    println("Singular values: " + svd.s)
    val topConceptTerms = topTermsInTopConcepts(svd, 4, 10, termIds)
    val topConceptDocs = topDocsInTopConcepts(svd, 4, 10, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      println("Concept docs: " + docs.map(_._1).mkString(", "))
      println()
    }

    // terms for term
    val VS = multiplyByDiagonalMatrix(svd.V, svd.s)
    val normalizedVS = rowsNormalized(VS)
    printTopTermsForTerm(normalizedVS, "algorithm", termIds)
    printTopTermsForTerm(normalizedVS, "radiohead", termIds)

    // docs for doc
    val US = multiplyByDiagonalMatrix(svd.U, svd.s)
    val normalizedUS = rowsNormalized(US)
    printTopDocsForDoc(normalizedUS, "Romania", docIds)
    printTopDocsForDoc(normalizedUS, "Brad Pitt", docIds)

    // docs for term
    printTopDocsForTerm(US, svd.V, "fir", termIds, docIds)
    printTopDocsForTerm(US, svd.V, "graph", termIds, docIds)

    // docs for terms query
    printTopDocsForTermQuery(US, svd.V,
      Seq("factorization", "decomposition"), termIds, idfs, docIds)
  }

  // Concepts
  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                            numTerms: Int, termIds: Map[Int, String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map{case (score, id) => (termIds(id), score)}
    }
    topTerms
  }

  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                           numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId()
      topDocs += docWeights.top(numDocs).map {
        case (score, id) => (docIds(id), score)
      }
    }
    topDocs
  }

  // Print function
  def printIdWeights[T](idWeights: Seq[(Double, T)], entityIds: Map[T, String]): Unit = {
    println(idWeights.map{case (score, id) => (entityIds(id), score)}.mkString(", "))
  }

  // Term * Term relevance
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = {
    (0 until mat.cols).map(c => mat(index, c))
  }

  def topTermsForTerm(normalizedVS: BDenseMatrix[Double], termId: Int): Seq[(Double, Int)] = {
    val rowVec = new BDenseVector[Double](row(normalizedVS, termId).toArray)
    val termScores = (normalizedVS * rowVec).toArray.zipWithIndex
    termScores.sortBy(-_._1).take(10)
  }

  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs {case ((r, c), v) => v * sArr(c)}
  }

  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).foreach(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  def printTopTermsForTerm(normalizedVS: BDenseMatrix[Double],
                           term: String, termIds: Map[Int, String]): Unit = {
    println("Top terms for " + term)
    val idTerms = termIds.map(_.swap)
    printIdWeights(topTermsForTerm(normalizedVS, idTerms(term)), termIds)
  }

  // Doc * Doc relevance
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId().map(_.swap).lookup(id).head.toArray
  }

  def topDocsForDoc(normalizedUS: RowMatrix, docId: Long): Seq[(Double, Long)] = {
    val docRowArr = row(normalizedUS, docId)
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    val docScores = normalizedUS.multiply(docRowVec)
    docScores.rows.map(_.toArray(0)).zipWithUniqueId()
      .filter(!_._1.isNaN).top(10)
  }

  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = vecArr.indices.map(i => vecArr(i) * sArr(i)).toArray
      Vectors.dense(newArr)
    }))
  }

  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

  def printTopDocsForDoc(normalizedUS: RowMatrix,
                         doc: String, docIds: Map[Long, String]): Unit = {
    val idDocs = docIds.map(_.swap)
    printIdWeights(topDocsForDoc(normalizedUS, idDocs(doc)), docIds)
  }

  // Docs * term
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    val rowNum = mat.numRows
    (0 until rowNum).map(i => arr(index + i * rowNum))
  }

  def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int): Seq[(Double, Long)] = {
    val termRowArr = row(V, termId).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    val docScores = US.multiply(termRowVec)
    docScores.rows.map(_.toArray(0)).zipWithUniqueId().top(10)
  }

  def printTopDocsForTerm(US: RowMatrix, V: Matrix, term: String,
                          termIds: Map[Int, String],
                          docIds: Map[Long, String]): Unit = {
    val idTerms = termIds.map(_.swap)
    printIdWeights(topDocsForTerm(US, V, idTerms(term)), docIds)
  }

 // Multi-terms query
  def termsToQueryVector(terms: Seq[String],
                         idTerms: Map[String, Int],
                         idfs: Map[String, Double]): BSparseVector[Double] = {
   val indices = terms.map(idTerms(_)).toArray
   val values = terms.map(idfs(_)).toArray
   new BSparseVector[Double](indices, values, idTerms.size)
  }

  def topDocsForTermQuery(US: RowMatrix,
                          V: Matrix,
                          query: BSparseVector[Double]): Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
    val termRowArr = (breezeV.t * query).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    val docScores = US.multiply(termRowVec)
    docScores.rows.map(_.toArray(0)).zipWithUniqueId().top(10)
  }

  def printTopDocsForTermQuery(US: RowMatrix, V: Matrix, terms: Seq[String],
                               termIds: Map[Int, String], idfs: Map[String, Double],
                               docIds: Map[Long, String]) = {

    val idTerms = termIds.map(_.swap)
    val query = termsToQueryVector(terms, idTerms, idfs)
    printIdWeights(topDocsForTermQuery(US, V, query), docIds)
  }
}

