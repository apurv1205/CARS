package aas.ch6_lsa

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import ParseWikipedia._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.Map
import scala.collection.mutable.HashMap

/**
  * Created by roger19890107 on 7/1/16.
  */
object WikiConfig {
  val wikiDumpPath = "/Volumes/TOS_Mac/WikiDump/enwiki-20160501-pages-articles-multistream.xml"
  val fileRoot= "/Volumes/RogerDrive/Developer/dataset/aas/ch6-wiki/"
  val stopWordsFile = fileRoot + "stopwords.txt"
  val docfreqsFile = fileRoot + "docfreqs.tsv"
  val parsedObjPath = fileRoot + "wiki-parse/"
  val objSerFile = parsedObjPath + "wiki.ser"
  val vectorsPath = parsedObjPath + "vectors"
}

class WikiVSM extends Serializable {
  import WikiConfig._

  @transient var termVecs: Option[RDD[Vector]] = None
  var termIds: Map[Int, String] = Map()
  var docIds: Map[Long, String] = Map()
  var idfs: Map[String, Double] = Map()

  // TF-Idf Matrix
  def documentTermMatrix(sc: SparkContext,
                         docs: RDD[(String, Seq[String])],
                         stopWords: Set[String],
                         numTerms: Int) = {
    // doc title -> map of (term, counts)
    val docTermFreqs = docs.mapValues(terms => {
      terms.foldLeft(new HashMap[String, Int]()) {
        (map, term) => map += term -> map.getOrElse(term, 1)
      }
    })
    docTermFreqs.cache()

    // doc Id -> title
    val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap()

    // document frequecies, term -> freq
    val docFreqs = documentFrequenciesDistributed(docTermFreqs.map(_._2), numTerms)
    println("Number of terms: " + docFreqs.length)
    saveDocFreqs(docfreqsFile, docFreqs)

    // term -> idf
    val numDocs = docIds.size
    val idfs = inverseDocumentFrequecies(docFreqs, numDocs)

    // map term to index
    val idTerms = idfs.keys.zipWithIndex.toMap
    val termIds = idTerms.map(_.swap)

    // vectors
    val bIdfs = sc.broadcast(idfs)
    val bIdTerms = sc.broadcast(idTerms)

    val vecs = docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bIdTerms.value.contains(term)
      }.map {
        case (term, freq) =>
          (bIdTerms.value(term), bIdfs.value(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      Vectors.sparse(bIdTerms.value.size, termScores)
    })

    this.termVecs = Some(vecs)
    this.termIds = termIds
    this.docIds = docIds
    this.idfs = idfs
  }

  def loadAndParse(sc: SparkContext, sampleSize: Double, numTerms: Int) = {

    val pages = readFile(wikiDumpPath, sc)
      .sample(false, sampleSize, 11L)

    val plainText = pages.filter(_ != null).flatMap(wikiXmlToPlainText)

    val stopWords = loadStopWords(stopWordsFile)

    val lemmatized = plainText.mapPartitions { it =>
      val pipeline = createNLPPipeline
      it.map { case (title, contents) =>
        (title, plainTextToLemmas(contents, stopWords, pipeline))
      }
    }

    val filtered = lemmatized.filter(_._2.size > 1)

    documentTermMatrix(sc, filtered, stopWords, numTerms)

    save(sc)
  }

  def save(sc: SparkContext): Unit = {
    println("save parsed wiki ...")
    termVecs.get.saveAsObjectFile(vectorsPath)
    val oos = new ObjectOutputStream(new FileOutputStream(objSerFile))
    oos.writeObject(this)
    oos.close()

    println("finish save parsed wiki ...")
  }
}

object WikiVSM {
  import WikiConfig._

  def load(sc: SparkContext): WikiVSM = {
    val ois = new ObjectInputStream(new FileInputStream(objSerFile))
    val wikiVSM = ois.readObject().asInstanceOf[WikiVSM]
    ois.close()
    wikiVSM.termVecs = Some(sc.objectFile[Vector](vectorsPath))
    wikiVSM
  }
}
