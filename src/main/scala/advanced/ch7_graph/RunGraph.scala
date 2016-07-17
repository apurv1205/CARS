package advanced.ch7_graph

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

import com.databricks.spark.xml.XmlInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.SparkContext
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

import scala.xml._

/**
  * Created by roger19890107 on 7/17/16.
  */
object RunGraph {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/aas/ch7-medline/"

  def main(args: Array[String]) {
    context.Env.setLogger
    val sc = context.Env.setupContext("Graph")

    // load medline mesh
    val medline_raw = loadMedline(sc, base + "*.xml")
    val mxml: RDD[Elem] = medline_raw.map(XML.loadString)
    val medline: RDD[Seq[String]] = mxml.map(majorTopics).cache()

    // topics profiling
    val topics = medline.flatMap(mesh => mesh)
    val topciCounts = topics.countByValue()
    val tcSeq = topciCounts.toSeq
    println("number of topics: " + tcSeq.size)
    tcSeq.sortBy(_._2).reverse.take(10).foreach(println)
    val valueDist = topciCounts.groupBy(_._2).mapValues(_.size)
    valueDist.toSeq.sorted.take(10).foreach(println)

    // co-occur
    val topicPairs = medline.flatMap(t => t.sorted.combinations(2))
    val cooccurs = topicPairs.map(p => (p, 1)).reduceByKey(_ + _)
    cooccurs.cache()
    println("number of cooccurs: " + cooccurs.count())
    cooccurs.top(10)(Ordering.by(_._2)).foreach(println)

    // topic graph
    val vertices = topics.map(topic => (hashId(topic), topic))
    val edges = cooccurs.map(p => {
      val (topics, cnt) = p
      val ids = topics.map(hashId).sorted
      Edge(ids(0), ids(1), cnt)
    })
    val topicGraph = Graph(vertices, edges)
    println("vertices count: " + vertices.count())
    println("graph vertices count: " + topicGraph.vertices.count())

    // connected component
    val connectedComponentGraph: Graph[VertexId, Int] =
      topicGraph.connectedComponents()
    val componentCounts = sortedConnectedComponents(connectedComponentGraph)
    println("cc counts size: " + componentCounts.size)
    componentCounts.take(10).foreach(println)

    val nameCID = topicGraph.vertices
      .innerJoin(connectedComponentGraph.vertices) {
        (topicId, name, componentId) => (name, componentId)
      }

    val c1 = nameCID.filter(x => x._2._2 == componentCounts(1)._1)
    c1.collect().foreach(x => println(x._2._1))

    // degree
    val degrees: VertexRDD[Int] = topicGraph.degrees.cache()
    println("degrees: ")
    println(degrees.map(_._2).stats())
    topNamesAndDegrees(degrees, topicGraph).foreach(println)

    // chi square
    val T = medline.count()
    val topicCountsRdd = topics.map(x => (hashId(x), 1)).reduceByKey(_ + _)
    val topicCountGraph = Graph(topicCountsRdd, topicGraph.edges)
    val chiSquredGraph = topicCountGraph.mapTriplets(triplet => {
      chiSq(triplet.attr, triplet.srcAttr, triplet.dstAttr, T)
    })
    println("Chi-squared statics:")
    println(chiSquredGraph.edges.map(x => x.attr).stats())

    // interesting
    val chiCrital = 30
    val interesting = chiSquredGraph.subgraph(triplet => triplet.attr > chiCrital)

    val interestingComponentCounts = sortedConnectedComponents(interesting.connectedComponents())
    println("interesting cc size: " + interestingComponentCounts.size)
    interestingComponentCounts.take(10).foreach(println)

    val interestingDegrees = interesting.degrees.cache()
    println("interesting degrees: ")
    println(interestingDegrees.map(_._2).stats())
    topNamesAndDegrees(interestingDegrees, topicGraph).foreach(println)

    // clustering coef
    val avgCC = avgClusteringCoef(interesting)
    println("avg clustering coef: " + avgCC)

    // sample path
    val paths = samplePathLength(interesting)
    println("sample path statics: ")
    println(paths.map(_._3).filter(_ > 0).stats())

    val hist = paths.map(_._3).countByValue()
    println("sample path hist: ")
    hist.toSeq.sorted.foreach(println)
  }

  def mergeMap(m1: Map[VertexId, Int],
               m2: Map[VertexId, Int]): Map[VertexId, Int] = {
    def minThatExists(k: VertexId): Int = {
      math.min(
        m1.getOrElse(k, Int.MaxValue),
        m2.getOrElse(k, Int.MaxValue)
      )
    }

    (m1.keySet ++ m2.keySet).map {
      k => (k, minThatExists(k))
    }.toMap
  }

  def update(id: VertexId,
             state: Map[VertexId, Int],
             msg: Map[VertexId, Int]) = {
    mergeMap(state, msg)
  }

  def checkIncrement(a: Map[VertexId, Int],
                     b: Map[VertexId, Int],
                     bid: VertexId) = {
    val aplus = a.map {case (v, d) => v -> (d + 1)}
    if (b != mergeMap(aplus, b)) {
      Iterator((bid, aplus))
    } else {
      Iterator.empty
    }
  }

  def iterate(e: EdgeTriplet[Map[VertexId, Int], _]) = {
    checkIncrement(e.srcAttr, e.dstAttr, e.dstId) ++
    checkIncrement(e.dstAttr, e.srcAttr, e.srcId)
  }

  def samplePathLength[V, E](graph: Graph[V, E], fraction: Double = 0.001)
    : RDD[(VertexId, VertexId, Int)] = {
    val replacement = false
    val sample = graph.vertices.map(v => v._1).sample(replacement, fraction, 1792L)
    val ids = sample.collect().toSet

    val mapGraph = graph.mapVertices((id, v) => {
      if (ids.contains(id)) {
        Map(id -> 0)
      } else {
        Map[VertexId, Int]()
      }
    })

    val start = Map[VertexId, Int]()
    val res = mapGraph.ops.pregel(start)(update, iterate, mergeMap)
    res.vertices.flatMap {case (id, m) =>
      m.map { case (k, v) =>
        if (id < k) {
          (id, k, v)
        } else {
          (k, id, v)
        }
      }
    }.distinct().cache()
  }

  def avgClusteringCoef(graph: Graph[_, _]): Double = {
    val triCountGraph = graph.triangleCount()
    println("tri statics:")
    println(triCountGraph.vertices.map(_._2).stats())

    val maxTridGraph = graph.degrees.mapValues(d => d * (d - 1) / 2.0)

    val clusterCoefGraph = triCountGraph.vertices.innerJoin(maxTridGraph) {
      (vertexId, triCount, maxTris) => if (maxTris == 0) 0 else triCount / maxTris
    }

    clusterCoefGraph.map(_._2).sum() / graph.vertices.count()
  }

  def sortedConnectedComponents(connectedComponents: Graph[VertexId, _])
    : Seq[(VertexId, Long)] = {
    val componentCounts = connectedComponents.vertices.map(_._2).countByValue()
    componentCounts.toSeq.sortBy(_._2).reverse
  }

  def topNamesAndDegrees(degrees: VertexRDD[Int], topicGraph: Graph[String, Int])
    : Array[(String, Int)] = {
    val namesAndDegrees = degrees.innerJoin(topicGraph.vertices) {
      (topicId, degree, name) => (name, degree)
    }
    namesAndDegrees.map(_._2).top(10)(Ordering.by(_._2))
  }

  def loadMedline(sc: SparkContext, path: String): RDD[String] = {
    val conf = new Configuration()
    conf.set(XmlInputFormat.START_TAG_KEY, "<MedlineCitation>")
    conf.set(XmlInputFormat.END_TAG_KEY, "</MedlineCitation>")
    val in = sc.newAPIHadoopFile(path, classOf[XmlInputFormat],
      classOf[LongWritable], classOf[Text], conf)
    in.map(line => line._2.toString)
  }

  def majorTopics(elem: Elem): Seq[String] = {
    val dn = elem \\ "DescriptorName"
    val mt = dn.filter(n => (n \ "@MajorTopicYN").text == "Y")
    mt.map(n => n.text)
  }

  def hashId(str: String): Long = {
    val bytes = MessageDigest.getInstance("MD5").digest(str.getBytes(StandardCharsets.UTF_8))
    (bytes(0) & 0xFFL) |
      ((bytes(1) & 0xFFL) << 8) |
      ((bytes(2) & 0xFFL) << 16) |
      ((bytes(3) & 0xFFL) << 24) |
      ((bytes(4) & 0xFFL) << 32) |
      ((bytes(5) & 0xFFL) << 40) |
      ((bytes(6) & 0xFFL) << 48) |
      ((bytes(7) & 0xFFL) << 56)
  }

  def chiSq(YY: Int, YB: Int, YA: Int, T: Long): Double = {
    val NB = T - YB
    val NA = T - YA
    val YN = YA - YY
    val NY = YB - YY
    val NN = T - NY - YN - YY
    val inner = YY * NN - YN * NY
    T * math.pow(inner, 2) / (YA * NA * YB * NB)
  }
}
