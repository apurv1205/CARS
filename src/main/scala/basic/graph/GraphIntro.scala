package basic.graph

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.rdd.RDD

/**
  * Created by roger19890107 on 7/8/16.
  */
object GraphIntro extends App {
  context.Env.setLogger
  val sc = context.Env.setupContext("GraphX intro")

  //==basic user graph==

  val users: RDD[(VertexId, (String, String))] =
    sc.parallelize(Array(
      (3L, ("rxin", "student")),
      (7L, ("jgonzal", "postdoc")),
      (5L, ("fanklin", "prof")),
      (2L, ("istoica", "prof")),
      (4L, ("peter", "student"))
    ))

  val relationships: RDD[Edge[String]] =
    sc.parallelize(Array(
      Edge(3L, 7L, "collab"),
      Edge(5L, 3L, "advisor"),
      Edge(2L, 5L, "colleague"),
      Edge(5L, 7L, "pi"),
      Edge(4L, 0L, "student"),
      Edge(5L, 0L, "colleague")
    ))

  val defaultUser = ("John Doe", "Missing")

  val graph = Graph(users, relationships, defaultUser)

  val countWithPost = graph.vertices.filter { case (id, (name, pos)) => pos == "postdoc" }.count()
  println("countWithPost: " + countWithPost)

  val edgeCount = graph.edges.filter { case Edge(src, dst, prop) => src > dst}.count()
  println("edge count: " + edgeCount)

  val facts = graph.triplets.map(triplet =>
    triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1)
  facts.foreach(println)

  // == Property Operators ==

  val inDegrees = graph.inDegrees.map{case (vertexId, degree) => vertexId + " in nums: " + degree}
    .foreach(println)

  val inputGraph: Graph[Int, String] =
    graph.outerJoinVertices(graph.outDegrees)((vid, _, degOpt) =>
      degOpt.getOrElse(0))

  val outputGraph: Graph[Double, Double] =
    inputGraph
      .mapTriplets(triplet => 1.0 / triplet.srcAttr)
      .mapVertices((id, _) => 1.0)

  // == Structural Operators ==

  println("validGraph: ")
  val validGraph = graph.subgraph(vpred = (id, attr) => attr._2 != "Missing")
  validGraph.vertices.collect().foreach(println)

  println("validCCGraph: ")
  val ccGraph = graph.connectedComponents()
  val validCCGraph = ccGraph.mask(validGraph)
  validCCGraph.vertices.collect().foreach(println)

  // == Neighborhood Aggregation ==
  val ageGraph = GraphGenerators
    .logNormalGraph(sc, numVertices = 100)
    .mapVertices((id, _) => id.toDouble)

  val olderFollowers = ageGraph.aggregateMessages[(Int, Double)](
    triplet => {
      if (triplet.srcAttr > triplet.dstAttr) {
        triplet.sendToDst(1, triplet.srcAttr)
      }
    },
    (a, b) => (a._1 + b._1, a._2 + b._2)
  )

  println("avgAgeOfOlderFollowers: ")
  val avgAgeOfOlderFollowers =
    olderFollowers.mapValues((id, value) => value match {
      case (count, totalAge) => totalAge / count
    })
  avgAgeOfOlderFollowers.collect().foreach(println)

  def max(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
    if (a._2 > b._2) a else b
  }
  val maxVertex = graph.inDegrees.reduce(max)
  println(maxVertex)

  // == Pregel API ==
  println("Pregel shortest path: ")
  val pGraph = GraphGenerators
    .logNormalGraph(sc, numVertices = 100)
    .mapEdges(e => e.attr.toDouble)
  val sourceId: VertexId = 42

  val initialGraph = pGraph.mapVertices((id,_) =>
    if (id == sourceId) 0.0 else Double.PositiveInfinity)

  val sssp = initialGraph.pregel(Double.PositiveInfinity)(
    (id, dist, newDist) => math.min(dist, newDist),
    triplet => {
      if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
        Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
      } else {
        Iterator.empty
      }
    },
    (a, b) => math.min(a, b)
  )
  println(sssp.vertices.collect().mkString("\n"))

  // == Vertex and Edge RDDs ==
  println("aggregateUsingIndex: ")
  val setA: VertexRDD[Int] = VertexRDD(sc.parallelize(0L until 100L).map(id => (id, 1)))
  val rddB: RDD[(VertexId, Double)] = sc.parallelize(0L until 100L).flatMap(id => List((id, 1.0), (id, 2.0)))
  println(rddB.count)
  val setB: VertexRDD[Double] = setA.aggregateUsingIndex(rddB, _ + _)
  println(setB.count())
  val setC: VertexRDD[Double] = setA.innerJoin(setB)((id, a, b) => a + b)
  setC.map {case (vertixId, value) => vertixId + ": " + value}.take(10).foreach(println)

  // == Graph Algorithms ==
  val pageGraph = GraphLoader.edgeListFile(sc, "data/datasets/graphx/followers.txt")
  val ranks = graph.pageRank(0.0001).vertices
  val pusers = sc.textFile("data/datasets/graphx/users.txt").map { line =>
    val fields = line.split(",")
    (fields(0).toLong, fields(1))
  }

  val ranksByUsername = pusers.join(ranks).map {
    case (id, (username, rank)) => (username, rank)
  }
  println("pagerank: ")
  println(ranksByUsername.collect().mkString("\n"))
}
