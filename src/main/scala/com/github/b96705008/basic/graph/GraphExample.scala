package com.github.b96705008.basic.graph

import com.github.b96705008.context.Env
import org.apache.spark.graphx.GraphLoader

/**
  * Created by roger19890107 on 7/9/16.
  */
object GraphExample extends App {
  Env.setLogger
  val sc = Env.setupContext("GraphX intro")

  val users = sc.textFile("data/datasets/graphx/users.txt")
    .map(line => line.split(",")).map(parts => (parts.head.toLong, parts.tail))

  val followerGraph = GraphLoader.edgeListFile(sc, "data/datasets/graphx/followers.txt")

  val graph = followerGraph.outerJoinVertices(users) {
    case (uid, deg, Some(attrList)) => attrList
    case (uid, deg, None) => Array.empty[String]
  }

  val subgraph = graph.subgraph(vpred = (vid, attr) => attr.size == 2)

  val pagerankGraph = subgraph.pageRank(0.001)

  val userInfoWithPageRank = subgraph.outerJoinVertices(pagerankGraph.vertices) {
    case (uid, attrList, Some(pr)) => (pr, attrList.toList)
    case (uid, attrList, None) => (0.0, attrList.toList)
  }

  println(userInfoWithPageRank.vertices
    .top(5)(Ordering.by(_._2._1)).mkString("\n"))
}
