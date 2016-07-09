package context

import scala.collection.mutable

/**
  * Created by roger19890107 on 7/9/16.
  */
object Test {
  def main(args: Array[String]) {
    val r = scala.util.Random
    val ratingCount = 1000
    val objMax = 20000
    val simCount = 10000

    val ratings = (for (i <- 1 to ratingCount) yield r.nextInt(objMax))
      .zip(for (i <- 1 to ratingCount) yield r.nextDouble).sortBy(_._1)
    val similarities = (for (i <- 1 to simCount) yield r.nextInt(objMax))
      .zip(for (i <- 1 to simCount) yield r.nextDouble).sortBy(_._1)

    // array
    val ratingsArr = ratings.toArray
    val similarArr = similarities.toArray
    val start1 = java.lang.System.currentTimeMillis()
    val s1 = getWeightScores(ratingsArr, similarArr)
    val end1 = java.lang.System.currentTimeMillis()
    println("arr score: " + s1)
    println("arr version duration: " + (end1 - start1))

    // map
    val ratingsHashMap = new mutable.HashMap[Int, Double]()
    ratings.foreach {
      case (key, value) => ratingsHashMap += (key -> value)
    }
    val similarHashMap = new mutable.HashMap[Int, Double]()
    similarities.foreach {
      case (key, value) => similarHashMap += (key -> value)
    }
    val start2 = java.lang.System.currentTimeMillis()
    val s2 = getWeightScores(ratingsHashMap, similarHashMap)
    val end2 = java.lang.System.currentTimeMillis()
    println("hash map score: " + s2)
    println("hash map version duration: " + (end2 - start2))
  }

  def getWeightScores(ratings: Array[(Int, Double)],
                      similarities: Array[(Int, Double)]): Double = {
//    // join buy key
//    val joinRatsAndSims = (ratings ++ similarities)
//      .groupBy(_._1)
//      .filter(_._2.length == 2)
//      .mapValues(arr => (arr(0)._2, arr(1)._2))
//      .values
//
//    if (joinRatsAndSims.isEmpty) return 0
//
//    // weight predict scores
//    val weightSum = joinRatsAndSims.map(_._2).sum
//
//    joinRatsAndSims.map {
//      case (rating, similarity) => rating * similarity
//    }.sum / weightSum

    var scores: Double = 0.0
    var total: Double = 0.0
    var i = 0
    var j = 0
    while (i < ratings.length && j < similarities.length) {
      val rat = ratings(i)
      val sim = similarities(j)
      if (rat._1 == sim._1) {
        scores += (rat._2 * sim._2)
        total += sim._2
        i += 1
        j += 1
      } else if (rat._1 < sim._1) {
        i += 1
      } else  {
        j += 1
      }
    }

    if (total == 0.0) 0.0 else scores / total
  }

  def getWeightScores(ratings: mutable.HashMap[Int, Double],
                      similarities: mutable.HashMap[Int, Double]): Double = {

    var scores: Double = 0.0
    var total: Double = 0.0

    if (ratings.size < similarities.size) {
      ratings.foreach {
        case (obj, rating) =>
          val sim = similarities.getOrElse(obj, 0.0)
          scores += (rating * sim)
          total += sim
      }
    } else {
      similarities.foreach {
        case (obj, sim) => {
          val rating = ratings.getOrElse(obj, 0.0)
          if (rating != 0.0) {
            scores += (rating * sim)
            total += sim
          }
        }
      }
    }

    if (total == 0.0) 0.0 else scores / total
  }
}
