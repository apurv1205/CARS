package aas.ch2_intro

import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter

/**
  * Created by roger19890107 on 6/3/16.
  */
case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)
case class Scored(md: MatchData, score: Double)

object RunIntro {
  def main(args: Array[String]) {
    context.Env.setLogger
    val sc = context.Env.setupContext("Intro")

    val rawblocks = sc.textFile("file:///Volumes/RogerDrive/Developer/dataset/aas/ch2-linkage")
    def isHeader(line: String) = line.contains("id_1")

    rawblocks.take(4).foreach(println)

    val noheader = rawblocks.filter(x => !isHeader(x))
    def toDouble(s: String) = if (s.equals("?")) Double.NaN else s.toDouble

    def parse(line: String) = {
      val pieces = line.split(",")
      val id1 = pieces(0).toInt
      val id2 = pieces(1).toInt
      val scores = pieces.slice(2, 11).map(toDouble)
      val matched = pieces(11).toBoolean
      MatchData(id1, id2, scores, matched)
    }

    val parsed = noheader.map(parse)
    parsed.cache()

    val matchCounts = parsed.map(_.matched).countByValue()
    val matchCountSeq = matchCounts.toSeq
    matchCountSeq.sortBy(_._2).reverse.foreach(println)

    val stats = (0 until 9).map { i =>
      parsed.map(_.scores(i)).filter(!_.isNaN).stats()
    }
    stats.foreach(println)

    val nasRDD = parsed.map(md => {
      md.scores.map(d => NAStatCounter(d))
    })

    val reduced = nasRDD.reduce((n1, n2) => {
      n1.zip(n2).map { case (a, b) => a.merge(b) }
    })
    reduced.foreach(println)

    val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
    val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))
    statsm.zip(statsn).map { case (m, n) =>
      (m.missing + n.missing, m.stats.mean - n.stats.mean)
    }.foreach(println)

    def naz(d: Double) = if (d.isNaN) 0.0 else d
    val ct = parsed.map(md => {
      val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
      Scored(md, score)
    })

    ct.filter(s => s.score >= 4.0)
      .map(_.md.matched).countByValue().foreach(println)

    ct.filter(s => s.score >= 2.0)
      .map(_.md.matched).countByValue().foreach(println)
  }

  def statsWithMissing(rdd: RDD[Array[Double]]): Array[NAStatCounter] = {
    val nastats = rdd.mapPartitions(iter => {
      val nas = iter.next().map(d => NAStatCounter(d))
      iter.foreach(arr => {
        nas.zip(arr).foreach { case (n, d) => n.add(d) }
      })
      Iterator(nas)
    })

    nastats.reduce((n1, n2) => {
      n1.zip(n2).map { case (a, b) => a.merge(b) }
    })
  }
}

class NAStatCounter extends Serializable {
  val stats: StatCounter = new StatCounter()
  var missing: Long = 0

  def add(x: Double): NAStatCounter = {
    if (x.isNaN) {
      missing += 1
    } else {
      stats.merge(x)
    }
    this
  }

  def merge(other: NAStatCounter): NAStatCounter = {
    stats.merge(other.stats)
    missing += other.missing
    this
  }

  override def toString = {
    "stats: " + stats.toString() + " NaN: " + missing
  }
}

object NAStatCounter extends Serializable {
  def apply(x: Double): NAStatCounter = new NAStatCounter().add(x)
}
