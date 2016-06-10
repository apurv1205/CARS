package advanced.ch2_intro

import org.apache.spark.util.StatCounter

/**
  * Created by roger19890107 on 6/4/16.
  */
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
