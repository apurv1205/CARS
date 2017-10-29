package context

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Env {
  def setLogger: Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    //System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger.setLevel(Level.OFF)
  }

  def setupContext(appName: String): SparkContext = {

    val sc = new SparkContext(new SparkConf()
      .setAppName(appName).setMaster("local[*]").setExecutorEnv("driver-memory", "6g"))
    sc.setCheckpointDir("data/checkpoint/")
    sc
  }
}
