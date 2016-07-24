package com.github.b96705008.context

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by roger19890107 on 5/8/16.
  */
object Env {
  def setLogger: Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    //System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger.setLevel(Level.OFF)
  }

  def setupContext(appName: String): SparkContext = {
    setLogger

    val sc = new SparkContext(new SparkConf()
      .setAppName(appName).setMaster("local[*]").setExecutorEnv("driver-memory", "6g"))
    sc.setCheckpointDir("data/checkpoint/")
    sc
  }
}
