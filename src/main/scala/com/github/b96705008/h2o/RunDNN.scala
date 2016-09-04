package com.github.b96705008.h2o

import hex.deeplearning.DeepLearning
import com.github.b96705008.context._
import org.apache.spark.h2o.H2OContext
import org.apache.spark.examples.h2o._
import java.io.File

import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import water.fvec.H2OFrame


/**
  * Created by roger19890107 on 9/4/16.
  */
object RunDNN extends App {
  val SMALL_DATA_PATH = "/Users/roger19890107/Developer/main/resources/h2o/sparkling-water-1.6.5/examples/smalldata/"

  // Initialize H2O services on top of Spark cluster:
  val sc = Env.setupContext("RunDNN")
  val sqlContext = SQLContext.getOrCreate(sc)
  val h2oContext = H2OContext.getOrCreate(sc)
  import h2oContext._
  import h2oContext.implicits._

  // Load weather data for Chicago international airport (ORD), with help from the RDD API
  val weatherDataFile = SMALL_DATA_PATH + "Chicago_Ohare_International_Airport.csv"
  val wrawdata = sc.textFile(weatherDataFile, 3).cache()
  val weatherTable = wrawdata.map(_.split(","))
    .map(row => WeatherParse(row))
    .filter(!_.isWrongRow())

  // Load airlines data using the H2O parser
  val dataFile = SMALL_DATA_PATH + "allyears2k_headers.csv.gz"
  val airlinesData = new H2OFrame(new File(dataFile))

  // Select flights destined for Chicago (ORD)
  val airlinesTable: RDD[Airlines] =  asRDD[Airlines](airlinesData)
  val flightsToORD = airlinesTable.filter(f => f.Dest == Some("ORD"))

  // Compute the number of these flights
  println(flightsToORD.count())

  // Use Spark SQL to join the flight data with the weather data
  import sqlContext.implicits._
  flightsToORD.toDF().registerTempTable("FlightsToORD")
  weatherTable.toDF().registerTempTable("WeatherORD")

  // Perform SQL JOIN on both tables:
  val bigTable = sqlContext.sql(
    """SELECT
      |f.Year,f.Month,f.DayofMonth,
      |f.CRSDepTime,f.CRSArrTime,f.CRSElapsedTime,
      |f.UniqueCarrier,f.FlightNum,f.TailNum,
      |f.Origin,f.Distance,
      |w.TmaxF,w.TminF,w.TmeanF,w.PrcpIn,w.SnowIn,w.CDD,w.HDD,w.GDD,
      |f.ArrDelay
      |FROM FlightsToORD f
      |JOIN WeatherORD w
      |ON f.Year=w.Year AND f.Month=w.Month AND f.DayofMonth=w.Day
    """.stripMargin)

  // Transform the first 3 columns containing date information into enum columns
  val bigDataFrame: H2OFrame = h2oContext.asH2OFrame(bigTable)
  for (i <- 0 to 2) bigDataFrame.replace(i, bigDataFrame.vec(i).toCategoricalVec)
  bigDataFrame.update()

  // Run deep learning to produce a model estimating arrival delay
  val dlParams = new DeepLearningParameters()
  dlParams._train = bigDataFrame
  dlParams._response_column = 'ArrDelay
  dlParams._epochs = 5
  dlParams._activation = Activation.RectifierWithDropout
  dlParams._hidden = Array[Int](100, 100)

  // Create a job
  val dl = new DeepLearning(dlParams)
  val dlModel = dl.trainModel().get()

  // Use the model to estimate the delay on the training data
  val predictionH2OFrame = dlModel.score(bigTable)('predict)
  val predictionsFromModel = asDataFrame(predictionH2OFrame)(sqlContext)
      .collect.map(row => if (row.isNullAt(0)) Double.NaN else row.getAs[Double](0))

}
