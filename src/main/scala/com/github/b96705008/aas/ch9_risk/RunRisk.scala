package com.github.b96705008.aas.ch9_risk

import java.text.SimpleDateFormat
import java.util.Locale

import com.github.b96705008.context.Env
import com.github.nscala_time.time.Implicits._
import org.joda.time.DateTime
import java.io.File

import breeze.plot.Figure
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.joda.time

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import breeze.plot._
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, MultivariateNormalDistribution}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.correlation.{Covariance, PearsonsCorrelation}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Created by roger19890107 on 7/30/16.
  */
object RunRisk {
  val dataPrefix = "/Volumes/RogerDrive/Developer/dataset/aas/ch9-data/"

  def main(args: Array[String]): Unit = {
    val sc = Env.setupContext("VaR")

    // load data
    println("*Read Stocks and Factors...")
    val (stocksReturns, factorsReturns) = readStocksAndFactors(dataPrefix)
    plotDistribution(factorsReturns(0), "Crude Oil")
    plotDistribution(factorsReturns(1), "US 30 yrs bonds")
    plotDistribution(factorsReturns(2), "S&P 500")
    plotDistribution(factorsReturns(3), "NASDAQ")

    // trials
    val numTrials = 10000000
    val parallelism = 1000
    val baseSeed = 1001L
    val trials = computeTrialReturns(stocksReturns, factorsReturns,
      sc, baseSeed, numTrials, parallelism)
    trials.cache()
    plotDistribution(trials, "Trials")

    // VaR
    val valueAtRisk = fivePercentVaR(trials)
    val conditionValueAtRisk = fivePercentCVaR(trials)
    println("VaR 5%: " + valueAtRisk)
    println("CVaR 5%: " + conditionValueAtRisk)

    // confidence level
    val varConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentVaR, 100, .05)
    val cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentCVaR, 100, .05)
    println("VaR confidence interval: " + varConfidenceInterval)
    println("CVaR confidence interval: " + cvarConfidenceInterval)

    // Kupiec test
    println("Kupiec test p-value: " + kupiecTestPValue(stocksReturns, valueAtRisk, 0.05))

    sc.stop()
  }

  // == Evaluating Our Results ==
  def bootstrappedConfidenceInterval(trials: RDD[Double],
                                     computeStatistic: RDD[Double] => Double,
                                     numResamples: Int,
                                     pValue: Double): (Double, Double) = {
    val stats = (0 until numResamples).map {i =>
      val resample = trials.sample(true, 1.0)
      computeStatistic(resample)
    }.sorted
    val lowerIndex = (numResamples * pValue / 2).toInt
    val upperIndex = (numResamples * (1 - pValue / 2)).toInt
    (stats(lowerIndex), stats(upperIndex))
  }

  def countFailures(stockReturns: Seq[Array[Double]], valueAtRisk: Double): Int = {
    var failures = 0
    for (i <- stockReturns.head.indices) {
      val loss = stockReturns.map(_(i)).sum
      if (loss < valueAtRisk) {
        failures += 1
      }
    }
    failures
  }

  /**
    * Kupiec’s proportion-of-failures (POF)
    * @param total T, the total number of historical intervals
    * @param failures x, the number of historical intervals over which the losses exceeded the VaR
    * @param confidenceLevel p, the confidence level parameter of the VaR calculation
    * @return
    */
  def kupiecTestStatistic(total: Int, failures: Int, confidenceLevel: Double): Double = {
    val failureRatio = failures.toDouble / total
    val logNumber = (total - failures) * math.log1p(-confidenceLevel) +
      failures * math.log(confidenceLevel)
    val logDenom = (total - failures) * math.log1p(-failureRatio) +
      failures * math.log(failureRatio)
    -2 * (logNumber - logDenom)
  }

  def kupiecTestPValue(stockReturns: Seq[Array[Double]],
                       valueAtRisk: Double,
                       confidenceLevel: Double): Double = {
    val failures = countFailures(stockReturns, valueAtRisk)
    val total = stockReturns.head.length
    val testStatistic = kupiecTestStatistic(total, failures, confidenceLevel)
    1 - new ChiSquaredDistribution(1.0).cumulativeProbability(testStatistic)
  }

  // == Running the Trials ==
  /**
    * Return of a instrument in one trial
    * @param instrument a set of regression coefficients
    * @param trial a sample of factor features
    * @return return of the value
    */
  def instrumentTrialReturn(instrument: Array[Double],
                           trial: Array[Double]): Double = {
    var theReturn = instrument(0) // intercept term
    var i = 0
    while (i < trial.length) {
      theReturn += trial(i) * instrument(i + 1)
      i += 1
    }
    theReturn
  }

  def trialReturn(trial: Array[Double],
                  instruments: Seq[Array[Double]]): Double = {
    var totalReturn = 0.0
    for (instrument <- instruments) {
      totalReturn += instrumentTrialReturn(instrument, trial)
    }
    totalReturn
  }

  def trialReturns(seed: Long,
                   numTrials: Int,
                   instruments: Seq[Array[Double]],
                   factorMeans: Array[Double],
                   factorCovariances: Array[Array[Double]]): Seq[Double] = {
    // random generator
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(
      rand, factorMeans, factorCovariances)

    val trialReturns = new Array[Double](numTrials)
    for (i <- 0 until numTrials) {
      val trialFactorReturns = multivariateNormal.sample()
      val trialFeatures = featurize(trialFactorReturns)
      trialReturns(i) = trialReturn(trialFeatures, instruments)
    }
    trialReturns
  }

  def computeTrialReturns(stocksReturns: Seq[Array[Double]],
                          factorsReturns: Seq[Array[Double]],
                          sc: SparkContext,
                          baseSeed: Long,
                          numTrials: Int,
                          parallelism: Int): RDD[Double] = {
    // factor features
    val factorMat = factorMatrix(factorsReturns)
    val factorFeatures = factorMat.map(featurize)

    // corrlation
    val factorCor =
      new PearsonsCorrelation(factorMat).getCorrelationMatrix.getData
    println("*Factor correlations:")
    println(factorCor.map(_.mkString("\t")).mkString("\n"))

    // multivariate normal distribution
    val factorCov = new Covariance(factorMat).getCovarianceMatrix.getData
    val factorMeans = factorsReturns.map(f => f.sum / f.length).toArray

    // factor weights
    val factorWeights = computeFactorWeights(stocksReturns, factorFeatures)
    val bInstruments = sc.broadcast(factorWeights)

    // seed
    val seeds = baseSeed until baseSeed + parallelism
    val seedRdd = sc.parallelize(seeds, parallelism)

    // main computation
    seedRdd.flatMap(trialReturns(_,
      numTrials / parallelism,
      bInstruments.value,
      factorMeans,
      factorCov))
  }

  def fivePercentVaR(trials: RDD[Double]): Double = {
    val topLosses = trials.takeOrdered(math.max(trials.count().toInt / 20, 1))
    topLosses.last
  }

  def fivePercentCVaR(trials: RDD[Double]): Double = {
    val topLosses = trials.takeOrdered(math.max(trials.count().toInt / 20, 1))
    topLosses.sum / topLosses.length
  }

  // == Determining the Factor Weights ==
  def factorMatrix(histories: Seq[Array[Double]]): Array[Array[Double]] = {
    val mat = new Array[Array[Double]](histories.head.length)
    for (i <- histories.head.indices) {
      mat(i) = histories.map(_(i)).toArray
    }
    mat //row length is number of time window, col length is number of factors
  }

  def featurize(factorReturns: Array[Double]): Array[Double] = {
    val squaredReturns = factorReturns
      .map(x => math.signum(x) * x * x)
    val squaredRootReturns = factorReturns
      .map(x => math.signum(x) * math.sqrt(math.abs(x)))

    squaredReturns ++ squaredRootReturns ++ factorReturns
  }

  def linearModel(instrument: Array[Double],
                  factorMatrix: Array[Array[Double]]): OLSMultipleLinearRegression = {
    val regression = new OLSMultipleLinearRegression()
    regression.newSampleData(instrument, factorMatrix)
    regression
  }

  def computeFactorWeights(stocksReturns: Seq[Array[Double]],
                           factorFeatures: Array[Array[Double]]): Array[Array[Double]] = {
    stocksReturns
      .map(linearModel(_, factorFeatures))
      .map(_.estimateRegressionParameters())
      .toArray
  }

  // == Preprocessing ==
  def readStocksAndFactors(prefix: String): (Seq[Array[Double]], Seq[Array[Double]]) = {
    val start = new time.DateTime(2009, 10, 23, 0, 0)
    val end = new time.DateTime(2014, 10, 23, 0, 0)

    // instruments: stocks
    val rawStocks = readHistories(new File(prefix + "stocks/")).filter(_.length > 260*5 + 10)
    val stocks = rawStocks
      .map(trimToRegion(_, start, end))
      .map(fillInHistory(_, start, end))

    // market factors
    val factorsPrefix = prefix + "factors/"
    val factors1 = Array("crudeoil.tsv", "us30yeartreasurybonds.tsv")
      .map(x => new File(factorsPrefix + x))
      .map(readInvestingDotComHistory)
    val factors2 = Array("SNP.csv", "NDX.csv")
      .map(x => new File(factorsPrefix + x))
      .map(readYahooHistory)

    val factors = (factors1 ++ factors2)
      .map(trimToRegion(_, start, end))
      .map(fillInHistory(_, start, end))

    println("All datetime span is equal: " +
      (stocks ++ factors).forall(_.length == stocks(0).length))

    val stocksReturns = stocks.map(twoWeekReturns)
    val factorsReturns = factors.map(twoWeekReturns)
    (stocksReturns, factorsReturns)
  }

  def readInvestingDotComHistory(file: File): Array[(DateTime, Double)] = {
    val format = new SimpleDateFormat("MMM d, yyyy", Locale.ENGLISH)
    val lines = Source.fromFile(file).getLines().toSeq
    lines.map(line => {
      val cols = line.split("\t")
      val date = new time.DateTime(format.parse(cols(0)))
      val value = cols(1).toDouble
      (date, value)
    }).reverse.toArray
  }

  def readYahooHistory(file: File): Array[(DateTime, Double)] = {
    val format = new SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH)
    val lines = Source.fromFile(file).getLines().toSeq
    lines.tail.map(line => {
      val cols = line.split(",")
      val date = new time.DateTime(format.parse(cols(0)))
      val value = cols(1).toDouble
      (date, value)
    }).reverse.toArray
  }

  def readHistories(dir: File): Seq[Array[(DateTime, Double)]] = {
    val files = dir.listFiles()
    files.flatMap(file => {
      try {
        Some(readYahooHistory(file))
      } catch {
        case e: Exception => None
      }
    })
  }

  def trimToRegion(histroy: Array[(DateTime, Double)],
                   start: DateTime, end: DateTime): Array[(DateTime, Double)] = {
    // filter by time region
    var trimmed = histroy
      .dropWhile(_._1 < start).takeWhile(_._1 <= end)
    // first datetime should be "start"
    if (trimmed.head._1 != start) {
      trimmed = Array((start, trimmed.head._2)) ++ trimmed
    }
    // last datetime should be "end"
    if (trimmed.last._1 != end) {
      trimmed = trimmed ++ Array((end, trimmed.last._2))
    }
    trimmed
  }

  def fillInHistory(history: Array[(DateTime, Double)],
                    start: DateTime, end: DateTime): Array[(DateTime, Double)] = {
    val filled = new ArrayBuffer[(DateTime, Double)]()
    var cur = history
    var curDate = start
    while (curDate < end) {
      if (cur.tail.nonEmpty && cur.tail.head._1 == curDate) {
        cur = cur.tail
      }
      filled += ((curDate, cur.head._2))

      curDate += 1.days
      if (curDate.dayOfWeek().get() > 5) curDate += 2.days // skip weekends
    }
    filled.toArray
  }

  def twoWeekReturns(history: Array[(DateTime, Double)]): Array[Double] = {
    history.sliding(10)
      .map(window => window.last._2 - window.head._2)
      .toArray
  }

  // == Visualizing the Distribution of Returns ==
  def plotDistribution(samples: Array[Double], title: String): Figure = {
    val min = samples.min
    val max = samples.max

    val domain = Range.Double(min, max, (max - min) / 100)
      .toList.toArray
    val densities = KernalDensity.estimate(samples, domain)

    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two Week Return ($) of " + title
    p.ylabel = "Density"
    f
  }

  def plotDistribution(samples: RDD[Double], title: String): Figure = {
    val stats = samples.stats()
    val min = stats.min
    val max = stats.max

    val domain = Range.Double(min, max, (max - min) / 100)
      .toList.toArray
    val densities = KernalDensity.estimate(samples, domain)

    val f = Figure()
    val p = f.subplot(0)
    p += plot(domain, densities)
    p.xlabel = "Two Week Return ($) of " + title
    p.ylabel = "Density"
    f
  }
}
