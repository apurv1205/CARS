package advanced.ch5_kmeans

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Created by roger19890107 on 6/19/16.
  */
object RunKMeans {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/aas/ch5-network/"

  def main(args: Array[String]) {
    context.Env.setLogger
    val sc = context.Env.setupContext("K-means")
    val rawData = sc.textFile(base + "kddcup.data")

    //clusteringTake0(rawData)
    //clusteringTake1(rawData)
    //visualizationInR(rawData)
    //clusteringTake2(rawData)
    //normalVisualInR(rawData)
    //clusteringTake3(rawData)
    //clusteringTake4(rawData)
    //anomalies(rawData)
    detectAnomaliesInStream(sc, rawData)
  }

  def prepareLabelAndVectors(rawData: RDD[String]) = {
    rawData.map { line =>
      val buffer = line.split(",").toBuffer
      buffer.remove(1, 3)
      val label = buffer.remove(buffer.length - 1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
  }


  // take 0
  def clusteringTake0(rawData: RDD[String]): Unit = {
    rawData.map(_.split(",").last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

    val labelsAndData = prepareLabelAndVectors(rawData)

    val data = labelsAndData.values.cache()

    println("Run KMeans with default k=2 ...")
    val kmeans = new KMeans()
    val model = kmeans.run(data)

    model.clusterCenters.foreach(println)

    val clusterLableCount = labelsAndData.map { case (label, datum) =>
      val cluster = model.predict(datum)
      (cluster, label)
    }.countByValue()

    clusterLableCount.toSeq.sorted.foreach { case ((cluster, label), count) =>
      println(f"$cluster%1s$label%18s$count%8s")
    }

    data.unpersist()
  }

  // take 1
  def distance(a: Vector, b: Vector) =
    math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  def clusteringScore(data: RDD[Vector], k: Int): Double = {
    println("clusteringScore for " + k)
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

  def clusteringScore2(data: RDD[Vector], k: Int): Double = {
    println("clusteringScore2 for " + k)
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setMaxIterations(10)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

  def clusteringTake1(rawData: RDD[String]): Unit = {
    val data = prepareLabelAndVectors(rawData).values.cache()

    println("5 -> 30 scores:")
    (5 to 30 by 5).map(k => (k, clusteringScore(data, k)))
      .foreach(println)

    (30 to 100 by 10).par.map(k => (k, clusteringScore2(data, k)))
      .toList.foreach(println)

    data.unpersist()
  }

  def visualizationInR(rawData: RDD[String]): Unit = {
    val data = prepareLabelAndVectors(rawData).values.cache()

    val kmeans = new KMeans()
    kmeans.setK(100)
    kmeans.setMaxIterations(10)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(data)

    val sample = data.map(datum =>
      model.predict(datum) + "," + datum.toArray.mkString(",")
    ).sample(false, 0.05)

    sample.saveAsTextFile(base + "sample")

    data.unpersist()
  }

  // take 2
  def buildNormalizationFunction(data: RDD[Vector]): (Vector => Vector) = {
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(data)
    (datum: Vector) => stdScaler.transform(datum)
  }

  def clusteringTake2(rawData: RDD[String]): Unit = {
    val data = prepareLabelAndVectors(rawData).values
    val normalizeFunction = buildNormalizationFunction(data)
    val normalizedData = data.map(normalizeFunction).cache()

    (60 to 120 by 10).par
      .map(k => (k, clusteringScore2(normalizedData, k)))
      .toList.foreach(println)

    normalizedData.unpersist()
  }

  def normalVisualInR(rawData: RDD[String]): Unit = {
    val data = prepareLabelAndVectors(rawData).values
    val normalizeFunction = buildNormalizationFunction(data)
    val normalizedData = data.map(normalizeFunction).cache()

    val kmeans = new KMeans()
    kmeans.setK(120)
    kmeans.setMaxIterations(10)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedData)

    val sample = normalizedData.map(datum =>
      model.predict(datum) + "," + datum.toArray.mkString(",")
    ).sample(false, 0.05)

    sample.saveAsTextFile(base + "sample-norm")

    normalizedData.unpersist()
  }

  // take 3
  def buildCategoricalAndLabelFunction(rawData: RDD[String]): (String => (String, Vector)) = {
    val splitData = rawData.map(_.split(","))
    val protocols = splitData.map(_(1)).distinct().collect().zipWithIndex.toMap
    val services = splitData.map(_(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = splitData.map(_(3)).distinct().collect().zipWithIndex.toMap

    (line: String) => {
      val buffer = line.split(",").toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)
      val label = buffer.remove(buffer.length - 1)
      val vector = buffer.map(_.toDouble)

      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      val newServiceFeatures = new Array[Double](services.size)
      newServiceFeatures(services(service)) = 1.0
      val newTcpStateFeatures = new Array[Double](tcpStates.size)
      newTcpStateFeatures(tcpStates(tcpState)) = 1.0

      vector.insertAll(1, newTcpStateFeatures)
      vector.insertAll(1, newServiceFeatures)
      vector.insertAll(1, newProtocolFeatures)

      (label, Vectors.dense(vector.toArray))
    }
  }

  def clusteringTake3(rawData: RDD[String]): Unit = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    val data = rawData.map(parseFunction).values
    val normalizeFunction = buildNormalizationFunction(data)
    val normalizedData = data.map(normalizeFunction).cache()

    (80 to 160 by 10).map(k => (k, clusteringScore2(normalizedData, k)))
      .toList.foreach(println)

    normalizedData.unpersist()
  }

  // take 4
  def entropy(counts: Iterable[Int]) = {
    val values = counts.filter(_ > 0)
    val n: Double = values.sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  def clusteringScore3(normalizedLabelsAndData: RDD[(String, Vector)], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setMaxIterations(10)
    kmeans.setEpsilon(1.0e-6)

    val model = kmeans.run(normalizedLabelsAndData.values)

    // Predict cluster for each datum
    val labelsAndClusters = normalizedLabelsAndData.mapValues(model.predict)

    val clustersAndLabels = labelsAndClusters.map(_.swap)

    val labelsInCluster = clustersAndLabels.groupByKey().values

    val labelCouns = labelsInCluster.map(_.groupBy(l => l).map(_._2.size))

    val n = normalizedLabelsAndData.count()

    labelCouns.map(m => m.sum * entropy(m)).sum / n
  }

  def clusteringTake4(rawData: RDD[String]): Unit = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    val labelsAndData = rawData.map(parseFunction)
    val normalizeFunction = buildNormalizationFunction(labelsAndData.values)
    val normalizedLabelsAndData = labelsAndData.mapValues(normalizeFunction).cache()

    (80 to 160 by 10).map(k => (k, clusteringScore3(normalizedLabelsAndData, k)))
      .toList.foreach(println)

    normalizedLabelsAndData.unpersist()
  }

  // Detect Anomalies
  def buildAnomalyDetector(data: RDD[Vector],
                           normalizeFuction: (Vector => Vector)): (Vector => Boolean) = {
    val normalizedData = data.map(normalizeFuction).cache()

    val kmeans = new KMeans()
    kmeans.setK(150)
    kmeans.setMaxIterations(10)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedData)

    normalizedData.unpersist()

    val distances = normalizedData.map(datum => distToCentroid(datum, model))
    val threshold = distances.top(100).last

    (datum: Vector) => distToCentroid(datum, model) > threshold
  }

  def anomalies(rawData: RDD[String]) = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
    val data = originalAndData.values
    val normalizeFunction = buildNormalizationFunction(data)

    println("build anomaly detector ...")
    val anomalyDector = buildAnomalyDetector(data, normalizeFunction)
    val anomalies = originalAndData.filter {
      case (original, datum) => anomalyDector(datum)
    }.keys
    anomalies.take(10).foreach(println)
  }

  /* test in => nc -lk 9999
  testing data:
  0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,239,486,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,19,19,1.00,0.00,0.05,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,235,1337,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,29,29,1.00,0.00,0.03,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,219,1337,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,6,6,0.00,0.00,0.00,0.00,1.00,0.00,0.00,39,39,1.00,0.00,0.03,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,217,2032,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,6,6,0.00,0.00,0.00,0.00,1.00,0.00,0.00,49,49,1.00,0.00,0.02,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,217,2032,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,6,6,0.00,0.00,0.00,0.00,1.00,0.00,0.00,59,59,1.00,0.00,0.02,0.00,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,212,1940,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,2,0.00,0.00,0.00,0.00,1.00,0.00,1.00,1,69,1.00,0.00,1.00,0.04,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,159,4087,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.00,0.00,0.00,0.00,1.00,0.00,0.00,11,79,1.00,0.00,0.09,0.04,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,210,151,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,89,1.00,0.00,0.12,0.04,0.00,0.00,0.00,0.00,normal
  0,tcp,http,SF,212,786,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,99,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00,normal
  0,tcp,private,REJ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,107,6,0.00,0.00,1.00,1.00,0.06,0.07,0.00,255,6,0.02,0.05,0.00,0.00,0.00,0.00,1.00,1.00,normal
  */
  def detectAnomaliesInStream(sc: SparkContext, rawData: RDD[String]): Unit = {
    println("build anomaly detector ...")
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    val trainData = rawData.map(parseFunction).values
    val normalizeFunction = buildNormalizationFunction(trainData)
    val anomalyDector = buildAnomalyDetector(trainData, normalizeFunction)

    println("start streaming ...")
    val ssc = new StreamingContext(sc, Seconds(2))
    val lines = ssc.socketTextStream("localhost", 9999)
    val anomalies = lines
      .map(parseFunction)
      .filter(d => anomalyDector(d._2))
      .print()

    ssc.start()             // Start the computation
    ssc.awaitTermination()  // Wait for the computation to terminate
  }
}
