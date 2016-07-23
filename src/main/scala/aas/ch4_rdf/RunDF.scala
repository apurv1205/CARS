package aas.ch4_rdf

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

/**
  * Created by roger19890107 on 6/13/16.
  */
object RunDF {
  val base = "file:///Volumes/RogerDrive/Developer/dataset/aas/ch4-covtype/"

  def main(args: Array[String]) {
    context.Env.setLogger
    val sc = context.Env.setupContext("RDF")

    // load covtype data
    val rawData = sc.textFile(base + "covtype.data")

    //evaluateSimple(rawData)
    //evaluateCategorical(rawData)
    evaluateForest(rawData)
  }

  def simpleDecisionTree(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)

    val metrics = getMetrics(model, cvData)

    println(metrics.confusionMatrix)
    println(metrics.precision)

    (0 until 7).map(
      category => (metrics.precision(category), metrics.recall(category))
    ).foreach(println)
  }

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def randomClassifier(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    val trainPriorProbs = classProbabilities(trainData)
    val cvPriorProbs = classProbabilities(cvData)
    val accuracy = trainPriorProbs.zip(cvPriorProbs).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
    println(accuracy)
  }

  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }

  def evaluate(trainData: RDD[LabeledPoint],
               cvData: RDD[LabeledPoint],
               testData: RDD[LabeledPoint]): Unit = {

    val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth <- Array(1, 20);
           bins <- Array(10, 300))
        yield {
          val model = DecisionTree.trainClassifier(
            trainData, 7, Map[Int, Int](), impurity, depth, bins)
          val accuracy = getMetrics(model, cvData).precision
          ((impurity, depth, bins), accuracy)
        }

    evaluations.sortBy(_._2).reverse.foreach(println)

    val model = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map[Int, Int](), "entropy", 20, 300)
    println(getMetrics(model, testData).precision)
  }

  def evaluateSimple(rawData: RDD[String]): Unit = {
    val data = rawData.map { line =>
      val values = line.split(",").map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }

    // split dataset
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    // train model
    simpleDecisionTree(trainData, cvData)
    randomClassifier(trainData, cvData)
    evaluate(trainData, cvData, testData)

    // unpersist
    trainData.unpersist()
    cvData.unpersist()
    testData.unpersist()
  }

  def uncodeOneHot(rawData: RDD[String]): RDD[LabeledPoint] = {
    rawData.map { line =>
      val values = line.split(",").map(_.toDouble)
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      val featureVector =
        Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
  }

  def evaluateCategorical(rawData: RDD[String]): Unit = {
    val data = uncodeOneHot(rawData)

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth <- Array(10, 20, 30);
           bins <- Array(40, 300))
        yield {
          val model = DecisionTree.trainClassifier(
            trainData, 7, Map[Int, Int](10 -> 4, 11 -> 40), impurity, depth, bins)
          val trainAccuracy = getMetrics(model, trainData).precision
          val cvAccuracy = getMetrics(model, cvData).precision
          ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
        }

    evaluations.sortBy(_._2._2).reverse.foreach(println)

    val model = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map[Int, Int](10 -> 4, 11 -> 40), "entropy", 30, 300)
    println(getMetrics(model, testData).precision)

    trainData.unpersist()
    cvData.unpersist()
    testData.unpersist()
  }

  def evaluateForest(rawData: RDD[String]): Unit = {
    val data = uncodeOneHot(rawData)

    val Array(trainData, cvData) = data.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    val forest = RandomForest.trainClassifier(
      trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)

    val predictionAndLabels = cvData.map(example =>
      (forest.predict(example.features), example.label))
    println(new MulticlassMetrics(predictionAndLabels).precision)

    val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
    val vector = Vectors.dense(input.split(",").map(_.toDouble))
    println(forest.predict(vector))

    trainData.unpersist()
    cvData.unpersist()
  }
}
