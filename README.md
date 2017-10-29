# CARS
Context Aware Recommendation System

It is a scala project to be run on spark clusters. SBT is needed to build and compile the project

#TO RUN :
cd into directory

run the following commands :
1. sbt clean
2. sbt compile
3. sbt package
4.  $SPARK_HOME/bin/spark-submit --class com.github.b96705008.basic.recommender.knn.KNNExample --master local[4] target/scala-2.11/simple-project_2.11-1.0.jar

Can run other objects containing main also in the similar way.

Can also be used to open via intellij idea 17 community edition