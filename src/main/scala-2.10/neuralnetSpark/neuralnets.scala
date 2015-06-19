package neuralnetSpark 

import org.apache.spark.mllib.ann._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.ann.{FeedForwardTrainer, Topology}
import org.apache.spark.mllib.classification.ANNClassifier
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
 
object neuralnets {
  
     def main(args: Array[String])   { 
      var myArray = Array.ofDim[Double](100, 60)
      val logFile = "/home/user/workspace/cass_spark/src/main/scala-2.11/cass_spark/abc" // Should be some file on your system
      val conf = new SparkConf().setAppName("Simple Application").setMaster("local[8]")
      val sc = new SparkContext(conf)
      var path = "/home/user/MEGA/workspace/workspace/neuralnetSpark/src/main/scala-2.10/neuralnetSpark/xor"
      val xordataset =  sc.textFile(path)
      
     val data= xordataset.map{ line => 
       val parts = line.split(';')
       ( Vectors.dense(parts(1).split(' ').toArray.map{_.toDouble}),Vectors.dense(parts(0).toDouble))
      }
      data.cache()
      data.foreach(println)
      println("---------------------------------->")
      val topology = FeedForwardTopology.multiLayerPerceptron(Array[Int](2,4, 1), false)
      val initialWeights = FeedForwardModel(topology, 23124).weights()
       
      val trainer = new FeedForwardTrainer(topology, 2, 1)
      trainer.setWeights(initialWeights)
       
      trainer.SGDOptimizer.setNumIterations(100000).setStepSize(0.1) 
       
      val model = trainer.train(data)
      
     
      println(model.predict(Vectors.dense(1.0, 1.0)).toString())
      println(model.predict(Vectors.dense(1.0, 0)).toString())
      println(model.predict(Vectors.dense(0, 1.0)).toString())
      println(model.predict(Vectors.dense(0, 0)).toString())
      
      /*
      val logData = sc.textFile(logFile, 2).cache()
      val numAs = logData.filter(line => line.contains("a")).count()
      val numBs = logData.filter(line => line.contains("b")).count()
      println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
 */
      }
     
     }