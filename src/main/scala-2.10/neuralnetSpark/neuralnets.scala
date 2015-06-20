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
 
      val conf = new SparkConf().setAppName("Simple Application").setMaster("local[8]")
      val sc = new SparkContext(conf)
      var path = "./src/main/scala-2.10/neuralnetSpark/xor"
      val xordataset =  sc.textFile(path)
      
      val data= xordataset.map{ line => 
       val parts = line.split(';')
       ( Vectors.dense(parts(1).split(' ').toArray.map{_.toDouble}),Vectors.dense(parts(0).toDouble))
      }
      data.persist()
      data.foreach(println)
      println("---------------------------------->")
      
      
      /* A simple example for XOR training
      */
      //Multilayerperceptron takes an Integer array of layers, for. eg : (2,2,1) Input = 2, hidden =2, output =1 
      val topology = FeedForwardTopology.multiLayerPerceptron(Array[Int](2,2, 1), false)
 
      //FeedforwardTrainer takes topology and input and output dimensions
      val trainer = new FeedForwardTrainer(topology, 2, 1)
  
      //Though SGD converges slowly, but step size of 20 ??????? 
      trainer.SGDOptimizer.setNumIterations(10000).setStepSize(10)
       
      val model = trainer.train(data)
 
      
      println(model.predict(Vectors.dense(1.0, 1.0)).toString())
      println(model.predict(Vectors.dense(1.0, 0)).toString())
      println(model.predict(Vectors.dense(0, 1.0)).toString())
      println(model.predict(Vectors.dense(0, 0)).toString())
 
      }
     
     }