package Regression
package Regression

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.SQLContext
object LogisticRegression extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark= SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read.option("header","true").format("csv").load("src/main/scala/Resources/data.csv")
  data.printSchema()

  import spark.implicits._
  val logregdataall= data.select(data("Position").as("label") , $"ID", $"Age", $"Overall", $"Special",
    $"Weak Foot", $"Skill Moves", $"crossing", $"Finishing", $"HeadingAccuracy", $"ShortPassing",
    $"Volleys", $"Dribbling", $"Curve", $"FKAccuracy", $"LongPassing", $"BallControl", $"Acceleration", $"SprintSpeed", $"Agility",
    $"Reactions", $"Balance", $"ShotPower", $"Jumping", $"Stamina", $"Strength", $"LongShots", $"Aggression", $"Interceptions", $"Positioning", $"Vision",
    $"Penalties", $"Composure", $"Marking", $"StandingTackle", $"SlidingTackle", $"GKDiving", $"GKHandling", $"GKKicking",
    $"GKPositioning", $"GKReflexes", $"Preferred Foot")

  logregdataall.printSchema()




}
