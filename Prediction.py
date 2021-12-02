#!/usr/bin/env python
# coding: utf-8


import pyspark 
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Wine_Quality_Prediction").enableHiveSupport().getOrCreate()


df = spark.read.csv('TrainingDataset.csv',header='true', inferSchema='true', sep=';')


new_column_name_list= list(map(lambda x: x.replace("\"\"", ""), df.columns))

df = df.toDF(*new_column_name_list)

df = df.withColumnRenamed("quality\"", "quality")


def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0



from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
tasty_udf_int = udf(isTasty, IntegerType())



df_tasty = df.withColumn("tasty", tasty_udf_int('quality'))



featureColumns = ["alcohol", "volatile acidity", "sulphates", "citric acid", "total sulfur dioxide", "density"]



from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")



trainingData = assembler.transform(df_tasty).select('features', 'tasty')



df_test = spark.read.csv('ValidationDataset.csv',header='true', inferSchema='true', sep=';')



new_column_name_list= list(map(lambda x: x.replace("\"\"", ""), df.columns))

df_test = df_test.toDF(*new_column_name_list)

df_test = df_test.withColumnRenamed("quality\"", "quality")



df_test_tasty = df_test.withColumn("tasty", tasty_udf_int('quality'))



featureColumns_test = ["alcohol", "volatile acidity", "sulphates", "citric acid", "total sulfur dioxide", "density"]



assembler_test = VectorAssembler(inputCols=featureColumns_test, 
                            outputCol="features")




testData = assembler_test.transform(df_test_tasty).select('features', 'tasty')




from pyspark.ml.classification import RandomForestClassifier

rfc = RandomForestClassifier(featuresCol='features',labelCol='tasty', numTrees=28)


rfc_model = rfc.fit(trainingData)


rfc_preds = rfc_model.transform(testData)


rfc_evaluator = MulticlassClassificationEvaluator(
    labelCol='tasty', predictionCol="prediction", metricName="f1")
rfc_f1 = rfc_evaluator.evaluate(rfc_preds)
print("f-score on RFC = %g" % rfc_f1)








