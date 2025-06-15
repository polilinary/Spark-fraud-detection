from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.ml import Pipeline
import mlflow
import os

# Инициализация Spark 
spark = SparkSession.builder \
    .appName("FraudDetection") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.2.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4G") \
    .getOrCreate()

# Создание структуры папок
os.makedirs("data/bronze", exist_ok=True)
os.makedirs("data/silver", exist_ok=True)
os.makedirs("data/gold", exist_ok=True)

# =============================================
# 1. Загрузка в Bronze (исходные данные)
# =============================================
print("Загрузка данных в Bronze...")
df = spark.read.csv("data/fraud_test.csv", header=True, inferSchema=True)
df.write.format("delta").mode("overwrite").save("data/bronze/fraud_data")

# =============================================
# 2. Обработка в Silver (очистка)
# =============================================
print("Обработка данных в Silver...")
bronze_df = spark.read.format("delta").load("data/bronze/fraud_data")

# Удаление ненужных колонок + фильтрация
silver_df = bronze_df.drop("Unnamed: 0", 'trans_date_trans_time', "trans_num", 
                           "unix_time").filter(col("amt") < 10000)

# Кодирование категорий
category_cols = ["merchant", "category", "gender"]
for col_name in category_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx")
    silver_df = indexer.fit(silver_df).transform(silver_df)

silver_df.write.format("delta").mode("overwrite").save("data/silver/fraud_cleaned")


# =============================================
# 3. Агрегации в Gold
# =============================================
print("Агрегация данных в Gold...")
silver_df = spark.read.format("delta").load("data/silver/fraud_cleaned")

# Агрегация по категориям
gold_agg = silver_df.groupBy("category") \
    .agg(
        count("*").alias("transaction_count"),
        sum("is_fraud").alias("fraud_count"),
        mean("amt").alias("avg_amount")
    )

gold_agg.write.format("delta").mode("overwrite").save("data/gold/fraud_analytics")

# =============================================
# 4. Машинное обучение
# =============================================
print("Обучение модели...")
mlflow.set_experiment("FraudDetection")

# Подготовка фичей - ИСКЛЮЧАЕМ merchant_idx из-за слишком большого числа категорий
feature_cols = [c for c in silver_df.columns 
               if c.endswith("_idx") and c != "merchant_idx" 
               or c in ["amt", "age"]]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Разделение данных
train, test = silver_df.randomSplit([0.8, 0.2], seed=42)

# Настройка модели с увеличенным maxBins
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=100,
    maxDepth=5,
    maxBins=700,
    subsamplingRate=0.8,
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])

# Обучение с обработкой ошибок
try:
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params({
            "numTrees": 100,
            "maxDepth": 5,
            "maxBins": 700,
            "features": ", ".join(feature_cols),
            "excluded_features": "merchant_idx"
        })
        
        model = pipeline.fit(train)
        predictions = model.transform(test)
        
        # Вычисляем метрики
        accuracy = predictions.filter(col("is_fraud") == col("prediction")).count() / test.count()
        evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction")
        auc = evaluator.evaluate(predictions)
        
        # Логируем метрики
        mlflow.log_metrics({
            "accuracy": accuracy,
            "AUC": auc
        })
        
        # Сохраняем модель
        mlflow.spark.log_model(model, "model")
        print(f"Точность модели: {accuracy:.2%}")
        print(f"AUC: {auc:.4f}")
        
except Exception as e:
    print(f"Ошибка при обучении модели: {str(e)}")



print("Лабораторная работа успешно завершена!")


spark.stop() 