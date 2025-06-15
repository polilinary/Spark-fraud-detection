# Spark-fraud-detection

Активация окружения
   ```bash
source ~/spark_env/bin/activate
   ```

Запуск скрипта
   ```bash
python lakehouse_pipeline.py
   ```

По итогу создадутся нужные папки bronze, silver и gold, в них разложатся нужные данные в формате .parquette и сработает модель Random FOrest для предсказания банкротства, метрики (accuracy и точность) которой выведутся в терминале.
