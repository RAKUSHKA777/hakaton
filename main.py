import logging
import os
from modules import data_processing, feature_engineering, model_training, prediction, interpretability
import joblib

# Создание директорий для логов и данных, если они отсутствуют
os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Загрузка и обработка данных")
    clients, transactions = data_processing.load_data()

    logger.info("Создание признаков")
    data = feature_engineering.create_features(clients, transactions)

    # Проверка наличия целевой переменной
    if 'erly_pnsn_flg' in data.columns:
        # Если целевая переменная присутствует, обучаем модель
        logger.info("Обучение модели")
        model, f1_score_val = model_training.train_model(data)
        logger.info(f"Модель обучена. Значение F1-score: {f1_score_val:.4f}")
    else:
        # Если целевая переменная отсутствует, загружаем сохранённую модель для предсказания
        model_path = 'dataset/model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Модель загружена для предсказаний.")
            logger.info("Для предсказаний требуются тестовые данные: 'cntrbtrs_clnts_ops_tst.csv' и 'trnsctns_ops_tst.csv'")
        else:
            raise FileNotFoundError("Модель не найдена. Сначала обучите модель на данных с целевой переменной.")

    # Создание предсказаний и генерация файла sample_submission.csv
    logger.info("Создание предсказаний и генерация файла sample_submission.csv")
    prediction.create_submission_file(model, transactions)

    # Анализ важности признаков для интерпретируемости
    logger.info("Анализ важности признаков для интерпретируемости")
    interpretability.log_feature_importance(model)

    # Ожидание закрытия консоли
    input("Нажмите любую клавишу, чтобы выйти...")

if __name__ == "__main__":
    main()
