import pandas as pd
import logging

logger = logging.getLogger(__name__)


def log_feature_importance(model):
    # Извлечение важности признаков
    best_model = model.named_steps['classifier']
    importances = best_model.feature_importances_

    # Получение имен числовых и категориальных признаков
    feature_names = model.named_steps['preprocessor'].transformers_[0][2] + \
                    list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())

    # Создание DataFrame с важностью признаков
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    # Сохранение важности признаков в CSV файл
    feature_importances.to_csv("dataset/feature_importances.csv", index=False, encoding='utf-8', errors='ignore')

    # Логирование простого сообщения о завершении
    logger.info("Важность признаков получена.")
