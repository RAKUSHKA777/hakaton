import pandas as pd
import os

def create_submission_file(model, transactions):
    # Путь до тестового файла
    test_file_path = "dataset/test_cntrbtrs_clnts.csv"

    # Проверка наличия файла
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    # Загрузка тестового файла
    test_data = pd.read_csv(test_file_path, sep=';', encoding='ISO-8859-1', low_memory=False)

    # Подготовка данных для прогнозирования
    test_data.fillna(0, inplace=True)  # Заполнение пропущенных значений

    # Создание признаков для прогнозирования
    transaction_features = transactions.groupby('accnt_id').agg({
        'sum': ['count', 'sum'],
        'mvmnt_type': 'nunique'
    }).reset_index()
    transaction_features.columns = ['accnt_id', 'transaction_count', 'total_sum', 'unique_movement_types']

    # Объединение клиентских данных с агрегированными данными о транзакциях
    test_data = test_data.merge(transaction_features, on='accnt_id', how='left')
    test_data.fillna(0, inplace=True)  # Заполните все оставшиеся пропущенные значения

    # Извлекайте функции для прогнозирования
    x_test = test_data.drop(columns=['accnt_id'])

    # Генерация прогнозы
    predictions = model.predict(x_test)

    # Создайте фрейм данных с результатами
    submission = pd.DataFrame({
        'accnt_id': test_data['accnt_id'],  # Использование accnt_id из тестовых данных
        'erly_pnsn_flg': predictions
    })

    # Save predictions
    submission_file_path = "dataset/submission.csv"
    submission.to_csv(submission_file_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {submission_file_path}")
