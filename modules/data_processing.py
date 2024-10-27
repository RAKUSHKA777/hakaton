import pandas as pd
import os

def load_data():
    # Определение путей к основным и альтернативным файлам
    clients_train_path = "dataset/cntrbtrs_clnts_ops_trn.csv"
    transactions_train_path = "dataset/trnsctns_ops_trn.csv"
    clients_test_path = "dataset/cntrbtrs_clnts_ops_tst.csv"
    transactions_test_path = "dataset/trnsctns_ops_tst.csv"

    # Загрузка данных о клиентах
    if os.path.exists(clients_train_path):
        clients = pd.read_csv(clients_train_path, sep=';', encoding='ISO-8859-1', low_memory=False)
    else:
        clients = pd.read_csv(clients_test_path, sep=';', encoding='ISO-8859-1', low_memory=False)

    # Загрузка данных о транзакциях
    if os.path.exists(transactions_train_path):
        transactions = pd.read_csv(transactions_train_path, sep=';', encoding='ISO-8859-1', low_memory=False)
    else:
        transactions = pd.read_csv(transactions_test_path, sep=';', encoding='ISO-8859-1', low_memory=False)

    # Сохраняем все данные о клиентах в качестве тестового файла (для создания submission)
    clients.to_csv("dataset/test_cntrbtrs_clnts.csv", index=False, encoding='utf-8', sep=';')

    return clients, transactions
