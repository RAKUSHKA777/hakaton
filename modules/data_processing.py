import pandas as pd

def load_data():
    # Загрузка данных о клиентах и транзакциях
    clients = pd.read_csv("dataset/cntrbtrs_clnts_ops_trn.csv", sep=';', encoding='ISO-8859-1', low_memory=False)
    transactions = pd.read_csv("dataset/trnsctns_ops_trn.csv", sep=';', encoding='ISO-8859-1', low_memory=False)

    # Создание тестового файла (30% от данных)
    clients.to_csv("dataset/test_cntrbtrs_clnts.csv", index=False, encoding='utf-8', sep=';')


    # Возвращаем обработанные данные
    return clients, transactions
