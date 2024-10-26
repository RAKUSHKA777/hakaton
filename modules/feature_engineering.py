def create_features(clients, transactions):
    # Агрегируем данные транзакций по клиенту
    transaction_features = transactions.groupby('accnt_id').agg({
        'sum': ['count', 'sum'],
        'mvmnt_type': 'nunique'
    }).reset_index()
    transaction_features.columns = ['accnt_id', 'transaction_count', 'total_sum', 'unique_movement_types']

    # Объединяем данные клиентов с признаками транзакций
    data = clients.merge(transaction_features, on='accnt_id', how='left')
    data.fillna(0, inplace=True)

    # Приводим категориальные признаки к строковому типу
    categorical_features = ['gndr', 'rgn', 'accnt_status']
    for feature in categorical_features:
        data[feature] = data[feature].astype(str)
    return data
