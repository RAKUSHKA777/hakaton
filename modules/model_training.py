import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def train_model(data):
    x = data.drop(columns=['erly_pnsn_flg', 'accnt_id'])
    y = data['erly_pnsn_flg']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    numerical_features = ['prsnt_age', 'transaction_count', 'total_sum', 'unique_movement_types']
    categorical_features = ['gndr', 'rgn', 'accnt_status']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            max_depth=None,
            min_samples_split=5,
            n_estimators=100,
            random_state=42
        ))
    ])

    model_path = 'dataset/model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)

        # Удаление файлов обучающих данных
        training_files = ["dataset/cntrbtrs_clnts_ops_trn.csv", "dataset/trnsctns_ops_trn.csv"]
        for file in training_files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Файл {file} успешно удалён.")

        # Сообщение о необходимости тестовых данных
        logger.info(
            "Для предсказаний требуются тестовые данные: 'cntrbtrs_clnts_ops_tst.csv' и 'trnsctns_ops_tst.csv'.")

    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Model training completed with F1-score: {f1:.4f}")
    print(f"Model training completed with F1-score: {f1:.4f}")

    return model, f1
