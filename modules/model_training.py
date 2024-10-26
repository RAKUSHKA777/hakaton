import os
import joblib
import logging  # Add logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score

# Initialize a logger
logger = logging.getLogger(__name__)


def train_model(data):
    # Выделение целевой переменной и идентификаторов
    x = data.drop(columns=['erly_pnsn_flg', 'accnt_id'])
    y = data['erly_pnsn_flg']

    # Разделение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Пайплайн предобработки
    numerical_features = ['prsnt_age', 'transaction_count', 'total_sum', 'unique_movement_types']
    categorical_features = ['gndr', 'rgn', 'accnt_status']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Модель
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            max_depth=None,
            min_samples_split=5,
            n_estimators=100,
            random_state=42
        ))
    ])

    # Сохранение модели
    model_path = 'dataset/model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)

    # Calculate F1 score
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)

    # Log and print F1 score
    logger.info(f"Model training completed with F1-score: {f1:.4f}")
    print(f"Model training completed with F1-score: {f1:.4f}")

    return model, f1
