# Прогнозирование Флага Досрочного Пенсионного Обеспечения

Этот проект представляет собой систему машинного обучения для прогнозирования, имеет ли клиент флаг досрочного пенсионного обеспечения (early pension flag). Модель обучается на данных о клиентах и их транзакциях и предсказывает вероятный флаг на основе различных характеристик.

## Структура проекта

Проект состоит из нескольких модулей, отвечающих за этапы обработки данных, обучения модели, предсказания и интерпретации результатов.

### Содержание корневой папки:

- **`main.py`** — основной исполняемый файл, запускающий весь процесс: загрузку данных, обработку, обучение модели, предсказание и анализ важности признаков.
- **`dataset`** — папка для хранения данных и результатов работы модели.
- **`logs`** — папка для хранения логов, отслеживающих работу программы.
- **`modules`** — папка с основными модулями проекта, каждый из которых выполняет одну из задач.
- **`install_requirements.bat`** — бат-файл для автоматической установки всех необходимых библиотек, указанных в **`requirements.txt`** .
- **`readme.md`** — файл документации проекта, описывающий его назначение, структуру, порядок установки и использования.
- **`requirements.txt`** — файл с зависимостями Python, необходимыми для работы проекта.

### Структура папок

- **modules/** — папка с основными модулями проекта:
  - **`data_processing.py`** — модуль для загрузки данных о клиентах и транзакциях. Создает тестовый файл на основе клиентских данных.
  - **`feature_engineering.py`** — модуль для создания признаков из данных клиентов и транзакций.
  - **`model_training.py`** — модуль для обучения модели `RandomForestClassifier`. Сохраняет обученную модель в файл.
  - **`prediction.py`** — модуль для создания предсказаний на основе тестовых данных.
  - **`interpretability.py`** — модуль для анализа важности признаков модели, сохраняет результаты для интерпретации.

- **dataset/** — папка для хранения данных и результатов работы модели:
  - `cntrbtrs_clnts_ops_trn.csv` — основной файл с данными о клиентах.
  - `trnsctns_ops_trn.csv` — основной файл с данными о транзакциях.
  - `test_cntrbtrs_clnts.csv` — автоматически создается из `cntrbtrs_clnts_ops_trn.csv`, содержит 30% случайной выборки для тестирования.
  - `model.pkl` — обученная модель, сохраняется здесь после обучения.
  - `submission.csv` — файл предсказаний, создаваемый после выполнения скрипта, содержит идентификаторы клиентов и предсказанный флаг досрочного пенсионного обеспечения.
  - `feature_importances.csv` — файл с важностью признаков для интерпретации результатов.

- **logs/** — папка для хранения логов:
  - `training.log` — основной лог-файл, где записываются этапы обработки данных, обучения модели и результаты анализа.
## Установка и Настройка

### Шаг 1: Установка зависимостей
В проекте используются библиотеки Python, такие как `pandas`, `scikit-learn`, и `joblib`. Все зависимости можно установить автоматически, запустив следующий бат-файл:

```bash
install_requirements.bat