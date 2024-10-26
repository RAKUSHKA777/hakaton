import pandas as pd
import os

def create_submission_file(model, transactions):
    # Path to the test file
    test_file_path = "dataset/test_cntrbtrs_clnts.csv"

    # Check if the test file exists
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    # Load test data
    test_data = pd.read_csv(test_file_path, sep=';', encoding='ISO-8859-1', low_memory=False)

    # Prepare data for prediction
    test_data.fillna(0, inplace=True)  # Fill missing values

    # Create features for test data
    transaction_features = transactions.groupby('accnt_id').agg({
        'sum': ['count', 'sum'],
        'mvmnt_type': 'nunique'
    }).reset_index()
    transaction_features.columns = ['accnt_id', 'transaction_count', 'total_sum', 'unique_movement_types']

    # Combine customer data with aggregated transaction data
    test_data = test_data.merge(transaction_features, on='accnt_id', how='left')
    test_data.fillna(0, inplace=True)  # Fill any remaining missing values

    # Extract features for prediction
    x_test = test_data.drop(columns=['accnt_id'])

    # Generate predictions
    predictions = model.predict(x_test)

    # Create a DataFrame with results
    submission = pd.DataFrame({
        'accnt_id': test_data['accnt_id'],  # Using accnt_id from test data
        'erly_pnsn_flg': predictions
    })

    # Save predictions
    submission_file_path = "dataset/submission.csv"
    submission.to_csv(submission_file_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {submission_file_path}")
