import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from memory_profiler import profile
import gc
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define artifacts directory
ARTIFACTS_DIR = Path("/app/artifacts")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {mb:.2f} MB")

def batch_train(model, X, y, batch_size=1000):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        model.fit(X_batch, y_batch)
        gc.collect()

@profile
def train(processed_data):
    logging.info("Training Started")
    try:
        train_data, test_data = processed_data
        log_memory_usage()
        
        # Initialize the model
        rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
        
        # Train the model
        X_train = train_data.drop('leadQualified', axis=1)
        y_train = train_data['leadQualified']
        
        # Compute class weights for balanced learning
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        rf.class_weight = class_weight_dict
        
        batch_train(rf, X_train, y_train)
        del X_train, y_train, train_data
        gc.collect()
        log_memory_usage()
        
        # Score on Test Data
        X_test = test_data.drop('leadQualified', axis=1)
        y_test = test_data['leadQualified']
        
        predictions = rf.predict(X_test)
        
        report = classification_report(y_test, predictions)
        logger.info(f"Classification Report:\n{report}")

        # Save Model
        joblib.dump(rf, ARTIFACTS_DIR / 'rf_model.joblib')
        logger.info("Model saved successfully")

        # Save feature importances
        feature_importance = pd.DataFrame({
            'feature': rf.feature_names_in_,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance.to_csv(ARTIFACTS_DIR / 'feature_importance.csv', index=False)
        logger.info("Feature importances saved successfully")

        del X_test, y_test, test_data, rf
        gc.collect()
        log_memory_usage()

    except Exception as e:
        logger.exception("An error occurred during training")
        raise
    finally:
        gc.collect()
# import os
# import pandas as pd
# import numpy as np
# import joblib
# import logging
# from pathlib import Path
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("Log.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Define artifacts directory
# ARTIFACTS_DIR = Path("/app/artifacts")

# def train():
#     logging.info("Training Started")
#     try:
#         # Check if there is enough data to train the model
#         if not (ARTIFACTS_DIR / 'train_data.csv').exists():
#             logger.warning("Model will not be trained. Possible reason: not enough data")
#             return

#         # Read the Dataset in chunks
#         chunk_size = 10000  # Adjust this based on your available memory
#         chunks = pd.read_csv(ARTIFACTS_DIR / 'train_data.csv', chunksize=chunk_size)

#         rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
        
#         # Train the model in batches
#         for i, chunk in enumerate(chunks):
#             X = chunk.drop('leadQualified', axis=1)
#             y = chunk['leadQualified']
            
#             if i == 0:
#                 rf.fit(X, y)
#             else:
#                 rf.n_estimators += 10  # Increase trees with each chunk
#                 rf.fit(X, y)
            
#             del X, y  # Free up memory
#             logger.info(f"Processed chunk {i+1}")

#         # Score on Test Data
#         test_chunks = pd.read_csv(ARTIFACTS_DIR / 'test_data.csv', chunksize=chunk_size)
#         all_predictions = []
#         all_true_values = []

#         for chunk in test_chunks:
#             X_test = chunk.drop('leadQualified', axis=1)
#             y_test = chunk['leadQualified']
            
#             predictions = rf.predict(X_test)
#             all_predictions.extend(predictions)
#             all_true_values.extend(y_test)
            
#             del X_test, y_test  # Free up memory

#         report = classification_report(all_true_values, all_predictions)
#         logger.info(f"Classification Report:\n{report}")

#         # Save Model
#         joblib.dump(rf, ARTIFACTS_DIR / 'rf_model.joblib')
#         logger.info("Model saved successfully")

#         # Save feature importances
#         feature_importance = pd.DataFrame({
#             'feature': rf.feature_names_in_,
#             'importance': rf.feature_importances_
#         }).sort_values('importance', ascending=False)

#         feature_importance.to_csv(ARTIFACTS_DIR / 'feature_importance.csv', index=False)
#         logger.info("Feature importances saved successfully")

#     except Exception as e:
#         logger.exception("An error occurred during training")
#         raise