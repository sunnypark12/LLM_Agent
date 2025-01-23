import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from helper import calculate_other_factors, calculate_stage_score, adjust_probability, get_column_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("/app/artifacts")

def preprocessing(df):
    try:
        model_path = ARTIFACTS_DIR / "rf_model.joblib"
        scaler_path = ARTIFACTS_DIR / "scaler.joblib"
        encoder_path = ARTIFACTS_DIR / "encoder.joblib"
        imputer_mode_path = ARTIFACTS_DIR / "imputer_mode.joblib"
        imputer_median_path = ARTIFACTS_DIR / "imputer_median.joblib"

        if not all(path.exists() for path in [model_path, scaler_path, encoder_path, imputer_mode_path, imputer_median_path]):
            raise FileNotFoundError("One or more required model files are missing")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        enc = joblib.load(encoder_path)
        imputer_mode = joblib.load(imputer_mode_path)
        imputer_median = joblib.load(imputer_median_path)

        feature_columns = [
            'sourceName', 'industryName', 'annualRevenueName', 'ratingName', 'ownershipName', 
            'employeesRangeName', 'country', 'state', 'city', 'leadsStatus', 'doNotCall', 
            'doNotEmail', 'daysSinceLastActivity', 'numberOfActivities', 'emailAvailable', 
            'phoneAvailable', 'socialProfileAvailable', 'contactEmailAvailable', 
            'contactPhoneAvailable', 'stageScore'
        ]

        cat_cols = [
            'sourceName', 'industryName', 'annualRevenueName', 'ratingName', 'ownershipName', 
            'employeesRangeName', 'country', 'state', 'city', 'leadsStatus'
        ]

        numeric_cols = ['daysSinceLastActivity', 'numberOfActivities', 'stageScore']

        var_cols = [
            'doNotCall', 'doNotEmail', 'emailAvailable', 'phoneAvailable', 
            'socialProfileAvailable', 'contactEmailAvailable', 'contactPhoneAvailable'
        ]

        df = df[feature_columns]
        df = df.fillna(np.nan)

        mode_cols = cat_cols + var_cols
        df[mode_cols] = imputer_mode.transform(df[mode_cols])
        df[numeric_cols] = imputer_median.transform(df[numeric_cols])
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Handle the encoder output
        encoded_features = enc.get_feature_names_out(cat_cols)
        encoded_data = enc.transform(df[cat_cols])
        
        # Convert sparse matrix to dense array
        encoded_data_dense = encoded_data.toarray()
        
        # Create a new DataFrame with encoded columns
        encoded_df = pd.DataFrame(encoded_data_dense, columns=encoded_features, index=df.index)
        
        # Drop original categorical columns and concatenate with encoded columns
        df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

        # Ensure all columns from training are present
        with open(ARTIFACTS_DIR / 'feature_names.txt', 'r', encoding='utf-8') as f:
            training_features = f.read().splitlines()

        for col in training_features:
            if col not in df.columns:
                df[col] = 0  # or another appropriate default value

        df = df[training_features]  # Reorder columns to match training data

        return df, model
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise
# def preprocessing(df):
#     try:
#         model_path = ARTIFACTS_DIR / "rf_model.joblib"
#         scaler_path = ARTIFACTS_DIR / "scaler.joblib"
#         encoder_path = ARTIFACTS_DIR / "encoder.joblib"
#         imputer_mode_path = ARTIFACTS_DIR / "imputer_mode.joblib"
#         imputer_median_path = ARTIFACTS_DIR / "imputer_median.joblib"

#         if not all(path.exists() for path in [model_path, scaler_path, encoder_path, imputer_mode_path, imputer_median_path]):
#             raise FileNotFoundError("One or more required model files are missing")

#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         enc = joblib.load(encoder_path)
#         imputer_mode = joblib.load(imputer_mode_path)
#         imputer_median = joblib.load(imputer_median_path)

#         feature_columns = [
#             'sourceName', 'industryName', 'annualRevenueName', 'ratingName', 'ownershipName', 
#             'employeesRangeName', 'country', 'state', 'city', 'leadsStatus', 'doNotCall', 
#             'doNotEmail', 'daysSinceLastActivity', 'numberOfActivities', 'emailAvailable', 
#             'phoneAvailable', 'socialProfileAvailable', 'contactEmailAvailable', 
#             'contactPhoneAvailable', 'stageScore'
#         ]

#         cat_cols = [
#             'sourceName', 'industryName', 'annualRevenueName', 'ratingName', 'ownershipName', 
#             'employeesRangeName', 'country', 'state', 'city', 'leadsStatus'
#         ]

#         numeric_cols = ['daysSinceLastActivity', 'numberOfActivities', 'stageScore']

#         var_cols = [
#             'doNotCall', 'doNotEmail', 'emailAvailable', 'phoneAvailable', 
#             'socialProfileAvailable', 'contactEmailAvailable', 'contactPhoneAvailable'
#         ]

#         df = df[feature_columns]
#         df = df.fillna(np.nan)

#         mode_cols = cat_cols + var_cols
#         df[mode_cols] = imputer_mode.transform(df[mode_cols])
#         df[numeric_cols] = imputer_median.transform(df[numeric_cols])
#         df[numeric_cols] = scaler.transform(df[numeric_cols])

#         # Handle the encoder output directly (it's already an array)
#         encoded_features = enc.get_feature_names_out(cat_cols)
#         encoded_data = enc.transform(df[cat_cols])
#         df = df.drop(columns=cat_cols)
#         for i, col in enumerate(encoded_features):
#             df[col] = encoded_data[:, i]

#         # Ensure all columns from training are present
#         with open(ARTIFACTS_DIR / 'feature_names.txt', 'r', encoding='utf-8') as f:
#             training_features = f.read().splitlines()

#         for col in training_features:
#             if col not in df.columns:
#                 df[col] = 0  # or another appropriate default value

#         df = df[training_features]  # Reorder columns to match training data

#         return df, model
#     except Exception as e:
#         logger.error(f"Error in preprocessing: {str(e)}")
#         raise

# def score(data):
#     try:
#         result = []
#         data['stageScore'] = data.apply(calculate_stage_score, axis=1)

#         with open(ARTIFACTS_DIR / "mappings.json", "r", encoding="utf-8") as file:
#             mappings = json.load(file)

#         with open(ARTIFACTS_DIR / "statistics.json", 'r') as json_file:
#             statistics_data = json.load(json_file)
            
#         raw_values = {col: [f'{col}_{val}' for val in data[col].unique()] for col in ['state', 'city', 'country', 'industryName', 'sourceName', 'annualRevenueName', 'employeesRangeName']}
#         top_values = {col: list(statistics_data[col].keys())[:3] for col in ['state', 'city', 'country', 'industryName', 'sourceName', 'annualRevenueName', 'employeesRangeName']}
        
#         processed_data, model = preprocessing(data)
#         predictions = model.predict_proba(processed_data)
#         feature_importance = zip(model.feature_importances_, model.feature_names_in_)
#         sorted_feature_importance = sorted(feature_importance, key=lambda x: x[0], reverse=True)

#         factors = ["emailAvailable", "phoneAvailable", "socialProfileAvailable",
#                    "contactEmailAvailable", "contactPhoneAvailable",
#                    "sourceName", "country"]

#         for idx, res in enumerate(predictions):
#             feature_scores = []
#             other_factors = calculate_other_factors(data, idx)
#             mapping_result = get_column_mapping(data.iloc[idx], top_values, raw_values)

#             for feature_score, feature_name in sorted_feature_importance:
#                 if feature_name not in factors and \
#                     '_' not in feature_name and \
#                     feature_name in mappings and \
#                     feature_name not in ['doNotCall', 'doNotEmail']:

#                     feature_scores.append({
#                         "feature": feature_name,
#                         "name": mappings[feature_name],
#                         "score": round(processed_data.iloc[idx][feature_name] * feature_score, 4)
#                     })

#             if pd.notna(data.loc[idx, "stageId"]) and str(data.loc[idx, "stageId"]) == '5':
#                 score = 100
#             elif pd.notna(data.loc[idx, "stageId"]) and str(data.loc[idx, "stageId"]) == '6':
#                 score = 0
#             else:
#                 min_prob = float(data.loc[idx, 'currentPosition']) / float(data.loc[idx, 'maxPosition'])
#                 max_prob = min(1, (float(data.loc[idx, 'currentPosition'] + 1) / float(data.loc[idx, 'maxPosition'])))
#                 score = adjust_probability(res[1], min_prob, max_prob) * 100

#             result.append({
#                 "id": int(data.loc[idx, "id"]),
#                 "score": round(score, 2),
#                 "featureScores": feature_scores,
#                 "otherFactors": other_factors,
#                 "staticFactors": mapping_result
#             })

#         return result
#     except Exception as e:
#         logger.exception("An error occurred during scoring")
#         raise

def score(data):
    try:
        """
        This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
        In the example we extract the data from the json input and call the scikit-learn model's predict()
        method and return the result back
        """
        result = []
        data['stageScore'] = data.apply(calculate_stage_score, axis=1)

        with open(ARTIFACTS_DIR / "mappings.json", "r", encoding="utf-8") as file:
            mappings = json.load(file)

        with open(ARTIFACTS_DIR / "statistics.json", 'r') as json_file:
            statistics_data = json.load(json_file)
            
        # Get unique values from raw data for the specified columns
        raw_values = {col: [f'{col}_{val}' for val in data[col].unique()] for col in ['state', 'city', 'country', 'industryName', 'sourceName', 'annualRevenueName', 'employeesRangeName']}

        # Extract top 3 keys for each column from statistics_data
        top_values = {col: list(statistics_data[col].keys())[:3] for col in ['state', 'city', 'country', 'industryName', 'sourceName', 'annualRevenueName', 'employeesRangeName']}
        
        processed_data, model = preprocessing(data)
        predictions = model.predict_proba(processed_data)
        feature_importance = zip(model.feature_importances_, model.feature_names_in_)
        sorted_feature_importance = sorted(feature_importance, key=lambda x: x[0], reverse=True)

        factors = ["emailAvailable", "phoneAvailable", "socialProfileAvailable",
                "contactEmailAvailable", "contactPhoneAvailable",
                "sourceName", "country"]

        for idx, res in enumerate(predictions):
            feature_scores = []
            other_factors = calculate_other_factors(data, idx)

            # Call find_column_mapping for the current row
            mapping_result = get_column_mapping(data.iloc[idx], top_values, raw_values)

            for feature_score, feature_name in sorted_feature_importance:
                if feature_name not in factors and \
                    '_' not in feature_name and \
                    feature_name in mappings and \
                    feature_name not in ['doNotCall', 'doNotEmail']:

                    feature_scores.append({
                        "feature": feature_name,
                        "name": mappings[feature_name],
                        "score": round(processed_data.loc[idx, feature_name] * feature_score, 4)
                    })

            if int(data.loc[idx, "stageId"]) == 5:
                result.append({
                    "id": int(data.loc[idx, "id"]),
                    "score": round(1 * 100, 2),
                    "featureScores": feature_scores,
                    "otherFactors": other_factors,
                    "staticFactors": mapping_result  # Include mapping result in the output
                })
            elif int(data.loc[idx, "stageId"]) == 6:
                result.append({
                    "id": int(data.loc[idx, "id"]),
                    "score": round(0 * 100, 2),
                    "featureScores": feature_scores,
                    "otherFactors": other_factors,
                    "staticFactors": mapping_result  # Include mapping result in the output
                })
            else:
                min_prob = data.loc[idx, 'currentPosition'] / data.loc[idx, 'maxPosition']
                if (data.loc[idx, 'currentPosition'] + 1) < data.loc[idx, 'maxPosition']:
                    max_prob = (data.loc[idx, 'currentPosition'] + 1) / data.loc[idx, 'maxPosition']
                else:
                    max_prob = 1

                score = adjust_probability(res[1], min_prob, max_prob)

                result.append({
                    "id": int(data.loc[idx, "id"]),
                    "score": round(score * 100, 2),
                    "featureScores": feature_scores,
                    "otherFactors": other_factors,
                    "staticFactors": mapping_result  # Include mapping result in the output
                })

        # logging.info("Request processed")
        return result
    except Exception as e:
        logging.exception("An error occurred during scoring")
        print(f"Error occured during scoring: {str(e)}")
