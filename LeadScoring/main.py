import logging
import uvicorn
from fastapi import FastAPI
from processing import preprocess
from training import train
from a2wsgi import ASGIMiddleware
from scoring import score
from helper import DataModel, get_recent_data
import pandas as pd
import warnings 
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore')

app = FastAPI(
    title="LeadScoreV2",
    docs_url="/LeadScoreV2/api/docs",
    redoc_url="/LeadScoreV2/api/redoc",
    openapi_url="/LeadScoreV2/api/openapi.json"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Log.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# CORS settings
origins = ["*"]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Wrap the FastAPI app in ASGIMiddleware
wsgi_app = ASGIMiddleware(app)
logging.info("App created successfully")

# Define the root endpoint
@app.get("/LeadScoreV2")
async def read_root():
    logging.info("Connection Build Successfully")
    return {"message": "Connection Build Successfully"}

# Define the train endpoint
@app.post("/LeadScoreV2/api/train")
async def modeltrain():
    try:
        logging.info("Starting training process")
        
        # Get the latest data
        leads_df, dataset_name = get_recent_data()
        logging.info(f"Retrieved data from: {dataset_name}")
        
        # Call the preprocess function
        processed_data = preprocess(leads_df)
        logging.info("Preprocessing completed")

        # Call the train function
        train(processed_data)
        logging.info("Model training completed")

        return {"message": "Preprocess and train functions executed successfully"}
    except Exception as e:
        logging.exception("An error occurred during training")
        return {"message": str(e), "statusCode": 500}

# Define the score endpoint
@app.post("/LeadScoreV2/api/score")
async def leadscore(data_model: DataModel):
    try:
        # Convert the input data to a DataFrame
        test_df = pd.DataFrame(data_model.data)
        
        # Call the score function
        response = score(test_df)

        return response
    except Exception as e:
        logging.exception("An error occurred during scoring")
        return {"message": str(e), "result": None, "statusCode": 500}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)