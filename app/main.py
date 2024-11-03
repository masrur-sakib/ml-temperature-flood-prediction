# app/main.py
from fastapi import FastAPI
from app.api import predict

app = FastAPI()

# Include routes from predict.py
app.include_router(predict.router)
