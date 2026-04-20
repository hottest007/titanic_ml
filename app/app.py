from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.inference import predict

app = FastAPI()

# ----------------------
# Request Schema
# ----------------------
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Name: str
    Cabin: str

# ----------------------
# API Routes
# ----------------------
@app.get("/")
def root():
    return {"message": "Titanic API running"}

@app.post("/predict")
def api_predict(data: Passenger):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------
# Gradio Interface
# ----------------------
def gradio_fn(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Name, Cabin):
    payload = {
        "Pclass": int(Pclass),
        "Sex": Sex,
        "Age": float(Age),
        "SibSp": int(SibSp),
        "Parch": int(Parch),
        "Fare": float(Fare),
        "Embarked": Embarked,
        "Name": Name,
        "Deck": Cabin,
    }
    
    return predict(payload)

demo = gr.Interface(
    fn=gradio_fn,
    inputs=[
        gr.Dropdown([1,2,3], label="Passenger Class"),
        gr.Dropdown(["male","female"], label="Sex"),
        gr.Number(label="Age"),
        gr.Number(label="Siblings/Spouses aboard"),
        gr.Number(label="Parents/Children aboard"),
        gr.Number(label="Fare"),
        gr.Dropdown(["S","C","Q"], label="Embarked"),
        gr.Textbox(label="Name (e.g Elon, Mr. Musk)"),
        gr.Textbox(label="Cabin (A-G e.g C85)")
    ],
    outputs="json",
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict survival."
)

# Mount UI
app = gr.mount_gradio_app(app, demo, path="/ui")