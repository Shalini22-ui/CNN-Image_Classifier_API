from fastapi import FastAPI, UploadFile, File
from predict import predict

app = FastAPI()
@app.get('/')
def home():
    return{"message" : "Successfully deployed CNN Image classifier"}
@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    with open("temp.jpg", "wb") as f:
        f.write(await file.read())
    
    # Use predict function from predict.py
    result = predict("temp.jpg")
    
    # Return prediction
    return {"prediction": result}