from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model
MODEL_POTATO = tf.keras.models.load_model("C:\\Users\\aswin\\Projects\\Plant disease classification\\Models\\potato06-11.keras")

MODEL_PEPPER = tf.keras.models.load_model("C:\\Users\\aswin\\Projects\\Plant disease classification\\Models\\pepper06-11.keras", custom_objects={'Functional': tf.keras.Model})
    

CLASS_NAMES_POTATO = ["Early Blight", "Late Blight", "Healthy"]

CLASS_NAMES_PEPPER = ["Bacterial Spot" , "Healthy" ]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # Resize to the model input size if needed
    image = np.array(image) / 255.0  # Normalize the image if required by the model
    return image

@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Please upload a JPEG or PNG image."})
    
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension
        predictions = MODEL_POTATO.predict(img_batch)
        predicted_class = CLASS_NAMES_POTATO[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing the image: " + str(e)})

@app.post("/predict/pepper")
async def predict_pepper(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Please upload a JPEG or PNG image."})
    
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension
        predictions = MODEL_PEPPER.predict(img_batch)
        predicted_class = CLASS_NAMES_PEPPER[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing the image: " + str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
