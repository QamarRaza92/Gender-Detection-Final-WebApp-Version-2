from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supress all logs except warning
import tensorflow as tf
import shutil
import gdown

# MODEL_URL = "https://drive.google.com/uc?id=1EDXO3Flu9Ze71pvg0SQLWeJJ7nlzLAHu"
# MODEL_PATH = "GenderDetection.keras"
MODEL_ID = "1EDXO3Flu9Ze71pvg0SQLWeJJ7nlzLAHu"
MODEL_PATH = "GenderDetection.keras"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please make sure the Google Drive file permission is set to 'Anyone with the link'.")
        exit()




templates = Jinja2Templates(directory='templates')

def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image_arr = np.array(image) / 255.0
    return np.expand_dims(image_arr, axis=0)

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/ImagesAndVideos", StaticFiles(directory="ImagesAndVideos"), name="ImagesAndVideos")
app.mount('/static/uploads', StaticFiles(directory='static/uploads'), name='static_uploads')



try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Make sure the model file is not corrupted and is a valid .keras file.")
    exit()

UPLOAD_FOLDER = "uploads"
STATIC_UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_UPLOAD_FOLDER, exist_ok=True)

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        'request': request,
        'prediction': None,
        'confidence': None,
        'uploaded_image': None
    })

@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save in uploads (for processing)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    await file.seek(0)
    # Copy to static/uploads (for serving on frontend)
    static_path = os.path.join(STATIC_UPLOAD_FOLDER, file.filename)
    shutil.copy(file_path, static_path)

    # Prediction
    processed_image = preprocess(file_path)
    prediction = model.predict(processed_image)
    labels = {0: "Male", 1: "Female"}
    predicted_class = labels[int(np.argmax(prediction, axis=1)[0])]
    confidence = float(np.max(prediction))

    return templates.TemplateResponse("index.html", {
        'request': request,
        'prediction': predicted_class,
        'confidence': f"{confidence:.2f}",
        "uploaded_image": f"/static/uploads/{file.filename}"
    })


@app.get("/about-me")
async def about_me(request: Request):
    return templates.TemplateResponse("about-me.html", {"request": request})

@app.get("/about-project")
async def about_project(request: Request):
    return templates.TemplateResponse("about-project.html", {"request": request})

@app.get("/about-gender-detection")
async def about_gender_detection(request: Request):
    return templates.TemplateResponse("about-gender-detection.html", {"request": request})

@app.get("/references")
async def references(request: Request):
    return templates.TemplateResponse("references.html", {"request": request})