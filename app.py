import os
import logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# Load pre-trained model
MODEL_PATH = os.getenv('MODEL_PATH', 'ecovision_model2.0.h5')
try:
    model = load_model(MODEL_PATH)
    class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    logging.info(f"Model loaded successfully from {MODEL_PATH}.")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_PATH}: {e}")
    raise RuntimeError("Model loading failed. Check logs for details.")

# AWS S3 client setup
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET]):
    logging.warning("AWS credentials or bucket name not fully set.")
    s3 = None
else:
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

# Disposal instructions mapping
DISPOSAL_INSTRUCTIONS = {
    'battery': "Dispose of batteries at designated hazardous waste collection points.",
    'biological': "Compost biological waste if possible, otherwise dispose in general waste.",
    'brown-glass': "Place brown glass in recycling bins.",
    'cardboard': "Flatten cardboard and place in recycling bin.",
    'clothes': "Donate wearable clothes or recycle at textile recycling centers.",
    'green-glass': "Place green glass in recycling bins.",
    'metal': "Recycle metal items at local recycling centers.",
    'paper': "Place paper in recycling bin.",
    'plastic': "Place plastic in recycling bin.",
    'shoes': "Donate shoes or recycle through appropriate programs.",
    'trash': "Dispose of non-recyclable trash in general waste bins.",
    'white-glass': "Place white glass in recycling bins."
}

# Preprocess image for model inference
def preprocess_image(image):
    resized_image = cv2.resize(image, (160, 160))  # Resize to match model input size
    normalized_image = resized_image / 255.0       # Match preprocessing used during training
    return np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def uploadpage():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.warning("No file uploaded.")
            return jsonify({'error': 'No file uploaded.'}), 400
        file = request.files['file']
        if file.filename == '':
            logging.warning("Empty filename provided.")
            return jsonify({'error': 'Empty filename.'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            logging.warning(f"Unsupported file type uploaded: {file.filename}")
            return jsonify({'error': 'Unsupported file type. Allowed types: png, jpg, jpeg.'}), 400

        file_content = file.read()
        if not file_content:
            logging.warning(f"Uploaded file is empty: {file.filename}")
            return jsonify({'error': 'Empty file uploaded.'}), 400

        image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.warning(f"Failed to decode image: {file.filename}")
            return jsonify({'error': 'Invalid image file.'}), 400

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]

        if s3:
            try:
                from io import BytesIO
                BytesIO_obj = BytesIO(file_content)
                s3.upload_fileobj(BytesIO_obj, S3_BUCKET, f"uploads/{file.filename}")
                logging.info(f"Image uploaded to S3: {file.filename}")
            except Exception as e:
                logging.error(f"Failed to upload image to S3: {e}")

        return jsonify({
            'prediction': predicted_class,
            'disposal_instructions': DISPOSAL_INSTRUCTIONS.get(predicted_class, "No specific instructions available.")
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)