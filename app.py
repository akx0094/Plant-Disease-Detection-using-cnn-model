from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
import uuid  # For generating unique file names

# Create the Flask application instance
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure the uploads directory exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = tf.keras.models.load_model('my_h5_model.h5')

# Print the model's input shape for debugging
print("Model input shape:", model.input_shape)

def process_image(image_path):
    # Resize image to match the model input (256x256 instead of 224x224)
    img = load_img(image_path, target_size=(256, 256))  
    img_array = img_to_array(img)

    # Normalize the image array to [0, 1] if required by the model
    img_array = img_array / 255.0

    # Add batch dimension (the model expects shape (None, 256, 256, 3))
    img_array = tf.expand_dims(img_array, 0)  # Now img_array has shape (1, 256, 256, 3)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        processed_image = process_image(file_path)
        try:
            prediction = model.predict(processed_image)
            print("Raw prediction output:", prediction)  # Log raw prediction
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

        predicted_class = tf.argmax(prediction[0]).numpy()
        confidence = tf.reduce_max(prediction[0]).numpy() * 100  # Get the confidence percentage
        print("Predicted class index:", predicted_class)  # Log the predicted class index

        # Update class mapping based on your model's output
        classes = ["Apple___Apple_scab", 
                   "Apple___Black_rot", 
                   "Apple___Cedar_apple_rust", 
                   "Apple___healthy", 
                   "Blueberry___healthy", 
                   "Cherry_(including_sour)___Powdery_mildew", 
                   "Cherry_(including_sour)___healthy",
                   "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
                   "Corn_(maize)___Common_rust",  
                   "Corn_(maize)___Northern_Leaf_Blight", 
                   "Corn_(maize)___healthy",
                   "Grape___Black_rot", 
                   "Grape___Esca_(Black_Measles)", 
                   "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  
                   "Grape___healthy",
                   "Orange___Haunglongbing_(Citrus_greening)", 
                   "Peach___Bacterial_spot", 
                   "Peach___healthy", 
                   "Pepper__bell___Bacterial_spot", 
                   "Pepper__bell___healthy", 
                   "Potato___Early_blight",
                   "Potato___Late_blight",
                   "Potato___healthy",
                   "Raspberry___healthy", 
                   "Soybean___healthy", 
                   "Squash___Powdery_mildew", 
                   "Strawberry___Leaf_scorch",
                   "Strawberry___healthy",
                   "Tomato___Bacterial_spot",
                   "Tomato___Early_blight",
                   "Tomato___Late_blight",
                   "Tomato___Leaf_Mold",
                   "Tomato___Septoria_leaf_spot",
                   "Tomato___Spider_mites Two-spotted_spider_mite",
                   "Tomato___Target_Spot",
                   "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                   "Tomato___Tomato_mosaic_virus",
                   "Tomato___healthy",
                  ]

        predicted_label = classes[predicted_class] if predicted_class < len(classes) else 'Unknown'

        return jsonify({'prediction': predicted_label, 'accuracy': round(confidence, 2)})  # Return accuracy

    return jsonify({'error': 'Failed to process image'}), 500


if __name__ == '__main__':
    app.run(debug=True)
