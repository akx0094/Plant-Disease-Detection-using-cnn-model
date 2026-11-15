<div align="center">

# 🌱 Plant Disease Identifier

**An AI-powered plant leaf disease detection system using Convolutional Neural Networks (CNN).**

Upload a leaf image and let the trained deep-learning model predict whether the plant is healthy or affected by a disease.

</div>

---

## 📌 About the Project

Plant Disease Identifier is a machine-learning application that analyzes images of plant leaves to identify possible diseases. It uses a **CNN model built with TensorFlow and Keras** for image classification and a **Flask REST API** to receive images and return predictions.

The project includes a simple HTML interface where users can select a leaf image, preview it, view the predicted class and confidence score, and search online for treatment information when a disease is predicted.

> **Note:** This project is an educational image-classification tool. Predictions are not a substitute for diagnosis by an agricultural expert.

## ✨ Features

- Upload an image of a plant leaf.
- Preview the selected image before submitting it.
- Classify the image into one of the model's supported classes.
- Display the prediction and model confidence percentage.
- Offer a web search for treatment information when the prediction is not labelled healthy.
- Flask-based prediction API.
- CNN model trained for 38 plant/disease classes.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep-learning framework |
| Keras | CNN model and image processing |
| Flask | Backend REST API |
| Flask-CORS | Cross-origin requests |
| HTML, CSS, JavaScript | Frontend interface |
| NumPy | Image array processing |
| Jupyter Notebook | Model development and experimentation |

## 🌿 Supported Classes

The prediction code maps the model output to the following 38 classes:

<details>
<summary><strong>View all supported classes</strong></summary>

1. Apple — Apple scab
2. Apple — Black rot
3. Apple — Cedar apple rust
4. Apple — Healthy
5. Blueberry — Healthy
6. Cherry — Powdery mildew
7. Cherry — Healthy
8. Corn — Cercospora leaf spot / Gray leaf spot
9. Corn — Common rust
10. Corn — Northern leaf blight
11. Corn — Healthy
12. Grape — Black rot
13. Grape — Esca (Black Measles)
14. Grape — Leaf blight (Isariopsis Leaf Spot)
15. Grape — Healthy
16. Orange — Huanglongbing (Citrus greening)
17. Peach — Bacterial spot
18. Peach — Healthy
19. Bell pepper — Bacterial spot
20. Bell pepper — Healthy
21. Potato — Early blight
22. Potato — Late blight
23. Potato — Healthy
24. Raspberry — Healthy
25. Soybean — Healthy
26. Squash — Powdery mildew
27. Strawberry — Leaf scorch
28. Strawberry — Healthy
29. Tomato — Bacterial spot
30. Tomato — Early blight
31. Tomato — Late blight
32. Tomato — Leaf Mold
33. Tomato — Septoria leaf spot
34. Tomato — Spider mites (Two-spotted spider mite)
35. Tomato — Target Spot
36. Tomato — Tomato Yellow Leaf Curl Virus
37. Tomato — Tomato mosaic virus
38. Tomato — Healthy

</details>

## 🗂️ Project Structure

```text
plant-disease-identifier-v2/
│
├── app.py                         # Flask prediction API
├── INdex.html                     # Frontend interface
├── my_h5_model.h5                 # Trained CNN model
├── plant-disease-identifier-v2.ipynb # Model development notebook
│
└── images/
    ├── bg.jfif                    # Frontend background image
    ├── flowchart.png              # Project workflow diagram
    ├── README.md                  # Dataset/model notes
    ├── Test/                      # Test leaf images by class
    └── my_flask_app/
        └── uploads/
            └── backend.py         # Additional backend example
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/plant-disease-identifier-v2.git
cd plant-disease-identifier-v2
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow flask flask-cors pillow numpy
```

The application uses Keras image utilities through TensorFlow. If your installed TensorFlow/Keras version does not provide the legacy `keras.preprocessing.image` imports used in `app.py`, use a compatible TensorFlow/Keras environment or update those imports.

## 🚀 Run the Application

### Step 1: Start the Flask backend

Make sure `my_h5_model.h5` is in the project root, alongside `app.py`.

```bash
python app.py
```

The Flask API starts at:

```text
http://127.0.0.1:5000
```

### Step 2: Open the frontend

Open `INdex.html` in your browser.

### Step 3: Identify a plant disease

1. Select an image of a plant leaf.
2. Preview the selected image.
3. Click **Upload**.
4. View the predicted disease class and confidence percentage.
5. If a disease is predicted, use **Search Treatment** to open a web search for that prediction.

## 🔌 API Reference

### Predict a plant disease

**Endpoint**

```http
POST /predict
```

**Request format:** `multipart/form-data`

| Parameter | Type | Description |
|---|---|---|
| `file` | Image file | Leaf image to classify |

**Example using cURL**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@leaf.jpg"
```

**Example successful response**

```json
{
  "prediction": "Apple___Apple_scab",
  "accuracy": 96.25
}
```

The values above are illustrative; actual predictions and confidence scores depend on the uploaded image and model output.

**Possible error responses**

| Status | Description |
|---|---|
| `400` | No file was uploaded |
| `500` | Prediction or image processing failed |

## 🧠 Model & Image Processing

The backend performs the following steps:

1. Receives the uploaded image.
2. Saves it in the `uploads/` directory.
3. Resizes it to **256 × 256** pixels.
4. Converts the image into a numerical array.
5. Normalizes pixel values to the range **0–1**.
6. Passes the image to the trained CNN model.
7. Selects the class with the highest predicted score.
8. Returns the predicted class and confidence percentage as JSON.

The model is loaded from `my_h5_model.h5`.

## 📊 Dataset & Training

The project notes describe a plant leaf image dataset sourced from Kaggle:

**Dataset:** [New Plant Diseases Dataset — Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

- Image type: RGB
- Image dimensions used for model development: 256 × 256
- Classes: 38
- Model architecture: Convolutional Neural Network (CNN)
- Training tools: TensorFlow, Keras, Kaggle / Google Colab

The project notes report the following results:

| Metric | Reported result |
|---|---:|
| Training accuracy | 92.32% |
| Testing accuracy | 83.41% |

These are the results documented in the project's existing notes, not a newly run evaluation.

## 🔄 Project Workflow

```text
        Upload Leaf Image
                |
                v
        Image Preprocessing
       (256 x 256, Normalize)
                |
                v
          Trained CNN Model
                |
                v
        Predict Disease Class
                |
                v
       Display Prediction and
        Confidence Percentage
                |
                v
       Search Treatment Info
       (if disease is predicted)
```

## ⚠️ Important Notes

- Run `app.py` from the project root so the relative model path resolves correctly.
- The frontend currently sends requests to `http://127.0.0.1:5000/predict`. Update this URL if the backend is hosted elsewhere.
- The frontend and backend must both be accessible for image prediction to work.
- The backend currently runs with Flask debug mode enabled. Disable debug mode before deploying publicly.
- The current upload handler does not enforce file type or file size restrictions. Add validation and secure upload handling before public deployment.
- The confidence value is the model's top predicted score; it should not automatically be interpreted as a calibrated probability or a guarantee of correctness.
- Treatment information is opened as a general web search, not generated or verified by this application.

## 🚧 Future Improvements

- Improve model accuracy and evaluate it on independent real-world images.
- Add confidence thresholds and an uncertain-result message.
- Add image validation and secure upload handling.
- Display disease descriptions and verified treatment guidance.
- Improve the frontend for mobile devices.
- Deploy the application using a production-ready server.
- Add prediction history and downloadable reports.

## 🤝 Contributing

Contributions and suggestions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Commit and push your branch.
5. Open a Pull Request.

## 📄 License

No license file was included in the supplied project. Add a `LICENSE` file and select a license before allowing others to reuse, modify, or distribute the project.

---

<div align="center">

**Built with 🌱 to support smarter plant health monitoring.**

</div>
