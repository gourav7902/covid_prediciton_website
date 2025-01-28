from flask import Flask, request, render_template
import pickle
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.flatten().reshape(1, -1)  # Flatten and reshape the image
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')  # Use get() to avoid UnboundLocalError
        if file:
            image = Image.open(file)
            img = preprocess_image(image)
            prediction = model.predict(img)
            labels = ['COVID', 'Normal', 'Viral Pneumonia']
            result = labels[prediction[0]]
            return render_template('result.html', result=result)
        else:
            return render_template('index.html', error="No file selected")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
