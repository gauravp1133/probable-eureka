from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
import joblib
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    # Load your trained KNN model
    knn_model = joblib.load('model.pkl')  
    # Load your scaler used during training
    scaler = joblib.load('scaler.pkl')

    # Get the image data from the POST request
    image_data = request.get_json()['image']
    # print(image_data)

    # # Convert the base64 image to a NumPy array
    # image_np = np.array(image_data, dtype=np.int8)
    image_content = image_data.split(",")[-1]
    image_bytes = base64.b64decode(image_content)

    # Convert the image bytes to a NumPy array
    image_np = np.array(Image.open(BytesIO(image_bytes)))

    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


    # Resize the drawn digit to match the training data size
    resized_digit = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    # print('resized_digit shape: ', resized_digit.shape)
    # Flatten the digit and standardize using the same scaler used during training
    flattened_digit = scaler.transform(resized_digit.flatten().reshape(1, -1))
    # print('flattened digit shape: ', flattened_digit.shape)
    # Make a prediction using the KNN model
    prediction = knn_model.predict(flattened_digit)
    print(prediction)
    # Return the recognized digit as JSON response
    return jsonify({'digit': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
