from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

def read_image(file):
    # Read the image file as a numpy array
    image = np.frombuffer(file.read(), np.uint8)
    # Decode the numpy array to an image
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "Please provide both img1 and img2 files"}), 400
    
    # Get the image files from the request
    img1_file = request.files['img1']
    img2_file = request.files['img2']
    
    # Convert the image files to OpenCV format
    img1 = read_image(img1_file)
    img2 = read_image(img2_file)
    
    # Perform face verification using DeepFace
    result = DeepFace.verify(img1_path=img1, img2_path=img2, detector_backend='opencv')
    
    # Convert numpy bool_ to native Python bool for JSON serialization
    result['verified'] = bool(result['verified'])
    
    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
