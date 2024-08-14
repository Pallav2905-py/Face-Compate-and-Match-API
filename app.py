from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import io

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route("/verify", methods=["POST"])
def verify():
    try:
        # Get the image files from the request
        img1 = request.files["image1"]
        img2 = request.files["image2"]
        backend = request.form.get("backend", "ssd")  # Default to 'ssd' if not provided

        # Convert the images to a format that DeepFace can use (e.g., numpy arrays)
        img1 = Image.open(io.BytesIO(img1.read()))
        img2 = Image.open(io.BytesIO(img2.read()))

        # Perform face verification
        result = DeepFace.verify(
            img1_path=img1, img2_path=img2, detector_backend=backend
        )

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def hello():
    return jsonify("Server is Running.")

if __name__ == "__main__":
    app.run(debug=True)
