from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from text_extraction import extract_text
from image_processing import process_image
from search_service import generate_embeddings, perform_search
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

import uuid

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# Initialize Flask app
app = Flask(__name__)
api = Api(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.semanticsearchapi
collection = db.documents

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Serve the HTML form
@app.route("/")
def index():
    return render_template("index.html")

# API Endpoints
class UploadAPI(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"message": "No selected file"}, 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            text = extract_text(file_path)
            doc_id = str(uuid.uuid4())
            collection.insert_one({
                "_id": doc_id,
                "filename": filename,
                "filepath": file_path,
                "text": text
            })
            return {"message": "File uploaded and processed", "doc_id": doc_id}, 200
        except Exception as e:
            return {"message": f"Error processing file: {str(e)}"}, 500

class SearchAPI(Resource):
    def get(self):
        query = request.args.get("query")

        if not query:
            return {"message": "Query parameter is required"}, 400

        try:
            results = perform_search(query, collection)
            return jsonify(results)
        except Exception as e:
            return {"message": f"Error in search: {str(e)}"}, 500


class ImageProcessingAPI(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"message": "No selected file"}, 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            result = process_image(file_path)
            return {"message": "Image processed successfully", "result": result}, 200
        except Exception as e:
            return {"message": f"Error processing image: {str(e)}"}, 500


class GenerateEmbeddingsAPI(Resource):
    def post(self):
        # Get the doc_id from the form data
        doc_id = request.form.get("doc_id")
        if not doc_id:
            return {"message": "Document ID is required"}, 400

        # Fetch the document from the MongoDB collection using doc_id
        document = collection.find_one({"_id": doc_id})
        if not document:
            return {"message": "Document not found"}, 404

        # Now pass the doc_id and collection to generate_embeddings function
        try:
            # Call the generate_embeddings function with the correct parameters
            result = generate_embeddings(doc_id, collection)
            return {"message": "Embeddings generated successfully", "result": result}, 200
        except Exception as e:
            return {"message": f"Error generating embeddings: {str(e)}"}, 500

# Register API resources
api.add_resource(UploadAPI, "/upload")
api.add_resource(SearchAPI, "/search")
api.add_resource(ImageProcessingAPI, "/image-processing")
api.add_resource(GenerateEmbeddingsAPI, "/generate-embeddings")

if __name__ == "__main__":
    app.run(debug=True)

