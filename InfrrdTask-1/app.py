from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from werkzeug.utils import secure_filename
from text_extraction import extract_text
from sentence_transformers import SentenceTransformer, util
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

        # Fetch documents from the database
        documents = list(collection.find({}, {"_id": 1, "text": 1, "filename": 1}))  # Include filename
        results = []

        # Loop through each document
        for doc in documents:
            doc_id = doc["_id"]
            filename = doc["filename"]
            text = doc["text"]

            # Split the document text into lines
            lines = text.split("\n")
            relevant_lines = []

            # Encode the query
            query_embedding = model.encode(query)

            # Check each line for relevance
            for line in lines:
                if query.lower() in line.lower():  # Direct match
                    relevant_lines.append(line)
                else:  # Semantic similarity
                    line_embedding = model.encode(line)
                    similarity = util.cos_sim(query_embedding, line_embedding).item()
                    if similarity > 0.6:  # Adjust threshold as needed
                        relevant_lines.append(line)

                if len(relevant_lines) >= 5:  # Limit to 5 lines
                    break

            # Add the result if relevant lines are found
            if relevant_lines:
                results.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "relevant_text": "\n".join(relevant_lines)
                })

        # Return the results
        return jsonify(results)

# Register endpoints
api.add_resource(UploadAPI, "/upload")
api.add_resource(SearchAPI, "/search")

if __name__ == "__main__":
    app.run(debug=True)
