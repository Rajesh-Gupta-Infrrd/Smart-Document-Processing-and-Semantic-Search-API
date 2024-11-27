from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import open_clip
import numpy as np
from sentence_transformers import util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

def generate_embeddings(doc_id, collection):
    text_embedding = None
    image_embedding = None
    doc = collection.find_one({"_id": doc_id})

    if not doc:
        raise ValueError("Document not found")

    filepath = doc["filepath"]

    try:
        # Generate text embeddings
        if filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = file.read()
                if text:
                    text_embedding = text_model.encode(text)

        # Generate image embeddings
        elif filepath.endswith((".png", ".jpg", ".jpeg")):
            image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = clip_model.encode_image(image).cpu().numpy()

        # Convert embeddings to lists to make them JSON serializable
        text_embedding = text_embedding.tolist() if text_embedding is not None else None
        image_embedding = image_embedding.tolist() if image_embedding is not None else None

        # Update document with embeddings
        collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "text_embedding": text_embedding,
                "image_embedding": image_embedding
            }}
        )

    except Exception as e:
        raise ValueError(f"Error generating embeddings: {e}")

    return {"text_embedding": text_embedding, "image_embedding": image_embedding}


def perform_search(query, collection):
    try:
        # Step 1: Generate embedding for the query (query_embedding)
        query_embedding = text_model.encode(query)
        
        # Ensure query_embedding is 2D (1, embedding_dim)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Normalize dimensionality if necessary (for instance, if query_embedding has a different dimension)
        query_embedding = resize_embedding(query_embedding, 512)  # Assuming 512 is the expected dimension

        # Step 2: Fetch all documents from the collection
        documents = collection.find({})  # Find all documents in the collection

        # Step 3: Initialize a list to store results with similarity scores
        similarity_scores = []

        # Step 4: Compare query embedding with each document's embeddings
        for doc in documents:
            doc_id = doc.get("_id")
            text_embedding = doc.get("text_embedding")
            image_embedding = doc.get("image_embedding")

            if text_embedding:
                # Ensure text_embedding is 2D (1, embedding_dim)
                text_embedding = np.array(text_embedding).reshape(1, -1)
                
                # Resize text_embedding to match query_embedding's dimension if necessary
                text_embedding = resize_embedding(text_embedding, 512)

                # Calculate cosine similarity between query embedding and document's text embedding
                text_similarity = cosine_similarity(query_embedding, text_embedding)
                similarity_scores.append((doc_id, "text", text_similarity[0][0], doc))
            
            if image_embedding:
                # Ensure image_embedding is 2D (1, embedding_dim)
                image_embedding = np.array(image_embedding).reshape(1, -1)
                
                # Resize image_embedding to match query_embedding's dimension if necessary
                image_embedding = resize_embedding(image_embedding, 512)

                # Calculate cosine similarity between query embedding and document's image embedding
                image_similarity = cosine_similarity(query_embedding, image_embedding)
                similarity_scores.append((doc_id, "image", image_similarity[0][0], doc))

        # Step 5: Sort results by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[2], reverse=True)

        # Step 6: Get top results (adjust the number as needed)
        top_results = similarity_scores[:5]  # Top 5 most relevant documents

        # Step 7: Return the results in a desired format
        results = []
        for score in top_results:
            doc_id, content_type, similarity, doc = score
            results.append({
                "doc_id": doc_id,
                "content_type": content_type,
                "similarity_score": similarity,
                "filename": doc.get("filename")
            })

        return {"results": results}

    except Exception as e:
        return {"message": f"Error in search: {str(e)}"}

# Helper function to resize embeddings to a consistent dimension
def resize_embedding(embedding, target_dim):
    """Resize the embedding to the target dimension by padding or truncating."""
    current_dim = embedding.shape[1]
    if current_dim == target_dim:
        return embedding
    elif current_dim < target_dim:
        # Pad the embedding with zeros to reach the target dimension
        padding = np.zeros((embedding.shape[0], target_dim - current_dim))
        return np.hstack((embedding, padding))
    else:
        # Truncate the embedding to match the target dimension
        return embedding[:, :target_dim]

