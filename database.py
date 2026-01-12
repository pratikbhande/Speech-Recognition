"""Database managers for MongoDB and Qdrant."""

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Optional, List, Dict
import numpy as np
from datetime import datetime
import config


class MongoManager:
    """MongoDB manager for user metadata."""
    
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DB]
        self.users = self.db.users
    
    def create_user(self, client_id: str, name: str) -> Dict:
        """Create new user."""
        user = {
            "client_id": client_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "is_sample": False
        }
        self.users.insert_one(user)
        return user
    
    def get_user(self, client_id: str) -> Optional[Dict]:
        """Get user by client_id."""
        return self.users.find_one({"client_id": client_id})
    
    def get_all_users(self) -> List[Dict]:
        """Get all users."""
        return list(self.users.find())
    
    def user_exists(self, client_id: str) -> bool:
        """Check if user exists."""
        return self.users.find_one({"client_id": client_id}) is not None


class QdrantManager:
    """Qdrant manager for voice embeddings."""
    
    def __init__(self):
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if config.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
    
    def insert(self, point_id: int, embedding: np.ndarray, payload: Dict):
        """
        Insert or update embedding in Qdrant.
        
        Args:
            point_id: Unique point ID (integer)
            embedding: numpy array of embeddings
            payload: metadata dict (must include client_id)
        """
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=[point]
        )
    
    def add_embedding(self, client_id: str, embedding: np.ndarray):
        """
        Add voice embedding to collection (legacy method).
        Uses client_id hash as point ID.
        """
        point_id = hash(client_id) & 0x7FFFFFFF
        self.insert(point_id, embedding, {"client_id": client_id})
    
    def search(self, query_vector: np.ndarray, limit: int = 1):
        """
        Search for similar embeddings.
        
        Args:
            query_vector: numpy array to search for
            limit: number of results to return
            
        Returns:
            List of search results with score and payload
        """
        results = self.client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        return results
    
    def search_similar(self, embedding: np.ndarray, limit: int = 1) -> List[Dict]:
        """
        Search for similar voice embeddings (legacy method).
        Returns formatted dict results.
        """
        results = self.search(embedding, limit)
        
        return [
            {
                "client_id": r.payload["client_id"],
                "score": r.score
            }
            for r in results
        ]
    
    def delete_embedding(self, client_id: str):
        """Delete embedding by client_id."""
        point_id = hash(client_id) & 0x7FFFFFFF
        self.client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=[point_id]
        )
    
    def delete_all(self):
        """Delete all embeddings (for testing)."""
        self.client.delete_collection(config.COLLECTION_NAME)
        self._initialize_collection()