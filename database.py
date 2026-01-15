"""
Database managers for MongoDB and Qdrant.

MongoDB: Stores user metadata (name, client_id, timestamps)
Qdrant: Stores voice embeddings with vector similarity search
"""

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Optional, List, Dict
import numpy as np
from datetime import datetime
import logging
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoManager:
    """
    MongoDB manager for user metadata.
    
    Stores user information including:
    - client_id: Unique identifier
    - name: User's name
    - created_at: Registration timestamp
    - num_embeddings: Number of voice embeddings stored
    """
    
    def __init__(self):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(config.MONGODB_URI)
            self.db = self.client[config.MONGODB_DB]
            self.users = self.db.users
            logger.info("✅ MongoDB connected successfully")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    def create_user(self, client_id: str, name: str, num_embeddings: int = 1) -> Dict:
        """
        Create new user record.
        
        Args:
            client_id: Unique client identifier
            name: User's name
            num_embeddings: Number of embeddings stored (default: 1)
            
        Returns:
            Created user document
        """
        user = {
            "client_id": client_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "is_sample": False,
            "num_embeddings": num_embeddings
        }
        self.users.insert_one(user)
        logger.info(f"✅ Created user: {name} ({client_id})")
        return user
    
    def get_user(self, client_id: str) -> Optional[Dict]:
        """
        Get user by client_id.
        
        Args:
            client_id: Client identifier
            
        Returns:
            User document or None if not found
        """
        return self.users.find_one({"client_id": client_id})
    
    def get_all_users(self) -> List[Dict]:
        """
        Get all registered users.
        
        Returns:
            List of user documents
        """
        return list(self.users.find())
    
    def user_exists(self, client_id: str) -> bool:
        """
        Check if user exists.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if user exists, False otherwise
        """
        return self.users.find_one({"client_id": client_id}) is not None
    
    def delete_user(self, client_id: str) -> bool:
        """
        Delete user record.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if deleted, False if not found
        """
        result = self.users.delete_one({"client_id": client_id})
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted user: {client_id}")
            return True
        return False


class QdrantManager:
    """
    Qdrant manager for voice embeddings.
    
    Stores multiple embeddings per user for improved robustness.
    Uses cosine similarity for vector search.
    """
    
    def __init__(self):
        """Initialize Qdrant connection and collection."""
        try:
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            self._initialize_collection()
            logger.info("✅ Qdrant connected successfully")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            raise
    
    def _initialize_collection(self):
        """Initialize or verify Qdrant collection exists."""
        try:
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
                logger.info(f"✅ Created collection: {config.COLLECTION_NAME}")
            else:
                logger.info(f"✅ Using existing collection: {config.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ Collection initialization failed: {e}")
            raise
    
    def insert(self, point_id: int, embedding: np.ndarray, payload: Dict):
        """
        Insert or update single embedding in Qdrant.
        
        Args:
            point_id: Unique point ID (integer)
            embedding: Embedding vector (numpy array)
            payload: Metadata dict (must include client_id)
        """
        try:
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=[point]
            )
        except Exception as e:
            logger.error(f"❌ Failed to insert point {point_id}: {e}")
            raise
    
    def insert_multiple(self, client_id: str, embeddings: List[np.ndarray]):
        """
        Insert multiple embeddings for same user.
        
        This is the KEY improvement for robustness. Multiple embeddings
        from the same audio sample capture different aspects and variations,
        improving matching under different conditions.
        
        Args:
            client_id: User's client ID
            embeddings: List of embedding arrays
        """
        try:
            points = []
            base_id = hash(client_id) & 0x7FFFFFFF  # Ensure positive int
            
            for idx, embedding in enumerate(embeddings):
                # Create unique point ID for each embedding
                # base_id ensures same user, idx differentiates embeddings
                point_id = base_id + idx
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "client_id": client_id,
                        "embedding_idx": idx,
                        "total_embeddings": len(embeddings)
                    }
                )
                points.append(point)
            
            # Batch insert for efficiency
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            logger.info(f"✅ Stored {len(embeddings)} embeddings for {client_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert multiple embeddings: {e}")
            raise
    
    def add_embedding(self, client_id: str, embedding: np.ndarray):
        """
        Add single voice embedding (legacy method for backward compatibility).
        
        Args:
            client_id: User's client ID
            embedding: Embedding vector
        """
        point_id = hash(client_id) & 0x7FFFFFFF
        self.insert(point_id, embedding, {
            "client_id": client_id,
            "embedding_idx": 0,
            "total_embeddings": 1
        })
    
    def search(self, query_vector: np.ndarray, limit: int = 50):
        """
        Search for similar embeddings.
        
        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of search results with score and payload
        """
        try:
            results = self.client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_vector.tolist(),
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def search_with_max_score(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict]:
        """
        Search and group by client_id, returning MAX score per user.
        
        This is CRITICAL for multi-embedding support:
        - Each user has multiple embeddings
        - We compare against ALL of them
        - Return the BEST match (max score)
        - This handles voice variation naturally
        
        Example:
            User has 4 embeddings: [0.72, 0.68, 0.81, 0.70]
            We return: 0.81 (the best match)
            
        Args:
            query_vector: Query embedding
            limit: Number of top users to return
            
        Returns:
            List of {client_id, score} sorted by max score (descending)
        """
        try:
            # Search broadly to capture all embeddings per user
            # We use limit=50 to ensure we get multiple embeddings per user
            results = self.search(query_vector, limit=50)
            
            if not results:
                logger.warning("No search results found")
                return []
            
            # Group by client_id and take MAX score
            user_scores = {}
            for r in results:
                client_id = r.payload.get("client_id")
                if not client_id:
                    continue
                    
                score = r.score
                
                # Keep track of max score for each user
                if client_id not in user_scores:
                    user_scores[client_id] = {
                        "score": score,
                        "embedding_idx": r.payload.get("embedding_idx", 0)
                    }
                else:
                    # Update if this embedding has higher score
                    if score > user_scores[client_id]["score"]:
                        user_scores[client_id] = {
                            "score": score,
                            "embedding_idx": r.payload.get("embedding_idx", 0)
                        }
            
            # Sort by score (descending) and return top N
            sorted_users = sorted(
                user_scores.items(), 
                key=lambda x: x[1]["score"], 
                reverse=True
            )
            
            result = [
                {
                    "client_id": client_id,
                    "score": data["score"],
                    "best_embedding_idx": data["embedding_idx"]
                }
                for client_id, data in sorted_users[:limit]
            ]
            
            logger.debug(f"Found {len(result)} users from {len(results)} embeddings")
            return result
            
        except Exception as e:
            logger.error(f"❌ Max score search failed: {e}")
            return []
    
    def search_similar(self, embedding: np.ndarray, limit: int = 1) -> List[Dict]:
        """
        Search for similar voice embeddings (legacy method).
        
        Args:
            embedding: Query embedding
            limit: Number of results
            
        Returns:
            List of {client_id, score} results
        """
        return self.search_with_max_score(embedding, limit)
    
    def delete_embedding(self, client_id: str):
        """
        Delete single embedding by client_id (legacy method).
        
        Args:
            client_id: Client identifier
        """
        self.delete_all_embeddings(client_id)
    
    def delete_all_embeddings(self, client_id: str):
        """
        Delete ALL embeddings for a client.
        
        Args:
            client_id: Client identifier
        """
        try:
            base_id = hash(client_id) & 0x7FFFFFFF
            # Delete up to 20 possible embeddings (generous upper bound)
            point_ids = [base_id + i for i in range(20)]
            
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=point_ids
            )
            logger.info(f"✅ Deleted embeddings for {client_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete embeddings: {e}")
            raise
    
    def delete_all(self):
        """Delete entire collection (for testing/reset)."""
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
            self._initialize_collection()
            logger.info("✅ Collection reset complete")
        except Exception as e:
            logger.error(f"❌ Failed to reset collection: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dict with collection info
        """
        try:
            collection_info = self.client.get_collection(config.COLLECTION_NAME)
            return {
                "total_points": collection_info.points_count,
                "vector_dim": config.EMBEDDING_DIM,
                "collection_name": config.COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}