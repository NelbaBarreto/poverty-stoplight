"""
Embeddings manager supporting multiple providers (OpenAI and HuggingFace).
"""
import os
import requests
from typing import List, Literal, Optional, Dict
from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings


# Define available embedding models
EMBEDDING_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "dimension": 1536,
            "description": "OpenAI - Más rápido y económico"
        }
    },
    "huggingface": {
        "Qwen/Qwen3-Embedding-0.6B": {
            "provider": "huggingface",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "dimension": 1024,
            "description": "HuggingFace - Qwen3 Embedding 0.6B, modelo ligero (1024 dim)",
            "endpoint_url": "https://ddz32oohf81bvew1.us-east-1.aws.endpoints.huggingface.cloud"
        },
        "BAAI/bge-m3": {
            "provider": "huggingface",
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "description": "HuggingFace - BGE M3 LWY, modelo ligero (1024 dim)",
            "endpoint_url": "https://g1vqumtf7p0plple.us-east-1.aws.endpoints.huggingface.cloud"
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "description": "HuggingFace - All MPNet Base V2, modelo ligero (768 dim)",
            "endpoint_url": "https://shrk7i1bsvrjsaxk.us-east-1.aws.endpoints.huggingface.cloud"
        }
    }
}


class BaseEmbeddingsProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass


class OpenAIEmbeddingsProvider(BaseEmbeddingsProvider):
    """OpenAI embeddings provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings."""
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get dimension from model config."""
        for provider_models in EMBEDDING_MODELS.values():
            if self.model_name in provider_models:
                return provider_models[self.model_name]["dimension"]
        return 1536  # Default
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embeddings.embed_documents(texts)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


class HuggingFaceEmbeddingsProvider(BaseEmbeddingsProvider):
    """HuggingFace Inference API embeddings provider."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", endpoint_url: str = None):
        """Initialize HuggingFace embeddings.
        
        Args:
            model_name: Name of the model
            endpoint_url: Custom inference endpoint URL. If None, gets it from EMBEDDING_MODELS config
        """
        print(f"[DEBUG] HuggingFaceEmbeddingsProvider.__init__ llamado con model_name: {model_name}")
        self.model_name = model_name
        
        # Get endpoint URL from parameter or model config
        if endpoint_url:
            self.endpoint_url = endpoint_url
        else:
            # Get endpoint from model configuration
            model_info = None
            for provider_models in EMBEDDING_MODELS.values():
                if model_name in provider_models:
                    model_info = provider_models[model_name]
                    break
            
            if model_info and "endpoint_url" in model_info:
                self.endpoint_url = model_info["endpoint_url"]
            else:
                self.endpoint_url = None
        
        # Get HuggingFace API token
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN or HF_TOKEN not found in environment variables")
        
        self.hf_token = hf_token
        
        # Use endpoint if configured
        if self.endpoint_url:
            print(f"Using HuggingFace Endpoint for {model_name}: {self.endpoint_url[:50]}...")
        else:
            print(f"No endpoint configured for {model_name}")
        
        self.dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get dimension from model config."""
        for provider_models in EMBEDDING_MODELS.values():
            if self.model_name in provider_models:
                return provider_models[self.model_name]["dimension"]
        return 384  # Default
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        print(f"[DEBUG] HuggingFaceEmbeddingsProvider.embed_query - model: {self.model_name}, endpoint: {self.endpoint_url[:50] if self.endpoint_url else 'None'}...")
        if not self.endpoint_url:
            raise ValueError(f"No endpoint configured for model {self.model_name}")
        
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": text,
                "parameters": {}
            }
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            embeddings = response.json()
            
            # Normalize response: ensure we return a 1-D array
            # Some endpoints return [[...]] while others return [...]
            if isinstance(embeddings, list):
                if len(embeddings) > 0 and isinstance(embeddings[0], list):
                    # It's a 2-D array [[...]], take the first element
                    return embeddings[0]
                else:
                    # It's already a 1-D array [...]
                    return embeddings
            else:
                raise ValueError(f"Unexpected embedding format: {type(embeddings)}")
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not self.endpoint_url:
            raise ValueError(f"No endpoint configured for model {self.model_name}")
        
        embeddings = []
        for text in texts:
            try:
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": text,
                    "parameters": {}
                }
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                embedding = response.json()
                
                # Normalize response: ensure each embedding is a 1-D array
                if isinstance(embedding, list):
                    array_len = len(embedding)
                    if array_len > 0 and isinstance(embedding[0], list):
                        print("Array lenght", array_len)
                        embeddings.append(embedding[0])
                    else:
                        embeddings.append(embedding)
                else:
                    raise ValueError(f"Unexpected embedding format: {type(embedding)}")
            except Exception as e:
                print(f"Error embedding document: {str(e)}")
                raise
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


class EmbeddingsManager:
    """Manager for creating and managing different embedding providers."""
    
    @staticmethod
    def get_available_models() -> dict:
        """Get all available embedding models."""
        return EMBEDDING_MODELS
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get information about a specific model."""
        for provider_models in EMBEDDING_MODELS.values():
            if model_name in provider_models:
                return provider_models[model_name]
        raise ValueError(f"Model {model_name} not found")
    
    @staticmethod
    def create_embeddings(
        model_name: str,
        provider: Literal["openai", "huggingface"] = None
    ) -> BaseEmbeddingsProvider:
        """
        Create an embeddings provider.
        
        Args:
            model_name: Name of the model
            provider: Provider name (openai or huggingface). If None, auto-detect from model_name.
        
        Returns:
            Embeddings provider instance
        """
        print(f"[DEBUG] EmbeddingsManager.create_embeddings llamado con model_name: {model_name}, provider: {provider}")
        # Auto-detect provider if not specified
        if provider is None:
            model_info = EmbeddingsManager.get_model_info(model_name)
            provider = model_info["provider"]
            print(f"[DEBUG] Provider auto-detectado: {provider}")
        
        # Create provider
        if provider == "openai":
            return OpenAIEmbeddingsProvider(model_name=model_name)
        elif provider == "huggingface":
            return HuggingFaceEmbeddingsProvider(model_name=model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def get_models_by_provider(provider: str) -> dict:
        """Get all models for a specific provider."""
        return EMBEDDING_MODELS.get(provider, {})
    
    @staticmethod
    def list_all_models() -> List[dict]:
        """List all available models with their metadata."""
        all_models = []
        for provider, models in EMBEDDING_MODELS.items():
            for model_name, info in models.items():
                all_models.append({
                    "name": model_name,
                    "provider": provider,
                    "dimension": info["dimension"],
                    "description": info["description"]
                })
        return all_models
