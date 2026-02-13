"""
Embeddings manager supporting multiple providers (OpenAI and HuggingFace).
"""
import os
from typing import List, Literal
from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings
from huggingface_hub import InferenceClient


# Define available embedding models
EMBEDDING_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "dimension": 1536,
            "description": "OpenAI - Más rápido y económico"
        },
        "text-embedding-3-large": {
            "provider": "openai",
            "model_name": "text-embedding-3-large",
            "dimension": 3072,
            "description": "OpenAI - Mayor precisión"
        },
        "text-embedding-ada-002": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimension": 1536,
            "description": "OpenAI - Modelo clásico"
        }
    },
    "huggingface": {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "description": "HuggingFace - Rápido y ligero (384 dim)"
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "description": "HuggingFace - Balance precisión/velocidad (768 dim)"
        },
        "BAAI/bge-small-en-v1.5": {
            "provider": "huggingface",
            "model_name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "description": "HuggingFace - BGE Small, optimizado (384 dim)"
        },
        "BAAI/bge-base-en-v1.5": {
            "provider": "huggingface",
            "model_name": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "description": "HuggingFace - BGE Base, alta calidad (768 dim)"
        },
        "intfloat/multilingual-e5-small": {
            "provider": "huggingface",
            "model_name": "intfloat/multilingual-e5-small",
            "dimension": 384,
            "description": "HuggingFace - Multilingüe E5 Small (384 dim)"
        },
        "intfloat/multilingual-e5-base": {
            "provider": "huggingface",
            "model_name": "intfloat/multilingual-e5-base",
            "dimension": 768,
            "description": "HuggingFace - Multilingüe E5 Base (768 dim)"
        },
        "Qwen/Qwen3-Embedding-0.6B": {
            "provider": "huggingface",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "dimension": 1024,
            "description": "HuggingFace - Qwen3 Embedding 0.6B, modelo ligero (1024 dim)"
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
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize HuggingFace embeddings."""
        self.model_name = model_name
        
        # Get HuggingFace API token
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            # Try alternative token name
            hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN or HF_TOKEN not found in environment variables")
        
        # Initialize HuggingFace InferenceClient with hf-inference provider
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )
        self.dimension = self._get_model_dimension()
    
    def _get_model_dimension(self) -> int:
        """Get dimension from model config."""
        for provider_models in EMBEDDING_MODELS.values():
            if self.model_name in provider_models:
                return provider_models[self.model_name]["dimension"]
        return 384  # Default
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            result = self.client.feature_extraction(
                text,
                model=self.model_name
            )
            # InferenceClient returns a numpy array or list, ensure it's a list
            if hasattr(result, 'tolist'):
                return result.tolist()
            elif isinstance(result, list):
                return result
            else:
                return list(result)
        except Exception as e:
            print(f"Error embedding query with {self.model_name}: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            try:
                result = self.client.feature_extraction(
                    text,
                    model=self.model_name
                )
                # Convert to list if needed
                if hasattr(result, 'tolist'):
                    embeddings.append(result.tolist())
                elif isinstance(result, list):
                    embeddings.append(result)
                else:
                    embeddings.append(list(result))
            except Exception as e:
                print(f"Error embedding document with {self.model_name}: {str(e)}")
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
        # Auto-detect provider if not specified
        if provider is None:
            model_info = EmbeddingsManager.get_model_info(model_name)
            provider = model_info["provider"]
        
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
