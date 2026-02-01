#!/usr/bin/env python
"""
Script de prueba para validar la integración de pgvector-db
Verifica la conexión a la base de datos y las funcionalidades básicas.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test connection to PostgreSQL database."""
    print("🔍 Testing database connection...")
    try:
        from src.pgvector_manager import PGVectorManager
        manager = PGVectorManager()
        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False

def test_tables_exist():
    """Test if required tables exist."""
    print("\n🔍 Checking database tables...")
    try:
        from src.pgvector_manager import PGVectorManager
        import psycopg2
        
        manager = PGVectorManager()
        conn = manager.get_connection()
        cursor = conn.cursor()
        
        # Check for documents table
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'documents'
            )
        """)
        documents_exists = cursor.fetchone()[0]
        
        # Check for chunks table
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'chunks'
            )
        """)
        chunks_exists = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        if documents_exists and chunks_exists:
            print("Required tables exist!")
            return True
        else:
            print("Missing tables. Run schema.sql first.")
            if not documents_exists:
                print("  - documents table missing")
            if not chunks_exists:
                print("  - chunks table missing")
            return False
            
    except Exception as e:
        print(f"Error checking tables: {str(e)}")
        return False

def test_pgvector_extension():
    """Test if pgvector extension is enabled."""
    print("\n🔍 Checking pgvector extension...")
    try:
        from src.pgvector_manager import PGVectorManager
        
        manager = PGVectorManager()
        conn = manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        pgvector_exists = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        if pgvector_exists:
            print("pgvector extension is enabled!")
            return True
        else:
            print("pgvector extension not found. Install it in PostgreSQL.")
            return False
            
    except Exception as e:
        print(f"Error checking pgvector: {str(e)}")
        return False

def test_openai_api():
    """Test OpenAI API connectivity."""
    print("\n🔍 Testing OpenAI API...")
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Test with a simple query
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) == 1536:
            print(f"OpenAI API working! Embedding dimension: {len(test_embedding)}")
            return True
        else:
            print(f"Unexpected embedding dimension: {len(test_embedding)}")
            return False
            
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        print("   Make sure OPENAI_API_KEY is set in .env")
        return False

def test_vectorstore_manager():
    """Test VectorStoreManager initialization."""
    print("\n🔍 Testing VectorStoreManager...")
    try:
        from src.vectorstore import VectorStoreManager
        
        manager = VectorStoreManager(use_pgvector=True)
        vectorstore_type = manager.get_vectorstore_type()
        
        if vectorstore_type == "pgvector":
            print("VectorStoreManager initialized with pgvector!")
            return True
        else:
            print(f"Unexpected vectorstore type: {vectorstore_type}")
            return False
            
    except Exception as e:
        print(f"VectorStoreManager error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 PGVECTOR-DB INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Database Connection", test_database_connection()))
    results.append(("pgvector Extension", test_pgvector_extension()))
    results.append(("Database Tables", test_tables_exist()))
    results.append(("OpenAI API", test_openai_api()))
    results.append(("VectorStoreManager", test_vectorstore_manager()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use pgvector-db integration.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
