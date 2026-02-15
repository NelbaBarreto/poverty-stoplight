"""
RAGAS evaluation module for comparing embedding model performance.
"""
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.embeddings_manager import EmbeddingsManager
from src.pgvector_manager import PGVectorManager


class RAGASEvaluator:
    """Evaluates RAG system performance using RAGAS metrics."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            llm_model: OpenAI model to use for evaluation
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.pgvector_manager = PGVectorManager()

    def create_test_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        contexts_list: List[List[str]],
        answers: List[str]
    ) -> Dataset:
        """
        Create a RAGAS evaluation dataset.
        
        Args:
            questions: List of test questions
            ground_truths: List of ground truth answers
            contexts_list: List of retrieved context lists
            answers: List of generated answers
            
        Returns:
            RAGAS Dataset
        """
        data = {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts_list,
            "answer": answers
        }
        return Dataset.from_dict(data)

    def generate_synthetic_questions(
        self, 
        documents: List[Document], 
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic test questions from documents.
        
        Args:
            documents: List of documents to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of test cases with questions and ground truths
        """
        test_cases = []
        
        # Simple heuristic: create questions from document chunks
        # In production, you'd use a more sophisticated approach
        for i, doc in enumerate(documents[:num_questions]):
            content = doc.page_content[:500]  # Use first 500 chars
            
            # Generate question using LLM
            prompt = f"""Based on the following text, generate a factual question that can be answered from this text.
            
Text: {content}

Generate only the question, nothing else."""
            
            try:
                question_response = self.llm.invoke(prompt)
                question = question_response.content.strip()
                
                # Generate ground truth answer
                answer_prompt = f"""Based on the following text, answer this question concisely.

Text: {content}

Question: {question}

Answer:"""
                
                answer_response = self.llm.invoke(answer_prompt)
                ground_truth = answer_response.content.strip()
                
                test_cases.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "source_document": doc
                })
                
            except Exception as e:
                print(f"Error generating question {i}: {str(e)}")
                continue
        
        return test_cases

    def evaluate_embedding_model(
        self,
        embedding_model: str,
        document_id: int = None,
        test_cases: List[Dict[str, Any]] = None,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluate an embedding model using RAGAS metrics.
        
        Args:
            embedding_model: Name of the embedding model to evaluate
            document_id: Optional document ID to filter by
            test_cases: Optional list of test cases. If None, will generate synthetic ones.
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with evaluation metrics and metadata
        """
        print(f"\n=== Evaluating model: {embedding_model} ===")
        
        # Create embeddings provider
        embeddings = EmbeddingsManager.create_embeddings(embedding_model)
        
        # If no test cases provided, generate them
        if test_cases is None:
            print("Generating synthetic test questions...")
            # Get some documents to generate questions from
            conn = self.pgvector_manager.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT chunk_text FROM chunks"
            params = []
            
            if document_id:
                query += " WHERE document_id = %s"
                params.append(document_id)
            
            query += " LIMIT 10"
            cursor.execute(query, params)
            
            docs = [Document(page_content=row[0]) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            test_cases = self.generate_synthetic_questions(docs, num_questions=5)
        
        if not test_cases:
            raise ValueError("No test cases available for evaluation")
        
        # Evaluate each test case
        questions = []
        ground_truths = []
        contexts_list = []
        answers = []
        retrieval_times = []
        
        print(f"Running evaluation on {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            # Retrieve contexts
            start_time = time.time()
            query_embedding = embeddings.embed_query(question)
            retrieved_docs = self.pgvector_manager.search_similar(
                query_embedding,
                k=k,
                document_id=document_id,
                embedding_model=embedding_model
            )
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # Generate answer using LLM
            context_str = "\n\n".join(contexts)
            answer_prompt = f"""Answer the following question based on the provided context.

Context:
{context_str}

Question: {question}

Answer:"""
            
            try:
                answer_response = self.llm.invoke(answer_prompt)
                answer = answer_response.content.strip()
            except Exception as e:
                print(f"Error generating answer for question {i}: {str(e)}")
                answer = "Error generating answer"
            
            questions.append(question)
            ground_truths.append(ground_truth)
            contexts_list.append(contexts)
            answers.append(answer)
            
            print(f"  Processed {i+1}/{len(test_cases)} questions (retrieval: {retrieval_time:.3f}s)")
        
        # Create dataset
        dataset = self.create_test_dataset(
            questions, ground_truths, contexts_list, answers
        )
        
        # Run RAGAS evaluation
        print("Running RAGAS evaluation...")
        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=self.llm,
            )
            
            # Extract metrics - EvaluationResult object can be accessed like a dict or converted to pandas
            # Try to convert to pandas DataFrame first, then to dict
            try:
                if hasattr(result, 'to_pandas'):
                    result_df = result.to_pandas()
                    # Get mean of each metric column
                    metrics_dict = {
                        "faithfulness": float(result_df['faithfulness'].mean()) if 'faithfulness' in result_df.columns else 0.0,
                        "answer_relevancy": float(result_df['answer_relevancy'].mean()) if 'answer_relevancy' in result_df.columns else 0.0,
                        "context_precision": float(result_df['context_precision'].mean()) if 'context_precision' in result_df.columns else 0.0,
                        "context_recall": float(result_df['context_recall'].mean()) if 'context_recall' in result_df.columns else 0.0,
                    }
                elif hasattr(result, '__getitem__'):
                    # Try indexing directly
                    metrics_dict = {
                        "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else 0.0,
                        "answer_relevancy": float(result["answer_relevancy"]) if "answer_relevancy" in result else 0.0,
                        "context_precision": float(result["context_precision"]) if "context_precision" in result else 0.0,
                        "context_recall": float(result["context_recall"]) if "context_recall" in result else 0.0,
                    }
                else:
                    # Try accessing as attributes
                    metrics_dict = {
                        "faithfulness": float(getattr(result, "faithfulness", 0.0)),
                        "answer_relevancy": float(getattr(result, "answer_relevancy", 0.0)),
                        "context_precision": float(getattr(result, "context_precision", 0.0)),
                        "context_recall": float(getattr(result, "context_recall", 0.0)),
                    }
            except Exception as e:
                print(f"Warning: Error extracting metrics from result: {e}")
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
                # Fallback to zeros
                metrics_dict = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                }
            
            metrics = {
                **metrics_dict,
                "context_relevancy": None,  # Not available in current RAGAS version
                "test_questions_count": len(test_cases),
                "average_retrieval_time": sum(retrieval_times) / len(retrieval_times),
                "metadata": {
                    "k": k,
                    "llm_model": getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown')),
                    "embedding_model": embedding_model,
                }
            }
            
            print(f"\n✓ Evaluation complete!")
            print(f"  Faithfulness: {metrics['faithfulness']:.3f}")
            print(f"  Answer Relevancy: {metrics['answer_relevancy']:.3f}")
            print(f"  Context Precision: {metrics['context_precision']:.3f}")
            print(f"  Context Recall: {metrics['context_recall']:.3f}")
            print(f"  Avg Retrieval Time: {metrics['average_retrieval_time']:.3f}s")
            
            return metrics
            
        except Exception as e:
            print(f"Error in RAGAS evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def compare_embedding_models(
        self,
        models: List[str],
        document_id: int = None,
        test_cases: List[Dict[str, Any]] = None,
        k: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple embedding models.
        
        Args:
            models: List of model names to compare
            document_id: Optional document ID to filter by
            test_cases: Optional list of test cases
            k: Number of documents to retrieve
            
        Returns:
            Dictionary mapping model names to their evaluation metrics
        """
        results = {}
        
        for model in models:
            try:
                metrics = self.evaluate_embedding_model(
                    model, document_id, test_cases, k
                )
                results[model] = metrics
                
                # Save to database
                model_info = EmbeddingsManager.get_model_info(model)
                embedding_model_id = self.pgvector_manager.get_or_create_embedding_model(model)
                
                if document_id:
                    self.pgvector_manager.save_ragas_evaluation(
                        embedding_model_id, document_id, metrics
                    )
                
            except Exception as e:
                print(f"Error evaluating model {model}: {str(e)}")
                results[model] = {"error": str(e)}
        
        return results

    def get_best_model(
        self, 
        results: Dict[str, Dict[str, Any]], 
        metric: str = "answer_relevancy"
    ) -> str:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            results: Evaluation results from compare_embedding_models
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        valid_results = {
            model: metrics 
            for model, metrics in results.items() 
            if "error" not in metrics and metric in metrics
        }
        
        if not valid_results:
            raise ValueError("No valid results to compare")
        
        best_model = max(valid_results.items(), key=lambda x: x[1][metric])
        return best_model[0]
