from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict, Tuple
from langchain_core.documents import Document
import os


class VectorStoreManager:
    def __init__(
        self,
        index_name="car-articles",
        persist_directory="./vector_db",
        dense_weight=0.7,
        sparse_weight=0.3,
    ):
        self.persist_directory = persist_directory
        self.index_name = index_name
        self.embeddings = self._initialize_embeddings()
        self.sparse_retriever = None
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.vector_store = self._load_or_create_store()

    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-large-en-v1.5",
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )

    def _load_or_create_store(self):
        if os.path.exists(
            os.path.join(self.persist_directory, f"{self.index_name}.faiss")
        ):
            try:
                store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True,
                )
                print(
                    f"Vector store loaded successfully with {len(store.docstore._dict)} documents"
                )
                return store
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
        return None

    def add_documents(self, documents: List[Document]):
        """Add documents to both dense and sparse retrievers."""
        if self.vector_store is None:
            print("Creating new vector store...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print(f"Created vector store with {len(documents)} documents")
        else:
            print(f"Adding {len(documents)} documents to existing vector store")
            self.vector_store.add_documents(documents)

        # BM25Retriever is used for sparse retrieval
        self.sparse_retriever = BM25Retriever.from_documents(documents)

        # Persist the dense index
        print("Saving vector store...")
        self.vector_store.save_local(self.persist_directory, index_name=self.index_name)
        print("Vector store saved successfully")

    def similarity_search(
        self, query: str, k: int = 5, filter: Dict = None
    ) -> List[Document]:
        """Hybrid search combining dense and sparse retrievers."""
        print(f"Searching with k={k} and filters={filter}")

        if filter:
            print("Applying filters:", filter)

        dense_docs = self.vector_store.similarity_search(query, k=k, filter=filter)
        dense_scores = [1 / (i + 1) for i in range(len(dense_docs))]

        if self.sparse_retriever:
            sparse_docs = self.sparse_retriever.get_relevant_documents(query)[: k * 2]
            sparse_scores = self.sparse_retriever.get_scores(query)[: k * 2]
        else:
            sparse_docs, sparse_scores = [], []

        return self._combine_results(
            dense_docs, dense_scores, sparse_docs, sparse_scores, k
        )

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: Dict = None
    ) -> List[Tuple[Document, float]]:
        """Hybrid search returning documents with scores."""
        dense_docs = self.vector_store.similarity_search_with_score(
            query, k=k * 2, filter=filter
        )

        if self.sparse_retriever:
            sparse_docs = self.sparse_retriever.get_relevant_documents(query)[: k * 2]
            sparse_scores = self.sparse_retriever.get_scores(query)[: k * 2]
        else:
            sparse_docs, sparse_scores = [], []

        return self._combine_results_with_score(
            dense_docs, sparse_docs, sparse_scores, k
        )

    def _combine_results(self, dense_docs, dense_scores, sparse_docs, sparse_scores, k):
        combined_results = {}

        for doc, score in zip(dense_docs, dense_scores):
            doc_key = doc.page_content
            combined_results[doc_key] = {"doc": doc, "score": score * self.dense_weight}

        for doc, score in zip(sparse_docs, sparse_scores):
            doc_key = doc.page_content
            if doc_key in combined_results:
                combined_results[doc_key]["score"] += score * self.sparse_weight
            else:
                combined_results[doc_key] = {
                    "doc": doc,
                    "score": score * self.sparse_weight,
                }

        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["doc"] for item in sorted_results[:k]]

    def _combine_results_with_score(
        self, dense_docs, sparse_docs, sparse_scores, k
    ) -> List[Tuple[Document, float]]:
        """Combine dense and sparse search results with scores."""
        combined_results = {}

        # Add dense documents
        for doc_score_tuple in dense_docs:
            doc = doc_score_tuple[0]
            score = doc_score_tuple[1]
            doc_key = doc.page_content
            combined_results[doc_key] = {"doc": doc, "score": score * self.dense_weight}

        # Add sparse documents
        for doc, score in zip(sparse_docs, sparse_scores):
            doc_key = doc.page_content
            if doc_key in combined_results:
                combined_results[doc_key]["score"] += score * self.sparse_weight
            else:
                combined_results[doc_key] = {
                    "doc": doc,
                    "score": score * self.sparse_weight,
                }

        # Sort by score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["score"], reverse=True
        )

        # Return top k results
        return [(item["doc"], item["score"]) for item in sorted_results[:k]]
