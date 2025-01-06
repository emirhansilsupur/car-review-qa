from pathlib import Path
import json
from typing import List, Dict, Union, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain_community.document_loaders import JSONLoader
from vector_store import VectorStoreManager  # type: ignore


class DocumentProcessor:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

        # Define category-specific chunk sizes
        self.chunk_sizes = {"expert_review": 1000, "long_term_review": 1200}

    def get_chunk_size(self, category: str) -> int:
        """Get appropriate chunk size for category."""
        return self.chunk_sizes.get(
            category.lower().replace(" ", "_").replace("-", "_")
        )

    def create_metadata(self, file_path: str, raw_json: Dict) -> Dict:
        """Create metadata from JSON content."""
        category = raw_json.get("category", "").lower()
        car_details = raw_json.get("car_details", {})

        metadata = {
            "title": raw_json.get("title", ""),
            "category": category,
            "car": car_details.get("make", "").lower(),
            "model": car_details.get("model", "").lower(),
            "body_type": str(car_details.get("body_type", "")).lower(),
            "model_year": car_details.get("year", ""),
        }

        return metadata

    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document and store in vector database."""
        try:
            # Load and parse the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            # Create metadata
            metadata = self.create_metadata(file_path, raw_json)

            # Initialize JSONLoader with metadata
            loader = JSONLoader(
                file_path=Path(file_path),
                jq_schema=".sections[].content",
                metadata_func=lambda *args: metadata,
                content_key=None,
                text_content=False,
            )

            # Load and split documents
            documents = loader.load()
            chunk_size = self.get_chunk_size(metadata.get("category", ""))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 8,
                length_function=len,
            )

            chunks = text_splitter.split_documents(documents)

            # Store documents
            self.vector_store.add_documents(chunks)

            return chunks

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise

    def process_directory(
        self, directory_path=r"articles\raw"
    ) -> Dict[str, List[Document]]:
        """Process all JSON files in a directory."""
        path = Path(directory_path)
        json_files = list(path.glob("**/*.json"))
        print(len(json_files))

        file_documents = {}
        all_chunks = []

        with tqdm(total=len(json_files), desc="Processing files") as pbar:
            for json_file in json_files:
                try:
                    documents = self.process_document(str(json_file))
                    file_documents[str(json_file)] = documents
                    all_chunks.extend(documents)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue

        # Add all documents to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)

        return file_documents

    def search(
        self, query: str, k: int = 5, filter: Dict = None, include_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Search the vector store with a text query."""
        if include_scores:
            return self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter
            )
        return self.vector_store.similarity_search(query, k=k, filter=filter)
