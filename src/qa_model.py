import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from src.vector_store import VectorStoreManager

warnings.filterwarnings("ignore", category=UserWarning, module="torch.classes")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class CarReviewQA:
    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize the Q&A system with vector store and LLM."""
        self.vector_store = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.2)
        # self.llm = OllamaLLM(model="llama3.2:latest", temperature=0.2)
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a car expert analyzing expert reviews and long-term ownership experiences. Provide clear, specific answers based ONLY on the review content.
                  If information is missing, explicitly state that. Never add assumptions or external information.

                For follow-up questions:
                1. Maintain context using pronouns (it, this car, etc.).
                2. Avoid repeating previous information.
                3. If the question is ambiguous, ask for clarification.

                Format responses clearly:
                - Use bullet points for lists.
                - Separate expert and long-term review insights.
                - Be concise and avoid repetition.

                If information is unavailable, say: "The reviews don't mention this. Would you like to ask something else?"
                If the question is off-topic, redirect to car-related topics.
                
                Current car being discussed: {current_car}
                """,
                ),
                (
                    "user",
                    """Previous context: {previous_context}
                
                Current question: {question}
                
                Relevant review sections:
                {context}
                """,
                ),
            ]
        )

        self.qa_chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "current_car": lambda x: x["current_car"],
                "previous_context": lambda x: x["previous_context"],
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def find_matching_documents(self, make: str, model: str = None) -> List[Document]:
        """Find documents matching make/model to understand stored format."""
        # Try to find any documents matching just the make
        make_filter = {"make": make.lower()}
        docs = self.vector_store.similarity_search("", k=10, filter=make_filter)

        if docs:
            print("\nFound documents with metadata:")
            for doc in docs:
                print(f"Make: {doc.metadata.get('make')}")
                print(f"Model: {doc.metadata.get('model')}")
                print(f"Body Type: {doc.metadata.get('body_type')}")
                print(f"Year: {doc.metadata.get('year')}")
                print("---")
        else:
            print(f"\nNo documents found for make: {make}")

        return docs

    def normalize_filters(self, filter_metadata: Dict) -> Dict:
        """Normalize filter values to match the stored metadata format."""
        if not filter_metadata:
            return {}

        normalized = {}

        # First, try to find example documents to understand the format
        if "car" in filter_metadata:
            example_docs = self.find_matching_documents(filter_metadata["car"])
            if example_docs:
                # Use the first document's format as a reference
                ref_doc = example_docs[0].metadata
                print(f"\nReference document format: {ref_doc}")

        # Handle make/car - convert to lowercase
        if "car" in filter_metadata:
            normalized["make"] = filter_metadata["car"].lower()

        # Handle model - normalize format
        if "model" in filter_metadata:
            model = filter_metadata["model"].lower()
            # Remove any year information from model name
            model = " ".join([part for part in model.split() if not part.isdigit()])
            # Convert spaces to hyphens and clean up
            normalized["model"] = model.strip().replace(" ", "-")

        # Handle body type
        if "body_type" in filter_metadata:
            normalized["body_type"] = filter_metadata["body_type"].lower()

        # Handle year
        if "model_year" in filter_metadata:
            normalized["year"] = filter_metadata["model_year"]

        print(f"Final normalized filters: {normalized}")
        return normalized

    def answer_question(
        self, question: str, filter_metadata: Dict = None, previous_context: str = None
    ) -> str:
        """Answer a question about car reviews using RAG."""

        # First, try with normalized filters
        normalized_filters = (
            self.normalize_filters(filter_metadata) if filter_metadata else None
        )
        relevant_docs = self.vector_store.similarity_search(
            question, k=5, filter=normalized_filters
        )

        # If no results, try with only make and model
        if not relevant_docs and normalized_filters:
            simplified_filters = {
                k: v for k, v in normalized_filters.items() if k in ["make", "model"]
            }
            print(f"\nTrying with simplified filters: {simplified_filters}")
            relevant_docs = self.vector_store.similarity_search(
                question, k=5, filter=simplified_filters
            )

        # If still no results, search without filters
        if not relevant_docs:
            print("\nTrying without filters...")
            relevant_docs = self.vector_store.similarity_search(question, k=5)

        if not relevant_docs:
            if filter_metadata:
                make = filter_metadata.get("car", "").upper()
                model = filter_metadata.get("model", "")
                return f"I couldn't find any information about {make} {model}. Please try another question or select a different car."

        # Format the context and generate response
        context = self.format_documents(relevant_docs)
        current_car = "None"
        if filter_metadata:
            make = filter_metadata.get("car", "").upper()
            model = filter_metadata.get("model", "")
            if make and model:
                current_car = f"{make} {model}"

        response = self.qa_chain.invoke(
            {
                "context": context,
                "question": question,
                "current_car": current_car,
                "previous_context": (previous_context if previous_context else "None"),
            }
        )

        return response

    def format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents into a structured string."""
        formatted_sections = []

        for doc in docs:
            metadata = doc.metadata
            make = metadata.get("make", "").upper()
            model = metadata.get("model", "")
            year = metadata.get("year", "")

            car_info = f"{year} {make} {model}".strip()
            formatted_sections.append(
                f"Section from {car_info} review:\n{doc.page_content}\n"
            )

        return "\n---\n".join(formatted_sections)
