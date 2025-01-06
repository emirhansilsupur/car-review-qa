import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from vector_store import VectorStoreManager  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning, module="torch.classes")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class CarReviewQA:
    def __init__(self, vector_store_manager: VectorStoreManager):
        """Initialize the Q&A system with vector store and LLM."""
        self.vector_store = vector_store_manager

        # Initialize Google's Generative AI
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.1)
        # self.llm = OllamaLLM(model="llama3.2:latest", temperature=0.1)

        # Create the Q&A prompt template
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a car expert analyzing expert reviews and long-term ownership experiences. Provide clear, specific answers based ONLY on the review content.
                      If information is missing, explicitly state that. Never add assumptions or external information.

            Distinguish between:
            - Expert reviews: Professional evaluations from test drives.
            - Long-term reviews: Real-world experiences over time (costs, maintenance, usability).

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

        # Create the QA chain
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

    def format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents into a structured string."""
        formatted_sections = []

        for doc in docs:
            metadata = doc.metadata
            car_info = (
                f"{metadata.get('model_year', '')} "
                f"{metadata.get('car', '').lower()} "
                f"{metadata.get('model', '')}"
            )

            formatted_sections.append(
                f"Section from {car_info} review:\n{doc.page_content}\n"
            )

        return "\n---\n".join(formatted_sections)

    def answer_question(
        self, question: str, filter_metadata: Dict = None, previous_context: str = None
    ) -> str:
        """Answer a question about car reviews using RAG."""
        try:
            # Add car name to question if not already present
            if (
                filter_metadata
                and "car" in filter_metadata
                and "model" in filter_metadata
            ):
                car_name = f"{filter_metadata['car']} {filter_metadata['model']}"
                if "car" in question.lower():
                    question = question.lower().replace("car", car_name)
                elif car_name.lower() not in question.lower():
                    question = f"{question} for {car_name}"

            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(
                question, k=10, filter=filter_metadata
            )

            if not relevant_docs:
                return f"I couldn't find any information about {car_name if filter_metadata else 'the car you asked about'}. Please try another question or select a different car."

            # Format the context
            context = self.format_documents(relevant_docs)

            # Get current car info from filters
            current_car = "None"
            if filter_metadata:
                make = filter_metadata.get("car", "").lower()
                model = filter_metadata.get("model", "")
                if make and model:
                    current_car = f"{make} {model}"

            # Generate answer using the QA chain
            response = self.qa_chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "current_car": current_car,
                    "previous_context": (
                        previous_context if previous_context else "None"
                    ),
                }
            )

            return response

        except Exception as e:
            return f"Error processing your question: {str(e)}"
