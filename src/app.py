import streamlit as st
import os
from typing import Dict, Any
from pathlib import Path
import json
from vector_store import VectorStoreManager
from qa_model import CarReviewQA


VECTOR_STORE_SETTINGS = {
    "index_name": "car-articles",
    "persist_directory": "./vector_db",
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
}


def safe_get(dictionary: Dict, key: str, default: Any = "") -> str:
    """Safely get a value from dictionary, handling None values."""
    value = dictionary.get(key, default)
    return str(value).strip().lower() if value is not None else ""


def load_available_cars(reviews_dir: str) -> Dict[str, Dict]:
    """Load all available car details and maintain make-model relationships."""
    car_data = {
        "makes_models": {},  # Hierarchical make -> models mapping
        "models": set(),  # All unique models
    }

    if not os.path.exists(reviews_dir):
        st.error(f"Reviews directory {reviews_dir} not found!")
        return car_data

    for json_file in Path(reviews_dir).glob("**/*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                details = data.get("car_details", {})
                if details:
                    make = safe_get(details, "make").upper()
                    model = safe_get(details, "model").lower().replace("-", " ")

                    # Add to makes_models hierarchy
                    if make:
                        if make not in car_data["makes_models"]:
                            car_data["makes_models"][make] = set()
                        if model:
                            car_data["makes_models"][make].add(model)

                    # Add to all models
                    if model:
                        car_data["models"].add(model)

        except Exception as e:
            st.error(f"Error loading {json_file}: {str(e)}")
            continue

    # Convert sets to sorted lists for UI display
    car_data["makes_models"] = {
        make: sorted(models) for make, models in car_data["makes_models"].items()
    }
    car_data["models"] = sorted(car_data["models"])

    return car_data


def initialize_qa_system():
    """Initialize the Q&A system with proper error handling."""
    try:
        vector_store = VectorStoreManager(**VECTOR_STORE_SETTINGS)
        qa_system = CarReviewQA(vector_store)
        return qa_system
    except Exception as e:
        st.error(f"Error initializing Q&A system: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Car Review Q&A", page_icon="🚗", layout="wide")
    st.title("🚗 Car Review Q&A System")

    # Initialize chat history with an initial assistant message
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    reviews_dir = os.path.join("articles", "raw", "expert_review")

    if not os.path.exists(reviews_dir):
        st.error(f"Reviews directory not found at {reviews_dir}!")
        st.info("Please create a 'reviews' directory and add your JSON review files.")
        return

    qa_system = initialize_qa_system()
    if not qa_system:
        st.error("Failed to initialize Q&A system. Please check the errors above.")
        return

    # Load available cars
    car_data = load_available_cars(reviews_dir)

    # Display available cars in the sidebar
    st.sidebar.title("Available Cars")
    if car_data["makes_models"]:
        st.sidebar.markdown("### Makes and Models")
        for make, models in car_data["makes_models"].items():
            with st.sidebar.expander(f"{make}"):
                for model in models:
                    st.write(f"- {model}")
    else:
        st.sidebar.warning("No car reviews found in the reviews directory!")

    # Example questions section
    st.sidebar.markdown("### Example Questions")
    example_questions = [
        "What is the reliability of the BMW M5?",
        "How comfortable is the Tesla Model S?",
        "What are the features of the Toyota Camry?",
        "What are the running costs for the Ford Mustang?",
        "How safe is the Volvo XC90?",
    ]

    # Buttons for example questions
    for q in example_questions:
        if st.sidebar.button(q):
            st.session_state.chat_history.append({"role": "user", "content": q})

            # Process the question
            with st.spinner("Finding answer..."):
                try:
                    answer = qa_system.answer_question(q)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")

    # Chat input
    if user_question := st.chat_input(
        placeholder="e.g., What is the reliability of the BMW M5?"
    ):
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Get previous context if available
        previous_context = None
        if len(st.session_state.chat_history) > 1:
            for message in reversed(st.session_state.chat_history[:-1]):
                if message.get("role") == "assistant":
                    previous_context = f"Previous question: {message['content']}"
                    break

        with st.spinner("Finding answer..."):
            try:
                answer = qa_system.answer_question(
                    user_question, previous_context=previous_context
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear History button
    if st.sidebar.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()
