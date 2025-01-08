import streamlit as st
import os
from typing import Dict, Any
from pathlib import Path
import json
from collections import defaultdict
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
        "model_body_types": defaultdict(set),  # Model -> body types mapping
        "body_type_years": defaultdict(lambda: defaultdict(set)),
        "body_types": set(),
        "model_years": defaultdict(lambda: defaultdict(lambda: defaultdict(set))),
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
                    body_type = str(safe_get(details, "body_type")).lower()
                    model_year = str(safe_get(details, "year"))

                    # Add to makes_models hierarchy
                    if make:
                        if make not in car_data["makes_models"]:
                            car_data["makes_models"][make] = set()
                        if model:
                            car_data["makes_models"][make].add(model)

                    # Add to model_body_types mapping
                    if model and body_type:
                        car_data["model_body_types"][model].add(body_type)

                    # Add to body types
                    if body_type:
                        car_data["body_types"].add(body_type)

                    # Add to hierarchical year mapping
                    if make and model and body_type and model_year:
                        car_data["model_years"][make][model][body_type].add(model_year)
                        car_data["body_type_years"][model][body_type].add(model_year)

        except Exception as e:
            st.error(f"Error loading {json_file}: {str(e)}")
            continue

    # Convert sets to sorted lists for UI display
    car_data["makes_models"] = {
        make: sorted(models) for make, models in car_data["makes_models"].items()
    }
    car_data["model_body_types"] = {
        model: sorted(body_types)
        for model, body_types in car_data["model_body_types"].items()
    }
    car_data["body_types"] = sorted(car_data["body_types"])

    # Convert nested defaultdicts to regular dicts with sorted lists
    car_data["model_years"] = {
        make: {
            model: {
                body_type: sorted(years, reverse=True)
                for body_type, years in body_types.items()
            }
            for model, body_types in models.items()
        }
        for make, models in car_data["model_years"].items()
    }

    car_data["body_type_years"] = {
        model: {
            body_type: sorted(years, reverse=True)
            for body_type, years in body_types.items()
        }
        for model, body_types in car_data["body_type_years"].items()
    }

    return car_data


def initialize_vector_store(reviews_dir: str):
    """Initialize and populate vector store if needed."""
    vector_store = VectorStoreManager(
        index_name="car-articles", persist_directory="./vector_db"
    )
    return vector_store


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

    reviews_dir = r"articles\raw\expert_review"

    if not os.path.exists(reviews_dir):
        st.error(f"Reviews directory not found at {reviews_dir}!")
        st.info("Please create a 'reviews' directory and add your JSON review files.")
        return

    qa_system = initialize_qa_system()
    if not qa_system:
        st.error("Failed to initialize Q&A system. Please check the errors above.")
        return

    car_data = load_available_cars(reviews_dir)

    if not car_data["makes_models"]:
        st.warning("No car reviews found in the reviews directory!")
        return

    # Sidebar filters in a single column
    st.sidebar.title("Search Filters")

    # Car Make selection
    selected_make = st.sidebar.selectbox(
        "Car Make", ["All"] + list(car_data["makes_models"].keys()), key="make"
    )

    # Model selection based on selected make
    if selected_make != "All":
        available_models = car_data["makes_models"][selected_make]
        model_options = ["All"] + available_models
    else:
        all_models = set()
        for models in car_data["makes_models"].values():
            all_models.update(models)
        model_options = ["All"] + sorted(all_models)

    selected_model = st.sidebar.selectbox("Model", model_options, key="model")

    # Body type selection based on selected model
    if selected_model != "All":
        normalized_model = selected_model.lower().replace("-", " ")
        if normalized_model in car_data["model_body_types"]:
            available_body_types = car_data["model_body_types"][normalized_model]
        else:
            available_body_types = []
        body_type_options = ["All"] + sorted(available_body_types)
    else:
        body_type_options = ["All"] + car_data["body_types"]

    selected_body_type = st.sidebar.selectbox(
        "Body Type", body_type_options, key="body_type"
    )

    # Dynamic year selection based on selected make, model, and body type
    available_years = set()
    if (
        selected_make != "All"
        and selected_model != "All"
        and selected_body_type != "All"
    ):
        # Get years for specific make, model, and body type
        available_years = (
            car_data["model_years"]
            .get(selected_make, {})
            .get(selected_model, {})
            .get(selected_body_type, set())
        )
    elif selected_model != "All" and selected_body_type != "All":
        # Get years for specific model and body type
        available_years = (
            car_data["body_type_years"]
            .get(selected_model, {})
            .get(selected_body_type, set())
        )
    else:
        # Get all available years across all combinations
        for make_data in car_data["model_years"].values():
            for model_data in make_data.values():
                for years in model_data.values():
                    available_years.update(years)

    year_options = ["All"] + sorted(available_years, reverse=True)
    selected_year = st.sidebar.selectbox("Model Year", year_options, key="year")

    # Example questions section
    st.sidebar.markdown("### Example Questions")
    example_questions = [
        f"What is the reliability of the {selected_make} {selected_model}?",
        f"How comfortable is the {selected_make} {selected_model}?",
        f"What are the features of the {selected_make} {selected_model}?",
        f"What are the running costs for the {selected_make} {selected_model}?",
        f"How safe is the {selected_make} {selected_model}?",
    ]

    # Buttons for example questions
    for q in example_questions:
        if st.sidebar.button(q):

            st.session_state.chat_history.append({"role": "user", "content": q})

            # Process the question
            filters = {}
            if selected_make != "All":
                filters["car"] = selected_make.lower()
            if selected_model != "All":
                filters["model"] = selected_model.lower()
            if selected_body_type != "All":
                filters["body_type"] = selected_body_type
            if selected_year != "All":
                filters["model_year"] = selected_year

            # Get current context
            current_context = {
                "make": selected_make,
                "model": selected_model,
                "body_type": selected_body_type,
                "year": selected_year,
            }

            with st.spinner("Finding answer..."):
                try:
                    answer = qa_system.answer_question(
                        q, filter_metadata=filters if filters else None
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "filters": filters,
                            "context": current_context,
                        }
                    )

                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")

    # Chat context for session state
    if "current_context" not in st.session_state:
        st.session_state.current_context = {
            "make": "All",
            "model": "All",
            "body_type": "All",
            "year": "All",
        }

    # Update context when filters change
    if (
        selected_make != st.session_state.current_context["make"]
        or selected_model != st.session_state.current_context["model"]
        or selected_body_type != st.session_state.current_context["body_type"]
        or selected_year != st.session_state.current_context["year"]
    ):
        st.session_state.current_context = {
            "make": selected_make,
            "model": selected_model,
            "body_type": selected_body_type,
            "year": selected_year,
        }

    # Show current context
    if st.session_state.current_context["make"] != "All":
        current_car = (
            f"{st.session_state.current_context['make']} "
            f"{st.session_state.current_context['model']}"
        )
        st.info(f"Currently discussing: {current_car}")

    # Chat input
    if user_question := st.chat_input(
        placeholder="e.g., What is the reliability of the BMW M5?"
    ):
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Get filters from current context
        filters = {}
        if selected_make != "All":
            filters["car"] = selected_make.lower()
        if selected_model != "All":
            filters["model"] = selected_model.lower()
        if selected_body_type != "All":
            filters["body_type"] = selected_body_type.lower()
        if selected_year != "All":
            filters["model_year"] = selected_year

        # Get previous context if available
        previous_context = None
        if len(st.session_state.chat_history) > 1:
            for message in reversed(st.session_state.chat_history[:-1]):
                if message.get("role") == "assistant":
                    previous_context = f"Previous question: {message['content']}"
                    break

        with st.spinner("Finding answer..."):

            if selected_make != "All" and selected_model != "All":
                car_name = f"{selected_make} {selected_model}"
                if "car" not in user_question.lower():
                    modified_question = user_question.replace("car", car_name)
                else:
                    modified_question = user_question
            else:
                modified_question = user_question

            print(f"Question: {modified_question}")
            print(f"Filters: {filters}")
            print(f"Selected make: {selected_make}")
            print(f"Selected model: {selected_model}")

            answer = qa_system.answer_question(
                modified_question,
                filter_metadata=filters if filters else None,
                previous_context=previous_context,
            )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear History button
    if st.sidebar.button("Clear History"):
        st.session_state.chat_history = []
        st.session_state.current_context = {
            "make": "All",
            "model": "All",
            "body_type": "All",
            "year": "All",
        }
        st.rerun()


if __name__ == "__main__":
    main()
