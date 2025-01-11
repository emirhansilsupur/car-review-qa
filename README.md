# Car Review Q&A System

An intelligent Q&A system that combines expert car reviews and long-term ownership experiences to provide comprehensive answers about vehicles.

## Demo
![Demo](./assets/car-review-gif.gif)

## Installation

```bash
# Clone the repository
git clone https://github.com/emirhansilsupur/car-review-qa.git
cd car-review-qa

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r scraping/scraper_requirements.txt

# Run the scraper
python scraping/scraper.py

# Run the data processing notebook
python src/car_review_processor.py

# Pull the image
docker pull emirhnslspr/car-review-qa:v1.0.0

# Create .env file with your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run the container
docker run -d \
  -p 8501:8501 \
  -v ${pwd}/articles:/app/articles \
  -v ${pwd}/vector_db:/app/vector_db \
  -v ${pwd}/src:/app/src \
  emirhnslspr/car-review-qa:v1.0.0
```
The application will be available at http://localhost:8501

Go to the [Google AI Studio](https://aistudio.google.com/apikey?hl=tr&_gl=1*e6ens6*_ga*MTI5MjA2Mzk4Ny4xNzM2NjE2MzU1*_ga_P1DBVKWT6V*MTczNjYxNjM1NC4xLjEuMTczNjYxNjU3My42MC4wLjE1MDA4NzAzNDU.) > Click **Get API key**

## Features

- Hybrid search combining dense (FAISS) and sparse (BM25) retrievers
- Support for both expert reviews and long-term ownership experiences
- Context-aware responses that distinguish between different types of reviews
- Interactive chat interface with conversation history
- Example questions for easy exploration

## Usage

1. Start the application
2. Ask questions about cars using either:
   - The chat input field
   - Example questions provided in the sidebar


## Data Processing
The 'car_reviews_data_processing.ipynb' notebook contains code for processing and analyzing car review data. It includes functions for:

- Parsing car review filenames to extract metadata (make, model, body type, year).
- Cleaning JSON files to remove unwanted characters and normalize text.
- Computing statistics on the processed data.

## Scraping
The scraper.py file contains utilities for scraping car review data from AutoTrader. It supports scraping both expert reviews and long-term reviews.

## Example Questions

- What is the reliability of the BMW M5?
- How comfortable is the Tesla Model S?
- What are the features of the Toyota Camry?
- What are the running costs for the Ford Mustang?
- How safe is the Volvo XC90?

## Data Sources

The system uses two types of car reviews:
- Expert reviews: Professional evaluations from test drives
- Long-term reviews: Real ownership experiences over time

## Technologies Used

- LangChain for RAG implementation
- FAISS for dense vector search
- BM25 for sparse retrieval
- gemini-1.5-flash-8b for llm
- Streamlit for web interface
- HuggingFace embeddings (Alibaba-NLP/gte-large-en-v1.5)
- Docker for containerization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
