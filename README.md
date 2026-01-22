# Sentiment Analysis Chat App

This is a Python Flask application that analyzes chat messages for sentiment. 
You can analyze individual messages or upload a CSV of chat messages to get batch sentiment analysis with confidence scores and visualizations.

## Features

- Single message sentiment analysis with emoji indicators 😊😐😠
- CSV batch analysis for multiple chat messages
- Generates sentiment distribution charts
- Download analyzed CSV with sentiment labels and confidence
- Clean, responsive UI with chat bubbles for better UX

## Installation

1. Clone the repository:
	git clone https://github.com/Vanshika1435/Sentiment_Analysis_.git
2. Create and activate a virtual environment:
	python -m venv venv
	venv\Scripts\activate
3. Install dependencies:
	pip install -r requirements.txt
4. Run the Flask app:
	python app.py

## Usage

- Open your browser and go to `http://127.0.0.1:5000/`
- Type a message or upload a CSV to see sentiment analysis

## License

This project is for learning and demonstration purposes.
