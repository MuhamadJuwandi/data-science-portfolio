.PHONY: setup run-dashboard run-api clean

setup:
	pip install -r requirements.txt

run-dashboard:
	streamlit run dashboard/app.py

run-api:
	uvicorn api.main:app --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
