.PHONY: install train test app
install:
	pip install -r requirements.txt
train:
	python -m src.train --trials 25 --repeats 3
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
