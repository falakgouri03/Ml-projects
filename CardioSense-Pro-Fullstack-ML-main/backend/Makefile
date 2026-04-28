.PHONY: install download train run

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download:
	. .venv/bin/activate && python -m ml.download_kaggle_data

train:
	. .venv/bin/activate && python -m ml.train

run:
	. .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
