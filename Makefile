.PHONY: help install test run-api run-ui clean
help:
	@echo"Available commands:"
	@echo"  make install   - Install project dependencies"
	@echo"  make test      - Run automated test suite via pytest"
	@echo"  make run-api   - Start FastAPI inference server on port 8000"
	@echo"  make run-ui    - Start Streamlit dashboard on port 8501"
	@echo"  make clean     - Clean temporary Python cache and build artifacts"
install:
	.venv/bin/pip install -r requirements.txt
test:
	.venv/bin/python -m unittest discover -s tests -p"test_*.py"
run-api:
	.venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
run-ui:
	.venv/bin/streamlit run app/dashboard/main.py --server.port 8501
clean:
	find . -type d -name"__pycache__" -exec rm -rf {} +
	find . -type f -name"*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
