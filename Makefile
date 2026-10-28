.PHONY: test lint format clean

test:
	PYTHONPATH=src:. python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name '*.pyc' -delete 2>/dev/null; \
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info; true
