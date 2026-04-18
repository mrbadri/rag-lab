.PHONY: help loader splitter

help:
	@echo "Available commands:"
	@echo "  make loader    - Run document loader"
	@echo "  make splitter  - Run text splitter"
	@echo "  make help      - Show this help message"

loader:
	uv run python document_loader/main/main.py

splitter:
	uv run python splitter/main/main.py
