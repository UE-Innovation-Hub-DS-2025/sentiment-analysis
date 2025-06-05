lint:
	poetry run flake8 src
	poetry run black src/modules/pipeline/extract.py

test:
	poetry run pytest 

