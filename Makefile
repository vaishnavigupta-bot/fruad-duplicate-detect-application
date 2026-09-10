PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup fetch-search data match eval decide search screen dashboard test all clean

help:
	@echo "setup      create venv and install dependencies"
	@echo "fetch-search  download Elasticsearch for this platform into vendor/"
	@echo "data       generate the synthetic applicant dataset"
	@echo "match      run blocking + scoring, write results/scored_pairs.csv"
	@echo "eval       blocking recall, threshold sweep, recall by difficulty"
	@echo "decide     three-tier decision simulation"
	@echo "search     index into Elasticsearch and benchmark query latency"
	@echo "screen     demo: screen applicants one at a time against a live index"
	@echo "dashboard  launch the Streamlit review console"
	@echo "test       run the unit test suite"
	@echo "all        data -> match -> eval -> decide"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fetch-search:
	./scripts/fetch_search.sh

data:
	$(PY) src/generate_data.py

match:
	cd src && ../$(PY) matching.py

eval:
	cd src && ../$(PY) evaluate.py

decide:
	cd src && ../$(PY) decision.py

search:
	cd src && ../$(PY) search_index.py

screen:
	cd src && ../$(PY) match_streaming.py

dashboard:
	.venv/bin/streamlit run src/app.py

test:
	$(PY) -m pytest tests/ -q

all: data match eval decide

clean:
	rm -rf results/*.csv results/*.json __pycache__ src/__pycache__ .pytest_cache
