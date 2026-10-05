PY ?= python
LAB ?= 01_autograd

.PHONY: help lab progress solutions exercises links gpu docs docs-serve

help:
	@echo "make lab LAB=01_autograd   test YOUR implementation of one lab"
	@echo "make progress              scoreboard of your implementations"
	@echo "make solutions             verify the reference solutions (what CI runs)"
	@echo "make exercises             create missing exercise stubs from solutions"
	@echo "make links                 check relative links in all markdown files"
	@echo "make check                 check labs, stubs and tests (what CI runs)"
	@echo "make docs                  build the docs site into site/ (pip install -r requirements-docs.txt)"
	@echo "make docs-serve            live preview of the docs site at http://127.0.0.1:8000"

lab:
	$(PY) -m pytest labs/$(LAB) -x

progress:
	$(PY) tools/progress.py

solutions:
	$(PY) -m pytest --impl=solution

exercises:
	$(PY) tools/make_exercises.py

links:
	$(PY) tools/check_links.py

check:
	$(PY) tools/check_labs.py

gpu:
	$(PY) tools/measure_gpu.py

docs:
	$(PY) tools/build_docs.py

docs-serve:
	$(PY) tools/build_docs.py --serve
