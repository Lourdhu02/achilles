PY ?= python
LAB ?= 01_autograd

.PHONY: help lab progress solutions exercises links gpu

help:
	@echo "make lab LAB=01_autograd   test YOUR implementation of one lab"
	@echo "make progress              scoreboard of your implementations"
	@echo "make solutions             verify the reference solutions (what CI runs)"
	@echo "make exercises             create missing exercise stubs from solutions"
	@echo "make links                 check relative links in all markdown files"
	@echo "make gpu                   measure your GPU's roofline (matmul FLOP/s, bandwidth)"

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

gpu:
	$(PY) tools/measure_gpu.py
