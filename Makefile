# Docker Variables
IMAGE_NAME = document-management-app
CONTAINER_NAME = document_management_container
DOCKERFILE = Dockerfile

# Python Variables
PYTHON_DEPS = requirements.in

# Docker Targets
build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

run:
	docker run --name $(CONTAINER_NAME) -p 8501:8501 $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)

rm:
	docker rm $(CONTAINER_NAME)

clean: stop rm
	docker rmi $(IMAGE_NAME)

logs:
	docker logs $(CONTAINER_NAME)

# Python Dependency Targets
.PHONY: deps-compile deps-install install install-dev

deps-install: deps-compile install-dev

install-dev:
	pip install -e .[dev]

install:
	pip install -e .

deps-compile:
	pip-compile --upgrade $(PYTHON_DEPS) --resolver backtracking --no-emit-index-url --no-emit-trusted-host

deps-sync:
	pip-sync requirements.txt

use-pip-tools:
	pip install --upgrade pip
	pip install pip-tools

use-pre-commit:
	pip install pre-commit
	pre-commit install

# Phony targets
.PHONY: build run stop rm clean logs