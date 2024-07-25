LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=spacy-textcat-reviews:${LOCAL_TAG}

test:
	pipenv run pytest train_model/tests/

quality_checks:
	pipenv run isort tests/
	pipenv run black tests/
# 	pipenv run pylint --recursive=y tests/

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash test-service.sh

publish: integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash deploy.sh

setup:
	pipenv install --dev
	pipenv run pre-commit install


