default_target: local

COMMIT_HASH := $(shell git log -1 --pretty=format:"%h"|tail -1)
VERSION = 0.4.8
#IMAGE_REPO ?= ghcr.io/irakhlin/amblegpt
IMAGE_REPO ?= irakhlin/amblegpt
GITHUB_REF_NAME ?= $(shell git rev-parse --abbrev-ref HEAD)
CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

version:
	echo 'VERSION = "$(VERSION)-$(COMMIT_HASH)"' > version.py

local: version
	DOCKER_BUILDKIT=1 docker buildx build --tag amblegpt:latest --load --file ./Dockerfile .

amd64:
	docker buildx build --platform linux/amd64 --tag $(IMAGE_REPO):$(VERSION)-$(COMMIT_HASH) --file ./Dockerfile .

arm64:
	docker buildx build --platform linux/arm64 --tag $(IMAGE_REPO):$(VERSION)-$(COMMIT_HASH) --file ./Dockerfile .

build:  version amd64 arm64
	docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $(IMAGE_REPO):$(VERSION)-$(COMMIT_HASH) --file ./Dockerfile .

push:
	docker buildx build --push --platform linux/arm64/v8,linux/amd64 --tag $(IMAGE_REPO):${GITHUB_REF_NAME}-$(COMMIT_HASH) --file ./Dockerfile .

push2:
	docker buildx build --push --platform linux/amd64 --tag $(IMAGE_REPO):${VERSION}-$(COMMIT_HASH) --file ./Dockerfile .