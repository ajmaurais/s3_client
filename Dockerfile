from python:3.9-slim

MAINTAINER "Aaron Maurais -- MacCoss Lab"

RUN apt-get update && \
    apt-get -y install procps

RUN mkdir -p /code/s3_client

# install python dependencies
RUN pip install boto3 tqdm

COPY s3_client.py /code/s3_client/s3_client.py

# add executables
RUN cd /usr/local/bin && \
    echo '#!/usr/bin/env bash\npython3 /code/s3_client/s3_client.py "$@"' > s3_client && \
    echo '#!/usr/bin/env bash\nset -e\nexec "$@"' > entrypoint && \
    chmod 755 s3_client entrypoint

# Git version information
ARG GIT_BRANCH
ARG GIT_REPO
ARG GIT_HASH
ARG GIT_SHORT_HASH
ARG GIT_UNCOMMITTED_CHANGES
ARG GIT_LAST_COMMIT
ARG DOCKER_IMAGE
ARG DOCKER_TAG

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_REPO=${GIT_REPO}
ENV GIT_HASH=${GIT_HASH}
ENV GIT_SHORT_HASH=${GIT_SHORT_HASH}
ENV GIT_UNCOMMITTED_CHANGES=${GIT_UNCOMMITTED_CHANGES}
ENV GIT_LAST_COMMIT=${GIT_LAST_COMMIT}
ENV DOCKER_IMAGE=${DOCKER_IMAGE}
ENV DOCKER_TAG=${DOCKER_TAG}

WORKDIR /data

CMD []
entrypoint ["/usr/local/bin/entrypoint"]

