from python:3.9-slim

MAINTAINER "Aaron Maurais -- MacCoss Lab"

RUN apt-get update && \
    apt-get -y install procps

RUN mkdir -p /code/s3_client

# install python dependencies
RUN pip install boto3

COPY s3_client.py /code/s3_client/s3_client.py

# add executables
RUN cd /usr/local/bin && \
    echo '#!/usr/bin/env bash\npython3 /code/s3_client/s3_client.py "$@"' > s3_client && \
    echo '#!/usr/bin/env bash\nset -e\nexec "$@"' > entrypoint && \
    chmod 755 s3_client entrypoint

WORKDIR /data

CMD []
entrypoint ["/usr/local/bin/entrypoint"]

