# syntax=docker/dockerfile:1.4

# Step 1: Use a Python image to install dependencies
FROM python:3.11 as builder

RUN pip install --upgrade pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install psutil \
    && apt-get purge -y --auto-remove gcc python3-dev

WORKDIR /src

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 wheel --wheel-dir=/wheels -r requirements.txt


FROM python:3.11-slim
ARG TARGETARCH
RUN <<EOF /bin/bash
apt-get update
apt-get install -y --no-install-recommends wget procps xz-utils
# btbn-ffmpeg -> amd64
if [[ "${TARGETARCH}" == "amd64" ]]; then
    mkdir -p /usr/lib/btbn-ffmpeg
    wget -qO btbn-ffmpeg.tar.xz "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.0-latest-linux64-gpl-6.0.tar.xz"
    tar -xf btbn-ffmpeg.tar.xz -C /usr/lib/btbn-ffmpeg --strip-components 1
    rm -rf btbn-ffmpeg.tar.xz /usr/lib/btbn-ffmpeg/doc /usr/lib/btbn-ffmpeg/bin/ffplay
fi

# ffmpeg -> arm64
if [[ "${TARGETARCH}" == "arm64" ]]; then
    mkdir -p /usr/lib/btbn-ffmpeg
    wget -qO btbn-ffmpeg.tar.xz "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.0-latest-linuxarm64-gpl-6.0.tar.xz"
    tar -xf btbn-ffmpeg.tar.xz -C /usr/lib/btbn-ffmpeg --strip-components 1
    rm -rf btbn-ffmpeg.tar.xz /usr/lib/btbn-ffmpeg/doc /usr/lib/btbn-ffmpeg/bin/ffplay
fi
EOF

RUN --mount=type=bind,from=builder,source=/wheels,target=/deps/wheels \
    pip3 install -U /deps/wheels/*.whl

WORKDIR /app

ENV PATH /usr/lib/btbn-ffmpeg/bin:$PATH

# Copy the application files
COPY --link . .

# Set the entry point to the application
CMD ["python3", "./mqtt_client.py"]
