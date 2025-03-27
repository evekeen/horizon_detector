FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    unzip \
    wget \
    openssh-server \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/setup_auth.sh /workspace/setup_auth.sh
RUN chmod +x /workspace/setup_auth.sh

COPY scripts/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

COPY unpack_dataset.sh /workspace/unpack_dataset.sh
RUN chmod +x /workspace/unpack_dataset.sh

RUN mkdir /run/sshd
RUN ssh-keygen -A

EXPOSE 22
EXPOSE 6006

ENTRYPOINT ["/workspace/entrypoint.sh"]