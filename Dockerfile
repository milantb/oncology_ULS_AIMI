FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base
ARG LOCAL_BUILD="false"

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        wget \
        unzip \
        libopenblas-dev \
        python3.9 \
        python3.9-dev \
        python3-pip \
        python3-gdcm \
        libgdcm-dev \
        nano && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Make GDCM bindings available to Python 3.9
RUN ln -s /usr/lib/python3/dist-packages/gdcm.py /usr/lib/python3.9/gdcm.py 2>/dev/null || true && \
    ln -s /usr/lib/python3/dist-packages/gdcmswig.py /usr/lib/python3.9/gdcmswig.py 2>/dev/null || true && \
    ln -s /usr/lib/x86_64-linux-gnu/python3/dist-packages/_gdcmswig.so /usr/lib/python3.9/ 2>/dev/null || true

RUN python3.9 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install --no-cache-dir -r /tmp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN git config --global advice.detachedHead false && \
    git clone --no-checkout https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ && \
    cd /opt/algorithm/nnunet/ && \
    git checkout v2.5.1

RUN pip3 install \
        -e /opt/algorithm/nnunet \
        graphviz \
        onnx \
        SimpleITK && \
    rm -rf ~/.cache/pip

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN chown -R user /opt/algorithm/
RUN mkdir -p /opt/app /input /output /opt/ml/model \
    && chown -R user:user /opt/app /input /output /opt/ml

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user config.json /opt/app/
COPY --chown=user:user ./architecture/ /tmp/architecture/

COPY --chown=user:user ./architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/

RUN if [ "$LOCAL_BUILD" = "true" ]; then \
        echo "Building for local testing - copying weights and data"; \
        mkdir -p /opt/ml/model && cp -r /tmp/architecture/nnUNet_results /opt/ml/model/ 2>/dev/null || echo "No weights found"; \
        cp -r /tmp/architecture/input/* /input/ 2>/dev/null || echo "No test data found"; \
    fi

ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT [ "python3.9", "-m", "process" ]
