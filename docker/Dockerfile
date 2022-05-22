ARG CUDA_FULL=11.3.1
FROM nvidia/cuda:${CUDA_FULL}-cudnn8-runtime-ubuntu20.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA_FULL
ARG CUDA=11.3

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA/./-} \
      git \
      wget \
      software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c pytorch -c conda-forge -c defaults -c anaconda -c bioconda -c pyg -c salilab \
      python=3.8 \
      pip=22.0.3 \
      pytorch=1.10.0 \
      cudatoolkit=${CUDA} \
      numpy=1.20.3 \
      scipy=1.7.1 \
      pandas=1.3.4 \
      requests=2.26.0 \
      networkx=2.6.3 \
      msms=2.6.1 \
      pyg=2.0.4 \
      modeller=10.2

# Mirror the curated directory structure in the Docker image's application execution directory.
COPY . /app/DeepRefine

# Install pip packages.
WORKDIR /app/DeepRefine
RUN pip3 install --upgrade pip \
    && pip3 install -e . \
    && pip3 install -r /app/DeepRefine/requirements.txt

# Explicitly install DGL 0.8.0post1 (CUDA 11.3) with Conda.
RUN conda install -c dglteam https://anaconda.org/dglteam/dgl-cuda11.3/0.8.0post1/download/linux-64/dgl-cuda11.3-0.8.0post1-py38_0.tar.bz2

# Explicitly install latest version of BioPython with pip.
RUN pip3 install git+https://github.com/biopython/biopython@1dd950aec08ed3b63d454fea662697f6949f8dfa

# Remove Git artifacts from local repository clone.
RUN rm -rf .git/

# Replace XXXX with license key for Modeller (Note: Must be provided by user to enable Ca-atom model support).
RUN sed -i '2s/.*/license = r\x27'XXXX'\x27/' /opt/conda/lib/modeller-10.2/modlib/modeller/config.py

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/DeepRefine/project/lit_model_predict_docker.py "$@"' > /app/run_deeprefine.sh \
  && chmod +x /app/run_deeprefine.sh
ENTRYPOINT ["/app/run_deeprefine.sh"]