FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    htop \
    wget \
    curl \
    git 

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y python=3.8
# install pip
RUN apt-get update && apt-get install -y python3-pip

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools wheel
RUN python -m ensurepip --upgrade


RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh \n" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# install conda
# RUN conda install mpi4py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /workspace


CMD ["/bin/bash"]