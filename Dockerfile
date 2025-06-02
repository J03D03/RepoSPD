FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    python3-pip \
    python-is-python3 \
    cflow \
    openjdk-8-jre \
    maven \
    clang \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/x86_64-linux-gnu/libclang-*.so.1 /usr/lib/x86_64-linux-gnu/libclang.so

RUN pip install numpy==1.24.4
RUN pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html  
RUN pip install torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install tree-sitter==0.21.1
RUN pip install transformers==4.41.2
RUN pip install pandas
RUN pip install clang==6.0.0.2
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
RUN pip install jsonlines

# Create the working directory
RUN mkdir -p /RepoSPD
WORKDIR /RepoSPD

# Copy and set up the initialization script
COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

ENTRYPOINT ["/usr/local/bin/init.sh"]
