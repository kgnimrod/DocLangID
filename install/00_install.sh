
# Make data directory
RUN mkdir -p /data

export DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python-is-python3 \
    nano \
    vim \
    zsh \
    rsync \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-nld \
    tesseract-ocr-spa \
    tesseract-ocr-ces \
    tesseract-ocr-slv \
    tesseract-ocr-bul \
    tesseract-ocr-pol \
 
# Install requirements.txt
RUN pip install -r /install/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install development tool for PyCharm for remote debugging
RUN pip install pydevd-pycharm~=211.7628.24

# save some space
RUN rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"