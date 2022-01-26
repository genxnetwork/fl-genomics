Bootstrap: docker
From: neurodebian:latest

%help

    Simple container for launching python scripts in cluster environment

%files
  ./requirements.txt /requirements.txt

%post

  apt-get update
  
  DEBIAN_FRONTEND=noninteractive apt-get -yq install \
    build-essential \
    wget \
    unzip \
    git \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libgit2-dev \
    libssh2-1-dev \
    htop \
    top

  wget -c http://s3.amazonaws.com/plink2-assets/alpha2/plink2_linux_avx2.zip
  mkdir /plink
  mv plink2_linux_avx2.zip /plink/plink2_linux_avx2.zip
  cd plink && unzip plink2_linux_avx2.zip && mv plink2 /usr/bin/plink2 && cd /
  chmod 755 /usr/bin/plink2
  