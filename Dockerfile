FROM andrewosh/binder-base

MAINTAINER Ákos Laczkó <akos.laczko88@gmail.com>

USER root

RUN apt-get update
RUN apt-get -y install ttf-freefont
RUN echo "deb http://httpredir.debian.org/debian jessie main contrib" > /etc/apt/sources.list \
    && echo "deb http://security.debian.org/ jessie/updates main contrib" >> /etc/apt/sources.list \
    && echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get update \
    && apt-get install -y ttf-mscorefonts-installer \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy pandas matplotlib seaborn
