#!/usr/bin/env sh
# Download MNIST dataset into dataset/

cd dataset
wget https://data.deepai.org/mnist.zip
unzip mnist.zip -d dataset/
gzip -d dataset/*
rm mnist.zip
