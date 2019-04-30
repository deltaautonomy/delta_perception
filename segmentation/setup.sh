#!/bin/bash

sudo pip3 install numpy matplotlib pillow toposort
sudo apt-get -y install protobuf-compiler
make build-protos
mkdir weights
