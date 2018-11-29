#!/usr/bin/env bash

if [[ ! -d "data" ]]
then
    mkdir "data"
fi

curl https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar -O
tar -xzvf imdb_crop.tar -o data


