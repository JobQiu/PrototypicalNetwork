#!/usr/bin/env bash

DATADIR=data/tiny-imagenet/data

rm -rf $DATADIR
mkdir -p $DATADIR

if [[ -e  tiny-imagenet-200.zip ]]; then
  echo "File already downloaded" >&2
else
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi


unzip tiny-imagenet-200.zip -d $DATADIR