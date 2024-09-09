#!/usr/bin/env bash

export SPANBERT_DIR=/nethome/jsteuer/git/coref
export data_dir=/nethome/jsteuer/git/winogender-2.0/data/spanbert

# clone SpanBERT coref repo
rm -rf $SPANBERT_DIR
git clone https://github.com/justeuer/coref.git --branch add-accuracy-to-evaluation $SPANBERT_DIR

# install Python dependencies
pip install -r $SPANBERT_DIR/requirements.txt

# compile c++ dependencies
chmod u+x $SPANBERT_DIR/setup_all.sh
cd $SPANBERT_DIR && ./setup_all.sh

# download pretrained models
# ./download_pretrained.sh spanbert_base
# ./download_pretrained.sh spanbert_large