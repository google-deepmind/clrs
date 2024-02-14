#!/bin/bash
conda create -n clrs_env python=3.9
conda activate clrs_env
pip install -r requirements/requirements.txt
python setup.py install
pip install jinja2