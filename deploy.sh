#!/bin/sh
python3 evaluate.py -datapath pickled/data/test_dataset.pkl -predictor baseline
python3 run_server.py -predictor baseline