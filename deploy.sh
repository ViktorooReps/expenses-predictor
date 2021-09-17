#!/bin/sh
python3 evaluate.py -datapath pickled/data/test_dataset.pkl -predictor ema
python3 run_server.py -predictor ema