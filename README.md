# Category expenses prediction REST API service

## HOWTO:
### 1. Serialize data:
Make sure you moved all .csv files to `cache/tinkoff_hackathon_data`

`python serialize_data.py`

### 2. Deserialize data:

`Dataset.load('path/to/dataset.pkl')`

### 3. Run server locally:

`python run_server.py -predictor <registered PredictorName>`

### 4. Evaluate models:

`python evaluate.py -predictor <registered PredictorName> -datapath <path to serialized dataset>`

### 5. Evaluate best model + run server:

`./deploy.sh`