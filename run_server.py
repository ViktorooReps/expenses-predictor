from argparse import ArgumentParser

import uvicorn
from fastapi import FastAPI

from models.abstract import AbstractPredictor
from models.predictors import name2predictor
from models.registered import PredictorName
from server.helper import register_users_predictor

if __name__ == '__main__':
    predictor_names = [pn.value for pn in PredictorName]

    parser = ArgumentParser()
    parser.add_argument('-predictor', choices=predictor_names, type=str, default=PredictorName.BASELINE.value)

    args = parser.parse_args()

    app = FastAPI()
    predictor: AbstractPredictor = name2predictor[PredictorName(args.predictor)]()

    register_users_predictor(app, endpoint='/', predictor=predictor.predict_users)
    uvicorn.run(app, host='0.0.0.0', port=8001)
