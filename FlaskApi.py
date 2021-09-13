from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import ast
from joblib import load


class Prediction(Resource):
    def get(self):
        # load the model from file
        lgbr_cars_model = load("lgbr_cars.model")
        
        # create input array
        parser = reqparse.RequestParser()
        parser.add_argument('vehicleType', required=True)
        parser.add_argument('gearBox', required=True)
        parser.add_argument('powerPS', required=True)
        parser.add_argument('model', required=True)
        parser.add_argument('kilometer', required=True)
        parser.add_argument('monthOfRegistration', required=True)
        parser.add_argument('fuelType', required=True)
        parser.add_argument('brand', required=True)

        args = parser.parse_args()

        vehicleType = args['vehicleType']
        gearBox = args['gearBox']
        powerPS = args['powerPS']
        model = args['model']
        kilometer = args['kilometer']
        monthOfRegistration = args['monthOfRegistration']
        fuelType = args['fuelType']
        brand = args['brand']
        
        input = [vehicleType, gearBox, powerPS, model, 
                kilometer, monthOfRegistration, fuelType, brand]
        input = np.array(input, dtype=float)
        
        # make a prediction
        prediction = lgbr_cars_model.predict([input])[0]
        
        # return the data
        return {'prediction': prediction}, 200

app = Flask(__name__)
api = Api(app)
api.add_resource(Prediction, '/prediction') 

# http://127.0.0.1:5000/prediction?vehicleType=3&gearBox=1&powerPS=190&model=-1&kilometer=125000&monthOfRegistration=5&fuelType=3&brand=1 
# http://127.0.0.1:5000/prediction?vehicleType=-1&gearBox=1&powerPS=0&model=118&kilometer=150000&monthOfRegistration=0&fuelType=1&brand=38

if __name__ == '__main__':
    app.run()  # run our Flask app