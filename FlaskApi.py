import numpy as np
from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load


class Prediction(Resource):
    def __init__(self) -> None:
        super().__init__()
        # load the model from file
        self.lgbr_cars_model = load("lgbr_cars.model")

    def get(self):
        # create input array
        parser = reqparse.RequestParser()
        parser.add_argument('vehicleType', required=True, type=int)
        parser.add_argument('gearBox', required=True, type=int)
        parser.add_argument('powerPS', required=True, type=int)
        parser.add_argument('model', required=True, type=int)
        parser.add_argument('kilometer', required=True, type=int)
        parser.add_argument('monthOfRegistration', required=True, type=int)
        parser.add_argument('fuelType', required=True, type=int)
        parser.add_argument('brand', required=True, type=int)

        args = parser.parse_args()

        vehicle_type = args['vehicleType']
        gear_box = args['gearBox']
        power_PS = args['powerPS']
        model = args['model']
        kilometer = args['kilometer']
        month_of_registration = args['monthOfRegistration']
        fuel_type = args['fuelType']
        brand = args['brand']

        input = [vehicle_type, gear_box, power_PS, model, 
                kilometer, month_of_registration, fuel_type, brand]
        input = np.array(input, dtype=int)
        
        # make a prediction
        prediction = self.lgbr_cars_model.predict([input])[0]
        
        # return the data
        return {'prediction': prediction}, 200

app = Flask("CarPredictionApi")
api = Api(app)
api.add_resource(Prediction, '/prediction') 

# http://127.0.0.1:5000/prediction?vehicleType=3&gearBox=1&powerPS=190&model=-1&kilometer=125000&monthOfRegistration=5&fuelType=3&brand=1 
# http://127.0.0.1:5000/prediction?vehicleType=-1&gearBox=1&powerPS=0&model=118&kilometer=150000&monthOfRegistration=0&fuelType=1&brand=38

if __name__ == '__main__':
    app.run()  # run our Flask app