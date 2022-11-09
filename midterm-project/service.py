import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel


class InsuranceApplication(BaseModel):
    age: int
    anychronicdiseases: int
    anytransplants: int
    bloodpressureproblems: int
    diabetes: int
    height: int
    historyofcancerinfamily: int
    knownallergies: int
    numberofmajorsurgeries: int
    weight: int


model_ref = bentoml.xgboost.get("ins_price_xgb_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

srvc = bentoml.Service("ins_premium_prediction", runners = [model_runner])

@srvc.api(input = JSON(pydantic_model = InsuranceApplication), output = JSON())
def classify(ins_application):
    application_data = ins_application.dict()
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)

    result = np.expm1(prediction)
    print('Predicted premium price: ', result)

    return result
