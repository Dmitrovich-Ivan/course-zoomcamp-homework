import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from pydantic import BaseModel

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5") #1-mlzoomcamp_homework:qtzdz3slg6mwwdu5 ,2-mlzoomcamp_homework:jsi67fslz6txydu5
model_runner = model_ref.to_runner()
srvc = bentoml.Service("bento-model", runners = [model_runner])

@srvc.api(input = NumpyNdarray(), output = JSON())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)

    print(prediction)
    result = prediction[0]

    return result
