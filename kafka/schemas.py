from pydantic import BaseModel

class PredictIn_ppm(BaseModel):
    turbidity: float
    alkalinity: float
    conductivity: float
    pH: float
    temp: float
    year: int
    month: int
    turbidity_avg24: float

class PredictOut_ppm(BaseModel):
    target: float

class PredictIn_tur(BaseModel):
    turbidity: float
    turbidity_4h: float
    temp: float
    rainfall: float
    t_diff: float

class PredictOut_tur(BaseModel):
    target: float

