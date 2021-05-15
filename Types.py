from typing import Sequence, Dict, List
from dataclasses import dataclass


@dataclass
class Iris_Data_Sample:
    features: Sequence[float]
    category: str


@dataclass
class Summary:
    noOfSamples: float
    noOfFeatures: float
    mean: List[float]
    stddev: List[float]


Iris_Dataset = List[Iris_Data_Sample]
Summary_By_Class = Dict[str, Summary]
Matrix_Of_Strings = List[List[str]]
