from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Iris_Data_Sample:
    features: List[float]
    category: str

    def __repr__(self):
        return f"features: {self.features}, category: {self.category}"


@dataclass
class Summary:
    noOfSamples: float
    noOfFeatures: float
    mean: List[float]
    stddev: List[float]

    def __repr__(self):
        return f"""noOfSamples: {self.noOfSamples}\noOfFeatures: {self.noOfFeatures}\nmean: {self.mean}\nstddev: {self.stddev}\n"""


Iris_Dataset = List[Iris_Data_Sample]
Summary_By_Class = Dict[str, Summary]
Matrix_Of_Strings = List[List[str]]
