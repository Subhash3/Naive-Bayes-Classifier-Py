from typing import Sequence, Dict
from dataclasses import dataclass


@dataclass
class Iris_Data_Sample:
    features: Sequence[float]
    category: str


@dataclass
class IFC_Summary:
    noOfSamples: float
    noOfFeatures: float
    mean: Sequence[float]
    stddev: Sequence[float]


IFC_Summary_By_Class = Dict[str, IFC_Summary]
