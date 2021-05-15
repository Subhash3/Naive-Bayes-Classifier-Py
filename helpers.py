from Types import Iris_Data_Sample
from math import sqrt, exp, floor, pow, PI
import random
from typing import Dict, Sequence


def gaussianPdf(x: float, mean: float, stddev: float):
    return (1 / (stddev * (sqrt(2 * PI)))) * (exp((-1 / 2) * pow(((x - mean) / (stddev)), 2)))


def customArgmax(data: Dict[str, float]):
    maxKey = None
    maxValue = None

    for key in data:
        if maxKey == None:
            maxKey = key
        if maxValue == None or maxValue < data[key]:
            maxValue = data[key]
            maxKey = key

    return maxKey


def shuffleArray(array: list):
    arrayCopy = array.copy()
    random.shuffle(arrayCopy)

    return arrayCopy


def splitArr(array: list, ratio: float):
    n = len(array)

    m = floor(n * ratio)

    firstPart: list = array[0: m]
    secondPart: list = array[m: n]

    return [firstPart, secondPart]


def convertArrayToDatasamples(data: Sequence[Sequence[str]]):
    noOfSamples = 0
    noOfFeatures = 0

    noOfSamples = data.length
    noOfFeatures = len(data[0]) - 1 if (noOfSamples > 0) else 0
    dataset: Sequence[Iris_Data_Sample] = []

    for row in data:
        features: Sequence[float] = list(map(float, row[:noOfFeatures]))
        category: str = row[noOfFeatures]
        dataSample: Iris_Data_Sample = Iris_Data_Sample(
            features,
            category
        )

        dataset.push(dataSample)

    return [
        noOfFeatures,
        noOfSamples,
        dataset
    ]
