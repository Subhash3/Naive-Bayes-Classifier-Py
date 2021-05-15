import json
import helpers
from Types import Iris_Data_Sample, Iris_Dataset, Matrix_Of_Strings, Summary, Summary_By_Class
from typing import Set, Dict
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.dataset: Iris_Dataset = []
        self.noOfFeatures = 0
        self.noOfSamples = 0
        self.summaryByClass: Summary_By_Class = dict()

    def _parseSimpleMetadata(self):
        self.noOfSamples = len(self.dataset)
        self.noOfFeatures = 0 if len(
            self.dataset) <= 0 else len(self.dataset[0].features)

    def loadDatasetFromFile(self, filename):
        # Loads the dataset from a json file of data samples
        try:
            with open(filename) as fp:
                dataset = json.load(fp)
                for sample in dataset:
                    dataSample = Iris_Data_Sample(
                        sample["features"],
                        sample["category"]
                    )
                    self.dataset.append(dataSample)

        except Exception as e:
            raise e
        self._parseSimpleMetadata()

    def loadDataset(self, dataset: Iris_Dataset):
        # Assign the dataset to be an array of pre-loaded samples
        self.dataset = dataset
        self._parseSimpleMetadata()

    def loadDatasetFromArr(self, data: Matrix_Of_Strings):
        """ Load the dataset from an array(array(strings)) 
            Something like this
            [
                ["1.0", "3.2", "6.9",    "4.20"],
                ["1.0", "3.2", "6.9",    "4.20"],
                ["1.0", "3.2", "6.9",    "4.20"],
                <--    Features   --> <- Category ->
            ]
        """
        [_noOfFeatures, _noOfSamples,
            dataset] = helpers.convertArrayToDatasamples(data)
        self.dataset = dataset
        self._parseSimpleMetadata()

    def allCategories(self, data: Iris_Dataset):
        allCategories: Set[str] = set()

        for sample in data:
            allCategories.add(sample.category)

        return list(allCategories)

    @staticmethod
    def describeData(dataset: Iris_Dataset):
        noOfSamples = len(dataset)
        noOfFeatures = 0 if len(dataset) <= 0 else len(dataset[0].features)

        summary: Summary = Summary(
            noOfSamples=noOfSamples,
            noOfFeatures=noOfFeatures,
            mean=[],
            stddev=[]
        )

        for i in range(noOfFeatures):
            featureValues = []

            for sample in dataset:
                featureValues.append(sample.features[i])

            m = np.mean(featureValues)
            s = np.std(featureValues)
            summary.mean.append(m)
            summary.stddev.append(s)

        return summary

    @staticmethod
    def separateByClass(dataset: Iris_Dataset):
        separatedByClass: Dict[str, Iris_Dataset] = dict()

        for sample in dataset:
            if sample.category not in separatedByClass:
                separatedByClass[sample.category] = list()
            separatedByClass[sample.category].append(sample)

        return separatedByClass

    @staticmethod
    def describeByClass(dataset: Iris_Dataset):
        separatedByClass = NaiveBayesClassifier.separateByClass(dataset)
        summaryByClass: Summary_By_Class = dict()

        for category in summaryByClass:
            samples = separatedByClass[category]
            summaryOfThisCategory = NaiveBayesClassifier.describeData(samples)

            summaryByClass[category] = summaryOfThisCategory

    def train(self):
        self.summaryByClass = NaiveBayesClassifier.describeByClass(
            self.dataset)

    def computeClassProbabilities(self, newSample: Iris_Data_Sample):

        # Bayes theorem:
        # Posterior = (likelihood*prior)/evidence
        # P(class/X) = (P(X/class) * P(class))/P(X)
        # We generally ignore the denominator, i.e, evidence

        probabilities: Dict[str, float] = dict()

        for category in self.summaryByClass:
            priorProbability = self.summaryByClass[category].noOfSamples / \
                self.noOfSamples
            evidence = 1

            likelihood = 1

            for i in range(self.summaryByClass[category].noOfFeatures):
                # mean of ith feature of this class
                m = self.summaryByClass[category].mean[i]
                # standard deviation of ith feature of this class
                s = self.summaryByClass[category].stddev[i]
                x = newSample.features[i]  # ith feature of given sample
                p = helpers.gaussianPdf(x, m, s)

                likelihood *= p
            # P(class=category/newSample)
            probabilities[category] = (likelihood*priorProbability)/evidence

        return probabilities

    def predict(self, newSample):
        probabilities = self.computeClassProbabilities(newSample)

        return helpers.customArgmax(probabilities), probabilities
