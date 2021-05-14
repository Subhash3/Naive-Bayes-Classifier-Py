import { IFC_Iris_Data_Sample, IFC_Summary, IFC_Summary_By_Class } from './Types'
import * as nj from "numjs"
import fs from 'fs'
import { gaussianPdf, customArgmax, convertArrayToDatasamples } from './helpers'


class NaiveBayesClassifier {
    dataset: IFC_Iris_Data_Sample[]
    noOfFeatures: number = 0
    noOfSamples: number = 0
    summaryByClass: IFC_Summary_By_Class = {}

    constructor() {
        this.dataset = []
    }

    loadDatasetFromFile(file: string) {
        /* Loads the dataset from a json file of data samples*/
        try {
            this.dataset = JSON.parse(fs.readFileSync(file).toString())
            this.noOfSamples = this.dataset.length
            if (this.dataset.length > 0) {
                this.noOfFeatures = this.dataset[0].features.length
            }
            // console.log(this.dataset);
        } catch (err) {
            console.log("Error loading dataset!!", err.message)
        }
    }

    loadDataset(data: IFC_Iris_Data_Sample[]) {
        /* Assign the dataset to be an array of pre-loaded samples*/
        this.dataset = data
        this.noOfSamples = this.dataset.length
        if (this.dataset.length > 0) {
            this.noOfFeatures = this.dataset[0].features.length
        }
    }

    loadDatasetFromArr(data: (string[])[]) {
        /* Load the dataset from an array(array(strings)) 
            Something like this
            [
                ["1.0", "3.2", "6.9",    "4.20"],
                ["1.0", "3.2", "6.9",    "4.20"],
                ["1.0", "3.2", "6.9",    "4.20"],
                <--    Features   --> <- Category ->
            ]
        */
        let { noOfFeatures, noOfSamples, dataset } = convertArrayToDatasamples(data)

        this.noOfSamples = noOfSamples
        this.noOfFeatures = noOfFeatures
        this.dataset = dataset
    }

    private extractAllCategories(data: IFC_Iris_Data_Sample[]) {
        let allCategories: Set<string> = new Set<string>()

        for (let sample of data) {
            allCategories.add(sample.category)
        }

        return Array.from(allCategories)
    }

    private describeData(dataset: IFC_Iris_Data_Sample[], display: boolean = true): IFC_Summary {
        let noOfFeatures = (dataset.length > 0) ? dataset[0].features.length : 0

        let summary: IFC_Summary = {
            noOfSamples: dataset.length,
            noOfFeatures,
            mean: [] as number[], // mean of values of each feature
            stddev: [] as number[], // standard deviation of values each feature
        }


        // Iterate over each feature
        for (let i = 0; i < noOfFeatures; i++) {
            let featureValues: number[] = []

            // Accumulate the values of feature i 
            for (let sample of dataset) {
                featureValues.push(sample.features[i])
            }
            // console.log(featureValues)
            let m = nj.mean(featureValues)
            let s = nj.std(featureValues)
            summary.mean.push(m)
            summary.stddev.push(s)
        }

        if (display) {
            console.log("No. of samples: ", summary.noOfSamples)
            for (let i = 0; i < noOfFeatures; i++) {
                console.log(`Mean of ${i}th feature's values: ${summary.mean[i]}`)
                console.log(`Standard Deviation of ${i}th feature's values: ${summary.stddev[i]}`)
            }
        }

        return summary
    }

    private separateByClass(dataset: IFC_Iris_Data_Sample[]) {
        let separatedByClass: { [category: string]: IFC_Iris_Data_Sample[] } = {}

        for (let dataSample of dataset) {
            let { category } = dataSample

            if (!(category in separatedByClass)) {
                separatedByClass[category] = []
            }

            separatedByClass[category].push(dataSample)
        }

        // console.log(separatedByClass);
        return separatedByClass
    }

    private describeByClass(display: boolean = false): IFC_Summary_By_Class {
        let separatedByClass = this.separateByClass(this.dataset)
        let categories = Object.keys(separatedByClass)
        let summaryByClass: IFC_Summary_By_Class = {}

        for (let category of categories) {
            let samples = separatedByClass[category]
            let summary = this.describeData(samples, display)

            summaryByClass[category] = summary
        }

        return summaryByClass
    }

    private getTargetVector(category: string, allCategories: string[]) {
        let target: { [key: string]: number } = {}

        for (let c of allCategories) {
            target[c] = (c == category) ? 1 : 0
        }

        return target
    }

    computeLoss() {
        // let allCategories = this.extractAllCategories(this.dataset)
        let inCorrect = 0

        for (let sample of this.dataset) {
            let prediction: any = this.predict(sample)
            // console.log(prediction[0], sample.category, prediction[0] == sample.category)
            if (prediction[0] != sample.category) {
                inCorrect += 1
            }
        }
        // console.log({ inCorrect })
        return inCorrect * 100 / this.noOfSamples
    }

    computeAccuracy() {
        let loss = this.computeLoss()

        return 100 - loss
    }

    private applySoftMax(output: { [category: string]: number }) {
        let keys = Object.keys(output)
        let values = Object.values(output)

        // Applying softmax twice inorder to bring the probabilities to a reasonable range
        let softMaxedValues = nj.softmax(values)
        softMaxedValues = nj.softmax(softMaxedValues)

        let revised: { [category: string]: number } = {}
        for (let i = 0; i < keys.length; i++) {
            revised[keys[i]] = softMaxedValues.get(i)
        }

        return revised
    }

    private computeClassProbabilities(newSample: IFC_Iris_Data_Sample) {
        // console.log(this.summaryByClass)

        let probabilities: { [category: string]: number } = {}

        let categories = Object.keys(this.summaryByClass)


        /* P(class/X) = P(x1/class)*P(x2/class) * .... * P(xn/class) * P(C) */

        for (let category of categories) {
            // Prior Probability
            let priorProbability = this.summaryByClass[category].noOfSamples / this.noOfSamples
            probabilities[category] = Math.log10(priorProbability)

            let summary = this.summaryByClass[category]

            // Bayes Chain
            for (let i = 0; i < summary.noOfFeatures; i++) {
                let m = summary.mean[i]
                let s = summary.stddev[i]
                let p = gaussianPdf(newSample.features[i], m, s)

                probabilities[category] += Math.log10(p)
            }
        }

        probabilities = this.applySoftMax(probabilities)

        return probabilities
    }

    /*Computes the mean and standard deviation of different features of each category*/
    train() {
        this.summaryByClass = this.describeByClass()
        return this.computeLoss()
    }

    /* Returns the probability of each class given a new data sample */
    predict(newSample: IFC_Iris_Data_Sample, returnProbabilities: boolean = false) {
        let probabilities = this.computeClassProbabilities(newSample)

        if (returnProbabilities) {
            return [customArgmax(probabilities), probabilities]
        }

        return [customArgmax(probabilities), null]
    }
}

export default NaiveBayesClassifier;