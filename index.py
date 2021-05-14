import NaiveBayesClassifier from './naive_bayes'
import fs from 'fs'
import { shuffleArray, splitArr, convertArrayToDatasamples } from './helpers'
import { IFC_Iris_Data_Sample } from './Types'
import * as nj from 'numjs'

let classifier = new NaiveBayesClassifier()

let buffer = fs.readFileSync('./iris_123.csv').toString()
let lines = buffer.split('\n')
// console.log(lines)
let data = lines.map(line => line.split(','))
let datasetInfo = convertArrayToDatasamples(data)
// console.log(dataset)
let dataset = datasetInfo.dataset

dataset = shuffleArray(dataset)
let training: IFC_Iris_Data_Sample[]
let testing: IFC_Iris_Data_Sample[]
[training, testing] = splitArr(dataset, 4 / 5)
console.log(training.length, testing.length)

classifier.loadDataset(training)
let loss = classifier.train()

for (let sample of testing) {
    let prediction = classifier.predict(sample, true)
    console.log(prediction[0], sample.category)
}

let accuracy = classifier.computeAccuracy()
console.log({ loss, accuracy })