import { IFC_Iris_Data_Sample } from './Types'

const { sqrt, PI, exp, pow, floor, random } = Math

export const gaussianPdf = (x: number, mean: number, stddev: number) => {
    return (1 / (stddev * (sqrt(2 * PI)))) * (exp((-1 / 2) * pow(((x - mean) / (stddev)), 2)))
}

export const customArgmax = (data: { [key: string]: number }) => {
    let maxKey: string | null = null

    let maxValue: number | null = null
    Object.keys(data).forEach(key => {
        if (maxKey == null) maxKey = key

        if (maxValue == null || maxValue < data[key]) {
            maxValue = data[key]
            maxKey = key
        }
    })

    return maxKey
}

export const shuffleArray = (array: any[]) => {
    var currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = floor(random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }

    return array;
}

export const splitArr = (array: any[], ratio: number) => {
    let n = array.length

    let m = floor(n * ratio)

    let firstPart: any[] = array.slice(0, m)
    let secondPart: any[] = array.slice(m, n)

    return [firstPart, secondPart]
}

export const convertArrayToDatasamples = (data: (string[])[]) => {
    let noOfSamples = 0
    let noOfFeatures = 0

    noOfSamples = data.length
    noOfFeatures = (noOfSamples > 0) ? data[0].length - 1 : 0
    let dataset: IFC_Iris_Data_Sample[] = []

    for (let row of data) {
        let features: number[] = row.slice(0, noOfFeatures).map(num => parseFloat(num))
        let category: string = row[noOfFeatures]
        let dataSample: IFC_Iris_Data_Sample = {
            features,
            category
        }

        dataset.push(dataSample)
    }

    return {
        noOfFeatures,
        noOfSamples,
        dataset
    }
}