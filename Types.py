export interface IFC_Iris_Data_Sample {
    features: number[],
    category: string
}

export interface IFC_Summary {
    noOfSamples: number,
    noOfFeatures: number,
    mean: number[],
    stddev: number[]
}

export interface IFC_Summary_By_Class {
    [category: string]: IFC_Summary
}
