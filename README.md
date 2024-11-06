# Passing Zone Prediction Model

This repository contains a machine learning model built to predict passing zones in football plays. Leveraging data from play-by-play analysis, the model identifies which areas of the field are most likely to receive a pass based on situational and tactical features.

## Project Overview

The Passing Zone Prediction Model is designed to support coaches, analysts, and strategists by providing insights into passing tendencies. Using input features such as play direction, formations, field position, and opponent specifics, the model generates a predicted passing zone and a probability distribution over all possible zones.

## Key Features

- **Input Data Preprocessing**: Standardization and encoding of categorical features.
- **Neural Network Architecture**: A TensorFlow/Keras-based neural network trained with features pertinent to football strategy.
- **Class Imbalance Handling**: Custom class weights are applied to address any imbalances in passing zone data.
- **Interpretability**: The model outputs both the predicted zone and a probability distribution across all zones.

## Project Structure

```plaintext
├── README.md                   # Project documentation
├── data/
│   ├── playdata_processed.csv   # Preprocessed play data (input features)
│   ├── passing_zone_labels.csv  # Labels of passing zones
├── model/
│   └── passing_zone_predictor_model.h5  # Trained neural network model
└── notebook/
    └── PassingZonePrediction.ipynb  # Jupyter notebook with data exploration, model training, and evaluation

