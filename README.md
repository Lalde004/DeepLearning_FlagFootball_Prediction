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

## Getting Started
Install Requirements: Ensure you have Python and the required libraries.

bash
Copy code
pip install -r requirements.txt
Dependencies: TensorFlow, pandas, numpy, and scikit-learn.

Load the Data: Use playdata_processed.csv in the data folder for testing and training. The dataset should align with the feature requirements of the model.

Run the Notebook: Open PassingZonePrediction.ipynb for data processing steps, model training, evaluation, and prediction examples.

Load the Model: The pre-trained model is saved as passing_zone_predictor_model.h5. Follow the notebook for usage with new data.

Model Features and Categorical Mappings
The model requires 44 input features, including numeric metrics, categorical indicators, and binary flags. Below are some examples:

1. Play Direction (PLAY DIR)
markdown
Copy code
- `0`: Left  
- `1`: Right  
- `2`: Center  
2. Passing Zones (PASS ZONE)
markdown
Copy code
- `0`: Short Left  
- `1`: Short Center  
- `2`: Short Right  
- `3`: Deep Left  
- `4`: Deep Center  
- `5`: Deep Right  
3. Coverage Types (COVERAGE)
markdown
Copy code
- `0`: Zone  
- `1`: Man-to-Man  
- `2`: Mixed  
- `3`: Blitz  
4. Play Types (PLAY TYPE)
markdown
Copy code
- `PLAY TYPE_Pass`: 1 if pass play, 0 otherwise  
- `PLAY TYPE_Run`: 1 if run play, 0 otherwise  
- `PLAY TYPE_Extra Pt.`: 1 if extra point, 0 otherwise  
For a complete list of feature mappings, refer to the Jupyter notebook.

How to Make Predictions
To predict passing zones for a new play:

Prepare the Play Data: Ensure it matches the required feature order as shown in the PassingZonePrediction.ipynb.
Standardize the Data: Use StandardScaler to maintain consistency with the training set.
Predict: Pass the processed data into the loaded model to get:
Predicted Zone: The most likely passing zone.
Probability Distribution: Probabilities for each zone, providing confidence insights.
Example Code
python
Copy code
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model/passing_zone_predictor_model.h5')

# Define play data (example input)
new_play_data = pd.DataFrame([[
    1, 1, 10, 20, 1, 1, 5, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 
    1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0
]], columns=feature_columns)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(new_play_data)

# Predict
predicted_zone = model.predict(X_scaled)
predicted_label = np.argmax(predicted_zone, axis=1) + 1  # Adjust index if needed

print("Predicted Passing Zone:", predicted_label)
print("Probability Distribution:", predicted_zone)
Results Interpretation
Predicted Zone: The passing zone label with the highest probability.
Probability Distribution: Shows the probability scores for each zone, giving insight into the model's confidence for each possible outcome.
Future Enhancements
Expand Feature Set: Additional situational and environmental data could enhance model accuracy.
Hyperparameter Tuning: Experiment with different optimizers and learning rates to improve performance.
Model Interpretability: Use tools like SHAP or LIME to explain model predictions.
License
This project is licensed under the MIT License.

Contact
For inquiries or support, please reach out to leonardo@sascoisimulation.com
