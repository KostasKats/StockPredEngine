# Stock Predictor Engine using TensorFlow LSTM

This repository contains a stock predictor engine built using TensorFlow LSTM (Long Short-Term Memory) model. The engine utilizes historical stock data to make predictions on future stock prices.

## Getting Started

### Prerequisites

- Python 3.11
- TensorFlow library
- Keras
- Pandas library
- Numpy library
- Matplotlib library


This repository focuses on the development and optimization of machine learning models for predictive analytics, specifically targeting deep learning approaches such as LSTM and Bidirectional LSTM networks. The project emphasizes the exploration and application of various techniques to improve model performance, efficiency, and training effectiveness. Key strategies used include:

## Batch Size Adjustment: 
Experimentation with different batch sizes to balance training speed and model performance. Smaller batch sizes lead to noisier updates, while larger batch sizes may speed up training at the cost of increased memory usage.

## Learning Rate Scheduling: 
Implementation of learning rate schedulers, such as ReduceLROnPlateau, to optimize the learning rate during training, helping the model converge faster and achieve better performance.

## Optimizer Tuning: 
Utilization of optimizers like Adam, RMSprop, and Nadam to determine the most suitable one for each specific problem, enhancing model efficiency and performance.

##  Early Stopping and Epoch Tuning: 
Incorporation of early stopping techniques to prevent overfitting while allowing the model sufficient time to learn by increasing the number of epochs.

## Shuffling and Data Augmentation: 
Application of data shuffling and augmentation techniques to ensure better generalization of models, particularly when working with time-series data.

## Model Monitoring with TensorBoard:
Real-time tracking and visualization of training metrics using TensorBoard to facilitate debugging, optimization, and performance analysis.

## Validation Splitting: 
Use of validation splits to evaluate model performance when separate validation sets are unavailable, improving the reliability of training metrics.

# TODO
## Parallelized Data Loading: 
Implementation of custom data generators to parallelize data loading, optimizing training time and handling large datasets more effectively.
## Gradient Clipping: 
Introduction of gradient clipping for RNNs and LSTMs to mitigate the issue of exploding gradients and stabilize the training process.



