# Advanced Time Series Forecasting with Uncertainty Quantification
This program implements a Bayesian LSTM for multi-step time series forecasting with Monte Carlo Dropout for uncertainty quantification.

1. Introduction & Objective
The objective of this project is to develop a robust time series forecasting model that not only predicts future values but also enables Uncertainty Quantification (UQ). Standard deep learning models often lack the ability to express confidence in their predictions. This project addresses that limitation by implementing a Bayesian LSTM using Monte Carlo (MC) Dropout, allowing us to estimate epistemic uncertainty (model uncertainty) and provide Prediction Intervals (PIs).

2. Methodology
2.1 Data Generation
Synthetic data was generated to simulate realistic time series characteristics, specifically incorporating:
Trend: A linear upward trend (slope=0.015).
Seasonality: A sinusoidal pattern (period=60).
Heteroscedastic Noise: Noise that varies with the signal amplitude, mimicking complex real-world environments.
Preprocessing: The data was split into Train, Validation, and Test sets (preserving temporal order) and scaled using StandardScaler to ensure stable training.

2.2 Model Architecture
We implemented a Bayesian LSTM network using TensorFlow/Keras.
Components:
Two stacked LSTM layers (64 units each) with MC Dropout.
One final LSTM layer.
Dense output layer for regression.
Uncertainty Mechanism: Unlike standard dropout which is active only during training, MC Dropout remains active during inference (training=True). This allows us to perform multiple stochastic forward passes (100 samples) for a single input, generating a distribution of predictions.

2.3 Training Strategy
Loss Function: Mean Squared Error (MSE).
Optimizer: Adam.
Hyperparameters:
Epochs: 5 (Demonstration).
Batch Size: 32.
Dropout Rate: 0.2.

3. Findings & Results
The model was evaluated on the unseen Test set. The following metrics were recorded:
3.1 Error Metrics
RMSE (Root Mean Squared Error): 2.975
MAE (Mean Absolute Error): 2.471
These metrics indicate the average deviation of the model's point forecast (mean of MC samples) from the actual values.

3.2 Uncertainty Metrics
PICP (Prediction Interval Coverage Probability): 0.342 (34.2%)
This measures the percentage of true values that fell within the generated confidence intervals. A value of ~34% suggests the model is currently under-confident or the intervals are too narrow, likely due to the limited training epochs (5 epochs) or the specific dropout rate used.
MPIW (Mean Prediction Interval Width): 3.044
This reflects the sharpness of the model's uncertainty. A width of ~3.04 indicates the average range of the confidence bounds.

4. Conclusion & Critical Analysis
We successfully implemented a pipeline for Probabilistic Time Series Forecasting.

Strengths
Viability of MC Dropout: The approach effectively generates prediction intervals without complex architectural changes (like Ensembles or Variational Inference), making it practical for existing LSTM pipelines.
Uncertainty Capture: The model captured the heteroscedastic nature of the data, with intervals widening in more volatile regions.

Weaknesses & Improvements
Calibration: The low PICP (34%) is the primary weakness. The model underestimates uncertainty.
Recommendation: Tune the dropout_rate. Higher dropout generally leads to wider intervals.
Recommendation: Implement Conformal Prediction as a post-processing step to calibrate the intervals to the desired coverage (e.g., 90%).
Aleatoric Uncertainty: We currently capture Model Uncertainty (Epistemic).
Recommendation: Update the final layer to predict distributional parameters (Mean and Variance) and use Negative Log Likelihood loss to capture data noise (Aleatoric).
