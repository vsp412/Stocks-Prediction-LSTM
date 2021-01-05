# Stocks-Prediction-LSTM
This code can be checked using a link constructed using the following format:
http://52.90.141.77/engine?algo=<ALGORITHM_NAME>&tick=<TICKER_SYMBOL>&ptype=<THE_METRIC_YOU_WANT_TO_PREDICT>&daysx=<PREDICTION_RANGE>
Example: http://52.90.141.77/engine?algo=Gradient&tick=AAL&ptype=Close&daysx=6. 
This returns the predicted values of the closing price of American Airlines stock (AAL) using Gradient boost for the next 6 consecutive days.
Allowed values for ticker symbols (tick): All currently listed NASDAQ stocks
Allowed values for algorithm (algo): 'Gradient', 'Meta', 'Delta' (remove single apostrophe)
Allowed values for price type (ptype): 'Volume', 'High', 'Low', 'Mid', 'Close'
Allowed values for prediction range (daysx): 1-10.
