This folder is empty until you run `python src/train.py`.
It will then contain:
  - churn_model.pkl   (trained XGBoost model + feature list)
  - encoders.pkl      (fitted LabelEncoders, needed by app.py / app_flask.py)
  - metrics.txt       (accuracy, precision, recall, F1, confusion matrix, report)
