from django.db import models
import json


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=500)
    ticker = models.CharField(max_length=50, default='UNKNOWN')
    total_rows = models.IntegerField(default=0)
    date_column = models.CharField(max_length=100, default='Date')
    target_column = models.CharField(max_length=100, default='Close')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns_json = models.TextField(default='[]')

    @property
    def columns(self):
        return json.loads(self.columns_json)

    def __str__(self):
        return f"{self.name} ({self.ticker})"


class PredictionModel(models.Model):
    MODEL_TYPES = [
        ('lstm', 'LSTM Deep Learning'),
        ('gru', 'GRU Deep Learning'),
        ('transformer', 'Transformer Model'),
        ('rf', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('ensemble', 'Ensemble (DL + ML)'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='models')
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    model_path = models.CharField(max_length=500, blank=True)
    scaler_path = models.CharField(max_length=500, blank=True)

    # Hyperparameters
    lookback_window = models.IntegerField(default=60)
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    forecast_days = models.IntegerField(default=30)

    # Metrics
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    trained_at = models.DateTimeField(null=True, blank=True)
    training_log = models.TextField(default='')
    error_message = models.TextField(blank=True)

    def __str__(self):
        return f"{self.name} - {self.model_type} [{self.status}]"


class Prediction(models.Model):
    model = models.ForeignKey(PredictionModel, on_delete=models.CASCADE, related_name='predictions')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='predictions')
    predicted_at = models.DateTimeField(auto_now_add=True)
    forecast_days = models.IntegerField(default=30)

    # Store as JSON
    dates_json = models.TextField(default='[]')
    predicted_values_json = models.TextField(default='[]')
    actual_values_json = models.TextField(default='[]')
    lower_bound_json = models.TextField(default='[]')
    upper_bound_json = models.TextField(default='[]')
    historical_dates_json = models.TextField(default='[]')
    historical_values_json = models.TextField(default='[]')

    confidence_score = models.FloatField(null=True, blank=True)
    trend = models.CharField(max_length=20, default='neutral')  # bullish/bearish/neutral
    signal = models.CharField(max_length=20, default='hold')    # buy/sell/hold

    @property
    def dates(self):
        return json.loads(self.dates_json)

    @property
    def predicted_values(self):
        return json.loads(self.predicted_values_json)

    @property
    def actual_values(self):
        return json.loads(self.actual_values_json)

    @property
    def lower_bound(self):
        return json.loads(self.lower_bound_json)

    @property
    def upper_bound(self):
        return json.loads(self.upper_bound_json)

    @property
    def historical_dates(self):
        return json.loads(self.historical_dates_json)

    @property
    def historical_values(self):
        return json.loads(self.historical_values_json)

    def __str__(self):
        return f"Prediction by {self.model.name} on {self.predicted_at.date()}"
