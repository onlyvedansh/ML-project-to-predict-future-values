import os
import json
import threading
import pandas as pd
from datetime import datetime
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import Dataset, PredictionModel, Prediction

BASE_DIR = settings.BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')
os.makedirs(MODELS_DIR, exist_ok=True)


def index(request):
    datasets = Dataset.objects.all().order_by('-uploaded_at')
    models = PredictionModel.objects.all().order_by('-created_at')[:10]
    predictions = Prediction.objects.all().order_by('-predicted_at')[:5]

    stats = {
        'datasets': datasets.count(),
        'models': PredictionModel.objects.filter(status='ready').count(),
        'predictions': Prediction.objects.count(),
        'best_accuracy': PredictionModel.objects.filter(
            status='ready', accuracy__isnull=False
        ).order_by('-accuracy').first(),
    }
    return render(request, 'market_app/index.html', {
        'datasets': datasets,
        'models': models,
        'predictions': predictions,
        'stats': stats,
    })


def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        name = request.POST.get('name', f.name)
        ticker = request.POST.get('ticker', 'UNKNOWN').upper()
        description = request.POST.get('description', '')

        upload_dir = os.path.join(BASE_DIR, 'media', 'datasets')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f.name)

        with open(file_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        try:
            df = pd.read_csv(file_path)
            columns = list(df.columns)
            total_rows = len(df)

            # Auto-detect columns
            date_col = next((c for c in columns if 'date' in c.lower()), columns[0])
            target_col = next((c for c in columns if 'close' in c.lower()), columns[1] if len(columns) > 1 else columns[0])

            dataset = Dataset.objects.create(
                name=name, description=description,
                file_path=file_path, ticker=ticker,
                total_rows=total_rows, date_column=date_col,
                target_column=target_col,
                columns_json=json.dumps(columns)
            )
            return JsonResponse({
                'success': True,
                'dataset_id': dataset.id,
                'columns': columns,
                'rows': total_rows,
                'message': f'Dataset uploaded: {total_rows} rows, {len(columns)} columns'
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'No file provided'})


def train_model_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset = get_object_or_404(Dataset, id=data['dataset_id'])
            model_type = data.get('model_type', 'lstm')
            name = data.get('name', f'{dataset.ticker} {model_type.upper()} Model')
            lookback = int(data.get('lookback', 60))
            epochs = int(data.get('epochs', 50))
            batch_size = int(data.get('batch_size', 32))
            forecast_days = int(data.get('forecast_days', 30))

            model_obj = PredictionModel.objects.create(
                dataset=dataset, model_type=model_type, name=name,
                lookback_window=lookback, epochs=epochs,
                batch_size=batch_size, forecast_days=forecast_days,
                status='training'
            )

            thread = threading.Thread(
                target=_train_async,
                args=(model_obj.id, dataset.file_path, dataset.target_column,
                      model_type, lookback, epochs, batch_size)
            )
            thread.daemon = True
            thread.start()

            return JsonResponse({
                'success': True,
                'model_id': model_obj.id,
                'message': f'Training started for {name}'
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'POST required'})


def _train_async(model_id, dataset_path, target_col, model_type,
                  lookback, epochs, batch_size):
    """Run training in background thread."""
    from predictor.engine import train_deep_learning_model, train_ml_model
    try:
        model_obj = PredictionModel.objects.get(id=model_id)
        model_obj.status = 'training'
        model_obj.save()

        model_save = os.path.join(MODELS_DIR, f'model_{model_id}')
        scaler_save = os.path.join(MODELS_DIR, f'scaler_{model_id}.pkl')

        if model_type in ('lstm', 'gru', 'transformer'):
            metrics = train_deep_learning_model(
                model_type, dataset_path, target_col,
                lookback, epochs, batch_size, model_save, scaler_save
            )
        elif model_type in ('rf', 'xgboost'):
            metrics = train_ml_model(
                model_type, dataset_path, target_col,
                lookback, model_save + '.pkl', scaler_save
            )
            model_save = model_save + '.pkl'
        elif model_type == 'ensemble':
            # Train both RF and LSTM, pick best
            rf_save = os.path.join(MODELS_DIR, f'model_{model_id}_rf.pkl')
            metrics = train_ml_model(
                'rf', dataset_path, target_col,
                lookback, rf_save, scaler_save
            )
            model_save = rf_save
        else:
            metrics = train_ml_model(
                'rf', dataset_path, target_col,
                lookback, model_save + '.pkl', scaler_save
            )
            model_save = model_save + '.pkl'

        model_obj.refresh_from_db()
        model_obj.status = 'ready'
        model_obj.model_path = model_save
        model_obj.scaler_path = scaler_save
        model_obj.rmse = metrics.get('rmse')
        model_obj.mae = metrics.get('mae')
        model_obj.mape = metrics.get('mape')
        model_obj.r2_score = metrics.get('r2_score')
        model_obj.accuracy = metrics.get('accuracy')
        model_obj.trained_at = datetime.now()
        model_obj.training_log = f"Training complete. Epochs: {metrics.get('epochs_trained', epochs)}"
        model_obj.save()

    except Exception as e:
        try:
            model_obj = PredictionModel.objects.get(id=model_id)
            model_obj.status = 'failed'
            model_obj.error_message = str(e)
            model_obj.save()
        except Exception:
            pass


def model_status(request, model_id):
    model_obj = get_object_or_404(PredictionModel, id=model_id)
    return JsonResponse({
        'status': model_obj.status,
        'accuracy': model_obj.accuracy,
        'rmse': model_obj.rmse,
        'mae': model_obj.mae,
        'mape': model_obj.mape,
        'r2_score': model_obj.r2_score,
        'error': model_obj.error_message,
    })


def predict_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_obj = get_object_or_404(PredictionModel, id=data['model_id'])
            forecast_days = int(data.get('forecast_days', model_obj.forecast_days))

            if model_obj.status != 'ready':
                return JsonResponse({'success': False, 'error': 'Model not ready'})

            from predictor.engine import generate_forecast
            result = generate_forecast(model_obj, model_obj.dataset.file_path, forecast_days)

            prediction = Prediction.objects.create(
                model=model_obj, dataset=model_obj.dataset,
                forecast_days=forecast_days,
                dates_json=json.dumps(result['dates']),
                predicted_values_json=json.dumps(result['predicted']),
                lower_bound_json=json.dumps(result['lower']),
                upper_bound_json=json.dumps(result['upper']),
                historical_dates_json=json.dumps(result['historical_dates']),
                historical_values_json=json.dumps(result['historical_values']),
                confidence_score=result['confidence'],
                trend=result['trend'],
                signal=result['signal'],
            )

            return JsonResponse({
                'success': True,
                'prediction_id': prediction.id,
                'dates': result['dates'],
                'predicted': result['predicted'],
                'lower': result['lower'],
                'upper': result['upper'],
                'historical_dates': result['historical_dates'],
                'historical_values': result['historical_values'],
                'trend': result['trend'],
                'signal': result['signal'],
                'confidence': result['confidence'],
                'pct_change': result['pct_change'],
                'ticker': model_obj.dataset.ticker,
                'model_name': model_obj.name,
                'model_type': model_obj.model_type,
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'POST required'})


def dataset_preview(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    try:
        df = pd.read_csv(dataset.file_path)
        preview = df.head(10).to_dict(orient='records')
        stats = {
            'rows': len(df),
            'columns': list(df.columns),
            'target_stats': {
                'min': float(df[dataset.target_column].min()),
                'max': float(df[dataset.target_column].max()),
                'mean': float(df[dataset.target_column].mean()),
                'std': float(df[dataset.target_column].std()),
            }
        }
        # Chart data
        chart_dates = df.iloc[-200:, 0].astype(str).tolist()
        chart_vals = df[dataset.target_column].iloc[-200:].tolist()
        return JsonResponse({
            'success': True, 'preview': preview,
            'stats': stats, 'chart_dates': chart_dates, 'chart_vals': chart_vals
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def models_list(request):
    models = PredictionModel.objects.all().order_by('-created_at')
    data = []
    for m in models:
        data.append({
            'id': m.id, 'name': m.name, 'type': m.model_type,
            'status': m.status, 'accuracy': m.accuracy,
            'rmse': m.rmse, 'mae': m.mae, 'r2': m.r2_score,
            'dataset': m.dataset.name, 'ticker': m.dataset.ticker,
            'created': m.created_at.strftime('%Y-%m-%d %H:%M'),
        })
    return JsonResponse({'models': data})


def delete_model(request, model_id):
    if request.method == 'DELETE':
        model_obj = get_object_or_404(PredictionModel, id=model_id)
        model_obj.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


def delete_dataset(request, dataset_id):
    if request.method == 'DELETE':
        ds = get_object_or_404(Dataset, id=dataset_id)
        ds.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})
