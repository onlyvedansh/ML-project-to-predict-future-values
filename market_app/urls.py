from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/upload/', views.upload_dataset, name='upload_dataset'),
    path('api/train/', views.train_model_view, name='train_model'),
    path('api/predict/', views.predict_view, name='predict'),
    path('api/model/<int:model_id>/status/', views.model_status, name='model_status'),
    path('api/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('api/dataset/<int:dataset_id>/preview/', views.dataset_preview, name='dataset_preview'),
    path('api/dataset/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('api/models/', views.models_list, name='models_list'),
]
