from django.urls import path
from generate import views
from . import views

urlpatterns = [
    path('', views.generate, name='generate'),
    path('event', views.event, name='event'),
]