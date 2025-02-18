from django.urls import path
from .views import*

urlpatterns =[
    path('input/',inputform),
    path('result/',result)

]