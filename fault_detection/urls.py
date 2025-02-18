from django.urls import path
from .views import*

urlpatterns =[
    path('',inputform),
    path('result/',result)

]