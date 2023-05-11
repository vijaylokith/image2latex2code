from django.urls import path
from . import views

urlpatterns = [
    path("",views.ExecuteCode.as_view()),
    path("monaco",views.ExecuteMonaco.as_view()),
    path("test",views.ExecuteHello.as_view())
]