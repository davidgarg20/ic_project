from django.urls import path


from . import views


urlpatterns = [
    path('home/', views.home, name="home"),
    path('pickdate/<int:pk>/<str:s_type>/', views.pickdate, name="pickdate"),
    path('graph/<int:pk>/<str:s_type>/',views.graph, name="graph")
]