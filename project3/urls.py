"""project3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path

from admn import views as admn
from user import views as user

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', user.index, name="index"),
    url(r'^index/',admn.index, name="index"),
    url(r'^adminlogin/',admn.adminlogin, name="adminlogin"),
    url(r'^adminloginaction/', admn.adminloginaction, name="adminloginaction"),
    url(r'^userdetails/', admn.userdetails, name="userdetails"),
    path('activateuser/', admn.activateuser, name='activateuser'),
    path('storecsvdata1/', admn.storecsvdata1, name='storecsvdata1'),
    path('logout/', admn.logout, name='logout'),
    url(r'^lr/',admn.lr,name="lr"),
    url(r'^lr1/',admn.lr1,name="lr1"),


    path('userlogin/',user.userlogin,name='userlogin'),
    path('userpage/',user.userpage,name='userpage'),
    path('userregister/',user.userregister,name='userregister'),
    path('userlogincheck/',user.userlogincheck,name='userlogincheck'),
    path('houseprediction/',user.houseprediction,name='houseprediction'),
    path('adddata/',user.adddata,name='adddata')
]


