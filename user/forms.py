from django.core import validators
from django import forms
from user.models import *

class userForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(), required=True, max_length=100,)
    passwd = forms.CharField(widget=forms.PasswordInput(), required=True, max_length=100)
    cwpasswd = forms.CharField(widget=forms.PasswordInput(), required=True, max_length=100)
    email = forms.CharField(widget=forms.TextInput(),required=True)
    mobileno= forms.CharField(widget=forms.TextInput(), required=True, max_length=10,validators=[validators.MaxLengthValidator(10),validators.MinLengthValidator(10)])
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting', max_length=100)

    def __str__(self):
        return self.email

    class Meta:
        model=usermodel
        fields=['name','passwd','cwpasswd','email','mobileno','status']

class csvdatamodelForm(forms.ModelForm):
    longitude = forms.CharField(widget=forms.TextInput(), required=True, max_length=100,)
    latitude = forms.CharField(widget=forms.TextInput(), required=True, max_length=100)
    housing_median_age = forms.CharField(widget=forms.TextInput(), required=True, max_length=100)
    total_rooms = forms.CharField(widget=forms.TextInput(),required=True)
    total_bedrooms = forms.CharField(widget=forms.TextInput(),required=True)
    population= forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    households= forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_income= forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_house_value= forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    ocean_proximity= forms.CharField(widget=forms.TextInput(), required=True, max_length=10)


    def __str__(self):
        return self.median_house_value

    class Meta:
        model=csvdatamodel
        fields=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value','ocean_proximity']








