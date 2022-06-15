from django.db import models

class usermodel(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField()
    passwd = models.CharField(max_length=40)
    cwpasswd = models.CharField(max_length=40)
    mobileno = models.CharField(max_length=50, default="", editable=True)
    status = models.CharField(max_length=40, default="", editable=True)

    def  __str__(self):
        return self.email

    class Meta:
        db_table='userregister'


class csvdatamodel(models.Model):
    longitude = models.CharField(max_length=50)
    latitude = models.EmailField()
    housing_median_age = models.CharField(max_length=40)
    total_rooms = models.CharField(max_length=40)
    total_bedrooms = models.CharField(max_length=50, default="", editable=True)
    population = models.CharField(max_length=40, default="", editable=True)
    households = models.CharField(max_length=40, default="", editable=True)
    median_income = models.CharField(max_length=40, default="", editable=True)
    median_house_value = models.CharField(max_length=40, default="", editable=True)
    ocean_proximity = models.CharField(max_length=40, default="", editable=True)

    class Meta:
        db_table='csvdatamodel'










