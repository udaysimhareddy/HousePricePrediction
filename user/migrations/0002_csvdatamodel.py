# Generated by Django 3.0.5 on 2020-10-05 13:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='csvdatamodel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('longitude', models.CharField(max_length=50)),
                ('latitude', models.EmailField(max_length=254)),
                ('housing_median_age', models.CharField(max_length=40)),
                ('total_rooms', models.CharField(max_length=40)),
                ('total_bedrooms', models.CharField(default='', max_length=50)),
                ('population', models.CharField(default='', max_length=40)),
                ('households', models.CharField(default='', max_length=40)),
                ('median_income', models.CharField(default='', max_length=40)),
                ('median_house_value', models.CharField(default='', max_length=40)),
                ('ocean_proximity', models.CharField(default='', max_length=40)),
            ],
            options={
                'db_table': 'csvdatamodel',
            },
        ),
    ]
