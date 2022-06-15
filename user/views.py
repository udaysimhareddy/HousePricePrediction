from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from pyforest import sns, plt

from user.forms import *
from user.models import *

def index(request):
    return render(request,'index.html')

def userpage(request):
    return render(request,'user/userpage.html')

def userregister(request):
    if request.method=='POST':
        form1=userForm(request.POST)
        if form1.is_valid():
            form1.save()
            print("succesfully saved the data")
            return render(request, "user/userlogin.html")
            #return HttpResponse("registreration succesfully completed")
        else:
            print("form not valied")
            return HttpResponse("form not valied")
    else:
        form=userForm()
        return render(request,"user/userregister.html",{"form":form})


def userlogincheck(request):
    if request.method == 'POST':
        mail = request.POST.get('mail')
        print(mail)
        spasswd = request.POST.get('spasswd')
        print(spasswd)
        try:
            check = usermodel.objects.get(email=mail, passwd=spasswd)
            # print('usid',usid,'pswd',pswd)
            print(check)
            request.session['name'] = check.name
            print("name",check.name)
            status = check.status
            print('status',status)
            if status == "Activated":
                request.session['email'] = check.email
                return render(request, 'user/userpage.html')
            else:
                messages.success(request, 'user  is not activated')
                return render(request, 'user/userlogin.html')
        except Exception as e:
            print('Exception is ',str(e))

        messages.success(request,'Invalid name and password')
        return render(request,'user/userlogin.html')


def userlogin(request): 
    messages.success(request,'Invalid name and password')
    return render(request,'user/userlogin.html')


def adddata(request):
    if request.method=='POST':
        longitude= request.POST.get('longitude')
        latitude= request.POST.get('latitude')
        housing_median_age= request.POST.get('housing_median_age')
        total_rooms= request.POST.get('total_rooms')
        total_bedrooms= request.POST.get('total_bedrooms')
        population= request.POST.get('population')
        households= request.POST.get('households')
        median_income= request.POST.get('median_income')
        median_house_value= request.POST.get('median_house_value')
        ocean_proximity= request.POST.get('ocean_proximity')
        print("longitude:",longitude,"latitude",latitude,"housing_median_age",housing_median_age)
        print("total_rooms:",total_rooms,"total_bedrooms",total_bedrooms,"population",population)
        print("households:",households,"median_income",median_income,"median_house_value",median_house_value,"ocean_proximity",ocean_proximity)
        csvdatamodel(longitude=longitude,latitude=latitude,housing_median_age=housing_median_age,total_rooms=total_rooms,total_bedrooms=total_bedrooms,population=population,households=households,median_income=median_income,median_house_value=median_house_value,ocean_proximity=ocean_proximity).save()
        return render(request,'user/adddata.html')

        #     return render(request, "user/adddata.html")
        #     #return HttpResponse("registreration succesfully completed")
    else:
        form=csvdatamodelForm()
        return render(request,"user/adddata.html",{"form":form})

def houseprediction(request):
    import numpy as np
    import pandas as pd
    import pyforest

    df = pd.read_csv('housing.csv')
    df.head()
    df.info()
    # print(df.shape)
    # print(df.isnull().sum())
    sns.heatmap(df.isnull())
    # plt.show()
    print(df.describe())
    plt.figure(figsize=(10, 8))
    sns.distplot(df['housing_median_age'], color='g')
    # plt.show()
    # print("house median age-min:",df['housing_median_age'].min())
    # print("house median age-min:",df['housing_median_age'].max())

    # corr between feartures
    corr_matrix = df.corr()
    corr_df = corr_matrix['median_house_value'].sort_values(ascending=False)
    print(corr_df)
    plt.figure(figsize=(12, 7))
    sns.heatmap(corr_matrix, annot=True)
    # plt.show()
    from pandas.plotting import scatter_matrix
    attr = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(df[attr], figsize=(16, 8), color='g', alpha=0.3)
    # plt.show()
    plt.figure(figsize=(16, 8))
    sns.pairplot(df[attr])
    # plt.show()
    df.plot(kind='scatter', x='median_income', y='median_house_value', c='g', figsize=(10, 7))
    plt.show()
    # handle categorical variable

    df['ocean_proximity'].value_counts()
    pd.get_dummies(df['ocean_proximity']).head(3)
    dummy = pd.get_dummies(df['ocean_proximity'])
    dummy.drop('ISLAND', axis=1, inplace=True)
    dummy.head(2)
    df.merge(dummy, left_index=True, right_index=True).isna().sum()
    df['<1H OCEAN'] = dummy['<1H OCEAN'].values
    df['INLAND'] = dummy['INLAND'].values
    df['NEAR BAY'] = dummy['NEAR BAY'].values
    df['NEAR OCEAN'] = dummy['NEAR OCEAN'].values
    print(df.head(2))
    print(df.isnull().sum())
    # fill null values
    # from sklearn.preprocessing import Imputer
    from sklearn.impute import SimpleImputer
    train_ft = df.drop(['ocean_proximity', 'median_house_value'], axis=1)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_ft)
    train_ft.median().values
    x = imputer.transform(train_ft)
    train_new_set = pd.DataFrame(x, columns=train_ft.columns)
    train_new_set.head()
    train_new_set.isna().sum()
    train_new_set.head()
    train_new_set.shape
    train_new_set.info()
    X = train_new_set.values
    Y = df['median_house_value']
    # split the data

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2020)
    x_train.shape
    y_test.shape
    x_test.shape

    # model linear regression

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test[:10])
    print("predictions:", predictions)
    y_train[:10]
    data = {'predicted': predictions, 'Actual': y_test[:10].values, 'Diff': (predictions - y_test[:10].values)}
    error_df = pd.DataFrame(data=data)
    print("error diff:", error_df)
    return render(request,"user/houseprediction.html",{"errordiff":error_df})

    # model evaluation





