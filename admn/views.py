from io import TextIOWrapper

from django.shortcuts import render
from django.contrib import messages
# Create your views here.
from django.shortcuts import render
from user.models import *
from user.forms import *
from django.http import HttpResponse
import csv
import io

def index(request):
    return render(request,"index.html")

def logout(request):
    return render(request,'index.html')

def adminlogin(request):
    return render(request,"adminlogin.html")

def adminloginaction(request):
    if request.method == "POST":
        #if request.method == "POST":
            usid = request.POST.get('username')
            pswd = request.POST.get('password')
            if usid == 'admin' and pswd == 'admin':
                return render(request,'admin/adminhome.html')
            else:
                messages.success(request, 'Invalid user id and password')
    #messages.success(request, 'Invalid user id and password')
    return render(request,'adminlogin.html')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline



def lr(request):
    HouseDF = pd.read_csv('USA_Housing.csv')
    HouseDF.info()
    HouseDF.describe()
    HouseDF.columns
    sns.pairplot(HouseDF)
    sns.distplot(HouseDF['Price'])
    sns.heatmap(HouseDF.corr(), annot=True)

    X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                 'Avg. Area Number of Bedrooms', 'Area Population']]

    y = HouseDF['Price']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    from sklearn.linear_model import LinearRegression

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    coeff_df
    print(lm.intercept_)
    predictions = lm.predict(X_test)
    print("prediction:", predictions)
    plt.scatter(y_test, predictions)
    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    sns.distplot((y_test - predictions), bins=50)


def activateuser(request):
    if request.method == 'GET':
        uname = request.GET.get('pid')
        print(uname)
        status = 'Activated'
        print("pid=", uname, "status=", status)
        usermodel.objects.filter(id=uname).update(status=status)
        qs = usermodel.objects.all()
        return render(request, "admin/userdetails.html", {"qs": qs})


def userdetails(request):
    qs = usermodel.objects.all()
    return render(request, 'admin/userdetails.html', {"qs": qs})


def storecsvdata(request):
    if request.method == 'POST':
        # if request.method == "GET":
        # return render(request, template, prompt)
        csv_file = request.FILES['file']
        # let's check if it is a csv file
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
        data_set = csv_file.read().decode('UTF-8')

        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = csvdatamodel.objects.update_or_create(
                name=column[0],
                rating=column[1],
                reviews=column[2],
                type=column[3],
                hq=column[4],
                employees=column[5],

            )
        context = {}

        '''
        name = request.POST.get('name')
        csvfile = TextIOWrapper(request.FILES['file'])
        # columns = defaultdict(list)
        storecsvdata=csv.DictReader(csvfile)

        for row1 in storecsvdata:
            Date = row1["Date"]
            Day = row1["Day"]
            CodedDay = row1["CodedDay"]
            Zone = row1["Zone"]
            Weather = row1["Weather"]
            Temperature = row1["Temperature"]
            Traffic = row1["Traffic"]


            storetrafficdata.objects.create(Date=Date, Day=Day, CodedDay=CodedDay,
                                          Zone=Zone, Weather=Weather, Temperature=Temperature,
                                          Traffic=Traffic)

        print("Name is ", csvfile)
        return HttpResponse('CSV file successful uploaded')
    else:
'''
    return render(request, 'admin1/storecsvdata.html', {})


def storecsvdata1(request):
    if request.method == 'POST':
        # name = request.POST.get('name')
        csvfile = TextIOWrapper(request.FILES['file'])
        # columns = defaultdict(list)
        storecsvdata = csv.DictReader(csvfile)

        for row1 in storecsvdata:
            longitude = row1["longitude"]
            latitude = row1["latitude"]
            housing_median_age = row1["housing_median_age"]
            total_rooms = row1["total_rooms"]
            total_bedrooms = row1["total_bedrooms"]
            population = row1["population"]
            households = row1["households"]
            median_income = row1["median_income"]
            median_house_value = row1["median_house_value"]
            ocean_proximity = row1["ocean_proximity"]

            csvdatamodel.objects.create(longitude=longitude, latitude=latitude, housing_median_age=housing_median_age,
                                        total_rooms=total_rooms, total_bedrooms=total_bedrooms, population=population,households=households,median_income=median_income,median_house_value=median_house_value,ocean_proximity=ocean_proximity)

        print("Name is ", csvfile)
        return HttpResponse('CSV file successful uploaded')


    else:
        return render(request, 'admin/storecsvdata.html', {})



def lr1(request):
    import numpy as np
    import pandas as pd
    import  pyforest

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

    # model evaluation

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pred = lr.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print("r2 score:", r2)
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(lr, x_train, y_train, scoring="neg_mean_squared_error", cv=7)
    rmse_Score = np.sqrt(-scores)
    print("rmse_score:", rmse_Score)
    rmse_Score.mean()
    return render(request,'admin/lr.html',{"r2score":r2,"rmsescore":rmse_Score,"mean_absolute_error":mse,"mean_squared_error":mae})














