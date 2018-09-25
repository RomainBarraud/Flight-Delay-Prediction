import numpy as np
import pandas as pd
import sklearn
import flask
import json
from math import sin, cos, sqrt, atan2, radians
import pickle
import html5lib
import codecs

right_columns = ['flight_id','flight_no','Week','Departure','Arrival','Airline','std_hour','flight_date']
right_formats = [np.int64,'object',np.int64,'object','object','object',np.int64,'object']

airports_list = ['KIX', 'TNN', 'MNL', 'SIN', 'PEK', 'IST', 'HGH', 'KUL', 'BLR',
       'MXP', 'SYD', 'TPE', 'MEL', 'SGN', 'NRT', 'CNX', 'AKL', 'BKK',
       'HND', 'PVG', 'LHR', 'CTU', 'NGB', 'HAK', 'SHA', 'ICN', 'NGO',
       'DXB', 'LAX', 'HKT', 'HAN', 'KHN', 'CSX', 'DPS', 'NKG', 'CGK',
       'BKI', 'KHH', 'CEB', 'RMQ', 'JNB', 'BWN', 'OKA', 'PEN', 'USM',
       'PUS', 'CCU', 'DEL', 'WUH', 'FUK', 'MAA', 'CDG', 'CMB', 'JJN',
       'TAO', 'CKG', 'MAN', 'CJU', 'RGN', 'BNE', 'JFK', 'EWR', 'FRA',
       'HEL', 'CNS', 'SFO', 'XMN', 'ILO', 'ORD', 'ZRH', 'YVR', 'ADL',
       'BOM', 'KWE', 'PER', 'MLE', 'GUM', 'TSN', 'VVO', 'DOH', 'AUH',
       'KWL', 'AMS', 'TLV', 'YNZ', 'WUX', 'CTS', 'DMK', 'KMG', 'KOJ',
       'PNH', 'SYX', 'YYZ', 'YNT', 'CAN', 'BOS', 'CRK', 'SUB', 'WNZ',
       'FOC', 'DAC', 'CGO', 'ULN', 'AMM', 'REP', 'DFW', 'XIY', 'SVO',
       'ADD', 'DAD', 'TNA', 'SJW', 'KBV', 'NAN', 'NNG', 'DLC', 'KTM',
       'MUC', 'ZHA', 'FCO', 'RUH', 'MRU', 'YIW', 'HRB', 'SWA', 'SHE',
       'DUS', 'POM', 'LJG', 'OKJ', 'HFE', 'SEA', 'CXR', 'KLO', 'HIJ',
       'SPN', 'LYA', 'MXZ', 'WUS', 'XNN', 'HET', 'XUZ', 'OVB', 'ARN',
       'CGQ', 'LHW', 'IKT', 'KMJ', 'KCH', 'ALA', 'BAH', 'KMI', 'MAD',
       'NBO', 'ROR', 'YTY', 'DTW', 'ZYI', 'OOL', 'TAK', 'JHG', 'LGA']

airlines_list = ['UO', 'CX', 'NZ', 'AI', 'NH', 'GK', 'JL', 'QR', 'SA',
                 'MM', 'HX', 'LD', '9W', 'CI', 'PR', 'KA', '5J', 'Z2',
                 'DG', 'AY', 'TZ', 'SQ', 'AA', 'ET', 'TR', '3K', 'AC',
                 'S7', 'VA', 'UA', 'US', 'B6', 'TV', 'TT', 'HU', 'CA',
                 'CZ', 'BI', 'B5', 'TK', 'MU', 'FM', '9C', 'Y8', 'MH',
                 'WY', 'BA', 'EY', 'AK', 'MK', 'OD', 'OM', '7C', 'QF',
                 'VS', 'AF', 'BR', 'TG', 'HM', 'VN', 'LY', 'SV', 'DL',
                 'JW', 'NQ', 'FD', 'E8', 'LA', 'LH', 'PG', 'UL', 'MJ',
                 'O8', 'KQ', 'RJ', 'OX', 'EK', 'MD', 'OZ', 'PK', 'HO',
                 'JD', '3U', 'BL', 'VQ', 'KE', 'ZE', 'LJ', 'TP', '3V',
                 'SO', 'GA', 'SU', 'RI', 'JT', 'QG', 'AE', '2P', 'BX',
                 'KL', 'ZH', 'MF', 'UB', 'O3', 'LX', 'LV', 'HB', 'XF',
                 'HZ', 'AB', 'HG', 'JU', 'PQ', 'BG', 'FJ', 'RA', 'BO',
                 'PX', 'SK', 'KC', 'ED', 'P7']

hour_day = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

week_year = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52]

final_model_features = ['Airline_2P', 'Airline_3K', 'Airline_3U', 'Airline_3V',
       'Airline_5J', 'Airline_7C', 'Airline_9C', 'Airline_9W',
       'Airline_AA', 'Airline_AB', 'Airline_AC', 'Airline_AE',
       'Airline_AF', 'Airline_AI', 'Airline_AK', 'Airline_AY',
       'Airline_B5', 'Airline_B6', 'Airline_BA', 'Airline_BG',
       'Airline_BI', 'Airline_BL', 'Airline_BO', 'Airline_BR',
       'Airline_BX', 'Airline_CA', 'Airline_CI', 'Airline_CX',
       'Airline_CZ', 'Airline_DG', 'Airline_DL', 'Airline_E8',
       'Airline_ED', 'Airline_EK', 'Airline_ET', 'Airline_EY',
       'Airline_FD', 'Airline_FJ', 'Airline_FM', 'Airline_GA',
       'Airline_GK', 'Airline_HB', 'Airline_HG', 'Airline_HM',
       'Airline_HO', 'Airline_HU', 'Airline_HX', 'Airline_HZ',
       'Airline_JD', 'Airline_JL', 'Airline_JT', 'Airline_JU',
       'Airline_JW', 'Airline_KA', 'Airline_KC', 'Airline_KE',
       'Airline_KL', 'Airline_KQ', 'Airline_LA', 'Airline_LD',
       'Airline_LH', 'Airline_LJ', 'Airline_LV', 'Airline_LX',
       'Airline_LY', 'Airline_MD', 'Airline_MF', 'Airline_MH',
       'Airline_MJ', 'Airline_MK', 'Airline_MM', 'Airline_MU',
       'Airline_NH', 'Airline_NQ', 'Airline_NZ', 'Airline_O3',
       'Airline_O8', 'Airline_OD', 'Airline_OM', 'Airline_OX',
       'Airline_OZ', 'Airline_P7', 'Airline_PG', 'Airline_PK',
       'Airline_PQ', 'Airline_PR', 'Airline_PX', 'Airline_QF',
       'Airline_QG', 'Airline_QR', 'Airline_RA', 'Airline_RI',
       'Airline_RJ', 'Airline_S7', 'Airline_SA', 'Airline_SK',
       'Airline_SO', 'Airline_SQ', 'Airline_SU', 'Airline_SV',
       'Airline_TG', 'Airline_TK', 'Airline_TP', 'Airline_TR',
       'Airline_TT', 'Airline_TV', 'Airline_TZ', 'Airline_UA',
       'Airline_UB', 'Airline_UL', 'Airline_UO', 'Airline_US',
       'Airline_VA', 'Airline_VN', 'Airline_VQ', 'Airline_VS',
       'Airline_WY', 'Airline_XF', 'Airline_Y8', 'Airline_Z2',
       'Airline_ZE', 'Airline_ZH', 'weekday_0', 'weekday_1', 'weekday_2',
       'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'Week_1',
       'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6', 'Week_7',
       'Week_8', 'Week_9', 'Week_10', 'Week_11', 'Week_12', 'Week_13',
       'Week_14', 'Week_15', 'Week_16', 'Week_17', 'Week_18', 'Week_19',
       'Week_20', 'Week_21', 'Week_22', 'Week_23', 'Week_24', 'Week_25',
       'Week_26', 'Week_27', 'Week_28', 'Week_29', 'Week_30', 'Week_31',
       'Week_32', 'Week_33', 'Week_34', 'Week_35', 'Week_36', 'Week_37',
       'Week_38', 'Week_39', 'Week_40', 'Week_41', 'Week_42', 'Week_43',
       'Week_44', 'Week_45', 'Week_46', 'Week_47', 'Week_48', 'Week_49',
       'Week_50', 'Week_51', 'Week_52', 'std_hour_0', 'std_hour_1',
       'std_hour_2', 'std_hour_3', 'std_hour_4', 'std_hour_5',
       'std_hour_6', 'std_hour_7', 'std_hour_8', 'std_hour_9',
       'std_hour_10', 'std_hour_11', 'std_hour_12', 'std_hour_13',
       'std_hour_14', 'std_hour_15', 'std_hour_16', 'std_hour_17',
       'std_hour_18', 'std_hour_19', 'std_hour_20', 'std_hour_21',
       'std_hour_22', 'std_hour_23', 'period_afternoon', 'period_evening',
       'period_morning', 'period_night', 'country_Australia',
       'country_Bahrain', 'country_Bangladesh', 'country_Brunei',
       'country_Burma', 'country_Cambodia', 'country_Canada',
       'country_China', 'country_Ethiopia', 'country_Fiji',
       'country_Finland', 'country_France', 'country_Germany',
       'country_Guam', 'country_India', 'country_Indonesia',
       'country_Israel', 'country_Italy', 'country_Japan',
       'country_Jordan', 'country_Kazakhstan', 'country_Kenya',
       'country_Malaysia', 'country_Maldives', 'country_Mauritius',
       'country_Mongolia', 'country_Nepal', 'country_Netherlands',
       'country_New Zealand', 'country_Northern Mariana Islands',
       'country_Palau', 'country_Papua New Guinea', 'country_Philippines',
       'country_Qatar', 'country_Russia', 'country_Saudi Arabia',
       'country_Singapore', 'country_South Africa', 'country_South Korea',
       'country_Spain', 'country_Sri Lanka', 'country_Sweden',
       'country_Switzerland', 'country_Taiwan', 'country_Thailand',
       'country_Turkey', 'country_United Arab Emirates',
       'country_United Kingdom', 'country_United States',
       'country_Vietnam', 'not_worked_0', 'not_worked_1',
       'distance_group_1:<1000', 'distance_group_2:1000-1599',
       'distance_group_3:1000-2299', 'distance_group_4:2300-3499',
       'distance_group_5:>= 3500', 'arrival_latitude','arrival_longitude','distance','signal']

# external to data to add Hong Kong bank holidays
HK_bank_holidays = ['1-1','3-30','3-31','4-1','4-5','5-1','5-22','6-18','7-1','9-25','10-1','10-17','12-25','12-26']

def bank_holiday(date):
    """helper function to determine whether a date falls on a bank holiday"""
    if (date in HK_bank_holidays):
        return 1
    else:
        return 0

def not_worked(day):
    """helper function to regularize 'not worked to 0/1 values"""
    if day > 0:
        return 1
    else:
        return 0

def period_day(hour):
    """Bucketize the day into 4 periods."""
    if (hour >= 0 and hour < 6):
        return "night"
    elif (hour >= 6 and hour < 12):
        return "morning"
    elif (hour >= 12 and hour < 18):
        return "afternoon"
    else:
        return "evening"

def week_end(x):
    """create one categorical field weekend"""
    day = x.weekday()
    if (day < 5):
        return 0
    else:
        return 1

def airports_maker():
    """prepare airports DataFrame"""
    airports_col_names = ['name','country','airport_code','latitude','longitude']
    with codecs.open('./external_data/airports.csv', "r", encoding='utf-8', errors='ignore') as flights_source:
        airports = pd.read_csv(flights_source, encoding = 'utf-8', header=None)
        airports.drop(columns = [0,1,5,8,9,10,11,12,13], inplace = True)
    airports.columns = airports_col_names
    return airports

def lat_long_distance(dep_lat, dep_long, arr_lat, arr_long):
    """function to calculte the distance between 2 points on Earth based on latitude and longitude"""
    R = 6371.0 # Earth radius in km
    dep_lat = radians(dep_lat)
    dep_long = radians(dep_long)
    arr_lat = radians(arr_lat)
    arr_long = radians(arr_long)
    longitude = arr_long - dep_long
    latitude = arr_lat - dep_lat
    a = sin(latitude / 2)**2 + cos(dep_lat) * cos(arr_lat) * sin(longitude / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def distance_group_maker(distance):
    """bucketize distance"""
    if (distance < 1100):
        return "1:<1000"
    elif (1100 <= distance < 1600):
        return "2:1000-1599"
    elif (1600 <= distance < 2300):
        return "3:1000-2299"
    elif (2300 <= distance < 3500):
        return "4:2300-3499"
    else:
        return "5:>= 3500"

def typhoon_maker():
    """prepare typhoon DataFrame"""
    return pd.read_csv('./external_data/typhoon50.csv')

# checker function on input file
def file_checker(name_file):
    """succession of helper functions to check the file is in the right format"""

    # check that the file is a .csv and
    name_file = str(name_file)
    if (name_file[-4:] != '.csv'):
        print("The file is not a '.csv'")
        return 0
    else:
        print("The file is a '.csv'")

    # load the csv into a pandas DataFrame
    try:
        future_flights = pd.read_csv(name_file)
        print("The dataset is loaded")
    except:
        print("We could not load the file. Please check it.")
        return 0

    # check columns
    if len(future_flights.columns) != len(right_columns):
        print("The dataset does not have the same number of columns. Please check")
        return 0
    for i in range(len(right_columns)):
        if future_flights.columns[i] != right_columns[i]:
            print("The dataset does not have the same column names. Please check their names and order")
            print("We have detected the columns " + future_flights.columns.tolist())
            print("The columns should be " + right_columns)
            return 0
    print("The dataset has the right columns")

    # check format
    for i in range(len(right_formats)):
        if future_flights[right_columns[i]].dtype != right_formats[i]:
            print("The dataset have type inconsistency")
            return 0
    print("The dataset has the right types in each column")

    # check number of lines
    print("The dataset has " + str(future_flights.shape[0]) + " records")

    # check null value for Airline column
    if not(future_flights['Airline'].isin(airlines_list).all()):
        airline_null = np.invert(future_flights['Airline'].isin(airlines_list)).sum()
        future_flights['Airline'].replace('NULL', 'CX', inplace = True)
        future_flights['Airline'].fillna('CX', inplace = True)
        print(str(airline_null) + " incorrect values have been detected in Airline. OK to continue.")

    # check null values
    remaining_null = future_flights.isnull().sum().sum()
    if remaining_null != 0:
        print("The dataset has " + str(future_flights.isnull().sum().sum()) + " null value(s). Please review the data.")
        return 0
    else:
        print("The dataset has no blocking null value.")

    # specific column checker
    if (future_flights['Departure'] != 'HKG').any():
        print('There are Departures which are not Hong Kong. Please check.')
        return 0
    if not(future_flights['std_hour'].isin(hour_day).all()):
        print('There are standard hours out of the range [0-24]. Please check.')
        return 0
    if not(future_flights['Week'].isin(week_year).all()):
        print('There are weeks out of the range [1-52]. Please check.')
        return 0
    if not(future_flights['Arrival'].isin(airports_list).all()):
        print('Some arrivals are unknown. Please check.')
        return 0
    if not(future_flights['Airline'].isin(airlines_list).all()):
        print('Some airlines are unknown. Please check.')
        return 0

    try:
        pd.to_datetime(flights['date'])
        print("Data checked completed...")
    except:
        print("Date column cannot be read. Please check.")

    print("...the dataset is valid!")
    return 1

def transform_dataset(name_file):
    """preprocess the dataset to be able to apply prediction model"""
    flight_dataset = pd.read_csv(name_file)

    flight_dataset['flight_no'] = flight_dataset['flight_no'].astype('category')
    flight_dataset['Week'] = flight_dataset['Week'].astype('category')
    flight_dataset['Arrival'] = flight_dataset['Arrival'].astype('category')
    flight_dataset['Airline'] = flight_dataset['Airline'].astype('category')
    flight_dataset['std_hour'] = flight_dataset['std_hour'].astype('int8')

    airports = airports_maker()
    flight_dataset['departure_latitude'] = airports[airports['airport_code'] == 'HKG']['latitude'].values[0]
    flight_dataset['departure_longitude'] = airports[airports['airport_code'] == 'HKG']['longitude'].values[0]

    flight_dataset = flight_dataset.merge(airports, left_on='Arrival', right_on='airport_code', how = 'inner')
    flight_dataset.rename(columns={"latitude": "arrival_latitude", "longitude": "arrival_longitude"}, inplace = True)

    airports_list_2 = np.append(airports_list, 'HKG').tolist()
    airports_dict = {}
    dep_lat = airports[airports['airport_code'] == 'HKG']['latitude'].values[0] # origin is always Hong Kong latitude
    dep_long = airports[airports['airport_code'] == 'HKG']['longitude'].values[0] # origin is always Hong Kong longitude
    for airport in airports_list_2:
        arr_lat = airports[airports['airport_code'] == airport]['latitude'].values[0] # destination latitude
        arr_long = airports[airports['airport_code'] == airport]['longitude'].values[0] # destination longitude
        airports_dict[airport] = lat_long_distance(dep_lat, dep_long, arr_lat, arr_long) # distance to Hong Kong

    flight_dataset['distance'] = flight_dataset['Arrival'].map(airports_dict)
    flight_dataset['distance_group'] = flight_dataset['distance'].apply(distance_group_maker)
    flight_dataset['distance_group'] = flight_dataset['distance_group'].astype('category')

    flight_dataset['period'] = flight_dataset['std_hour'].apply(period_day)
    flight_dataset['period'] = flight_dataset['period'].astype('category')
    flight_dataset['date'] = pd.to_datetime(flight_dataset['flight_date'])
    flight_dataset['weekday'] = flight_dataset['date'].dt.dayofweek
    flight_dataset['weekday'] = flight_dataset['weekday'].astype('category')
    flight_dataset['month'] = flight_dataset['date'].dt.month
    flight_dataset['month'] = flight_dataset['month'].astype('category')
    flight_dataset['day'] = flight_dataset['date'].dt.day
    flight_dataset['day'] = flight_dataset['day'].astype('category')
    flight_dataset['weekend'] = flight_dataset['date'].apply(week_end)

    flight_dataset['date_key'] = flight_dataset['month'].astype(str) + "-" + flight_dataset['day'].astype(str)
    flight_dataset['not_worked'] = flight_dataset['date_key'].map(bank_holiday) + flight_dataset['weekend']
    flight_dataset['weekend'] = flight_dataset['weekend'].astype('category')
    flight_dataset['not_worked'] = flight_dataset['not_worked'].map(not_worked)

    flight_dataset['key'] = flight_dataset['month'].astype(str) + '-' + flight_dataset['day'].astype(str)
    typhoon50 = typhoon_maker()
    flight_dataset = flight_dataset.merge(typhoon50, how='left', left_on ='key', right_on = 'key')
    flight_dataset['signal'].fillna(0, inplace = True)
    flight_dataset.rename(columns={"key": "month-day"}, inplace = True)

    final_features = ['Airline','weekday', 'Week','std_hour','period', 'country', 'arrival_latitude',
            'arrival_longitude','distance', 'distance_group', 'not_worked', 'signal']

    flight_dataset = flight_dataset[final_features]

    for field in ['Airline','weekday','Week','std_hour','period','country','distance_group', 'not_worked']:
        flight_dataset[field] = flight_dataset[field].astype('category')
    categories = ['Airline', 'weekday', 'Week', 'std_hour', 'period', 'country', 'not_worked', 'distance_group']
    non_categories = ['arrival_latitude','arrival_longitude','distance','signal']

    flights_cat = pd.get_dummies(flight_dataset[categories])
    flights_non_cat = flight_dataset[non_categories]
    flight_dataset = pd.concat([flights_cat, flights_non_cat], axis = 1)

    # Add the dummy variables that we created in scikit-learn and that may be missing from the new dataset
    missing_features = set(final_model_features) - set(flight_dataset.columns)
    for feature in missing_features:
        flight_dataset[feature] = 0 # create new features and set them to 0
        flight_dataset[feature] = flight_dataset[feature].astype('category')
    flight_dataset = flight_dataset[final_model_features] # reorder the columns
    print("")
    print("Processing budget...")
    return flight_dataset

def predict_dataset(name_file, processed_dataset):
    """calculate budget to allocate to each flight and return the value in a new column of the dataframe"""
    flight_dataset = pd.read_csv(name_file)
    model = pickle.load(open("./external_data/model.pkl","rb")) # import model
    flight_dataset['budget'] = pd.DataFrame(model.predict_proba(processed_dataset))[0].map(budget_mapper)
    return flight_dataset

def budget_mapper(prob):
    """associate budget to prob_not_claimed probability"""
    budget_df = pd.read_csv("./external_data/budget_df.csv", index_col=0)
    prob = round(prob,2)
    amount = budget_df['budget'][prob]
    return amount

def flight_predictor(name_file):
    output_name = name_file[:-4] + "_output.csv"
    flight_dataset = file_checker(name_file)
    if flight_dataset:
        flight_dataset_output_csv = predict_dataset(name_file, transform_dataset(name_file)).to_csv(output_name, index = False)
        print("")
        print("... budget ready!")
        print("A csv file named output.csv with the budget figures is now in your folder.")
        print("Have a nice flight!")
        return output_name
    else:
        print("An error occured")
        return 0
