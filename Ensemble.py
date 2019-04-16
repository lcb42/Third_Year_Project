from numpy.core.multiarray import ndarray
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import datetime as dt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")


# functions required
def replace_space(element, char):
    if element != '':
        return element
    else:
        return char


def check_status(mins, status):
    if status == "LATE":
        return -mins
    else:
        return mins


def get_operator(operator):

    switcher = {
        '23': "Arriva Trains Northern",
        '71': "Transport for Wales",
        '79': "c2c",
        '35': "Caledonian Sleeper",
        '74': "Chiltern Railway",
        '27': "CrossCountry",
        '34': "Devon and Cornwall Railways",
        '28': "East Midlands Trains",
        '06': "Eurostar",
        '26': "First Capital Connect (defunct)",
        '55': "First Hull Trains",
        '81': "Gatwick Express",
        '54': "GB Railfreight",
        '88': "Govia Thameslink Railway / Southern",
        '22': "Grand Central",
        '25': "Great Western Railway",
        '21': "Greater Anglia",
        '24': "Heathrow Connect",
        '86': "Heathrow Express",
        '85': "Island Lines",
        '29': "London Midlands",
        '30': "London Overground",
        '64': "Merseyrail",
        '00': "Network Rail",
        '56': "Nexus",
        '51': "North Yorkshire Moors Railway",
        '60': "ScotRail",
        '84': "South Western Railway",
        '19': "South Yorkshire Supertram",
        '80': "Southeastern",
        '33': "TFL Rail",
        '20': "TransPennine Express",
        '65': "Virgin Trains",
        '61': "Virgin Trains East Coast",
        '50': "West Coast Railway Co."
    }

    return switcher.get(operator, "Invalid Station")


def get_stanox(stanox):

    switcher = {
        '87031': "AHT",
        '87021': "AON",
        '86074': "ADV",
        '87763': "AHS",
        '86066': "BSK",
        '86896': "BEU",
        '86339': "BDH",
        '87024': "BTY",
        '86216': "BTE",
        '87009': "BAW",
        '86202': "BOE",
        '86921': "BMH",
        '86061': "BMY",
        '86901': "BCU",
        '86223': "BUO",
        '86108': "CFR",
        '86915': "CHR",
        '86301': "CSA",
        '86112': "DEN",
        '86107': "DBG",
        '86087': "ESL",
        '86343': "EMS",
        '86241': "FRM",
        '86042': "FNB",
        '87010': "FNN",
        '87026': "FNH",
        '86045': "FLE",
        '86321': "FTN",
        '86077': "GRT",
        '86222': "HME",
        '87062': "HSL",
        '86341': "HAV",
        '86201': "HDE",
        '86332': "HLS",
        '86913': "HNA",
        '86049': "HOK",
        '87064': "LIP",
        '87065': "LIS",
        '86908': "LYP",
        '86907': "LYT",
        '86081': "MIC",
        '86527': "MBK",
        '86219': "NTL",
        '86911': "NWM",
        '87012': "NCM",
        '86070': "OVR",
        '87066': "PTR",
        '86917': "POK",
        '86248': "PTC",
        '86313': "PMS",
        '86311': "PMH",
        '86703': "RDB",
        '86101': "ROM",
        '87067': "RLN",
        '86122': "SAL",
        '86084': "SHW",
        '86218': "SHO",
        '86495': "SOA",
        '86520': "SOU",
        '86499': "SDN",
        '86225': "SNW",
        '86909': "SWY",
        '86497': "SWG",
        '86711': "TTN",
        '86342': "WBL",
        '86071': "WCH",
        '86083': "WIN",
        '86047': "WNF",
        '86215': "WLS"
    }

    return switcher.get(stanox, "Invalid Stanox")


# Import data
client = MongoClient()

dblist = client.list_database_names()

db = client["Third_Year_Project"]

real_time_data = db["Real_Time_Data_HANTS_2_2weeks"]

# Import Data

json_data = real_time_data.find({"header.msg_type": '0003'})
json_change_orig_data = real_time_data.find({"header.msg_type": '0006'})
json_cancel_data = real_time_data.find({"header.msg_type": "0002"})


data = []
change_orig_data = []
cancel_data = []

for x in json_change_orig_data:
    change_orig_data.append(list(x['body'].values()))

for x in json_cancel_data:
    cancel_data.append(list(x['body'].values()))


# Change data to neg data as in characterisation

neg_data = []
label_data = []

print("created array")

for x in json_data:
    if x['body']['planned_timestamp'] != "":
        neg_data.append([
            x['body']['event_type'],                       # event_type
            x['body']['planned_timestamp'],                       # planned timestamp
            # x[3],                       # Current train id
            # x[4],                       # likely mins until next stanox report WONT HAVE IN REAL INPUT DATA
            # x[5],                       # timestamp of arrival/departure WONT HAVE IN REAL INPUT DATA
            # x[6],                       # False if not a correction of a previous report, True if it is WONT HAVE IN REAL
            # x[7],                       # Platform number WONT HAVE IN REAL INPUT DATA
            # x['body']['train_id'],                       # train_id
            # x[9],                       # ONTIME/EARLY/LATE/OFFROUTE WONT HAVE IN REAL INPUT DATA
            x['body']['train_service_code'],                      # train_service_code
            # x['body']['toc_id'],                      # toc_id
            x['body']['loc_stanox']                      # station stanox (location currently)
            # x['body']['next_report_stanox'],                      # the next station
        ])
        label_data.append(
            int(check_status(int(x['body']['timetable_variation']), x['body']['variation_status']))
            # Minutes variation from timetabled time LABEL TO FIND
        )

# Label/data
X = np.array(neg_data)
print(np.shape(X))
y = label_data
print(np.shape(y))


# ********     Pre processing     ***********

df = pd.DataFrame(data=X, columns=["event_type", "planned_timestamp", "train_service_code", "loc_stanox"])

print(df.head(5))
# Convert timestamp

years = []
months = []
days = []
times = []
weekdays = []

for x in range(0, len(df)):
    temp_time = int(int(df['planned_timestamp'][x]) / 1000)
    readable = dt.datetime.fromtimestamp(temp_time).isoformat()

    # years.append(int(readable.split("-")[0]))
    # months.append(int(readable.split("-")[1]))
    daytime = readable.split("-")[2]
    days.append(daytime.split("T")[0])

    hh = (daytime.split("T")[1]).split(":")[0]
    mm = (daytime.split("T")[1]).split(":")[1]
    ss = (daytime.split("T")[1]).split(":")[2]
    times.append(int((dt.timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss))).total_seconds()))

    weekdays.append(dt.datetime.fromtimestamp(temp_time).weekday())

# df['year'] = pd.Series(np.array(years)).values
# df['month'] = pd.Series(np.array(months)).values
df['day'] = pd.Series(np.array(days)).values
df['time'] = pd.Series(np.array(times)).values
df['weekday'] = pd.Series(np.array(weekdays)).values



# One Hot Encoding using pandas
cat_cols = ['event_type', "train_service_code", "loc_stanox"]
df_processed = pd.get_dummies(df, prefix_sep="__", columns=cat_cols)


from sklearn.model_selection import train_test_split
X_train_test, X_valid, y_train_test, y_valid = train_test_split(df_processed, y, test_size=0.2)

X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2)

# K Fold Cross Validation
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

splits = 10

kf = KFold(n_splits=splits)

kf_X_train_test = np.array(X_train_test)
kf_y_train_test = np.array(y_train_test)

test_rmses = []
train_rmses = []

ns = [1]

for n in ns:

    print(n)

    train_rmse_agg = []
    test_rmse_agg = []

    for train_index, test_index in kf.split(kf_X_train_test):

        kf_X_train, kf_X_test = kf_X_train_test[train_index], kf_X_train_test[test_index]
        kf_y_train, kf_y_test = kf_y_train_test[train_index], kf_y_train_test[test_index]

        # Train Linear Regression
        lin_reg_1 = LinearRegression()
        lin_reg_1.fit(kf_X_train, kf_y_train)

        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(kf_X_train, kf_y_train)

        lin_reg_3 = LinearRegression()
        lin_reg_3.fit(kf_X_train, kf_y_train)

        lin_reg_4 = LinearRegression()
        lin_reg_4.fit(kf_X_train, kf_y_train)

        lin_reg_5 = LinearRegression()
        lin_reg_5.fit(kf_X_train, kf_y_train)

        lin_reg_6 = LinearRegression()
        lin_reg_6.fit(kf_X_train, kf_y_train)

        lin_reg_7 = LinearRegression()
        lin_reg_7.fit(kf_X_train, kf_y_train)

        lin_reg_8 = LinearRegression()
        lin_reg_8.fit(kf_X_train, kf_y_train)

        lin_reg_9 = LinearRegression()
        lin_reg_9.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_10 = LinearRegression()
        # lin_reg_10.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_11 = LinearRegression()
        # lin_reg_11.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_12 = LinearRegression()
        # lin_reg_12.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_13 = LinearRegression()
        # lin_reg_13.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_14 = LinearRegression()
        # lin_reg_14.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_15 = LinearRegression()
        # lin_reg_15.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_16 = LinearRegression()
        # lin_reg_16.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_17 = LinearRegression()
        # lin_reg_17.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_18 = LinearRegression()
        # lin_reg_18.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_19 = LinearRegression()
        # lin_reg_19.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_20 = LinearRegression()
        # lin_reg_20.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_21 = LinearRegression()
        # lin_reg_21.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_22 = LinearRegression()
        # lin_reg_22.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_23 = LinearRegression()
        # lin_reg_23.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_24 = LinearRegression()
        # lin_reg_24.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_25 = LinearRegression()
        # lin_reg_25.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_26 = LinearRegression()
        # lin_reg_26.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_27 = LinearRegression()
        # lin_reg_27.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_28 = LinearRegression()
        # lin_reg_28.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_29 = LinearRegression()
        # lin_reg_29.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_30 = LinearRegression()
        # lin_reg_30.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_31 = LinearRegression()
        # lin_reg_31.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_32 = LinearRegression()
        # lin_reg_32.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_33 = LinearRegression()
        # lin_reg_33.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_34 = LinearRegression()
        # lin_reg_34.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_35 = LinearRegression()
        # lin_reg_35.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_36 = LinearRegression()
        # lin_reg_36.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_37 = LinearRegression()
        # lin_reg_37.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_38 = LinearRegression()
        # lin_reg_38.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_39 = LinearRegression()
        # lin_reg_39.fit(kf_X_train, kf_y_train)
        #
        # lin_reg_40 = LinearRegression()
        # lin_reg_40.fit(kf_X_train, kf_y_train)

        # Train Random Forest Regression
        rf_reg_1 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_1.fit(kf_X_train, kf_y_train)

        rf_reg_2 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_2.fit(kf_X_train, kf_y_train)

        rf_reg_3 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_3.fit(kf_X_train, kf_y_train)

        rf_reg_4 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_4.fit(kf_X_train, kf_y_train)

        rf_reg_5 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_5.fit(kf_X_train, kf_y_train)

        rf_reg_6 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_6.fit(kf_X_train, kf_y_train)

        rf_reg_7 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_7.fit(kf_X_train, kf_y_train)

        rf_reg_8 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_8.fit(kf_X_train, kf_y_train)

        rf_reg_9 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        rf_reg_9.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_10 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1, random_state=0)
        # rf_reg_10.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_11 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_11.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_12 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_12.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_13 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_13.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_14 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_14.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_15 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_15.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_16 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_16.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_17 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_17.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_18 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_18.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_19 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_19.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_20 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_20.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_21 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_21.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_22 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_22.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_23 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_23.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_24 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_24.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_25 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_25.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_26 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_26.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_27 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_27.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_28 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_28.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_29 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                  random_state=0)
        # rf_reg_29.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_30 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_30.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_31 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_31.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_32 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_32.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_33 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_33.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_34 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_34.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_35 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_35.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_36 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_36.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_37 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_37.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_38 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_38.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_39 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_39.fit(kf_X_train, kf_y_train)
        #
        # rf_reg_40 = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
        #                                   random_state=0)
        # rf_reg_40.fit(kf_X_train, kf_y_train)

        # Put results of both into a pdframe
        # train pdframe
        df_train = pd.DataFrame()
        df_train['linear_1'] = lin_reg_1.predict(kf_X_train)
        df_train['linear_2'] = lin_reg_2.predict(kf_X_train)
        df_train['linear_3'] = lin_reg_3.predict(kf_X_train)
        # df_train['linear_4'] = lin_reg_4.predict(kf_X_train)
        # df_train['linear_5'] = lin_reg_5.predict(kf_X_train)
        # df_train['linear_6'] = lin_reg_6.predict(kf_X_train)
        # df_train['linear_7'] = lin_reg_7.predict(kf_X_train)
        # df_train['linear_8'] = lin_reg_8.predict(kf_X_train)
        # df_train['linear_9'] = lin_reg_9.predict(kf_X_train)
        # df_train['linear_10'] = lin_reg_10.predict(kf_X_train)
        # df_train['linear_11'] = lin_reg_11.predict(kf_X_train)
        # df_train['linear_12'] = lin_reg_12.predict(kf_X_train)
        # df_train['linear_13'] = lin_reg_13.predict(kf_X_train)
        # df_train['linear_14'] = lin_reg_14.predict(kf_X_train)
        # df_train['linear_15'] = lin_reg_15.predict(kf_X_train)
        # df_train['linear_16'] = lin_reg_16.predict(kf_X_train)
        # df_train['linear_17'] = lin_reg_17.predict(kf_X_train)
        # df_train['linear_18'] = lin_reg_18.predict(kf_X_train)
        # df_train['linear_19'] = lin_reg_19.predict(kf_X_train)
        # df_train['linear_20'] = lin_reg_20.predict(kf_X_train)
        # df_train['linear_21'] = lin_reg_21.predict(kf_X_train)
        # df_train['linear_22'] = lin_reg_22.predict(kf_X_train)
        # df_train['linear_23'] = lin_reg_23.predict(kf_X_train)
        # df_train['linear_24'] = lin_reg_24.predict(kf_X_train)
        # df_train['linear_25'] = lin_reg_25.predict(kf_X_train)
        # df_train['linear_26'] = lin_reg_26.predict(kf_X_train)
        # df_train['linear_27'] = lin_reg_27.predict(kf_X_train)
        # df_train['linear_28'] = lin_reg_28.predict(kf_X_train)
        # df_train['linear_29'] = lin_reg_29.predict(kf_X_train)
        # df_train['linear_30'] = lin_reg_30.predict(kf_X_train)
        # df_train['linear_31'] = lin_reg_31.predict(kf_X_train)
        # df_train['linear_32'] = lin_reg_32.predict(kf_X_train)
        # df_train['linear_33'] = lin_reg_33.predict(kf_X_train)
        # df_train['linear_34'] = lin_reg_34.predict(kf_X_train)
        # df_train['linear_35'] = lin_reg_35.predict(kf_X_train)
        # df_train['linear_36'] = lin_reg_36.predict(kf_X_train)
        # df_train['linear_37'] = lin_reg_37.predict(kf_X_train)
        # df_train['linear_38'] = lin_reg_38.predict(kf_X_train)
        # df_train['linear_39'] = lin_reg_39.predict(kf_X_train)
        # df_train['linear_40'] = lin_reg_40.predict(kf_X_train)
        df_train['random_forest_1'] = rf_reg_1.predict(kf_X_train)
        df_train['random_forest_2'] = rf_reg_2.predict(kf_X_train)
        df_train['random_forest_3'] = rf_reg_3.predict(kf_X_train)
        df_train['random_forest_4'] = rf_reg_4.predict(kf_X_train)
        df_train['random_forest_5'] = rf_reg_5.predict(kf_X_train)
        df_train['random_forest_6'] = rf_reg_6.predict(kf_X_train)
        df_train['random_forest_7'] = rf_reg_7.predict(kf_X_train)
        # df_train['random_forest_8'] = rf_reg_8.predict(kf_X_train)
        # df_train['random_forest_9'] = rf_reg_9.predict(kf_X_train)
        # df_train['random_forest_10'] = rf_reg_10.predict(kf_X_train)
        # df_train['random_forest_11'] = rf_reg_11.predict(kf_X_train)
        # df_train['random_forest_12'] = rf_reg_12.predict(kf_X_train)
        # df_train['random_forest_13'] = rf_reg_13.predict(kf_X_train)
        # df_train['random_forest_14'] = rf_reg_14.predict(kf_X_train)
        # df_train['random_forest_15'] = rf_reg_15.predict(kf_X_train)
        # df_train['random_forest_16'] = rf_reg_16.predict(kf_X_train)
        # df_train['random_forest_17'] = rf_reg_17.predict(kf_X_train)
        # df_train['random_forest_18'] = rf_reg_18.predict(kf_X_train)
        # df_train['random_forest_19'] = rf_reg_19.predict(kf_X_train)
        # df_train['random_forest_20'] = rf_reg_20.predict(kf_X_train)
        # df_train['random_forest_21'] = rf_reg_21.predict(kf_X_train)
        # df_train['random_forest_22'] = rf_reg_22.predict(kf_X_train)
        # df_train['random_forest_23'] = rf_reg_23.predict(kf_X_train)
        # df_train['random_forest_24'] = rf_reg_24.predict(kf_X_train)
        # df_train['random_forest_25'] = rf_reg_25.predict(kf_X_train)
        # df_train['random_forest_26'] = rf_reg_26.predict(kf_X_train)
        # df_train['random_forest_27'] = rf_reg_27.predict(kf_X_train)
        # df_train['random_forest_28'] = rf_reg_28.predict(kf_X_train)
        # df_train['random_forest_29'] = rf_reg_29.predict(kf_X_train)
        # df_train['random_rf_reg_30'] = rf_reg_30.predict(kf_X_train)
        # df_train['random_rf_reg_31'] = rf_reg_31.predict(kf_X_train)
        # df_train['random_rf_reg_32'] = rf_reg_32.predict(kf_X_train)
        # df_train['random_rf_reg_33'] = rf_reg_33.predict(kf_X_train)
        # df_train['random_rf_reg_34'] = rf_reg_34.predict(kf_X_train)
        # df_train['random_rf_reg_35'] = rf_reg_35.predict(kf_X_train)
        # df_train['random_rf_reg_36'] = rf_reg_36.predict(kf_X_train)
        # df_train['random_rf_reg_37'] = rf_reg_37.predict(kf_X_train)
        # df_train['random_rf_reg_38'] = rf_reg_38.predict(kf_X_train)
        # df_train['random_rf_reg_39'] = rf_reg_39.predict(kf_X_train)
        # df_train['random_forest_40'] = rf_reg_40.predict(kf_X_train)

        df_test = pd.DataFrame()
        df_test['linear_1'] = lin_reg_1.predict(kf_X_test)
        df_test['linear_2'] = lin_reg_2.predict(kf_X_test)
        df_test['linear_3'] = lin_reg_3.predict(kf_X_test)
        # df_test['linear_4'] = lin_reg_4.predict(kf_X_test)
        # df_test['linear_5'] = lin_reg_5.predict(kf_X_test)
        # df_test['linear_6'] = lin_reg_6.predict(kf_X_test)
        # df_test['linear_7'] = lin_reg_7.predict(kf_X_test)
        # df_test['linear_8'] = lin_reg_8.predict(kf_X_test)
        # df_test['linear_9'] = lin_reg_9.predict(kf_X_test)
        # df_test['linear_10'] = lin_reg_10.predict(kf_X_test)
        # df_test['linear_11'] = lin_reg_11.predict(kf_X_test)
        # df_test['linear_12'] = lin_reg_12.predict(kf_X_test)
        # df_test['linear_13'] = lin_reg_13.predict(kf_X_test)
        # df_test['linear_14'] = lin_reg_14.predict(kf_X_test)
        # df_test['linear_15'] = lin_reg_15.predict(kf_X_test)
        # df_test['linear_16'] = lin_reg_16.predict(kf_X_test)
        # df_test['linear_17'] = lin_reg_17.predict(kf_X_test)
        # df_test['linear_18'] = lin_reg_18.predict(kf_X_test)
        # df_test['linear_19'] = lin_reg_19.predict(kf_X_test)
        # df_test['linear_20'] = lin_reg_20.predict(kf_X_test)
        # df_test['linear_21'] = lin_reg_21.predict(kf_X_test)
        # df_test['linear_22'] = lin_reg_22.predict(kf_X_test)
        # df_test['linear_23'] = lin_reg_23.predict(kf_X_test)
        # df_test['linear_24'] = lin_reg_24.predict(kf_X_test)
        # df_test['linear_25'] = lin_reg_25.predict(kf_X_test)
        # df_test['linear_26'] = lin_reg_26.predict(kf_X_test)
        # df_test['linear_27'] = lin_reg_27.predict(kf_X_test)
        # df_test['linear_28'] = lin_reg_28.predict(kf_X_test)
        # df_test['linear_29'] = lin_reg_29.predict(kf_X_test)
        # df_test['linear_30'] = lin_reg_30.predict(kf_X_test)
        # df_test['linear_31'] = lin_reg_31.predict(kf_X_test)
        # df_test['linear_32'] = lin_reg_32.predict(kf_X_test)
        # df_test['linear_33'] = lin_reg_33.predict(kf_X_test)
        # df_test['linear_34'] = lin_reg_34.predict(kf_X_test)
        # df_test['linear_35'] = lin_reg_35.predict(kf_X_test)
        # df_test['linear_36'] = lin_reg_36.predict(kf_X_test)
        # df_test['linear_37'] = lin_reg_37.predict(kf_X_test)
        # df_test['linear_38'] = lin_reg_38.predict(kf_X_test)
        # df_test['linear_39'] = lin_reg_39.predict(kf_X_test)
        # df_test['linear_40'] = lin_reg_40.predict(kf_X_test)
        df_test['random_forest_1'] = rf_reg_1.predict(kf_X_test)
        df_test['random_forest_2'] = rf_reg_2.predict(kf_X_test)
        df_test['random_forest_3'] = rf_reg_3.predict(kf_X_test)
        df_test['random_forest_4'] = rf_reg_4.predict(kf_X_test)
        df_test['random_forest_5'] = rf_reg_5.predict(kf_X_test)
        df_test['random_forest_6'] = rf_reg_6.predict(kf_X_test)
        df_test['random_forest_7'] = rf_reg_7.predict(kf_X_test)
        # df_test['random_forest_8'] = rf_reg_8.predict(kf_X_test)
        # df_test['random_forest_9'] = rf_reg_9.predict(kf_X_test)
        # df_test['random_forest_10'] = rf_reg_10.predict(kf_X_test)
        # df_test['random_forest_11'] = rf_reg_11.predict(kf_X_test)
        # df_test['random_forest_12'] = rf_reg_12.predict(kf_X_test)
        # df_test['random_forest_13'] = rf_reg_13.predict(kf_X_test)
        # df_test['random_forest_14'] = rf_reg_14.predict(kf_X_test)
        # df_test['random_forest_15'] = rf_reg_15.predict(kf_X_test)
        # df_test['random_forest_16'] = rf_reg_16.predict(kf_X_test)
        # df_test['random_forest_17'] = rf_reg_17.predict(kf_X_test)
        # df_test['random_forest_18'] = rf_reg_18.predict(kf_X_test)
        # df_test['random_forest_19'] = rf_reg_19.predict(kf_X_test)
        # df_test['random_forest_20'] = rf_reg_20.predict(kf_X_test)
        # df_test['random_forest_21'] = rf_reg_21.predict(kf_X_test)
        # df_test['random_forest_22'] = rf_reg_22.predict(kf_X_test)
        # df_test['random_forest_23'] = rf_reg_23.predict(kf_X_test)
        # df_test['random_forest_24'] = rf_reg_24.predict(kf_X_test)
        # df_test['random_forest_25'] = rf_reg_25.predict(kf_X_test)
        # df_test['random_forest_26'] = rf_reg_26.predict(kf_X_test)
        # df_test['random_forest_27'] = rf_reg_27.predict(kf_X_test)
        # df_test['random_forest_28'] = rf_reg_28.predict(kf_X_test)
        # df_test['random_forest_29'] = rf_reg_29.predict(kf_X_test)
        # df_test['random_forest_30'] = rf_reg_30.predict(kf_X_test)
        # df_test['random_forest_31'] = rf_reg_31.predict(kf_X_test)
        # df_test['random_forest_32'] = rf_reg_32.predict(kf_X_test)
        # df_test['random_forest_33'] = rf_reg_33.predict(kf_X_test)
        # df_test['random_forest_34'] = rf_reg_34.predict(kf_X_test)
        # df_test['random_forest_35'] = rf_reg_35.predict(kf_X_test)
        # df_test['random_forest_36'] = rf_reg_36.predict(kf_X_test)
        # df_test['random_forest_37'] = rf_reg_37.predict(kf_X_test)
        # df_test['random_forest_38'] = rf_reg_38.predict(kf_X_test)
        # df_test['random_forest_39'] = rf_reg_39.predict(kf_X_test)
        # df_test['random_forest_40'] = rf_reg_40.predict(kf_X_test)

        #print(df_combo.head(10))
        # print(np.shape(df_combo))
        # print(np.shape(kf_y_train))

        # Linear Regress on that pdframe, with actual label as target
        lin_reg_final = LinearRegression()

        lin_reg_final.fit(df_train, kf_y_train)

        train_predict = lin_reg_final.predict(df_train)
        train_rmse = np.sqrt(((kf_y_train - train_predict) ** 2).mean())
        train_rmse_agg.append(train_rmse)

        test_predict = lin_reg_final.predict(df_test)
        test_rmse = np.sqrt(((kf_y_test - test_predict) ** 2).mean())
        test_rmse_agg.append(test_rmse)

    # Calculate Mean
    train_rmses.append(np.mean(train_rmse_agg))
    test_rmses.append(np.mean(test_rmse_agg))
    print("Train RMSE: ", np.mean(train_rmse_agg))
    print("Test RMSE: ", np.mean(test_rmse_agg))


plt.plot(ns, test_rmses, "b*", label="Test RMSE")
plt.plot(ns, train_rmses, "r*", label="Train RMSE")
plt.plot(ns, test_rmses, c='b')
plt.plot(ns, train_rmses, c='r')
plt.ylabel("RMSE Score")
plt.xlabel("max_depth Hyperparameter Values")
plt.xscale("log")
plt.legend()
plt.show()