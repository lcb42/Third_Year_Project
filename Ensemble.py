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
        lin_reg = LinearRegression()
        lin_reg.fit(kf_X_train, kf_y_train)

        # Train Random Forest Regression
        rf_reg = RandomForestRegressor(max_depth=10, max_features=0.6, n_estimators=20, bootstrap=False, n_jobs=-1,
                                    random_state=0)
        rf_reg.fit(kf_X_train, kf_y_train)

        # Put results of both into a pdframe
        # train pdframe
        df_combo = pd.DataFrame()
        df_combo['linear'] = lin_reg.predict(kf_X_train)
        df_combo['random_forest'] = lin_reg.predict(kf_X_train)
        df_combo['real'] = kf_y_train

        # Linear Regress on that pdframe, with actual label as target
        lin_reg_2 = LinearRegression()

        lin_reg_2.fit(df_combo[0:1], df_combo[2])

        train_predict = lin_reg_2.predict(kf_X_train)
        train_rmse = np.sqrt(((kf_y_train - train_predict) ** 2).mean())
        train_rmse_agg.append(train_rmse)

        test_predict = rf_reg.predict(kf_X_test)
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