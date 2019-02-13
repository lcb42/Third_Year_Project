# Data Characterisation
# Work out what the data consists of, and understand which elements of the data are useful

from pymongo import MongoClient
from sklearn.feature_extraction import DictVectorizer
import collections

# Function to replace a space with a specified character
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


def get_station(station):

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

    return switcher.get(station, "Invalid Station")


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

client = MongoClient()

dblist = client.list_database_names()

db = client["Third_Year_Project"]

real_time_data = db["Real_Time_Data_HANTS_1"]

# Import Data

json_data = real_time_data.find({"header.msg_type": '0003'})
json_change_orig_data = real_time_data.find({"header.msg_type": '0006'})
json_cancel_data = real_time_data.find({"header.msg_type": "0002"})

data = []
change_orig_data = []
cancel_data = []


for x in json_data:
    data.append([
        x['body']['event_type'],
        x['body']['planned_timestamp'],
        int(x['body']['timetable_variation']),  # Minutes variation from timetabled time
        # x['body']['original_loc_timestamp'],
        x['body']['current_train_id'],  # Indicates train has been reassigned -> does this have an impact on lateness
        int(replace_space(x['body']['next_report_run_time'], 0)),  # Likely minutes until next stanox report
        x['body']['actual_timestamp'],
        x['body']['correction_ind'],  # False if not a correction of a previous report, True if it is
        x['body']['platform'],
        x['body']['train_id'],
        x['body']['variation_status'],  # ON TIME, EARLY, LATE, OFF ROUTE
        x['body']['train_service_code'],
        x['body']['toc_id'],
        x['body']['loc_stanox'],
        x['body']['next_report_stanox']
    ])

for x in json_change_orig_data:
    change_orig_data.append(list(x['body'].values()))

for x in json_cancel_data:
    cancel_data.append(list(x['body'].values()))

# flatten_dict(data_body)

print(data)
print(change_orig_data)
print(cancel_data)


# ANALYSIS data using graphs/general stats
# E.g.  How many late trains, how many early trains etc
#       Break down of number of service instances per service code
#       Break down number of stops per train line etc
#       find lines with greatest amount of data etc
#       etc
early = 0
late = 0
ontime = 0
offroute = 0
for x in data:
    if x[9] == "EARLY":
        early += 1
    elif x[9] == "LATE":
        late += 1
    elif x[9] == "ON TIME":
        ontime += 1
    else:
        offroute += 1

print()
print("************************************* STATS ****************************************")
print()
print("EARLY: ", early, " | LATE: ", late, " | ON TIME: ", ontime, " | OFF ROUTE: ", offroute)

# Average number of minutes late/early
# Neg_data is form with mins in negative if appropriate
neg_data = []

for x in data:
    neg_data.append([
        x[0],
        x[1],
        check_status(x[2], x[9]),  # Minutes variation from timetabled time
        x[3],
        x[4],
        x[5],
        x[6],
        x[7],
        x[8],
        x[9],
        x[10],
        x[11],
        x[12],
        x[13]
    ])


total = 0
count = 0

for x in neg_data:
    total += x[2]
    count += 1

average = total/count

print("Average Mins off Timetable: ", average)


# Average delay based at station
stations = set()
for x in neg_data:
    stations.add(x[12])


print("Average Mins off Timetable: ")
for y in stations:
    temp_count = 0
    temp_sum = 0
    for x in neg_data:
        if x[12] == y:
            temp_count += 1
            temp_sum += x[2]
    print(get_stanox(y), ": ", temp_sum/temp_count)

print()

# Average delay based on toc
tocs = set()
for x in neg_data:
    tocs.add(x[11])


print("Average Mins off Timetable: ")
for y in tocs:
    temp_count = 0
    temp_sum = 0
    for x in neg_data:
        if x[11] == y:
            temp_count += 1
            temp_sum += x[2]
    print(get_station(y), ": ", temp_sum/temp_count)

print()

# Average delay based on Train Service Line

# Busiest Station

# Busiest Train Service

