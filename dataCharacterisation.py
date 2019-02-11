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


# Average delay based on TOC

# Average delay based on Train Service Line