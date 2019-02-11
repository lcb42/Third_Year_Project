from pymongo import MongoClient

client = MongoClient()

dblist = client.list_database_names()

db = client["Third_Year_Project"]

real_time_data = db["Real_Time_Data_HANTS_1"]


service_codes = []
for x in real_time_data.find():
    service_codes.append(x["body"]["train_service_code"])

num_service_codes = {}
for y in service_codes:
    num_service_codes[y] = real_time_data.find( { "body.train_service_code" : y } ).count()

print(max(num_service_codes.keys(), key=(lambda key: num_service_codes[key])))

docs = []
for z in real_time_data.find( { "body.train_service_code" : max(num_service_codes.keys(), key=(lambda key: num_service_codes[key])) } ):
    docs.append(z)


def by_train_id(json):
    return json["body"]["train_id"]


docs.sort(key=by_train_id, reverse=True)

for w in docs:
    print(w)