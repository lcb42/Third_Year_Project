from pymongo import MongoClient

client = MongoClient()

dblist = client.list_database_names()

db = client["Third_Year_Project"]


test_collection = db["test"]

real_time_data = db["Real_Time_Data"]


dict = {"name": "Lucy", "surname": "Blatherwick", "age": "20"}

x = test_collection.insert_one(dict)

print(x.inserted_id)

for x in test_collection.find():
    print(x)