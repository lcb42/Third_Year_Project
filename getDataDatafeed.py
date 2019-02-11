

# Security Tocken for National Rail Data Feeds account registered to lb13g16@soton.ac.uk
from pymongo import MongoClient

token = "2c1f3416-3510-443d-90a7-93d02d3254ce"


import json
import time

import stomp

import logging

username = 'lb13g16@soton.ac.uk'
password = 'NR0dpassword!'

client = MongoClient()
db = client["Third_Year_Project"]
real_time_data = db["Real_Time_Data_1"]
real_time_data_hants = db["Real_Time_Data_HANTS_1"]

stanox_hants_stn_list = ["87031", "87021", "86074", "87763", "86066", "86896", "86339", "87024", "86216", "87009", "86202", "86921", "86061", "86901", "86223", "86108", "86915", "86301", "86112", "86107", "86087", "86343", "86241", "86042", "87010", "87026", "86045", "86321", "86077", "86222", "87062", "86341", "86201", "86332", "86913", "86049", "87064", "87065", "86908", "86907", "86081", "86527", "86219", "86911", "87012", "86070", "87066", "86917", "86248", "86313", "86311", "86703", "86101", "87067", "86122", "86084", "86218", "86495", "86520", "86499", "86225", "86909", "86497", "86711", "86342", "86071", "86083", "86047", "86215"]
stanox_soton_stn_list = ['86520', '86216', '86527', '86703', '86499', '86218', '86495', '86497', '86215']

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


class MyListener(object):
    def on_error(self, headers, message):
        print("ERROR: {} {}".format(headers, message))

    def on_message(self, headers, message):
        #print("MESSAGE: {} {}".format(headers, message))
        data = json.loads(message)
        for event in data:
            self._handle_event(event)

    def _handle_event(self, data):

        stanox = data['body'].get('loc_stanox')

        # sou/bte/mbk/rdb/sdn/sho/soa/swg/wls
        if stanox in stanox_soton_stn_list:
            print("\n**** ", get_stanox(stanox), "****\n")
            print(data)
            # Insert data into mongoDB
            x = real_time_data.insert_one(data)
            print("Inserted at: ", x.inserted_id)

        # All Stations in Hampshire area
        if stanox in stanox_hants_stn_list:
            print("\n**** ", get_stanox(stanox), "****\n")
            print(data)
            # Insert data into mongoDB
            x = real_time_data_hants.insert_one(data)
            print("Inserted at: ", x.inserted_id)
        #else:
        #    print("somewhere else ({})".format(stanox))


def main():
    logging.basicConfig(level=logging.WARN)
    hostname = 'datafeeds.networkrail.co.uk'

    channel = 'TRAIN_MVT_ALL_TOC'

    conn = stomp.Connection(host_and_ports=[(hostname, 61618)])
    conn.set_listener('mylistener', MyListener())
    conn.start()
    conn.connect(username=username, passcode=password)

    conn.subscribe(destination='/topic/{}'.format(channel), id=1, ack='auto')

    keep_running = True
    try:
        while keep_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Quitting.")
        keep_running = False
    #else:
     #   raise

    conn.disconnect()


if __name__ == '__main__':
    main()