

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
real_time_data = db["Real_Time_Data"]

def get_stanox(stanox):

    switcher = {
        '86520': "SOU",
        '86216': "BTE",
        '86527': "MBK",
        '86703': "RDB",
        '86499': "SDN",
        '86218': "SHO",
        '86495': "SOA",
        '86497': "SWG",
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
        if (stanox == '86520') or (stanox == '86216') or (stanox == '86527') or (stanox == '86703') or (stanox == '86499') or (stanox == '86218') or (stanox == '86495') or (stanox == '86497') or (stanox == '86215'):
            print("\n**** ", get_stanox(stanox), "****\n")
            print(data)
            # Insert data into mongoDB
            x = real_time_data.insert_one(data)
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