"""

Parse XML output from open street map data.  Need to get create a mask for the
Charles River so that stations cannot be placed on water.

"""

from xml.etree import ElementTree as ET
import pandas as pd


# build the list of nodes associated with the Charles River
tree = ET.parse('../Data/Boston/charlesriver1.xml')
root = tree.getroot()
waylist = ['303073437', '303307344', '160693433', '261632627', \
        '27394613', '160693521', '198531431', '160693291', '160693304']

import json

with open('../Data/Boston/charlesriver.json') as data_file:    
    data = json.load(data_file)
    elements = data['elements']


    nodelist = []
    for way in data.findall('way'):
        for iwaylist in waylist:
            if way.get('id') == iwaylist:
                print("Found id" + way.get('id'))
                for nd in way.findall('nd'):
                    nodelist.append(nd.get('ref'))

# build the full list of latitudes and longitudes
fulllatitudes = []
fulllongitudes = []
fullids = []
for node in root.findall('node'):
    fullids.append(node.get('id'))
    fulllatitudes.append(node.get('lat'))
    fulllongitudes.append(node.get('lon'))

# turn the lists into dataframes and merge them together
charlesway = pd.DataFrame({'nodeid': nodelist})
fullloc = pd.DataFrame({'nodeid': fullids, 'latitude': fulllatitudes,\
        'longitude': fulllongitudes})

charlesway = charlesway.merge(fullloc, on='nodeid', how='left')
charlesway.to_csv('../Data/Boston/charlesrivercoast.csv')
import pdb; pdb.set_trace()
