"""

Parse XML output from open street map data.  Need to get create a mask for the
Charles River so that stations cannot be placed on water.

"""

import pandas as pd
import json
import numpy as np


# the list of nodes associated with the Charles River
waylist = ['303073437', '303307344', '160693433', '261632627', \
        '27394613', '160693521', '198531431', '160693291', '160693304']

with open('../Data/Boston/charlesriver.json') as data_file:    
    data = json.load(data_file)
    elements = data['elements']

    idlist = []
    latitudelist = []
    longitudelist = []
    for element in elements:

        # build the list of latitudes and longitudes
        if element['type'] == 'node':
            latitudelist.append(element['lat'])
            longitudelist.append(element['lon'])
            idlist.append(element['id'])

    node_df = pd.DataFrame({'id': idlist, 'latitude': latitudelist, 
            'longitude': longitudelist})


    for i, iway in enumerate(waylist):

        # build the list of nodes
        for element in elements:

            if element['type'] == 'way':
                if element['id'] == np.int(iway):
                    nodelist = element['nodes']

                    inode = pd.DataFrame({'id': nodelist})
                    inode = inode.merge(node_df, on='id', how='left')
                    print("Saving nodes for " + iway)
                    inode.to_csv('../Data/Boston/charlesriver_' + iway + '.csv', 
                            index=False)

    # do the corner of the river that won't fucking download from OSM for some
    # fucking reason
    inode = pd.read_csv('../Data/Boston/charlesrivertmp')
    inode = inode.merge(node_df, on='id', how='left')
    print("Saving nodes for 303307344")
    inode.to_csv('../Data/Boston/charlesriver_303307344.csv')

# build the full list of latitudes and longitudes
#fullids = []
#for node in root.findall('node'):
#    fullids.append(node.get('id'))
#    fulllatitudes.append(node.get('lat'))
#    fulllongitudes.append(node.get('lon'))

# turn the lists into dataframes and merge them together
#charlesway = pd.DataFrame({'nodeid': nodelist})
#fullloc = pd.DataFrame({'nodeid': fullids, 'latitude': fulllatitudes,\
#        'longitude': fulllongitudes})

#charlesway = charlesway.merge(fullloc, on='nodeid', how='left')
#charlesway.to_csv('../Data/Boston/charlesrivercoast.csv')
#import pdb; pdb.set_trace()
