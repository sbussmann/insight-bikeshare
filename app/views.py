import matplotlib
matplotlib.use('Agg')
from flask import Flask
from flask import render_template, request, make_response#, url_for
#from app import app
#from predictride import predict
import gridpredict
import loadutil
import folium
import numpy as np
import pandas as pd
import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import googlemaps
#from matplotlib.figure import Figure
#from subprocess import call


#print(Flask.root_path)
app = Flask(__name__)

#db = mdb.connect(user="root", host="localhost", password="password",
#        db="BostonFeaturesByStation_db", charset='utf8')
basedir = '../Data/Boston/'
growdir = '../Data/Boston/growing/'

#@app.route('/hubway')
#def station_hubway():
#def inline_map(m, width=650, height=500):
#    """Takes a folium instance and embed HTML."""
#    m._build_map()
#    srcdoc = m.HTML.replace('"', '&quot;')
#    embed = HTML('<iframe srcdoc="{}" '
#                 'style="width: {}px; height: {}px; '
#                 'border: none"></iframe>'.format(srcdoc, width, height))
#    return embed

def makemap():
    # generate the map
    latvec, longvec = loadutil.grid()
    lat0 = latvec.mean()
    long0 = longvec.mean()
    mapwidth = 600
    mapheight = 500
    map_hubway = folium.Map(location=[lat0, long0], width=mapwidth, height=mapheight, zoom_start=9)

    # generate the station locations
    dataload = loadutil.load(growdir)
    stationfeatures = dataload[6]
    station = dataload[2]
    nstation = len(station)
    for i in range(nstation):
        ilat = station['lat'][i]
        ilong = station['lng'][i]
        iride = stationfeatures['ridesperday'][i] * 10
        map_hubway.circle_marker(location=[ilat, ilong], radius=iride,
                fill_color='white', edge_color='none', fill_opacity=0.5)
    map_hubway.create_map(path="templates/hubway.html")
    #import fileinput
    #processing_foo1s = False
    #for i, line in enumerate(fileinput.input('templates/hubway.html', inplace=1)):
#	if i == 0:
#		print('{% extends "input.html" %}')
#		print('{% block map %}')
#        #if line.startswith('<head>'):
#        #    processing_foo1s = True
#        #else:
#        #    if processing_foo1s:
#        #        print('   <meta http-Equiv="Cache-Control" Content="no-cache" />')
#        #        print('   <meta http-Equiv="Pragma" Content="no-cache" />')
#        #        print('   <meta http-Equiv="Expires" Content="0" />')
#        #    processing_foo1s = False
#        print line,
#	if i == len(enumerate(fileinput.input('templates/hubway.html', inplace=1))) - 1:
#		print('{% endblock %}')

    #foliummap = 5#inline_map(map_hubway)

@app.route('/about')
def aboutpage():
    return render_template("about.html")

@app.route('/contact')
def contactpage():
    return render_template("contact.html")

@app.route('/')
@app.route('/index')
def station_input():

    # reset to existing Hubway stations only
    gridpredict.resetiteration(basedir, growdir)

    # reset the webapp results database
    useraddress = []
    riderate = []
    dictnew = {"address": useraddress, "ridesperday": riderate}
    stations = pd.DataFrame(dictnew)
    stations.to_csv(growdir + 'appresults.csv', index=False)

    #for line in fileinput.input('static/hubway.html', inplace=1):
    #    if line.startswith('<head>'):
    
    #    else:
    #        if processing_foo1s:
    #            print('   <meta http-Equiv="Cache-Control" Content="no-cache" />')
    #            print('   <meta http-Equiv="Pragma" Content="no-cache" />')
    #            print('   <meta http-Equiv="Expires" Content="0" />')
    #        processing_foo1s = False
    #    print line,
    ##render_template("hubway.html"
    return render_template("input.html")

@app.route('/output_auto')
def station_output_auto():
    gmaps = makegmap()
    the_results = gridpredict.autoinput(growdir)
    stationslistdict, riderate, ranking = makeoutput(the_results, gmaps)
    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

@app.route('/output_user')
def station_output_user():
    gmaps = makegmap()
    #pull 'ID' from input field and store it
    useraddress = request.args.get('ID1')
    geocode = gmaps.geocode(useraddress)
    latitude = geocode[0]['geometry']['location']['lat']
    longitude = geocode[0]['geometry']['location']['lng']
    #longitude = 
    #latitude = request.args.get('ID2')
    the_results = gridpredict.userinput(longitude, latitude, growdir)
    stationslistdict, riderate, ranking = makeoutput(the_results, gmaps)

    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

def makegmap():
    api_key = 'AIzaSyA1waGCAiSOdsKMI4mg_wrqAdouoVPIbXw'
    api_key = 'AIzaSyBM0FQfza4RMXKeN8rZpfk6--5RsRqWqyY'
    api_key = 'AIzaSyABlrd95eCKHV2tad5FmsfXBtlODsIZRWA'
    gmaps = googlemaps.Client(key=api_key)
    return(gmaps)

def makeoutput(the_results, gmaps):    
    latitude = the_results[0]
    longitude = the_results[1]
    riderate = the_results[2]
    ranking = the_results[3]
    location = gmaps.reverse_geocode((latitude, longitude))
    location = location[0]['formatted_address']

    # load old results
    stations = pd.read_csv(growdir + 'appresults.csv')
    newlocation = list(stations['address'].values)
    newriderate = list(stations['ridesperday'].values)

    # append the new results
    newlocation.append(location)
    newriderate.append(riderate)
    print(newlocation, newriderate)
    dictnew = {"address": newlocation, "ridesperday": newriderate}
    stationsnew = pd.DataFrame(dictnew)
    stationsnew.to_csv(growdir + 'appresults.csv', index=False)
    stationslistdict = []
    for i in range(len(stationsnew)):
        stindex = str(i + 1)
        rpd = stationsnew['ridesperday'].values[i]
        address = stationsnew['address'].values[i]
        stationslistdict.append({"index":stindex, "ridesperday": rpd, 
                "address": address})

    return stationslistdict, riderate, ranking

#@app.route("/osmmap")
#def osmmap():
#    return response

@app.route("/predictedridemap.png")
def predictedridemap():


    fig = plt.figure(figsize=(9,5))

    # plot predicted ride map
    growdir = '../Data/Boston/growing/'
    nrides = pd.read_csv(growdir + 'nridesmap.csv')
    longmin = nrides['longitude'].min()
    longmax = nrides['longitude'].max()
    latmin = nrides['latitude'].min()
    latmax = nrides['latitude'].max()
    nlat = np.sqrt(np.float(len(nrides)))
    ridemap = nrides['nrides'].values.reshape((nlat, nlat))

    plt.imshow(ridemap, vmin=0, cmap="Blues",
            extent=[longmin,longmax,latmin,latmax], origin='lower')
    cbar = plt.colorbar()
    #cbar = matplotlib.colorbar.ColorbarBase(ax)
    cbar.set_label('Predicted Daily Rides')

    # plot existing Hubway stations
    station = pd.read_csv(growdir + 'Station.csv')
    stationfeatures = pd.read_csv(growdir + 'Features.csv')
    plt.scatter(station['lng'], station['lat'], 
            s=stationfeatures['ridesperday'], alpha=0.4, 
            color='white', edgecolor='black', 
            label='Existing Hubway stations')
    stationnew = station[station['status'] == 'proposed']
    stationfeaturesnew = stationfeatures[station['status'] == 'proposed']
    plt.scatter(stationnew['lng'], stationnew['lat'], 
            s=stationfeaturesnew['ridesperday'], alpha=0.4, 
            color='red', edgecolor='black', 
            label='Proposed Hubway stations')
    plt.axis([longmin, longmax, latmin, latmax])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #ax = plt.gca()
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('black')

    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
