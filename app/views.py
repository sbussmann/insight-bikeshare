import matplotlib
matplotlib.use('Agg')
from flask import Flask
from flask import render_template, request, make_response, session, redirect, url_for
#from app import app
import gridpredict
import loadutil
import folium
import numpy as np
import pandas as pd
import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import googlemaps
import user
from subprocess import call
import datetime as dt


#print(Flask.root_path)
app = Flask(__name__)

app.secret_key = '\x96\\\xf7\x93\xc4\xae:\x8c{iL\x91\x12\x13^\xec\xad\xe2\x9a\xe6\x97{\x8dW'

#db = mdb.connect(user="root", host="localhost", password="password",
#        db="BostonFeaturesByStation_db", charset='utf8')

#def get_next_user_id():
#    f = open('users.stat', 'r')
#    match = re.search(r'total number of users = (\d+)', f.read())
#    if match:
#        next_user_id = int(match.group(1))
#    else:
#        next_user_id = 0
#    f.close()
#    return next_user_id

#def put_next_user_id(next_id):
#    f = open('users.stat', 'w')
#    f.write('total number of users = ' + str(next_id))
#    f.close()

# define a unique results directory for each user
next_user_id = user.get_next_user_id()
users = {}

basedir = '../Data/Boston/'
#growdir = '../Data/Boston/growing/'

#next_user_id = get_next_user_id()

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

def getgrowdir(next_user_id):

    """

    Generate a new directory to store results for each user.

    """

    gd = basedir + 'user' + str(next_user_id) + '/'

    return gd

def makemap():
    # generate the map
    latvec, longvec = loadutil.grid()
    lat0 = latvec.mean()
    long0 = longvec.mean()
    mapwidth = 600
    mapheight = 500
    map_hubway = folium.Map(location=[lat0, long0], width=mapwidth, height=mapheight, zoom_start=9)

    # generate the station locations
    growdir = getgrowdir(next_user_id)
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
    if 'userid' in session:
        uid = session['userid']
        if uid in users:
            users[uid].record_as_active()
    return render_template("about.html")

@app.route('/contact')
def contactpage():
    if 'userid' in session:
        uid = session['userid']
        if uid in users:
            users[uid].record_as_active()
    return render_template("contact.html")

@app.route('/')
@app.route('/index')
def station_input():
    # check existing users for activity, delete inactive users
    inactive_users = []
    now = dt.datetime.today()

    sessionlifetime = 24 * 3600
    for uid in users:
        if (now - users[uid].last_activity_time).seconds > sessionlifetime:
            inactive_users.append(uid)
    for uid in inactive_users:
        del users[uid]
        rottendir = getgrowdir(uid)
        cmd = 'rm -rf ' + rottendir
        call(cmd, shell=True)

    #try:
        #session['userid'] += 1
    #except KeyError:
        #session['userid'] = 1
    if 'userid' in session:
        uid = session['userid']
        growdir = getgrowdir(uid)
    else:
        #uid = user.get_next_user_id()
        uid = np.random.randint(1, high=1e6)
        #exec 'next_user_id += 1' in globals()
        #user.put_next_user_id(next_user_id)
        session['userid'] = uid
        growdir = getgrowdir(uid)
        # make a new directory for this user
        cmd = 'mkdir ' + growdir
        call(cmd, shell=True)

    
    # store user class in users dictionary
    users[uid] = user.User()

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
    if 'userid' not in session:
        return redirect(url_for('index'))
    uid = session['userid']
    if uid not in users:
        return redirect(url_for('index'))
    users[uid].record_as_active()    

    gmaps = makegmap()
    growdir = getgrowdir(uid)
    print(uid, growdir)
    the_results = gridpredict.autoinput(growdir)
    stationslistdict, riderate, ranking = makeoutput(the_results, gmaps)
    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

@app.route('/output_user')
def station_output_user():
    if 'userid' not in session:
        return redirect(url_for('index'))
    uid = session['userid']
    if uid not in users:
        return redirect(url_for('index'))
    users[uid].record_as_active()    

    gmaps = makegmap()
    #pull 'ID' from input field and store it
    useraddress = request.args.get('ID1')
    geocode = gmaps.geocode(useraddress)
    latitude = geocode[0]['geometry']['location']['lat']
    longitude = geocode[0]['geometry']['location']['lng']
    #longitude = 
    #latitude = request.args.get('ID2')
    growdir = getgrowdir(uid)
    print(uid, growdir)
    the_results = gridpredict.userinput(longitude, latitude, growdir)
    stationslistdict, riderate, ranking = makeoutput(the_results, gmaps)

    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

def makegmap():
    api_key = 'AIzaSyA1waGCAiSOdsKMI4mg_wrqAdouoVPIbXw'
    gmaps = googlemaps.Client(key=api_key)
    return(gmaps)

def makeoutput(the_results, gmaps):    

    latitude = the_results[0]
    longitude = the_results[1]
    riderate = the_results[2]
    ranking = the_results[3]
    location = gmaps.reverse_geocode((latitude, longitude))
    location = location[0]['formatted_address']

    if 'userid' not in session:
        return redirect(url_for('index'))
    uid = session['userid']
    if uid not in users:
        return redirect(url_for('index'))
    users[uid].record_as_active()    

    # load old results
    growdir = getgrowdir(uid)
    print(uid, growdir)
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

    if 'userid' not in session:
        return redirect(url_for('index'))
    uid = session['userid']
    if uid not in users:
        return redirect(url_for('index'))
    users[uid].record_as_active()    

    fig = plt.figure(figsize=(9,5))

    # plot predicted ride map
    growdir = getgrowdir(uid)
    print(uid, growdir)
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
