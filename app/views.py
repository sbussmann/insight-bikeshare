import matplotlib
matplotlib.use('Agg')
from flask import Flask
from flask import render_template, request, make_response, session, redirect, url_for
import gridpredict
import numpy as np
import pandas as pd
import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import googlemaps
import user
from subprocess import call
import datetime as dt


# instantiate the flask app
app = Flask(__name__)

# use a secret key so that users cannot see their cookie data directly
app.secret_key = '\x96\\\xf7\x93\xc4\xae:\x8c{iL\x91\x12\x13^\xec\xad\xe2\x9a\xe6\x97{\x8dW'

# instantiate a dictionary for users
users = {}

# get the next user id
next_user_id = user.get_next_user_id()

# directory in which user directories are stored
basedir = '../Data/Boston/'

def getgrowdir(user_id):

    """

    Generate a new directory to store results for each user.  Requires basedir
    to be defined globally.

    """

    gd = basedir + 'user' + str(user_id) + '/'

    return gd

def userisactive():

    """

    Check if the user has been active in the past 24 hours.  If so, update
    their most recent activity with the current time.

    """

    if 'userid' in session:
        uid = session['userid']
        if uid in users:
            users[uid].record_as_active()

def userisactive_output():

    """

    Make sure the user has been active in the past 24 hours.  If not, redirect
    them to the input page so they can start again.  If yes, update
    their most recent activity with the current time and return their user id.

    """

    # check for activity in session dictionary and in users dictionary
    if 'userid' not in session:
        return redirect(url_for('index'))
    uid = session['userid']
    if uid not in users:
        return redirect(url_for('index'))

    # if both tests pass, update their most recent activity with current time
    users[uid].record_as_active()    

    return uid

@app.route('/about')
def aboutpage():

    """

    Direct users to info on how I did this project.

    """

    # check for recent activity
    userisactive()

    return render_template("about.html")

@app.route('/contact')
def contactpage():

    """

    Direct users to my contact info.

    """

    # check for recent activity
    userisactive()

    return render_template("contact.html")

@app.route('/')
@app.route('/index')
def station_input():

    # check existing users for activity, delete inactive users
    inactive_users = []
    now = dt.datetime.today()

    # if inactive for longer than sessionlifetime, remove user from database
    sessionlifetime = 24 * 3600
    for uid in users:
        if (now - users[uid].last_activity_time).seconds > sessionlifetime:
            inactive_users.append(uid)
    for uid in inactive_users:
        del users[uid]
        rottendir = getgrowdir(uid)
        cmd = 'rm -rf ' + rottendir
        call(cmd, shell=True)

    # if this user has a history with Grow Hubway, get their directory
    if 'userid' in session:
        uid = session['userid']
        growdir = getgrowdir(uid)
    else:
        # otherwise, make a new userid with a random number generator
        # note, this solution may have problems related to collisions.  Need to
        # investigate this in the future
        uid = np.random.randint(1, high=1e8)

        # add the user id to the session dictionary
        session['userid'] = uid

        # get the users directory
        growdir = getgrowdir(uid)

        # make the directory
        cmd = 'mkdir ' + growdir
        call(cmd, shell=True)

    
    # store user class in users dictionary
    users[uid] = user.User()

    # reset to existing Hubway stations only
    gridpredict.resetiteration(basedir, growdir)

    # reset the results database for this user
    useraddress = []
    riderate = []
    dictnew = {"address": useraddress, "ridesperday": riderate}
    stations = pd.DataFrame(dictnew)
    stations.to_csv(growdir + 'appresults.csv', index=False)

    return render_template("input.html")

@app.route('/output_auto')
def station_output_auto():

    """

    Obtain predicted average daily rides at the best possible location.
    Add the new station to the station database.  Regenerate the predicted
    average daily rides matrix.

    """

    # get the user id number
    uid = userisactive_output()

    # google geo code API plugin to translate between address and lat/long
    googlegeo = makegeo()

    # get the user's directory
    growdir = getgrowdir(uid)

    # run the model on the best possible location
    prediction = gridpredict.autoinput(growdir)

    # get the list of stations, daily rides, and ranking of the new station
    stationslistdict, riderate, ranking = makeoutput(prediction, googlegeo)

    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

@app.route('/output_user')
def station_output_user():

    # get the user id number
    uid = userisactive_output()

    # google geo code API plugin to translate between address and lat/long
    googlegeo = makegeo()

    # convert address to latitude and longitude
    useraddress = request.args.get('ID1')
    geocode = googlegeo.geocode(useraddress)
    latitude = geocode[0]['geometry']['location']['lat']
    longitude = geocode[0]['geometry']['location']['lng']

    # get the user's directory
    growdir = getgrowdir(uid)

    # run the model on the user's chosen location
    prediction = gridpredict.userinput(longitude, latitude, growdir)

    # get the list of stations, daily rides, and ranking of the new station
    stationslistdict, riderate, ranking = makeoutput(prediction, googlegeo)

    return render_template("output.html", riderate=riderate, ranking=ranking,
          stations=stationslistdict)

def makegeo():

    """

    Apply Google geo code API key.

    """

    api_key = 'AIzaSyA1waGCAiSOdsKMI4mg_wrqAdouoVPIbXw'
    googlegeo = googlemaps.Client(key=api_key)

    return(googlegeo)

def makeoutput(prediction, googlegeo):    

    """

    Build the table of new stations that will be displayed on the output html
    page.

    """

    # convert predicted latitude and longitude to an address
    latitude = prediction[0]
    longitude = prediction[1]
    riderate = prediction[2]
    ranking = prediction[3]
    location = googlegeo.reverse_geocode((latitude, longitude))
    location = location[0]['formatted_address']

    # get the user id number
    uid = userisactive_output()

    # load old results
    growdir = getgrowdir(uid)
    stations = pd.read_csv(growdir + 'appresults.csv')
    newlocation = list(stations['address'].values)
    newriderate = list(stations['ridesperday'].values)

    # append the new results
    newlocation.append(location)
    newriderate.append(riderate)
    dictnew = {"address": newlocation, "ridesperday": newriderate}

    # store the new results in a csv file
    stationsnew = pd.DataFrame(dictnew)
    stationsnew.to_csv(growdir + 'appresults.csv', index=False)

    # build a list of dictionaries to pass to Flask
    stationslistdict = []
    for i in range(len(stationsnew)):
        stindex = str(i + 1)
        rpd = stationsnew['ridesperday'].values[i]
        address = stationsnew['address'].values[i]
        stationslistdict.append({"index":stindex, "ridesperday": rpd, 
                "address": address})

    return stationslistdict, riderate, ranking

@app.route("/predictedridemap.png")
def predictedridemap():

    """

    Generate the image showing the predicted daily rides after adding the new
    station.

    """

    # get the user id number
    uid = userisactive_output()

    # get the user's directory
    growdir = getgrowdir(uid)

    # set the figure size
    fig = plt.figure(figsize=(9,5))

    # load the predicted ride data
    nrides = pd.read_csv(growdir + 'nridesmap.csv')
    longmin = nrides['longitude'].min()
    longmax = nrides['longitude'].max()
    latmin = nrides['latitude'].min()
    latmax = nrides['latitude'].max()
    nlat = np.sqrt(np.float(len(nrides)))

    # generate the predicted ride matrix
    ridemap = nrides['nrides'].values.reshape((nlat, nlat))

    # plot the predicted ride matrix
    plt.imshow(ridemap, vmin=0, cmap="Blues",
            extent=[longmin,longmax,latmin,latmax], origin='lower')

    # add a colobar and label it
    cbar = plt.colorbar()
    cbar.set_label('Predicted Daily Rides')

    # plot the Hubway stations in white
    station = pd.read_csv(growdir + 'Station.csv')
    stationfeatures = pd.read_csv(growdir + 'Features.csv')
    plt.scatter(station['lng'], station['lat'], 
            s=stationfeatures['ridesperday'], alpha=0.4, 
            color='white', edgecolor='black', 
            label='Existing Hubway stations')

    # highlight the new stations added by this user in red
    stationnew = station[station['status'] == 'proposed']
    stationfeaturesnew = stationfeatures[station['status'] == 'proposed']
    plt.scatter(stationnew['lng'], stationnew['lat'], 
            s=stationfeaturesnew['ridesperday'], alpha=0.4, 
            color='red', edgecolor='black', 
            label='Proposed Hubway stations')

    # refine the axis range
    plt.axis([longmin, longmax, latmin, latmax])

    # add labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('black')

    # translate the figure into a format suitable for Flask
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

# run the app on the local machine on port 5000
if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
