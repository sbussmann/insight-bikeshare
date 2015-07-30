from flask import Flask
from flask import render_template, request#, url_for
#from app import app
#import pymysql as mdb
#from predictride import predict
import gridpredict
import loadutil
import folium
#from subprocess import call

#print(Flask.root_path)
app = Flask(__name__)

#db = mdb.connect(user="root", host="localhost", password="password",
#        db="BostonFeaturesByStation_db", charset='utf8')
basedir = '../Data/Boston/'
growdir = '../Data/Boston/growing/'

#@app.route('/hubway')
#def station_hubway():

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
@app.route('/index')
def station_input():

    # reset to existing Hubway stations only
    gridpredict.resetiteration(basedir, growdir)

    # generate the map
    latvec, longvec = loadutil.grid()
    lat0 = latvec.mean()
    long0 = longvec.mean()
    map_hubway = folium.Map(location=[lat0, long0], width=600, zoom_start=20)

    # generate the station locations
    dataload = loadutil.load(growdir)
    stationfeatures = dataload[6]
    station = dataload[2]
    nstation = len(station)
    for i in range(nstation):
        ilat = station['lat'][i]
        ilong = station['lng'][i]
        iride = stationfeatures['ridesperday'][i]
        map_hubway.circle_marker(location=[ilat, ilong], radius=iride,
                line_color='black', fill_color='red', fill_opacity=0.2)
    #cmd = 'rm -f app/template/hubway.html'
    #call(cmd, shell=True)
    #hubwayurl = url_for('static', filename='hubway.html')
    #print(hubwayurl)
    map_hubway.create_map(path="static/hubway.html")

    #import fileinput

    #processing_foo1s = False

    #for line in fileinput.input('static/hubway.html', inplace=1):
    #    if line.startswith('<head>'):
    #        processing_foo1s = True
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
    the_results = gridpredict.autoinput(growdir)
    latitude = the_results[0]
    longitude = the_results[1]
    riderate = the_results[2]
    ranking = the_results[3]
    return render_template("output.html", riderate=riderate, ranking=ranking,
          latitude=latitude, longitude=longitude)

@app.route('/output_user')
def station_output_user():
    #pull 'ID' from input field and store it
    longitude = request.args.get('ID1')
    latitude = request.args.get('ID2')
    the_results = gridpredict.userinput(longitude, latitude, growdir)
    latitude = the_results[0]
    longitude = the_results[1]
    riderate = the_results[2]
    ranking = the_results[3]
    return render_template("output.html", riderate=riderate, ranking=ranking,
          latitude=latitude, longitude=longitude)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
