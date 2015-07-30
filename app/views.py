import matplotlib
matplotlib.use('Agg')
from flask import Flask
from flask import render_template, request, make_response#, url_for
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

@app.route('/')
@app.route('/index')
def station_input():

    # reset to existing Hubway stations only
    gridpredict.resetiteration(basedir, growdir)

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

@app.route("/predictedridemap.png")
def predictedridemap():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter


    fig=Figure(figsize=(9,5))
    ax=fig.add_subplot(111)

    # plot predicted ride map
    growdir = '../Data/Boston/growing/'
    nrides = pd.read_csv(growdir + 'nridesmap.csv')
    longmin = nrides['longitude'].min()
    longmax = nrides['longitude'].max()
    latmin = nrides['latitude'].min()
    latmax = nrides['latitude'].max()
    nlat = np.sqrt(np.float(len(nrides)))
    ridemap = nrides['nrides'].values.reshape((nlat, nlat))

    ax.imshow(ridemap, vmin=0, cmap="Blues",
            extent=[longmin,longmax,latmin,latmax], origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Predicted Daily Rides')

    # plot existing Hubway stations
    station = pd.read_csv(growdir + 'Station.csv')
    stationfeatures = pd.read_csv(growdir + 'Features.csv')
    ax.scatter(station['lng'], station['lat'], 
            s=stationfeatures['ridesperday'], alpha=0.4, 
            color='white', edgecolor='black', 
            label='Existing Hubway stations')
    stationnew = station[station['status'] == 'proposed']
    stationfeaturesnew = stationfeatures[station['status'] == 'proposed']
    ax.scatter(stationnew['lng'], stationnew['lat'], 
            s=stationfeaturesnew['ridesperday'], alpha=0.4, 
            color='red', edgecolor='black', 
            label='Proposed Hubway stations')
    ax.axis([longmin, longmax, latmin, latmax])
    ax.xlabel('Longitude')
    ax.ylabel('Latitude')

    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
