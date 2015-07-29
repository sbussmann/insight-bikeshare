from flask import Flask
from flask import render_template, request
#from app import app
#import pymysql as mdb
#from predictride import predict
import gridpredict

app = Flask(__name__)

#db = mdb.connect(user="root", host="localhost", password="password",
#        db="BostonFeaturesByStation_db", charset='utf8')

@app.route('/')
@app.route('/index')
def station_input():
  basedir = '../Data/Boston/'
  growdir = '../Data/Boston/growing/'
  gridpredict.resetiteration(basedir, growdir)
  return render_template("input.html")

@app.route('/output_auto')
def station_output_auto():
  growdir = '../Data/Boston/growing/'
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
  growdir = '../Data/Boston/growing/'
  the_results = gridpredict.userinput(longitude, latitude, growdir)
  latitude = the_results[0]
  longitude = the_results[1]
  riderate = the_results[2]
  ranking = the_results[3]
  return render_template("output.html", riderate=riderate, ranking=ranking,
          latitude=latitude, longitude=longitude)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
