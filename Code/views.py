from flask import render_template, request, Flask
#from app import app
#import pymysql as mdb
#from predictride import predict
import gridpredict

app = Flask(__name__)

#db = mdb.connect(user="root", host="localhost", password="password",
#        db="BostonFeaturesByStation_db", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html",
        title = 'Home', user = { 'nickname': 'Shane' },
        )

@app.route('/input')
def station_input():
  return render_template("input.html")

@app.route('/output-auto')
def station_output_auto():
  iterstring = '0'
  the_results = gridpredict.autoinput(iterstring)
  riderate = the_results[0]
  ranking = the_results[1]
  iterstring = the_results[2]
  return render_template("output.html", riderate=riderate, ranking=ranking)

@app.route('/output-user')
def station_output_user():
  #pull 'ID' from input field and store it
  longitude = request.args.get('ID1')
  latitude = request.args.get('ID2')


  iterstring = '0'
  the_results = gridpredict.userinput(longitude, latitude, iterstring)
  riderate = the_results[0]
  ranking = the_results[1]
  iterstring = the_results[2]
  return render_template("output.html", riderate=riderate, ranking=ranking)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
