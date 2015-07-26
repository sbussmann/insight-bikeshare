from flask import render_template, request
from webapp import webapp
import pymysql as mdb
from predictride import predict

db = mdb.connect(user="root", host="localhost", password="password",
        db="BostonFeaturesByStation_db", charset='utf8')

@webapp.route('/')
@webapp.route('/index')
def index():
	return render_template("index.html",
        title = 'Home', user = { 'nickname': 'Shane' },
        )

@webapp.route('/db')
def cities_page():
	with db: 
		cur = db.cursor()
		cur.execute("SELECT Name FROM City LIMIT 15;")
		query_results = cur.fetchall()
	cities = ""
	for result in query_results:
		cities += result[0]
		cities += "<br>"
	return cities


@webapp.route("/db_fancy")
def cities_page_fancy():
	with db:
		cur = db.cursor()
		cur.execute("SELECT Name, CountryCode, \
			Population FROM City ORDER BY Population LIMIT 15;")

		query_results = cur.fetchall()
	cities = []
	for result in query_results:
		cities.append(dict(name=result[0], country=result[1], population=result[2]))
	return render_template('cities.html', cities=cities)

@webapp.route('/input')
def station_input():
  return render_template("input.html")

@webapp.route('/output')
def station_output():
  #pull 'ID' from input field and store it
  longitude = request.args.get('ID1')
  latitude = request.args.get('ID2')


  #with db:
  #  cur = db.cursor()
  #  #just select the city from the world_innodb that the user inputs
  #  cur.execute("SELECT * FROM BostonFeaturesByStation_tb;")
  #  query_results = cur.fetchall()

  #stations = []
  #for result in query_results:
  #  stations.append(dict(name=result[0], country=result[1], population=result[2]))
#call a function from predictride package. note we are only pulling one result in the query
  #pop_input = cities[0]['population']
  the_results = predict(longitude, latitude)
  riderate = the_results[0]
  ranking = the_results[1]
  return render_template("output.html", riderate=riderate, ranking=ranking)

