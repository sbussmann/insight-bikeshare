import folium
map_2 = folium.Map(location=[45.5236, -122.6750], tiles='Stamen Toner',
                  zoom_start=13)
map_2.simple_marker(location=[45.5244, -122.6699], popup='The Waterfront')
map_2.circle_marker(location=[45.5215, -122.6261], radius=500,
                   popup='Laurelhurst Park', line_color='#3186cc',
                   fill_color='#3186cc')
map_2.create_map(path='portland.html')

map_osm = folium.Map(location=[45.5236, -122.6750])
map_osm.create_map(path='osm.html')
