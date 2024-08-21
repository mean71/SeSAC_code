import json 

from data_structure.graph_datastructure import Vertex, Edge
from ADT.graph import Graph 
# graph example - subway map
class Station(Vertex):
    def __init__(self, station_name, **data):
        super().__init__(station_name, data)
        self.station_name = station_name 
        self.data = data 

    def __eq__(self, other):
        if isinstance(other, Station):
            return self.station_name == other.station_name
        return False 

    def __hash__(self):
        return hash(self.station_name) 

    def __str__(self):
        return str(self.station_name)

class StationEdge(Edge):
    def __init__(self, from_station, to_station, line, distance, time):
        super().__init__(from_station, to_station, is_directed = False, distance = distance, time = time)
        self.from_station = from_station
        self.to_station = to_station 
        self.distance = distance
        self.time = time
        self.line = line 

class SubwayMap(Graph):
    def __init__(self, stations_json = 'resources/vertices.json', 
                    station_edges_json = 'resources/edges.json'):
        with open(stations_json, 'r', encoding = 'utf-8') as stations:
            stations = json.load(stations)
            stations = [Station(s['station_nm'], **s) for s in \
                            stations['DATA']]

            self.stations = stations 
            station_dict = {}

            for station in stations:
                station_dict[station.station_name] = station 
        
        with open(station_edges_json, 'r', encoding = 'utf-8') as station_edges:
            station_edges = json.load(station_edges) 
            edges = []
            for line, station_edges in station_edges.items():
                for s in station_edges:
                    from_station = station_dict[s['from']]
                    to_station = station_dict[s['to']]
                    edges.append(
                        StationEdge(from_station, to_station, line, s['distance'], s['time'])
                    )
            self.station_edges = edges 
        super().__init__(stations, edges)

    def find_route(self, src):
        pass 

if __name__ == '__main__':
    s = SubwayMap()
    assert isinstance(s, Graph)
    s.show() # it takes very long time, with bad result. 


