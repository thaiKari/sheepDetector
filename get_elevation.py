# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:42:44 2019

@author: karim
"""

import requests
import xmltodict

    
def get_elevation_for_lat_lon(lat, lon):
    
    params = {
        'request': 'Execute',
        'service': 'WPS',
        'version': '1.0.0',
        'identifier': 'elevation',
        'datainputs': 'lat=%f;lon=%f;epsg=4326' % (lat, lon),
    }
    
    response = requests.get("http://openwps.statkart.no/skwms1/wps.elevation2", params=params)
    print(response)
    response_dict = xmltodict.parse(response.content)
    request_outputs = (response_dict['wps:ExecuteResponse']
                    ['wps:ProcessOutputs']
                    ['wps:Output'])
    elevation = (next(item for item in request_outputs if item['ows:Identifier'] =='elevation')
                    ['wps:Data']
                    ['wps:LiteralData']
                    ['#text'])
                     

    return float(elevation)


