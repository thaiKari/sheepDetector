# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:49:16 2019

@author: karim
"""

from labelbox import Client
import numpy as np
import json

api_key = open("lblbox_API_key_private.txt", "r").readline()
#api_key = open("lblbox_API_key.txt", "r").readline()
client = Client(api_key)


def generate_query_for_row_id( project_id, external_id):
    
    return '''\
      {\
        projects(where:{id:"%s"}) {\
          dataRows(where:{externalId:"%s"}){\
            id\
            externalId\
          }\
        }\
      }\
      
    '''%( project_id , external_id)


def get_row_id( project_id, external_id):
    query = generate_query_for_row_id(project_id, external_id)
    data = client.execute(query)
    print('data', data)
    return data['data']['projects'][0]['dataRows'][0]['id']


def get_label(row_id):
    data = client.execute('''
      {
        label(where:{id:"%s"}){
          label
          dataRow{
            externalId
          }
        }
      }
      
    '''%( row_id))

    return data['data']['label']
 
#    
#def update_label(row_id, label_str):    
#    data = client.execute('''
#      mutation {
#        updateLabel(
#            where:{
#              id:"%s"
#            }
#            data: {
#              label:"%s"
#            }
#        ) {
#          id
#          label
#        }
#      }
#      
#    '''%( row_id, label_str))
#        
#    print('''
#      mutation {
#        updateLabel(
#            where:{
#              id:"%s"
#            }
#            data: {
#              label:"%s"
#            }
#        ) {
#          id
#          label
#        }
#      }
#      
#    '''%( row_id, label_str))
#    
#    print(data)
#
#lbl = LABEL['label'].replace("\"", "\\\"")  
#update_label("ck1lrrwo0k73q0744glcdjok3",lbl)
#data3 = get_label("ck1lrrwo0k73q0744glcdjok3")


#data1 = get_label("")
        
    
    


def label_dict_to_json(label_dict):
    
    def get_geometry(geom):
        minx, miny = geom[0]
        maxx, maxy = geom[1]
        return[{
                "x": int(minx),
                "y": int(miny)
                },
                {
                "x": int(maxx),
                "y": int(miny)
                },
                {
                "x": int(maxx),
                "y": int(maxy)
                },
                {
                "x": int(minx),
                "y": int(maxy)
                }
            ]
    
    def get_label(label):
        if('labels' in label):
            if 'sheep_color' in label['labels'][0].keys():
                return {"Sheep": list(map( lambda l: {"sheep_color": l['sheep_color'],
                                  "geometry":get_geometry(l['geometry'])}  , label['labels']))}
            else:
                return {"Sheep": list(map( lambda l: {"sheep_color": 'White',
                                  "geometry":get_geometry(l['geometry'])}  , label['labels']))}
            
        
        
        else: return 'Skip'
        
        
    data = list(map( lambda k: {"Label": get_label(label_dict[k]), "External ID": k } , label_dict.keys()))    
    return data

def create_label(label, project_id, data_row_id):
    res_str = client.execute("""
    mutation CreateLabelFromApi($label: String!, $projectId: ID!, $dataRowId: ID!){
      createLabel(data:{
        label:$label,
        secondsToLabel:0,
        project:{
          connect:{
            id:$projectId
          }
        }
        dataRow:{
          connect:{
            id:$dataRowId
          }
        }
        type:{
          connect:{
            name:"Any"
          }
        }
      }){
      id
      }
    }
    """, {
        'label': label,
        'projectId': project_id,
        'dataRowId': data_row_id
    })
            
    print('LABEL UPLOAD RES:')
    print(res_str)
    print('=======================')