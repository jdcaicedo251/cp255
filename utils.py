import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
import geoplot



def cleaning_stations(data):
    '''
    Input: 
    - data: row Pandas DataFrame with information about the stations
    
    Output: 
    - a geopandas dataframe with all relevant stations and geometry(x,y coordinate as points) 
    '''

    mask = data.recaudoestacion.isin(['', '0', '00000', '01234', '06112', '22000', '12345'])
    exclude_stations = tuple(data.recaudoestacion[mask])
    
#     #Grupping by recaudoestacion because is a better id for the estation
#     df = data[~mask].groupby('recaudoestacion')\
#                 [['nombreestacion','idlinea','latitud','longitud']].first().reset_index()
    
#     #There are 10 duplicated stations (ID is different, but station should be the same)
#     duplicated = {'04100':'04004', '57503':'07503', '50001':'04100','08100':'40000','40004':'40003',
#                   '50002':'09100', '50003':'09001', '50004':'04003', '50006':'04100', '50007':'07006',}
    
#     df.recaudoestacion = df.recaudoestacion.replace(duplicated)
    df = data[~mask].copy()
    
    missing_xy = {'(50001) Portal Eldorado[Intermedium]' : (4.681520, -74.121143),
              '(40000) Cable Portal Tunal': (4.568481,-74.139379),
              '(40001) Juan Pablo II': (4.555476,-74.147523),
              '(40002) Manitas': (4.550445,-74.150598),
              '(40003) Mirador del Paraiso': (4.550019,-74.159009),
              '(40004) Bicicletero Mirador del Paraíso': (4.550019,-74.159009),
              '(08100) Portal Tunal Cable': (4.568481,-74.139379),
              'Cable Portal Tunal(40000)' : (4.568481,-74.139379),
              'Juan Pablo II(40001)' : (4.555476,-74.147523),
              'Manitas(40002)' : (4.550445,-74.150598),
              'Mirador del Paraiso(40003)': (4.550019,-74.159009),
              '(14005) Las Aguas': (4.60255, -74.068687),
              'Ampliacion San Mateo(57503)': (4.589146,-74.199496),
              'Corral Molinos(50003)': (4.556805,-74.121705),
              'Corral Avenida Ciudad de Cali(50004)':(4.702865,-74.100733),
              'Corral Calle 40 Sur(50002)':(4.575937,-74.119233),
              'EL CAMPIN(07106)':(4.645663,-74.078697),
              'Corral General Santander(50007)':(4.593200,-74.128343),
              'Corral Carrera 77(50006)': (4.698383,-74.094176),
              'Centro Comercial Santa Fe(02001)':(4.763741, -74.044402),
              'Las Aguas(14005)':(4.60255, -74.068687),
              '(50007) Corral General Santander': (4.593200,-74.128343),
              '(07106) EL CAMPIN':(4.645663,-74.078697),
              '(02001) Centro Comercial Santa Fe': (4.763741, -74.044402),
              '(50003) Corral Molinos':(4.556805,-74.121705),
              '(50004) Corral Avenida Ciudad de Cali':(4.702865,-74.100733),
              '(50002) Corral Calle 40 Sur':(4.575937,-74.119233),
              '(57503) Ampliacion San Mateo':(4.589146,-74.199496),
              '(50006) Corral Carrera 77':(4.698383,-74.094176)}

    for key, value in missing_xy.items():
        df.loc[df[df.nombreestacion == key].index[0], ['latitud', 'longitud']] = value
        
    gdf = gpd.GeoDataFrame(df, 
                 geometry = gpd.points_from_xy(df.longitud, df.latitud))
    
    return gdf

def assign_strata(stations, blocks):
    '''
    Input:
    - stations: row Pandas DataFrame with information about the stations
    - blocks: shp file with strata information by block
    Output:
    - Stations geoPandas dataframe with an assigned strata
    '''
    stations = cleaning_stations(stations)
    
    #Creating buffer ~500m 
    stations['catchment_area'] = stations.geometry.buffer(0.005) # 0.001° = 111 m
    stations_buffer = gpd.GeoDataFrame(stations, geometry = 'catchment_area')
    
    # Spatial Join stations buffer and blocks
    estacion_estrata = gpd.sjoin(stations_buffer, blocks, how = 'left', op = 'intersects' )
    
    # Average strata by BRT station 
    strata = estacion_estrata.groupby('idestacion').agg({'recaudoestacion':'first',
                                                     'nombreestacion':'first',
                                                     'ESTRATO':'mean',
                                                     'latitud':'first',
                                                     'longitud':'first',
                                                     'geometry':'first'})
    
    # BRT stations outside the Bogota Metro Region are only located in Soacha, with a strata of 1. 
    strata.ESTRATO.fillna(1, inplace = True) # Replace nan values with strata 1
    strata.ESTRATO = strata.ESTRATO.round(decimals = 0).astype(int)
    
    return gpd.GeoDataFrame(strata, geometry = 'geometry')

def cleaning_transactions(transactions, stations):
    '''
    Creates a timestamp colum with transaction datetime information 
    Orders the table by user_id and transaction timestamp
    Merge with a station unique Id (col = recaudoestacion)
    
    Input:
    -data: row transaccion data 
    -stations: clean stations with ids and unique "recaudoestacion"
    
    Output
    -Clean and ordered transactions dataframe
    '''
    data = transactions.copy()
    
    #Creating dataetime column with dataetime format 
    data.horatransaccion = data.horatransaccion.astype(str)
    data.fechatransaccion = data.fechatransaccion.astype(str)
    data['sec'] = data.horatransaccion.apply(lambda x: x[-2:])
    data['minute'] = data.horatransaccion.apply(lambda x: x[-4:-2])
    data['hour'] = data.horatransaccion.apply(lambda x: x[:-4])

    data['hour'].replace('', '0', inplace = True)
    data['minute'].replace('', '0', inplace = True)
    data['year'] = data.fechatransaccion.apply(lambda x: x[:4])
    data['month'] = data.fechatransaccion.apply(lambda x: x[4:6])
    data['day'] = data.fechatransaccion.apply(lambda x: x[6:8])

    #Converting to a stand alone dataaframe for conversion
    df = pd.DataFrame({'year': data['year'],
                       'month': data['month'],
                       'day': data['day'],
                       'hour': data['hour'],
                       'minutes':data['minute'] ,
                       'seconds': data['sec']})

    data['datetime'] = pd.to_datetime(df)
    
    df = data.merge(stations, how = 'left', on='idestacion') #merging station unique ID
    df = df[~df.recaudoestacion.isnull()] # Remove observations with invalid station ID
    df.sort_values(['idnumerotarjeta', 'datetime'], inplace = True ) 
    
    cols_order = ['idnumerotarjeta', 'datetime', 'recaudoestacion', 'idtipotarjeta',
                  'idtipotarifa','saldopreviotransaccion', 'valor', 'saldodespues_transaccion']
    
    return df[cols_order]





