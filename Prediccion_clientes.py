#!/usr/bin/env python
# coding: utf-8

# # Modelo de predicción a nivel cliente

# **LECTURA DE LIBRERIAS**

# In[18]:


# Librerías
import pyodbc
import optuna
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, silhouette_score
import plotly.express as px


print("Cambio")

# Filtrando las advertencias
warnings.filterwarnings("ignore")


# ## 1) Definición de parametros 

# In[2]:


fecha_inicio = "2021-07-18" # Esto es año, mes y día 
fecha_final = "2023-11-19"
filtro = ["D004"]#, "D002", "D003", "D004"]
umbral_z: float = 1.8
umbral_visitas_min: int = 65


# Tengo que ver que hace estos parametros 
optimizar: bool = False
n_trials: int = 5
val_dist: float = 0.1
val_porc_prods: int = 100
extra_litros: float = 4
output_mult = 1 + extra_litros / 100
umb_redondeo: float = 0.5
ruta_max: int = 100

filtro = ','.join(f"'{id_f}'" for id_f in filtro)
filtro


# **ESQUEMAS DE CLIENTES**

# In[3]:


esquemas_df = {
        'Nombre': {0: 'D02-Autoservicios', 1: 'D03-Clientes de contado', 2: 'D06-Mayoristas', 3: 'D10-Empleados',
                   4: 'D12-Ganaderos', 5: 'D13-Gobierno', 6: 'D17-Tiendas de conveniencia', 7: 'D18-Vending',
                   8: 'D19-Clientes', 9: 'D20-Intercompañias(filiales)', 10: 'D21-Tradicional',
                   11: 'D22-Institucionales', 12: 'D24-Otros (Provisional)', 13: 'D27-Horeca', 14: 'D28-E-Commerce',
                   15: 'D29-Expendio', 16: 'D30-Maquila', 17: 'D31-Club de Precios', 18: 'K01-Proveedor Ganadero',
                   19: 'K05-Acreedores Empleados'},
        'Id': {0: 'D02', 1: 'D03', 2: 'D06', 3: 'D10', 4: 'D12', 5: 'D13', 6: 'D17', 7: 'D18', 8: 'D19', 9: 'D20',
               10: 'D21', 11: 'D22', 12: 'D24', 13: 'D27', 14: 'D28', 15: 'D29', 16: 'D30', 17: 'D31', 18: 'K01',
               19: 'K05'}}
esquemas_df = pd.DataFrame(esquemas_df)
esquemas = list(esquemas_df["Id"])
esquemas = ",".join(f"'{id_f}'" for id_f in esquemas)
esquemas_df


# **PIEZAS POR CAJA**

# In[6]:


conversion_cajas = {
    'Material': {0: 6, 1: 9, 2: 12, 3: 14, 4: 17, 5: 20, 6: 21, 7: 22, 8: 23, 9: 24, 10: 25, 11: 26, 12: 27, 13: 30,
                 14: 37, 15: 53, 16: 54, 17: 55, 18: 56, 19: 58, 20: 60, 21: 61, 22: 62, 23: 63, 24: 64, 25: 65,
                 26: 66, 27: 68, 28: 71, 29: 72, 30: 75, 31: 80, 32: 81, 33: 82, 34: 83, 35: 84, 36: 85, 37: 86,
                 38: 97, 39: 100, 40: 101, 41: 102, 42: 103, 43: 104, 44: 117, 45: 132, 46: 133, 47: 134, 48: 135,
                 49: 149, 50: 155, 51: 166, 52: 173, 53: 179, 54: 180, 55: 182, 56: 183, 57: 187, 58: 188, 59: 192,
                 60: 193, 61: 194, 62: 195, 63: 198, 64: 203, 65: 204, 66: 205, 67: 206, 68: 207, 69: 217, 70: 218,
                 71: 219,72: 220, 73: 221, 74: 224, 75: 244, 76: 260, 77: 266, 78: 267, 79: 270, 80: 271, 81: 272,
                 82: 273, 83: 274, 84: 276, 85: 277, 86: 282, 87: 286, 88: 287, 89: 290, 90: 291, 91: 293, 92: 294,
                 93: 295, 94: 297, 95: 326, 96: 333, 97: 340, 98: 341, 99: 342, 100: 343, 101: 346, 102: 347,
                 103: 351, 104: 354, 105: 360, 106: 368, 107: 376, 108: 378, 109: 379, 110: 383, 111: 384, 112: 385,
                 113: 386, 114: 387,115: 412, 116: 415, 117: 417, 118: 418, 119: 419, 120: 420, 121: 421, 122: 422,
                 123: 423, 124: 486, 125: 487, 126: 488, 127: 515, 128: 519, 129: 520, 130: 522, 131: 539, 132: 540,
                 133: 541, 134: 542, 135: 543, 136: 545, 137: 546, 138: 580, 139: 583, 140: 584, 141: 585, 142: 586,
                 143: 591, 144: 594, 145: 605, 146: 607, 147: 608, 148: 609, 149: 610, 150: 611, 151: 612, 152: 613,
                 153: 614, 154: 619, 155: 620, 156: 621, 157: 622, 158: 623, 159: 624, 160: 625, 161: 626, 162: 627,
                 163: 628, 164: 629, 165: 632, 166: 633, 167: 635, 168: 638, 169: 640, 170: 653, 171: 655, 172: 656,
                 173: 683, 174: 685, 175: 686, 176: 687, 177: 688, 178: 689, 179: 690, 180: 691, 181: 693, 182: 695,
                 183: 705, 184: 706, 185: 707, 186: 730, 187: 735, 188: 736, 189: 738, 190: 741, 191: 750, 192: 825,
                 193: 826, 194: 827, 195: 828, 196: 829, 197: 830, 198: 831, 199: 833, 200: 850, 201: 851, 202: 852,
                 203: 853, 204: 854, 205: 855, 206: 857, 207: 859, 208: 860, 209: 861, 210: 862, 211: 863, 212: 864,
                 213: 865, 214: 866, 215: 867, 216: 868, 217: 869, 218: 870, 219: 871, 220: 872, 221: 873, 222: 874,
                 223: 881, 224: 882, 225: 883, 226: 884, 227: 885, 228: 886, 229: 887, 230: 888, 231: 889, 232: 890,
                 233: 900, 234: 918, 235: 931, 236: 933, 237: 959, 238: 960, 239: 961},

    'Unidad de Carga': {
        0: 16.0, 1: 24.0, 2: 24.0, 3: 25.0, 4: 25.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 25.0,
        10: 50.0, 11: 25.0, 12: 25.0, 13: 0.0, 14: 4.0, 15: 25.0, 16: 25.0, 17: 25.0, 18: 16.0, 19: 4.0,
        20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 25.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0, 30: 0.0,
        31: 9.0, 32: 9.0, 33: 9.0, 34: 25.0, 35: 25.0, 36: 25.0, 37: 25.0, 38: 30.0, 39: 0.0,
        40: 24.0, 41: 0.0, 42: 0.0, 43: 0.0, 44: 25.0, 45: 16.0, 46: 0.0, 47: 0.0, 48: 0.0, 49: 16.0,
        50: 12.0, 51: 0.0, 52: 24.0, 53: 9.0, 54: 9.0, 55: 9.0, 56: 9.0, 57: 24.0, 58: 15.0, 59: 72.0,
        60: 72.0, 61: 24.0, 62: 72.0, 63: 24.0, 64: 1.0, 65: 16.0, 66: 25.0, 67: 16.0, 68: 25.0, 69: 1.0,
        70: 1.0, 71: 1.0, 72: 1.0, 73: 1.0, 74: 4.0, 75: 1.0, 76: 36.0, 77: 25.0, 78: 36.0, 79: 42.0,
        80: 42.0, 81: 42.0, 82: 42.0, 83: 0.0, 84: 0.0, 85: 0.0, 86: 25.0, 87: 30.0, 88: 30.0, 89: 16.0,
        90: 12.0, 91: 0.0, 92: 0.0, 93: 0.0, 94: 16.0, 95: 0.0, 96: 16.0, 97: 0.0, 98: 9.0, 99: 0.0, 100: 9.0,
        101: 0.0, 102: 1.0, 103: 12.0, 104: 16.0, 105: 12.0, 106: 0.0, 107: 0.0, 108: 0.0, 109: 0.0,
        110: 42.0, 111: 42.0, 112: 42.0, 113: 0.0, 114: 9.0, 115: 15.0, 116: 9.0, 117: 9.0,
        118: 16.0, 119: 16.0, 120: 16.0, 121: 25.0, 122: 25.0, 123: 25.0, 124: 12.0, 125: 12.0, 126: 9.0,
        127: 0.0, 128: 0.0, 129: 0.0, 130: 12.0, 131: 0.0, 132: 0.0, 133: 100.0, 134: 100.0, 135: 100.0,
        136: 100.0, 137: 1.0, 138: 12.0, 139: 25.0, 140: 25.0, 141: 25.0, 142: 25.0, 143: 12.0,
        144: 1.0, 145: 4.0, 146: 9.0, 147: 9.0, 148: 16.0, 149: 16.0, 150: 16.0, 151: 25.0,
        152: 25.0, 153: 25.0, 154: 0.0, 155: 0.0, 156: 0.0, 157: 0.0, 158: 0.0, 159: 0.0, 160: 0.0, 161: 0.0,
        162: 0.0, 163: 0.0, 164: 0.0, 165: 0.0, 166: 0.0, 167: 0.0, 168: 0.0, 169: 30.0, 170: 30.0,
        171: 30.0, 172: 30.0, 173: 6.0, 174: 6.0, 175: 6.0, 176: 6.0, 177: 6.0, 178: 6.0, 179: 6.0,
        180: 6.0, 181: 12.0, 182: 16.0, 183: 36.0, 184: 6.0, 185: 6.0, 186: 12.0, 187: 12.0,
        188: 12.0, 189: 12.0, 190: 12.0, 191: 12.0, 192: 12.0, 193: 12.0, 194: 30.0, 195: 0.0, 196: 0.0,
        197: 12.0, 198: 12.0, 199: 6.0, 200: 6.0, 201: 6.0, 202: 6.0, 203: 6.0, 204: 0.0, 205: 6.0, 206: 6.0,
        207: 25.0, 208: 25.0, 209: 4.0, 210: 4.0, 211: 0.0, 212: 0.0, 213: 0.0, 214: 12.0,
        215: 12.0, 216: 12.0, 217: 12.0, 218: 12.0, 219: 12.0, 220: 12.0, 221: 12.0, 222: 12.0, 223: 12.0,
        224: 12.0, 225: 12.0, 226: 12.0, 227: 12.0, 228: 12.0, 229: 12.0, 230: 12.0, 231: 12.0,
        232: 12.0, 233: 0.0, 234: 120.0, 235: 16.0, 236: 9.0, 237: 24.0, 238: 24.0, 239: 24.0}}

conv_caja = pd.DataFrame(conversion_cajas)
conv_caja


# ## 2) Descarga de datos

# ### 2.1) Autentificación de la BD

# In[8]:


# Credenciales del servidor
server = "192.168.0.192"
database = "ASR"
usuarioDB = "consulta"
passwordDB = "Consult@"

# Generando la conexión
conexion = pyodbc.connect(f"DRIVER={{SQL SERVER}}; SERVER={server}; DATABASE={database}; UID={usuarioDB}; PWD={passwordDB}")
conexion


# ### 2.2) Alta de Queries 

# In[10]:


ventas_detalle_query = f"""
SELECT c.DiaClave as Fecha, d.AlmacenPadreid as UNE, d.AlmacenID as Ruta, c.ClienteClave, 
       CE.EsquemaID, TPD.ProductoClave, ROUND(TPD.Cantidad * lt.KgLts, 3) as KgLts_Total
FROM TransProd AS c
JOIN TransProdDetalle as TPD ON c.TransProdID = TPD.TransProdID
JOIN Almacen AS d ON c.MUsuarioID = d.AlmacenID
JOIN (select ClienteClave, EsquemaID
      from ClienteEsquema
      WHERE LEN(EsquemaID) = 3) as CE ON c.ClienteClave = CE.ClienteClave
JOIN ProductoUnidad as lt on (TPD.ProductoClave = lt.ProductoClave) and (TPD.TipoUnidad = lt.PruTipoUnidad)
WHERE c.Tipo = 1 AND c.TipoFaseIntSal = 1 AND TipoFase != 0 AND TipoMovimiento = 2
AND (DiaClave BETWEEN CONVERT(DATETIME, '{fecha_inicio}', 102) AND CONVERT(DATETIME, '{fecha_final}', 102))
AND d.AlmacenPadreid IN ({filtro})
AND CE.EsquemaID IN ({esquemas})
"""

query_secuencia = f"""
    select  C.ClienteClave, SEC.FrecuenciaClave, CE.EsquemaID
    From Cliente as C
    left join Secuencia as SEC ON C.ClienteClave = SEC.ClienteClave
    left join ClienteEsquema CE ON C.ClienteClave = CE.ClienteClave
    where C.TipoEstado = 1
    AND C.AlmacenID IN ({filtro})
    AND CE.EsquemaID IN ({esquemas})
"""

visitas_query = f"""
select V.ClienteClave, V.DiaClave, V.FechaHoraInicial, V.FechaHoraFinal
from Visita AS V
JOIN ClienteEsquema CE on V.ClienteClave = CE.ClienteClave
WHERE SUBSTRING(RUTClave, 1, 4) in ({filtro})
AND (DiaClave BETWEEN CONVERT(DATETIME, '{fecha_inicio}', 102) AND CONVERT(DATETIME, '{fecha_final}', 102))
AND EsquemaID IN ({esquemas})
"""

query_conv_prods = """
        Select pd.ProductoClave, pd.PRUTipoUnidad, pd.Factor, PU.KgLts
        from ProductoDetalle as pd
        join ProductoUnidad PU on pd.ProductoClave = PU.ProductoClave and pd.PRUTipoUnidad = PU.PRUTipoUnidad
"""

query_rutas = f"""
    Select DISTINCT AlmacenID
    from Secuencia AS c join Almacen AS d ON c.RUTClave = d.AlmacenID
    where d.AlmacenPadreid in ({filtro})
"""


# ## 3) Descarga de datos 

# ### 3.1) Rutas activas

# In[11]:


rutas_activas = pd.read_sql(sql=query_rutas, con=conexion)
rutas_activas["ID"] = rutas_activas["AlmacenID"].astype(str)
rutas_activas


# ### 3.2) Ventallas detalle

# In[12]:


ventas_detalle = pd.read_sql_query(sql=ventas_detalle_query, con=conexion, dtype_backend="pyarrow")
ventas_detalle


# ## 4) Análisis exploratorio de datos (AED) 

# ### 4.1) Distribución de venta de productos 

# In[21]:


# 1) Generamos la agrupación
distribucion_productos = ventas_detalle.groupby(by=["ProductoClave"])["KgLts_Total"].sum().reset_index()

# 2) Creamos una pie chart 
fig = px.pie(data_frame=distribucion_productos, values='KgLts_Total', names='ProductoClave', title='Distribución de demanda de productos')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# ### 4.2) Tendencia de la venta a traves del tiempo 
# Para esto tendremos que crear un par de columnas nuevas, ya que quiero la tendencia a nivel semana-año 
# 

# In[25]:


get_ipython().run_cell_magic('time', '', 'ventas_detalle["Fecha"] = pd.to_datetime(ventas_detalle["Fecha"], dayfirst=True)\nventas_detalle["Mes"] = ventas_detalle["Fecha"].dt.month\nventas_detalle["Year"] = ventas_detalle["Fecha"].dt.year\nventas_detalle["Dia"] = ventas_detalle["Fecha"].dt.weekday\nventas_detalle["Semana"] = ventas_detalle["Fecha"].dt.isocalendar().week\nventas_detalle\n')


# In[28]:


ventas_semana  = ventas_detalle.groupby(by=["Semana", "Year"])["KgLts_Total"].sum().reset_index()
ventas_semana.sort_values(by=["Year", "Semana"], inplace=True)
ventas_semana.reset_index(drop=True, inplace=True)
ventas_semana


# In[43]:


# 1) Función para la generación de una fecha concreta dado la seman y año  
def fecha(year:int, week:int):
    
    # 1) Primer día del año 
    primer_dia = datetime.date(year, 1, 1)

    # 2) A este día le sumo el num_semanas * 7 días
    dias_sumar = week * 7 
    nueva_fecha = primer_dia + timedelta(days=dias_sumar)
 
    return nueva_fecha

# 2) aplicamos la funcion con la función apply 
ventas_semana['Fecha'] = ventas_semana.apply(lambda row: fecha(row['Year'], row['Semana']), axis=1)
ventas_semana


# In[44]:


# Crear la gráfica con Plotly
fig = px.line(data_frame=ventas_semana, x='Fecha', y='KgLts_Total', color='Year', 
              labels={'semana_del_año': 'Semana del Año', 'ventas': 'Ventas Totales'},
              title='Ventas Totales por Semana')
fig.show() # Podemos ver una tendencia muy negatia, se ha reducico casi un 50 % las ventas en los ultimos 2 años


# ### 4.3) Distribución de ventas por cliente a nivel semana 

# In[48]:


ventas_semana_cliente = ventas_detalle.groupby(by=["ClienteClave", "Semana", "Year"])["KgLts_Total"].sum().reset_index()
ventas_semana_cliente.sort_values(by=["Year", "Semana"], inplace=True)
ventas_semana_cliente.reset_index(drop=True, inplace=True)
ventas_semana_cliente


# In[50]:


fig = px.histogram(data_frame=ventas_semana_cliente, x="KgLts_Total", title="Histograma de litros por semana/cliente")
fig.show()


# **DESCRIPCION ESTADISTICA SENCILLA**

# In[51]:


ventas_semana_cliente["KgLts_Total"].describe()


# In[ ]:


######### AQUI VAMOS, hacer una reflxiíon de esto.   SCAR TAMBIEN una variabilidad, desviacion estandar por categoria , que este dato lo traiga cada cliente tambien. 


# In[ ]:





# In[ ]:





# In[45]:


ventas_detalle.head()


# ### 4.4) Distribución de ventas por categorias a nivel semana 

# In[ ]:


fig = px.line(data_frame=, x='date', y="GOOG")
fig.show()


# In[22]:


# Using plotly.express
import plotly.express as px

df = px.data.stocks()


# In[24]:


df


# In[68]:


# Obteniendo el número de semanas que hay con base en las fechas seleccionadas
num_semanas = ((ventas_detalle["Fecha"].max() - ventas_detalle["Fecha"].min()) // 7).days

# Generamos los esquemas de los clientes que tenemos descargados.
esquemas_clientes = ventas_detalle[["ClienteClave", "EsquemaID"]].drop_duplicates("ClienteClave").reset_index(drop=True)


# In[72]:


visitas = pd.read_sql(visitas_query, conexion)
# Convertimos a dt
visitas["FechaHoraInicial"] = pd.to_datetime(visitas["FechaHoraInicial"])
visitas["FechaHoraFinal"] = pd.to_datetime(visitas["FechaHoraFinal"])
# Restamos tiempos
visitas["Visita"] = visitas["FechaHoraFinal"] - visitas["FechaHoraInicial"]
# Creamos datos dt
visitas = visitas[["ClienteClave", "DiaClave", "Visita"]]
visitas["DiaClave"] = pd.to_datetime(visitas["DiaClave"], dayfirst=True)
visitas["Mes"] = visitas["DiaClave"].dt.month
visitas["Year"] = visitas["DiaClave"].dt.year
visitas["Dia"] = visitas["DiaClave"].dt.weekday
visitas["Semana"] = visitas["DiaClave"].dt.isocalendar().week
# Convertimos a segundos
visitas["Visita_seg"] = visitas["Visita"].dt.total_seconds()
# Tiempos negativos los convertimos a 0
visitas.loc[visitas["Visita_seg"] <= 0,"Visita_seg"] = 0
# Filtramos algunos datos extremos
valor_filtro = visitas['Visita_seg'].quantile(0.95)
visitas = visitas[visitas["Visita_seg"] <= valor_filtro]
visitas.head()


# In[72]:


#### AQUI LA CORRIDA 





# In[73]:





# In[ ]:


# Obteniendo los datos en un DF
df_secuencia = pd.read_sql(query_secuencia, conexion, dtype_backend="pyarrow")
conv_prods = pd.read_sql(query_conv_prods, conexion, dtype_backend="pyarrow")
output_df = df_secuencia.copy()
# DF con las frecuencias agrupadas por clientes
df_secuencia = pd.DataFrame(df_secuencia.groupby(["ClienteClave"])["FrecuenciaClave"].unique()).reset_index()
#--------------------------------------------------------------------------------
# Obteniendo el número de visitas semanales
df_secuencia["Visitas_Semanales"] = df_secuencia["FrecuenciaClave"].apply(lambda x: len(x))
# Drop de la columna de FrecuenciaClave
df_secuencia.drop(columns=["FrecuenciaClave"], inplace=True)

#--------------------------------------------------------------------------------
# Multiplicando el número de Visitas_Semanales por el número de semanas para obtener el número de visitas totales
df_secuencia["Visitas_Totales"] = (df_secuencia["Visitas_Semanales"] * num_semanas)
# Generamos un df con los datos generales del cliente
datos_clientes = ventas_detalle[["ClienteClave", "Ruta", "UNE"]].drop_duplicates("ClienteClave")

# -------------------------------------------------------------------------------
frecuencia_df = pd.DataFrame(output_df["FrecuenciaClave"].drop_duplicates().sort_values(ascending=False).dropna()).reset_index(drop=True)
frecuencia_df["Dia"] = frecuencia_df.index
# Juntando la frecuencia
output_df = output_df.merge(frecuencia_df, on='FrecuenciaClave')
clientes_dia_visita = output_df.copy()
# DF con las frecuencias agrupadas por clientes
df_secuencia_clientes = pd.DataFrame(output_df.groupby(["ClienteClave"])["FrecuenciaClave"].unique()).reset_index()
#--------------------------------------------------------------------------------
# Obteniendo el número de visitas semanales
df_secuencia_clientes["Visitas_Semanales"] = df_secuencia_clientes["FrecuenciaClave"].apply(lambda x: len(x))
# Drop de la columna de FrecuenciaClave
df_secuencia_clientes = df_secuencia_clientes.drop(columns="FrecuenciaClave")
output_df = output_df.drop(columns="FrecuenciaClave")
# Juntamos con las visitas semanales
output_df = output_df.merge(df_secuencia_clientes, on="ClienteClave", how="left")


# ## Procesamiento y eliminación de outliers

# In[26]:


# Se heredan parámetros de fechas y df de la descarga de datos
df = ventas_detalle.copy()
visitas_semanales = df_secuencia.copy()
categorias_clientes = df[["ClienteClave", "EsquemaID"]].drop_duplicates()

semana_inicio = df['Semana'].min()
semana_fin = df['Semana'].max()

umbral_z_superior = 1.75
umbral_z_inferior = -1.55

# Función para eliminación de z-scores
def z_score(x):
    """
    Método para obtener el Z_Score de cada cliente
    :param x: Columna de KgLts_Total
    :return z_score: Z_Score de cada cliente
    """
    # Si la lóngitud de las filas del cliente es menor que 1 se le da un valor de Z de 10, se le da un valor a alto para que se filtre siempre
    if len(x) <= 1:
        return 10
    # Si la Desviación estándar de las filas = 0, se le da un valor de 0
    elif x.std() == 0:
        return 0
    # Cualquier otra cosa se le saca el Z_Score
    else:
        return (x - x.mean()) / x.std()

# Función para asignar categorías
def asignar_categoria(x):
    if x <= q1:
        return 'Bajo'
    elif q1 < x <= q2:
        return 'Medio Bajo'
    elif q2 < x <= q3:
        return 'Medio Alto'
    else:
        return 'Alto'


# In[27]:


# Generamos una variable de venta total
venta_dia = df.groupby(["ClienteClave", "Fecha"])["KgLts_Total"].sum().reset_index().rename(columns={"KgLts_Total": "Venta Total dia"})
df = df.merge(venta_dia, on=["ClienteClave", "Fecha"])
df.head()


# ### Filtro por quantiles del total por día.

# In[28]:


df[["Venta Total dia"]].boxplot()
plt.show()


# In[29]:


val_perc = df["Venta Total dia"].quantile(.95)
val_perc


# Aquí podemos ver que el 95% de los datos tienen una venta por debajo de los 282.96 litros y en la gráfica de bigotes podemos ver que hay muchísimo sesgo en la distribución con datos muy altos, vamos a eliminar ese 5% de datos atípicos.

# In[30]:


df = df[df["Venta Total dia"]<val_perc]
df.head()


# In[31]:


df[["Venta Total dia"]].boxplot()
plt.show()


# Seguimos con algo de sesgo, pero en mucho menor escala.

# ### Sacamos Z scores de venta por día por cliente.

# In[32]:


# Agregamos un nuevo filtro de Z scores por cliente por día
df['Valor_Z_general'] = df.groupby(["ClienteClave"])['Venta Total dia'].transform(z_score)
tam = len(df)
filt = df[df["Valor_Z_general"] <= umbral_z_superior]
filt = filt[filt["Valor_Z_general"] > umbral_z_inferior]
tam_fin = len(filt)

print(f"Se queda con un porcentaje de {np.round((tam_fin*100)/tam, 2)}% ")


# ### Sacamos Z scores por cliente por producto

# In[33]:


# Agregamos un nuevo filtro de Z scores por cliente por día
filt['Valor_Z_detalle'] = filt.groupby(["ClienteClave", "ProductoClave"])['KgLts_Total'].transform(z_score)
filt = filt[filt["Valor_Z_detalle"] <= umbral_z_superior]
filt = filt[filt["Valor_Z_detalle"] > umbral_z_inferior]
tam_fin = len(filt)

print(f"Se queda con un porcentaje de después del segundo filtro {np.round((tam_fin*100)/tam, 2)}% ")


# In[34]:


total_datos = len(filt) / len(ventas_detalle) * 100
print(f"En total se eliminó el {np.round(100 - total_datos, 2)}% de los datos")


# ## Generación del dataset para las predicciones.

# In[35]:


venta_dia = df.groupby(["ClienteClave", "Fecha"])["KgLts_Total"].sum().reset_index().rename(columns={"KgLts_Total": "Venta Total"})
venta_dia["Dia"] = venta_dia["Fecha"].dt.weekday
# Generamos la variable de media / mediana venta
venta_media = venta_dia.groupby(["ClienteClave", "Dia"])["Venta Total"].agg({"mean", "median"}).reset_index().rename(columns={"mean":"Venta media por dia", "median": "Venta mediana por dia"})
# Juntamos la media de venta con los datos filtrados, modo left para evitar nulos con clientes filtrados
data_producto = filt.merge(venta_media, on=["ClienteClave", "Dia"], how="left")
# Juntamos las visitas semanales
data_producto = data_producto.merge(visitas_semanales[["ClienteClave", "Visitas_Semanales"]], on="ClienteClave")
# Renombramos columnas
data_producto = data_producto.rename(columns={"KgLts_Total": "KgLts Producto"})
# Si lo queremos por día sin el producto, eliminamos duplicados y eliminamos ProductoClave y KgLts, la y sería "Venta Total Día"
dataset_diario = data_producto.drop_duplicates(subset=["ClienteClave", "Semana", "Dia"]).drop(columns=["ProductoClave", "KgLts Producto"]).reset_index(drop=True)
print(dataset_diario.shape)
dataset_diario.head()


# In[36]:


# Definimos los percentiles con respecto en la venta media
q1 = dataset_diario['Venta media por dia'].quantile(0.25)
q2 = dataset_diario['Venta media por dia'].quantile(0.5)
q3 = dataset_diario['Venta media por dia'].quantile(0.75)

print(f"El q1: {q1} \tEl q2: {q2} \tEl q3: {q3} ")

# Crear una nueva columna 'Categoria' basada en 'Venta Media'
dataset_diario['Categoria'] = dataset_diario['Venta Total dia'].apply(asignar_categoria)
dataset_diario.head()


# In[37]:


# Sacamos la moda de categoría por cliente por día
grupo_prediccion = dataset_diario.groupby(["ClienteClave", "Dia"])["Categoria"].agg(lambda x: x.mode().iloc[0]).reset_index()
# Eliminamos duplicados del dataset de predicciones con los datos que vamos a necesitar para predecir
datos_prediccion = dataset_diario[["ClienteClave", "Dia", "Venta media por dia"]].drop_duplicates().reset_index(drop=True)
# Juntamos los dfs anteriores
datos_prediccion = datos_prediccion.merge(grupo_prediccion, on=["ClienteClave", "Dia"])
# Juntamos con la frecuencia
output_df = output_df.merge(datos_prediccion, on=["ClienteClave", "Dia"], how="left")
# Eliminamos clientes sin datos históricos
df_eliminado = output_df[output_df["Categoria"].isna()]
# Filtramos columnas
df_eliminado = df_eliminado[["ClienteClave", "EsquemaID", "Visitas_Semanales"]].drop_duplicates(subset="ClienteClave")
# Seleccionamos los clientes que tienen datos nulos, para quitarlos de la generación por día
clientes_nan = list(output_df[output_df["Categoria"].isna()]["ClienteClave"].drop_duplicates())
# Filtramos datos nulos
output_df = output_df.dropna()
# Eliminamos clientes que puede que estén para las curvas = 
output_df = output_df[~output_df["ClienteClave"].isin(clientes_nan)]
# Agregamos el mes y semana de predicción
output_df["Mes"] = 11
output_df["Semana"] = 46
output_df.reset_index(drop=True, inplace=True)
output_df.head()


# In[38]:


# Definimos las variables
variables_categoricas = ["EsquemaID", "Semana", "Mes", "Dia", "Categoria"]
variables_numericas = ["Visitas_Semanales", "Venta media por dia"]

# Dividimos los datos en numéricos y categóricos.
features_numericos = dataset_diario[variables_numericas]
features_categoricos = dataset_diario[variables_categoricas]
# Creamos el objeto encoder
encoder = OneHotEncoder()
# Aplicamos el encoder con las variables categóricas
encoded_cols = encoder.fit_transform(features_categoricos)
# Convertimos a dataframe las variables encoded
data_encoded = pd.DataFrame(data=encoded_cols.toarray(), columns=encoder.get_feature_names_out())
# Juntamos las variables encoded con las variables numéricas
X = pd.concat(objs=[data_encoded, features_numericos], axis=1)
# Definimos la variable objetivo
y = dataset_diario[['Venta Total dia']]

# Repetimos el proceso con los datos de predicción
features_numericos = output_df[variables_numericas]
features_categoricos = output_df[variables_categoricas]
# Utilizamos el encoder para los datos de predicción
encoded = encoder.transform(features_categoricos)
# Convertimos a df las variables encoded
data_encoded_pred = pd.DataFrame(data=encoded.toarray(), columns=encoder.get_feature_names_out())
# Generamos un df con los clientes ya codificados para hacer predicciones.
predict_dataset = pd.concat(objs=[data_encoded_pred, features_numericos], axis=1, ignore_index=True)

# Separamos los datos en entrenamiento, prueba y validación, convertimos a numpy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
X_train = X_train.astype('float').to_numpy()
X_val = X_val.astype('float').to_numpy()
X_test = X_test.astype('float').to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()


# In[39]:


print(f"Tamaño de la muestra de entrenamiento X: {X_train.shape}, y: {y_train.shape}")
print(f"Tamaño de la muestra de validación X: {X_val.shape}, y: {y_val.shape}")
print(f"Tamaño de la muestra de prueba X: {X_test.shape}, y: {y_test.shape}")


# In[40]:


# Definimos el modelo que vamos a utilizar
XGB = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=10,
                   objective='reg:squarederror', n_jobs=-1)
# Entrenamos
XGB.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

# Prediciendo con el modelo
preds_xgb = XGB.predict(X_test)

# Métricas del Modelo
metrics_xgb = pd.DataFrame({'R2': [r2_score(y_test, preds_xgb)],
                           'MAE': [mean_absolute_error(y_test, preds_xgb)],
                           'RMSE': [np.sqrt(mean_squared_error(y_test, preds_xgb))]})

# Prediciendo con el Modelo
preds_xgb_complete = XGB.predict(predict_dataset.to_numpy())
# Agregamos la predicción a los datos originales
output_df['prediccion'] = preds_xgb_complete
metrics_xgb


# In[41]:


visitas_semanales_cat = dataset_diario.groupby(['EsquemaID'])['Visitas_Semanales'].unique().reset_index()
visitas_semanales_cat = visitas_semanales_cat.explode('Visitas_Semanales').sort_values(['EsquemaID', 'Visitas_Semanales']).reset_index(drop=True)
# Crear una lista para almacenar los DataFrames
dfs = []
# Iterar sobre cada fila
for idx, row in visitas_semanales_cat.iterrows():
    # Crear un DataFrame temporal para cada fila
    temp_df = pd.DataFrame({
        'EsquemaID': [row['EsquemaID']] * ((semana_fin + 1) - semana_inicio),
        'Visitas_Semanales': [row['Visitas_Semanales']] * ((semana_fin + 1) - semana_inicio),
        'Semana': range(semana_inicio, semana_fin + 1)
    })
    # Agregar el DataFrame temporal a la lista
    dfs.append(temp_df)
# Concatenar todos los DataFrames
visitas_semanales_cat = pd.concat(dfs).reset_index(drop=True)
# Obteniendo el número de visitas semanales por categoría
media_semana_esq = dataset_diario.copy()
# Obteniendo una fila por visitas semanales de cada categoría
media_semana_esq = media_semana_esq.groupby(['EsquemaID', 'Semana', 'Visitas_Semanales'])['Venta Total dia'].mean().reset_index().sort_values(['EsquemaID', 'Semana', 'Visitas_Semanales'])
# Merge de los datos
media_semana_esq = media_semana_esq.merge(visitas_semanales_cat, on=['EsquemaID', 'Semana', 'Visitas_Semanales'], how='right')

# Llenando los nulos
df_nuevo = pd.DataFrame()
for i in media_semana_esq['EsquemaID'].unique():
    df_temp = media_semana_esq[media_semana_esq['EsquemaID'] == i].copy()
    df_temp['Venta Total dia'] = df_temp['Venta Total dia'].fillna(df_temp['Venta Total dia'].mean())
    df_nuevo = pd.concat([df_nuevo, df_temp], ignore_index=True)

# Se saca una columna con los valores de Z por cada cliente
df_nuevo['Valor_Z'] = df_nuevo.groupby(['EsquemaID', 'Visitas_Semanales'])['Venta Total dia'].transform(z_score)
 # -----------------------------------------------------------------------------
# Si el z-score es mayor a 1 se pone un NaN en la columna de KgLts_Total
df_nuevo['Venta Total dia'] = np.where(df_nuevo['Valor_Z'] >= 1, np.nan, df_nuevo['Venta Total dia'])
# -----------------------------------------------------------------------------
# Rellenando la columna de KgLts_Total con la media del valor anterior y el valor siguiente
df_nuevo['Venta Total dia'] = df_nuevo.groupby(['EsquemaID', 'Visitas_Semanales'])['Venta Total dia'].transform(lambda x: x.fillna(x.interpolate()))
# Rellenando los nulos con la media del valor siguiente
df_nuevo['Venta Total dia'] = df_nuevo.groupby(['EsquemaID', 'Visitas_Semanales'])['Venta Total dia'].transform(lambda x: x.fillna(x.bfill()))
# Redondeando la columna a 3 decimales
df_nuevo['prediccion'] = np.round(df_nuevo['Venta Total dia'], decimals=3)
# -----------------------------------------------------------------------------
# Eliminando la columna de Valor_Z
df_nuevo.drop(columns=['Valor_Z'], inplace=True)
media_semana_esq = df_nuevo.copy()
# Obteniendo los KgLts por semana de cada categoría con base en las curvas de demanda generadas
clientes_curvas = df_eliminado.merge(media_semana_esq, on=['EsquemaID', 'Visitas_Semanales'])
clientes_curvas


# In[42]:


# Nos quedamos con la semana de interés
clientes_curvas = clientes_curvas[clientes_curvas["Semana"] == 46].reset_index(drop=True)
# Filtramos las visitas por cliente solo por los clientes con las curvas
curvas_dia = clientes_dia_visita[clientes_dia_visita["ClienteClave"].isin(list(set(clientes_curvas["ClienteClave"])))].reset_index(drop=True)
# Filtramos columnas
curvas_dia = curvas_dia[["ClienteClave", "Dia"]]
# Juntamos con las curvas
curvas_dia = curvas_dia.merge(clientes_curvas, on="ClienteClave")
# Ajustamos las curvas 
curvas_dia["prediccion"] = curvas_dia["prediccion"] / curvas_dia["Visitas_Semanales"]
# Ordenamos columnas para concatenar
curvas_dia = curvas_dia[["ClienteClave", "EsquemaID", "Semana", "Dia", "Visitas_Semanales", "prediccion"]]
# Lo mismo en el output
output_df = output_df[["ClienteClave", "EsquemaID", "Semana", "Dia", "Visitas_Semanales", "prediccion"]]
# Concatenamos los dfs
output_df = pd.concat([output_df, curvas_dia], ignore_index=True)
output_df


# In[43]:


output_df["prediccion"] = output_df["prediccion"] * 1.06


# In[44]:


output_df


# In[45]:


semanal = output_df.merge(datos_clientes, on="ClienteClave")
semanal = semanal.groupby(["ClienteClave", "UNE", "Ruta", "EsquemaID", "Semana", "Visitas_Semanales"])["prediccion"].sum().reset_index()
semanal


# ## Distribución histórica

# In[46]:


dist_dia_prods = df.groupby(["ClienteClave", "Dia", "ProductoClave"])["KgLts_Total"].sum().reset_index()
# Sacamos la suma por ruta por día total
total_unidades = dist_dia_prods.groupby("ClienteClave")["KgLts_Total"].sum().reset_index()
# Hacemos un rename para poder juntar los 2 dfs
total_unidades = total_unidades.rename(columns={"KgLts_Total": "Total_semanal"})
# Juntamos el total con la distribución de días.
dist_dia_prods = dist_dia_prods.merge(total_unidades, on="ClienteClave")
# Generamos un diccionario con los días de la semana para poder hacer un cambio de nombres.
dias_dict = {0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves", 4: "Viernes", 5: "Sabado"}
# Sacamos el porcentaje por producto a nivel semanal.
dist_dia_prods["Porcentaje"] = (dist_dia_prods["KgLts_Total"] * 100) / dist_dia_prods["Total_semanal"]
dist_porc_prods = dist_dia_prods.groupby(["ClienteClave", "ProductoClave"])["Porcentaje"].sum().reset_index()
# Hacemos un pivot para tener la Ruta, Producto por días de la semana.
dist_dia_prods = dist_dia_prods.pivot(index=["ClienteClave", "ProductoClave"],
                                      columns='Dia', values='Porcentaje').reset_index().fillna(0)
# Hacemos un rename para que esté por días de la semana
dist_dia_prods = dist_dia_prods.rename(columns=dias_dict)
dist_dia_prods = dist_dia_prods.merge(dist_porc_prods, on=["ClienteClave", "ProductoClave"])
# Definimos una lista con los días de la semana
dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"]
for i in dias:
    # Con una regla de 3 sacamos el total por día de la semana
    dist_dia_prods[i] = (dist_dia_prods[i] / dist_dia_prods["Porcentaje"])
dist_dia_prods


# ## Kmeans

# In[47]:


dist_kmeans = pd.DataFrame()
for une in sorted(set(df["UNE"])):
    df_kmeans = df[df["UNE"] == une] 
    # Obteniendo los z-values por cliente de cada fila
    df_kmeans['Valor_Z'] = df_kmeans.groupby(['ClienteClave'])['KgLts_Total'].transform(z_score)
    # Filtrando los Valor_Z que sean mayor a 2
    df_kmeans = df_kmeans[df_kmeans['Valor_Z'] <= 2]
    # Eliminando la columna de Valor_Z
    df_kmeans.drop(columns=['Valor_Z'], inplace=True)
    # Df con la suma de las ventas por cliente, producto y fecha
    prods_clientes_diario = df_kmeans.groupby(by=["ClienteClave", "Year", "Semana", "ProductoClave"])['KgLts_Total'].sum().reset_index()
    # Df con la mediana de las ventas por cliente, producto diaria
    mediana_clientes = prods_clientes_diario.groupby(by=["ClienteClave", "ProductoClave"])['KgLts_Total'].median().reset_index()
    # Obteniendo el total de ventas por cliente
    total_clientes = mediana_clientes.groupby("ClienteClave")['KgLts_Total'].sum().reset_index()
    # Renombrando la columna de KgLts_Total
    total_clientes.rename(columns={'KgLts_Total': "Total"}, inplace=True)
    # Merge de los dos métodos
    dist_clientes = mediana_clientes.merge(total_clientes, on='ClienteClave')
    # Obteniendo el porcentaje de ventas por cliente
    dist_clientes["Porcentaje"] = (dist_clientes['KgLts_Total'] / dist_clientes["Total"]) * 100
    # Drop de duplicados
    dist_clientes.drop_duplicates(subset=['ClienteClave', 'ProductoClave'], inplace=True)
    # Pivot Df
    dist_clientes = pd.pivot_table(data=dist_clientes, index='ClienteClave', columns='ProductoClave', values='Porcentaje', fill_value=0)
    # Reset los índices
    dist_clientes.reset_index(drop=False, inplace=True)
    # Quitando el nombre del índice
    dist_clientes.columns.name = None
    X = dist_clientes.drop(columns=['ClienteClave'])
    # Creando las variables que necesitamos
    clusters = []
    silueta = []
    # Iterando sobre el número de clusters
    for i in range(2, 10):
        # Creando el modelo de KMeans
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # Entrenando y prediciendo con el modelo
        cluster_labels = kmeans.fit_predict(X)
        # Obteniendo la silueta
        silhouette_avg = silhouette_score(X, cluster_labels)
        # Agregando los valores a las listas
        clusters.append(i)
        silueta.append(silhouette_avg)
    # Creando un Df con los valores
    df_silueta = pd.DataFrame({'Clusters': clusters, 'Silueta': silueta})
    # Ordenando el Df
    df_silueta = df_silueta.sort_values(by='Silueta', ascending=False).reset_index(drop=True)
    # Obteniendo el número óptimo de clusters
    num_cluster = df_silueta['Clusters'].iloc[0]
    # Haciendo el número entero
    num_cluster = int(num_cluster)
    # Creando el modelo de KMeans
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Entrenando el modelo
    kmeans.fit(X)
    # Diccionario con los nombres de los productos
    clave = pd.DataFrame(kmeans.cluster_centers_).columns
    valor = X.columns
    # Diccionario con los nombres de los productos
    dic_cluster = {}
    # Iterando sobre los nombres de los productos
    for i in range(len(clave)):
        # Agregando los nombres al diccionario
        dic_cluster[i] = valor[i]
    # Haciendo una copia del Df de distribución de ventas por cliente
    distribucion_clientes = dist_clientes.copy()
    # Obteniendo los clusters
    distribucion_clientes["Cluster"] = kmeans.predict(X)
    # Quitamos la columna CC para agrupar
    dist_clusters = distribucion_clientes.set_index("ClienteClave")
    # Sacamos la media por producto por grupo
    dist_clusters = dist_clusters.groupby("Cluster").mean().reset_index()
    # Creamos un df con el CC y el Grupo al que pertenece
    distribucion_clientes_centroides = distribucion_clientes[['ClienteClave', 'Cluster']]
    # Juntamos los datos del cliente con la distribución de la media de los productos
    result_df = distribucion_clientes_centroides.merge(dist_clusters, on='Cluster').drop(columns=['Cluster'])
    # Actualizamos la lista de los productos con los nuevos nombres.
    productos = [i for i in result_df.columns if i != "ClienteClave"]
    
    # Convertimos todos los productos a una sola columna
    distribucion_productos_ruta = pd.melt(result_df, id_vars="ClienteClave", value_vars=productos, 
                                          var_name="ProductoClave", 
                                          value_name="Porcentaje_kmeans").sort_values(by=["ClienteClave", 
                                                                                          "ProductoClave"]).reset_index(drop=True)
    # Concatenamos todas las iteraciones en un df general.
    dist_kmeans = pd.concat([dist_kmeans, distribucion_productos_ruta])
dist_kmeans


# ## Combinamos distribuciones históricas y las de recomendación (kmeans)

# In[48]:


# Juntamos las distribuciones historicas y de recomendación
dist_sem_prods_ruta = dist_dia_prods.merge(dist_kmeans, on=["ClienteClave", "ProductoClave"])
# Aplicamos el parámetro del filtro para ver a que se le da más prioridad, 0 → historico, 1 → recomendación, valores en medio una combinación 
dist_sem_prods_ruta["Porcentaje_ponderado"] = ((1 - val_dist) * dist_sem_prods_ruta["Porcentaje"] + val_dist * dist_sem_prods_ruta["Porcentaje_kmeans"])
# Eliminamos columnas que ya no nos interesan
dist_sem_prods_ruta = dist_sem_prods_ruta.drop(columns=["Porcentaje", "Porcentaje_kmeans"])
# Renombramos variable final
dist_sem_prods_ruta = dist_sem_prods_ruta.rename(columns={"Porcentaje_ponderado": "Porcentaje"})
# Concatenamos posibles clientes eliminados con el merge de kmeans con histórico.
cc = list(set(dist_dia_prods["ClienteClave"]) - set(dist_sem_prods_ruta["ClienteClave"]))
filt = dist_dia_prods[dist_dia_prods["ClienteClave"].isin(cc)].reset_index(drop=True)
dist_sem_prods_ruta = pd.concat([dist_sem_prods_ruta, filt], ignore_index=True)
# Ordenamos por CC y porcentaje
dist_sem_prods_ruta = dist_sem_prods_ruta.sort_values(by=["ClienteClave", "Porcentaje"], ascending=False)
# Realizamos una agregación sobre el porcentaje para aplicar el filtro de porcentaje de productos
dist_sem_prods_ruta["Agg"] = dist_sem_prods_ruta.groupby("ClienteClave")["Porcentaje"].cumsum()
# Generamos una copia para no eliminar clientes con porcentajes altos y agregarlos al final
copia = dist_sem_prods_ruta.copy()
# Filtramos productos por arriba del porcentaje de venta representativo
dist_sem_prods_ruta = dist_sem_prods_ruta[dist_sem_prods_ruta["Agg"]<val_porc_prods].reset_index(drop=True).drop(columns=["Agg"])
# Seleccionamos clientes eliminados
cc = (list(set(copia.ClienteClave) - set(dist_sem_prods_ruta.ClienteClave)))
# Filtramos por estos clientes y eliminamos duplicados por cliente para quedarnos únicamente con el producto representativo
fuera = copia[copia["ClienteClave"].isin(cc)].drop_duplicates("ClienteClave").drop(columns=["Agg"]).reset_index(drop=True)
# Convertimos a 1 el porcentaje, ya que solo quedará ese producto
fuera["Porcentaje"] = 1
# Generamos un df con el total del porcentaje por cliente para reajustar los porcentajes que quedaron posterior al filtro
total_reajuste = dist_sem_prods_ruta.groupby("ClienteClave")["Porcentaje"].sum().reset_index().rename(columns={"Porcentaje": "Total"})
# Juntamos dfs
dist_sem_prods_ruta = dist_sem_prods_ruta.merge(total_reajuste, on="ClienteClave")
# Reajustamos los porcentajes
dist_sem_prods_ruta["Porcentaje"] = ((dist_sem_prods_ruta["Porcentaje"] * 100) / dist_sem_prods_ruta["Total"]) / 100
# Eliminamos la variable del total
dist_sem_prods_ruta = dist_sem_prods_ruta.drop(columns="Total")
# Concatenamos los resultados
dist_sem_prods_ruta = pd.concat([dist_sem_prods_ruta, fuera], ignore_index=True)
cc_1 = list(output_df["ClienteClave"].unique())
# Filtramos por clientes activos
dist_sem_prods_ruta = dist_sem_prods_ruta[dist_sem_prods_ruta["ClienteClave"].isin(cc_1)]
dist_sem_prods_ruta.head()


# ## Juntamos la predicción con las distribuciones

# In[49]:


# Juntamos la predicción con la distribución
df_final = semanal.merge(dist_sem_prods_ruta, on="ClienteClave", how="left")
# Multiplicamos la predicción con la distribución por producto
df_final["Porcentaje"] = df_final["prediccion"] * df_final["Porcentaje"]

# Multiplicamos por cada día de la semana.
for dia in dias:
    df_final[dia] = df_final[dia] * df_final["Porcentaje"]
# Eliminamos columna que no nos sirve.
df_final = df_final.drop(columns=["prediccion"])
df_final


# In[50]:


umb_redondeo: float = 0.3


# In[51]:


# Nos quedamos con los que tengan PRUTipoUnidad = 1 (piezas)
piezas_litros = conv_prods.sort_values(by="PRUTipoUnidad", ascending=True).drop_duplicates(
    subset="ProductoClave").sort_index().reset_index(drop=True)
# Eliminamos columnas que ya no nos sirven
piezas_litros.drop(["PRUTipoUnidad", "Factor"], axis=1, inplace=True)
# Hacemos un melt a nuestra tabla para tener como una sola columna los días.
prods = pd.melt(df_final, id_vars=[ "UNE", "Ruta","ClienteClave", "ProductoClave"],
                             value_vars=["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"],
                             var_name="Dia", value_name="Total")
# Juntamos con los litros por pieza
prods = prods.merge(piezas_litros, on="ProductoClave")
# Generamos las piezas con los litros predichos / los KgLts por pieza
prods["Piezas"] = prods["Total"] / prods["KgLts"]

# Definimos una función para el redondeo
def custom_round(x, umbral):
    return np.where(x - np.floor(x) >= umbral, np.ceil(x), np.floor(x))

prods


# In[52]:


# Aplicamos el redondeo personalizado
prods["Piezas"] = np.round(prods["Piezas"])#, umbral=0.5).astype(int)
prods


# In[53]:


prods["Piezas"].sum()


# In[56]:


prods["Piezas_redondeo"].sum()


# In[57]:


# Aplicamos el redondeo personalizado
prods["Piezas"] = np.ceil(prods["Piezas"])#, umbral=0.5).astype(int)
# Sacamos los litros dadas las piezas
prods["Litros"] = prods["Piezas"] * prods["KgLts"]
# Filtramos días que no se visita el cliente
prods = prods[prods["Piezas"]>0]
# Ordenamos por UNE, Ruta, CC
prods = prods.sort_values(by=["UNE", "Ruta", "ClienteClave", "ProductoClave"]).reset_index(drop=True)
# Hacemos una agrupación para tener por Ruta las rejas 
agg_rutas = prods.groupby(["UNE", "Ruta", "ProductoClave", "Dia"])[["Piezas", "Litros"]].sum().reset_index()
# Juntamos con los litros por pieza
agg_rutas = agg_rutas.merge(piezas_litros, on="ProductoClave")
agg_rutas.rename(columns={"ProductoClave":"Material"}, inplace=True)
# Lo convertimos a entero para que sea solo el número sin ceros ej. 001 ⇾ 1
agg_rutas["Material"] = agg_rutas["Material"].astype(int)
agg_rutas = agg_rutas[agg_rutas["Piezas"]>0]
# Juntamos con el equivalente de rejas
agg_rutas = agg_rutas.merge(conv_caja, on="Material", how="inner")
# Sacamos el número de rejas que hay por cada unidad,
agg_rutas["Rejas"] = agg_rutas["Piezas"] / agg_rutas["Unidad de Carga"]
# Redondeamos
agg_rutas["Rejas"] = custom_round(agg_rutas["Rejas"], umb_redondeo)
# Sacamos cuantas piezas hay en las rejas
agg_rutas["Piezas en rejas"] = agg_rutas["Rejas"] * agg_rutas["Unidad de Carga"]
# Sacamos los litros totales ya con las rejas
agg_rutas["Litros en rejas"] = agg_rutas["KgLts"] * agg_rutas["Piezas en rejas"]
# Eliminamos columnas que ya no nos sirven
prods = prods.drop(columns=["Total", "KgLts"])
agg_rutas = agg_rutas.drop(columns=["KgLts", "Unidad de Carga"])
agg_rutas = agg_rutas[agg_rutas["Litros en rejas"]>0]
agg_rutas.head()


# In[62]:


agg_rutas["Litros en rejas"].sum()


# In[61]:


agg_rutas.to_excel("prueba_clientes_2.xlsx", index=False)

