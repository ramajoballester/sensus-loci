import pickle

# Abre el archivo .pkl y carga los datos en una variable
with open("/home/javier/prueba/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.pkl", "rb") as f:
    data = pickle.load(f)

# Imprime los datos para visualizar su contenido
print(data[0])
