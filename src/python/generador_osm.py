import osmnx as ox
import matplotlib.pyplot as plt

def probar_osmnx_laguna(lugar, distancia):
    
    print(f"Descargando red peatonal a {distancia}m de {lugar}...")
    
    # 2. Descargar el grafo. network_type='walk' porque los vecinos van caminando a la basura
    grafo = ox.graph_from_address(lugar, dist=distancia, network_type='walk')
    
    # 3. Extraer los Nodos y las Aristas en formato GeoDataFrame (tablas de datos espaciales)
    nodos, aristas = ox.graph_to_gdfs(grafo)
    
    print(f"¡Descarga completada!")
    print(f"Nodos encontrados (posibles esquinas J): {len(nodos)}")
    print(f"Calles encontradas (aristas): {len(aristas)}")
    
    # 4. Mostrar información del primer nodo (para que veas qué datos tiene)
    print("\nEjemplo de datos de un nodo:")
    print(nodos.iloc[0])
    
    # 5. Visualizar el grafo de forma rápida
    print("\nGenerando visualización...")
    # 5. Guardar la visualización como imagen en lugar de abrir ventana
    print("\nGuardando mapa en 'mapa_laguna.png'...")
    fig, ax = ox.plot_graph(
        grafo, 
        node_color='red', 
        node_size=15, 
        edge_color='gray', 
        show=False, 
        save=True, 
        filepath='mapa_laguna.png'
    )
    print("¡Mapa guardado! Ábrelo desde tu explorador de archivos de Windows.")

def obtener_edificios_demanda(lugar, distancia):
    print(f"\nDescargando edificios a {distancia}m de {lugar}...")
    
    # Descargar las geometrías que tengan la etiqueta 'building'
    # Nota: en versiones muy nuevas de osmnx se usa features_from_address
    try:
        edificios = ox.features_from_address(lugar, tags={'building': True}, dist=distancia)
    except AttributeError:
        # Por si tienes una versión un poco anterior de osmnx
        edificios = ox.geometries_from_address(lugar, tags={'building': True}, dist=distancia)
    
    # Limpiar datos: nos quedamos solo con los polígonos válidos
    edificios = edificios[edificios.geometry.type == 'Polygon'].copy()
    
    # Calcular el centroide (el punto central del edificio)
    edificios['centroide'] = edificios.geometry.centroid
    
    # Estimar población (h_i) basada en el área del edificio
    # Reproyectamos a un sistema métrico local (UTM) para medir el área en metros cuadrados
    edificios_utm = ox.project_gdf(edificios)
    edificios['area_m2'] = edificios_utm.geometry.area
    
    # Asumimos muy por encima: 1 persona por cada 30m2 construidos
    edificios['habitantes'] = (edificios['area_m2'] / 30).astype(int)
    # Aseguramos que al menos haya 1 persona si el edificio es muy pequeño
    edificios.loc[edificios['habitantes'] < 1, 'habitantes'] = 1 
    
    print(f"¡Edificios descargados! Total puntos de demanda (I): {len(edificios)}")
    print(f"Población total estimada en la zona: {edificios['habitantes'].sum()} habitantes")
    
    return edificios

if __name__ == "__main__":
    lugar = "Plaza del Cristo, San Cristóbal de La Laguna, España"
    distancia = 500
    
    # 1. Obtener la red de calles (Esquinas J)
    # (Comenta la llamada original y pon esto si prefieres separarlo)
    probar_osmnx_laguna(lugar, distancia)
    
    # 2. Obtener los edificios (Demanda I)
    edificios = obtener_edificios_demanda(lugar, distancia)
    
    print("\nEjemplo de un punto de demanda (I):")
    print(edificios[['centroide', 'area_m2', 'habitantes']].head(1))