"""

Este archivo se encarga de generar una instancia del problema.


"""

import json
import os
from datetime import datetime
from typing import Any
from dataclasses import dataclass

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point