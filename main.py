#!/home/vmagda/.anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os

from plot import *



# -----------------------------------------------------------------------------------------------------------
# données
# -----------------------------------------------------------------------------------------------------------
part = Particles('bulle30.h5')
print('# ----------------------------------------------------')
print('#', part.dir)
print('# ----------------------------------------------------')



# -----------------------------------------------------------------------------------------------------------
# Infos
# -----------------------------------------------------------------------------------------------------------
part.info_simu()



# -----------------------------------------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------------------------------------
# Quantité de mouvement
fig, ax = part.plot_mvt_quantity()
# Erreur relative à l'équilibre
fig, ax = part.plot_relative()
# Force de pression et force de tension de surface cote à cote
fig, ax = part.plot_ts_forces()
# graphe de courbure
fig, ax = part.plot_curvature()
# Évolution temporelle de la pression
fig, ax = part.plot_P()






# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# plt.show()
