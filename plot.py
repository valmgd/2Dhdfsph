import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
import h5py



def head_tail(vec) :
    chaine = '[ ' + str(vec[0]) + '   ' + str(vec[1]) + '   ' + str(vec[2])
    chaine += '   ...   '
    chaine += str(vec[-3]) + '   ' + str(vec[-2]) + '   ' + str(vec[-1]) + ' ]'

    return(chaine)
#}

def set_graph_style(fig, ax) :
    ax.set_aspect('equal', 'datalim')
    ax.set_facecolor('lavender')

    ax.legend(loc='best', fontsize='x-large', fancybox=True, framealpha=0.5)
    ax.grid(color='gray', linestyle='--')

    fig.tight_layout()
#}

def norm(x, y) :
    return(np.sqrt(x**2 + y**2))
#}



class Particles :
    """Classe Particles.
    Coordonnées, variables du système de N-S, tension de surface."""

    # -------------------------------------------------------------------------------------------------------
    # constructeur
    # -------------------------------------------------------------------------------------------------------
    def __init__(self, h5file) :
        # ---------------------------------------------------------------------------------------------------
        # pré-traitement
        # ---------------------------------------------------------------------------------------------------
        temps = 't' + sys.argv[1]
        mesh = 'I' + sys.argv[2] + 'dx' + sys.argv[3]

        # chemin d'accès aux données
        self.dir = temps + '/' + mesh
        os.chdir('/home/vmagda/Data/sph/' + self.dir)
        self.graphs = '/home/vmagda/Data/sph/graphs'
        # suffixe pour noms des graphes
        self.suf = temps + mesh

        # liste des fichier h5
        self.h5 = glob.glob('*.h5')
        # nombre de fichier h5
        self.nh5 = len(self.h5)
        # nom du fichier h5 demandé par l'utilisateur
        self.user_h5 = h5file

        # ensemble des données du fichier h5
        self.data = h5py.File(self.user_h5, 'r')
        # données concernant le fluide
        self.fluid = self.data['Fluid#0']

        # graphic parameters
        self.point_size = 5
        self.scale = 0.005
        self.scale2 = 0.05
        self.png = False

        # ---------------------------------------------------------------------------------------------------
        # lecture des données
        # ---------------------------------------------------------------------------------------------------
        # coordonnées des particules
        self.x = np.array(self.fluid['X'])
        self.y = np.array(self.fluid['Y'])
        self.n = len(self.x)
        # volume
        self.w = np.array(self.fluid['Volume'])
        # masse volumique
        # vitesse
        self.vx = np.array(self.fluid['VX'])
        self.vy = np.array(self.fluid['VY'])
        # pression
        self.P = np.array(self.fluid['P'])

        # courbure
        self.kappa = np.array(self.fluid['Curvature'])
        # quantité de mouvement
        self.mvx = np.array(self.fluid['mvx'])
        self.mvy = np.array(self.fluid['mvy'])

        # tension de surface
        self.FTSx = np.array(self.fluid['FTSx'])
        self.FTSy = np.array(self.fluid['FTSy'])
        # tension de surface volumique
        self.wFTSx = self.w * self.FTSx
        self.wFTSy = self.w * self.FTSy

        # gradient de pression volumique
        self.wGRPx = np.array(self.fluid['wGRPx'])
        self.wGRPy = np.array(self.fluid['wGRPy'])
        # gradient de pression
        self.GRPx = self.wGRPx / self.w
        self.GRPy = self.wGRPy / self.w

        self.rel = norm( self.mvx , self.mvy ) / norm( self.wGRPx , self.wGRPy )

        # évolution de la pression particule centrale
        file_name =  glob.glob('Solid_Kinematics*.csv')
        self.Pt = np.loadtxt(file_name[0], delimiter=',', skiprows=1)

        # ---------------------------------------------------------------------------------------------------
        # post-traitement
        # ---------------------------------------------------------------------------------------------------
        # indices des points de la couronne (indices pour lesquels la courbure est non nulle)
        self.ic = [i for i, elt in enumerate(self.kappa) if elt > 50]
    #}

    # -------------------------------------------------------------------------------------------------------
    # représentation
    # -------------------------------------------------------------------------------------------------------
    def __repr__(self) :
        chaine  = 'Read from files :'
        chaine += '\nx     : ' + head_tail(self.x)
        chaine += '\ny     : ' + head_tail(self.y)
        chaine += '\nmvx   : ' + head_tail(self.mvx)
        chaine += '\nmvy   : ' + head_tail(self.mvy)
        chaine += '\nP     : ' + head_tail(self.P)
        chaine += '\nvx    : ' + head_tail(self.vx)
        chaine += '\nvy    : ' + head_tail(self.vy)
        chaine += '\nFTSx  : ' + head_tail(self.FTSx)
        chaine += '\nFTSy  : ' + head_tail(self.FTSy)
        chaine += '\nwGRPx : ' + head_tail(self.wGRPx)
        chaine += '\nwGRPy : ' + head_tail(self.wGRPy)
        chaine += '\nw     : ' + head_tail(self.w)
        chaine += '\nkappa : ' + head_tail(self.kappa)

        return(chaine)
    #}

    # -------------------------------------------------------------------------------------------------------
    # save figure in pdf and png format
    # -------------------------------------------------------------------------------------------------------
    def save_figure(self, fig, fname) :
        fig.savefig(self.graphs + '/' + fname + '_' + self.suf + '.pdf')

        if self.png :
            fig.savefig(self.graphs + '/' + fname + '_' + self.suf + '.png', dpi=500)
        #}
    #}

    # -------------------------------------------------------------------------------------------------------
    # plot champs de vecteur : quantité de mvt
    # -------------------------------------------------------------------------------------------------------
    def plot_mvt_quantity(self) :
        fig, ax = plt.subplots()

        im = ax.scatter(self.x, self.y, s=self.point_size, c=self.P, cmap='jet', label='Particules')
        cax = fig.colorbar(im, ax=ax)
        cax.set_label('Pressure [Pa]')
        ax.quiver(self.x, self.y, self.mvx, self.mvy, angles='xy', scale=self.scale, label=r'$Dm_iu_i/Dt$', color='black')

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        t = 'Quantité de mouvement\n'
        t = t + r'$Dm_iu_i/Dt =-\omega_i \nabla P_i + \omega_i FTS_i$'
        ax.set_title(t)

        set_graph_style(fig, ax)
        self.save_figure(fig, 'dmv')

        return(fig, ax)
    #}

    # -------------------------------------------------------------------------------------------------------
    # affichage erreur relative par rapport à l'état d'équilibre
    # -------------------------------------------------------------------------------------------------------
    def plot_relative(self) :
        fig, ax = plt.subplots()
        # rel = norm( self.mvx , self.mvy ) / norm( self.wGRPx , self.wGRPy )
        im = ax.scatter(self.x, self.y, s=self.point_size, label='Particules', color='gray')
        im = ax.scatter(self.x[self.ic], self.y[self.ic], s=self.point_size, c=self.rel[self.ic], cmap='jet')
        cax = fig.colorbar(im, ax=ax)
        cax.set_label(r'$\|\|\omega \nabla P_i - \omega FTS\|\| \quad / \quad \|\|\omega \nabla P_i\|\|$')

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        t = 'Erreur relative\n'
        t = t + r'$\|\|\omega_i \nabla P_i - \omega_i FTS_i\|\| \quad / \quad \|\|\omega_i \nabla P_i\|\|$'
        ax.set_title(t)

        set_graph_style(fig, ax)
        self.save_figure(fig, 'rel')

        return(fig, ax)
    #}

    # -------------------------------------------------------------------------------------------------------
    # plot 2 champs de vecteurs : le gradient de pression et la tension de surface
    # -------------------------------------------------------------------------------------------------------
    def plot_ts_forces(self) :
        sc = 1
        r = 0.012

        fig, ax = plt.subplots(ncols=2)

        ax[0].scatter(self.x, self.y, s=self.point_size/4, label='Particules')
        ax[0].quiver(self.x, self.y, -self.wGRPx, -self.wGRPy, angles='xy', scale=self.scale2,
            label='$-\omega G^R_+(P)$', color='black')
        ax[0].set_xlabel('$x$ [m]')
        ax[0].set_ylabel('$y$ [m]')
        ax[0].set_title('Gradient de pression')
        ax[0].set_xlim((-r, r))
        set_graph_style(fig, ax[0])

        ax[1].scatter(self.x, self.y, s=self.point_size/4, label='Particules')
        ax[1].quiver(self.x, self.y, self.wFTSx, self.wFTSy, angles='xy', scale=self.scale2,
            label='$\omega FTS$', color='black')
        ax[1].set_xlabel('$x$ [m]')
        ax[1].set_ylabel('$y$ [m]')
        ax[1].set_title('Tension superficielle')
        ax[1].set_xlim((-r, r))

        set_graph_style(fig, ax[1])
        self.save_figure(fig, 'forces')

        return(fig, ax)
    #}

    # -------------------------------------------------------------------------------------------------------
    # plot champs de vecteur : quantité de mvt
    # -------------------------------------------------------------------------------------------------------
    def plot_curvature(self) :
        fig, ax = plt.subplots()
        im = ax.scatter(self.x, self.y, s=self.point_size, label='Particules', color='gray')
        im = ax.scatter(self.x[self.ic], self.y[self.ic], s=self.point_size, c=self.kappa[self.ic], cmap='jet')
        cax = fig.colorbar(im, ax=ax)
        cax.set_label('$\kappa$ [$m^{-1}$]')

        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('Courbure $\kappa$ à l\'interface')

        set_graph_style(fig, ax)
        self.save_figure(fig, 'curvature')

        return(fig, ax)
    #}

    # -------------------------------------------------------------------------------------------------------
    # plot évolution de la pression de la particule centrale
    # -------------------------------------------------------------------------------------------------------
    def plot_P(self) :
        if len(np.shape(self.Pt)) == 1 :
            return(0, 0)
        #}
        tf = self.Pt[-1, 0]
        fig, ax = plt.subplots()
        im = ax.plot(self.Pt[:, 0], self.Pt[:, 24], label='Pression', linewidth=0.25)

        ax.set_xlim((0., tf))
        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel('$P$ [Pa]')
        ax.set_title('Évolution de la pression')

        self.save_figure(fig, 'Pt')

        return(fig, ax)
    #}

    # -------------------------------------------------------------------------------------------------------
    # somme de la quantité de mouvement sur un quartier de la bulle
    # -------------------------------------------------------------------------------------------------------
    def quarter(self) :
        n = len(self.x)
        somme_x = somme_y = 0.
        for i in range(0, n) :
            if self.x[i] >= 0 and self.y[i] >= 0 :
                somme_x += self.mvx[i]
                somme_y += self.mvy[i]
            #}
        #}
        return(somme_x, somme_y)
    #}

    # -------------------------------------------------------------------------------------------------------
    # Infos sur la simulation
    # -------------------------------------------------------------------------------------------------------
    def info_simu(self) :
        # somme sur un quartier
        somme_x, somme_y = self.quarter()
        print('Somme Dmv/Dt sur un quartier : [', somme_x, ',', somme_y, ']')

        # intervalle de pression
        Pmin = str(min(self.P[self.ic]))
        Pmax = str(max(self.P[self.ic]))
        print('Intervalle de pression       : [', Pmin, ',', Pmax, ']')

        # Courbure attendue
        print('Courbure attendue            :', np.sqrt(np.pi / sum(self.w)))

        # table de courbure et erreur relative
        print(' _______________________________________')
        print('|         |         |         |         |')
        print('|         ', '|   min   ', '|  mean   ', '|   max   ', '|', sep='')
        print('|_________|_________|_________|_________|')
        print('|         |         |         |         |')
        print('| kappa   | %7.3f | %7.3f | %7.3f |' % (np.min(self.kappa[self.ic]), np.mean(self.kappa[self.ic]), np.max(self.kappa[self.ic])))
        print('|         |         |         |         |')
        print('| epsilon | %7.3f | %7.3f | %7.3f |' % (100*np.min(self.rel[self.ic]), 100*np.mean(self.rel[self.ic]), 100*np.max(self.rel[self.ic])))
        print('|_________|_________|_________|_________|')
    #}
#}
