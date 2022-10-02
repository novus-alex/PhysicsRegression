import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import csv
import numpy as np
from random import gauss


class Tools:
    '''
    Outils pour la creation de la regression linéaire

    '''

    @staticmethod
    def isNum(a):
        '''
        Fonction pour detecter si une chaîne de charactère est un nombre
        
        '''
        
        try:
            float(a)
        except ValueError:
            return False
        return True

    @staticmethod
    def plotStyle():
        tdir = 'in'
        major = 5.0
        minor = 2.5
        plt.rcParams['xtick.direction'] = tdir
        plt.rcParams['ytick.direction'] = tdir
        plt.rcParams['xtick.major.size'] = major
        plt.rcParams['xtick.minor.size'] = minor
        plt.rcParams['ytick.major.size'] = major
        plt.rcParams['ytick.minor.size'] = minor


class Regression:
    '''
    Classe de la regression linéaire

    On fait ici une regresison de type U(X) = a*X + b

    '''

    def __init__(self, csv_file, ordre=1, N=1000) -> None:
        self.csv_file = csv_file; self.N = N
        Tools.plotStyle()
        self.X, self.DX, self.U, self.DU = self.getData()
        self.X_fit, self.U_fit = self.regression(ordre)
        self.ordre = ordre
        self.U_pente, self.U_ordonee = self.regression_Monte_Carlo(self.X, self.U, np.mean(self.DX), np.mean(self.DU))

    def getData(self):
        '''
        Format CSV pour la regresion

        X / DX / U(X) / DU(X)
        . / .. / .... / .....

        '''

        X_temp, U_temp = [], []
        DX_temp, DU_temp = [], []
        with open(self.csv_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                if Tools.isNum(row[0]):
                    X_temp.append(float(row[0])); U_temp.append(float(row[2]))
                    DX_temp.append(float(row[1])); DU_temp.append(float(row[3]))
        return np.array(X_temp), np.array(DX_temp), np.array(U_temp), np.array(DU_temp)

    def regression_Monte_Carlo(self, x: np.array, y: np.array, D_x: float, D_y: float):
        '''
        Regression lineaire Monte-Carlo, pour obtenir l'incertitude sur la pente et l'ordonnée à l'origine

        '''

        liste_pente, liste_ordonnee = [], []
        for i in range(self.N):
            l = len(x)
            mx = x + D_x * np.random.uniform(-1, 1, l)
            my = y + D_y * np.random.uniform(-1, 1, l)

            p = np.polyfit (mx, my, 1)
            liste_pente.append(p[0])
            liste_ordonnee.append(p[1])

        u_pente = np.std(liste_pente, ddof = 1)
        u_ordonnee = np.std(liste_ordonnee, ddof = 1)

        return u_pente, u_ordonnee

    def regression(self, ordre=1):
        '''
        Regression linéaire, elle renvoie un modèle du type U(X) = a*X + b

        '''

        self.scal = np.polyfit(self.X, self.U, ordre)
        x_fit = np.linspace(min(self.X), max(self.X), 100)
        U_fit = np.array([self.scal[-1] for _ in x_fit])
        for d in range(1, ordre + 1):
            U_fit += self.scal[ordre-d]*x_fit**d
        
        return x_fit, U_fit

    def confidenceBand(self):
        '''
        Fonction pour calculer la bande de confiance de la regression
        
        '''
        
        Aalea, Balea = [], []
        for i in range(len(self.X_fit)):
            Xalea = [gauss(self.X[_], self.DX[_]) for _ in range(len(self.X))]
            Yalea = [gauss(self.U[_], self.DU[_]) for _ in range(len(self.U))]
            a, b = np.polyfit(Xalea, Yalea, 1)
            Aalea.append(a); Balea.append(b)
        M, m = [], []
        for i in self.X_fit:
            M.append(max([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))
            m.append(min([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))
        return m, M

    def plotData(self):
        '''
        Fonction pour crée le graphe, on affiche la regression, la bande de confiance et les incertitudes de chaques points
        
        '''
        
        plt.plot(self.X_fit, self.U_fit, "r", lw=1, label="Regression linéaire")

        dX, dY = [], []
        for i in range(len(self.X)):
            dX.append(self.DX[i]/np.sqrt(3)); dY.append(self.DU[i]/np.sqrt(3)) 
        plt.errorbar(self.X, self.U, xerr = np.array(dX), yerr = np.array(dY), fmt="o", color="k", lw=1, ms=3, capsize=3, zorder = 2, label = 'Mesures')

        if self.ordre == 1:
            m, M = self.confidenceBand()
            plt.fill_between(self.X_fit, m, M, alpha=0.2, label="Bande de confiance", color="r")
            plt.title(f"Regression linéaire : U(X) = ({round(self.scal[0], 3)} ± {round(self.U_pente, 3)})*X + ({round(self.scal[1], 3)} ± {round(self.U_ordonee, 3)})")
        else:
            plt.title(f"Regression linéaire : U(X) = {' + '.join(f'{round(self.scal[self.ordre - i], 4)}*X^{i}' for i in range(0, self.ordre + 1))}")

        plt.xlabel("X")
        plt.ylabel("U(X)")
        plt.axis([1.1*min(self.X), 1.1*max(self.X), 1.1*min(self.U), 1.1*max(self.U)])
        plt.legend()
        plt.grid()
        plt.show()

    def plotFullData(self):
        '''
        Fonction pour crée le graphe, on affiche la regression, la bande de confiance et les incertitudes de chaques points
        
        '''

        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        
        ax[0].plot(self.X_fit, self.U_fit, "r", lw=1, label="Regression linéaire")

        dX, dY = [], []
        for i in range(len(self.X)):
            dX.append(self.DX[i]/np.sqrt(3)); dY.append(self.DU[i]/np.sqrt(3)) 
        ax[0].errorbar(self.X, self.U, xerr = np.array(dX), yerr = np.array(dY), fmt="o", color="k", lw=1, ms=3, capsize=3, zorder = 2, label = 'Mesures')

        if self.ordre == 1:
            m, M = self.confidenceBand()
            ax[0].fill_between(self.X_fit, m, M, alpha=0.2, label="Bande de confiance", color="r")
            ax[0].set_title(f"Regression linéaire : U(X) = ({round(self.scal[0], 3)} ± {round(self.U_pente, 3)})*X + ({round(self.scal[1], 3)} ± {round(self.U_ordonee, 3)})")
        else:
            ax[0].set_title(f"Regression linéaire : U(X) = {' + '.join(f'{round(self.scal[self.ordre - i], 4)}*X^{i}' for i in range(0, self.ordre + 1))}")

        ax[0].set_xlabel("X")
        ax[0].set_ylabel("U(X)")
        ax[0].axis([1.1*min(self.X), 1.1*max(self.X), 1.1*min(self.U), 1.1*max(self.U)])
        #ax[0].xaxis.set_visible(False)
        ax[0].legend()
        ax[0].xaxis.set_minor_locator(MultipleLocator(abs(self.X[0]-self.X[1])/2))
        ax[0].yaxis.set_minor_locator(MultipleLocator(abs(self.U[0]-self.U[1])/2))
        ax[0].grid()

        U_fit_X = np.array([self.scal[-1] for _ in self.X])
        for d in range(1, self.ordre + 1):
            U_fit_X += self.scal[self.ordre-d]*self.X**d

        ax[1].plot(self.X, [0 for _ in self.X], "--k", lw=1)
        ax[1].xaxis.set_minor_locator(MultipleLocator(abs(self.X[0]-self.X[1])/2))
        ax[1].yaxis.set_minor_locator(MultipleLocator(abs(self.U[0]-self.U_fit[0] - self.U[1] + U_fit_X[1])/2))
        ax[1].errorbar(self.X, abs(self.U - U_fit_X), xerr = 0, yerr = np.array(dY), fmt="o", color="k", lw=1, ms=3, capsize=3, zorder = 2, label = 'Mesures')
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Uexp(X)-U(X)")
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def __str__(self) -> str:
        if self.ordre == 1:
            return f"Regression linéaire : U(X) = ({round(self.scal[0], 3)} ± {round(self.U_pente, 3)})*X + ({round(self.scal[1], 3)} ± {round(self.U_ordonee, 3)})"
        return f"Regression linéaire : U(X) = {' + '.join(f'({round(self.scal[self.ordre - i], 6)})*X^{i}' for i in range(0, self.ordre + 1))}"
