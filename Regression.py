import matplotlib.pyplot as plt
import csv
import numpy as np

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


class Regression:
    '''
    Classe de la regression linéaire

    On fait ici une regresison de type U(X) = a*X + b

    '''

    def __init__(self, csv_file, ordre=1, N=1000) -> None:
        self.csv_file = csv_file; self.N = N
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

    def plotData(self):
        '''
        Fonction pour crée le graphe, on affiche la regression, la bande de confiance et les incertitudes de chaques points
        
        '''
        
        plt.plot(self.X_fit, self.U_fit, label="Regression linéaire")

        dX, dY = [], []
        for i in range(len(self.X)):
            dX.append(self.DX[i]/np.sqrt(3)); dY.append(self.DU[i]/np.sqrt(3)) 
        plt.errorbar(self.X, self.U, xerr = np.array(dX), yerr = np.array(dY), fmt = 'r+', zorder = 2, label = 'Mesures')

        if self.ordre == 1:
            U_pente = np.linspace(self.scal[0] - self.U_pente, self.scal[0] + self.U_pente, 100)
            U_ordonnee = np.linspace(self.scal[1] - self.U_ordonee, self.scal[1] + self.U_ordonee, 100)
            plt.fill_between(self.X_fit, U_pente[0]*self.X_fit + U_ordonnee[0], U_pente[-1]*self.X_fit + U_ordonnee[-1], alpha=0.2, label="Bande de confiance")

        plt.xlabel("Valeurs de X")
        plt.ylabel("Valeurs de U(X)")
        plt.title(f"Regression linéaire : U(X) = {' + '.join(f'{round(self.scal[self.ordre - i], 4)}*X^{i}' for i in range(0, self.ordre + 1))}")
        plt.axis([min(self.X), 1.1*max(self.X), min(self.U), 1.1*max(self.U)])

        plt.legend()
        plt.grid()
        plt.show()
