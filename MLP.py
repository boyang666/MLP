""" programme à compléter du MLP """
import csv
import math
import random

nbExParClasse = 50
nbExApprent = 25
nbNeurones = [4, 10, 3]
nbClasse = nbNeurones[len(nbNeurones) - 1]

nbApprent = 5000
coeffApprent = 0.01
coeffSigmoide = 2/3

poids = []
inner = []
out = []

def lectureFichierCSV():
    with open("iris.data", 'r')as fic:
        lines = csv.reader(fic)
        datalist = list(lines)
    dataset = []
    for i in range(len(datalist)):
        data = []
        for j in range(nbNeurones[0]):
            data.append(float(datalist[i][j]))
        dataset.append(data)
    return dataset


def fSigmoide(x):
    return math.tanh(coeffSigmoide * x)
def dfSigmoide(x):
    return coeffSigmoide / math.cosh(coeffSigmoide * x) ** 2


def propagation(data):
    global out, inner
    inner = []
    out = []
    inner.append(data)
    out.append(data)
    for n in range(len(nbNeurones) - 1):
        N = []
        S = []
        for i in range(nbNeurones[n+1]):
            N.append(0)
            for j in range(nbNeurones[n]):
                N[i] = N[i] + data[j] * poids[n][i][j]
            S.append(fSigmoide(N[i]))
        inner.append(N)
        out.append(S)
        data = S

    #print(inner)

def retropropagation(classe):
    y = []
    for i in range(nbClasse):
        y.append(-1)
    y[classe] = 1

    # calculer les erreurs
    erreurCouche = []
    for n in range(len(nbNeurones) - 1, -1, -1):
        if n == len(nbNeurones) - 1:
            erreur = []
            for j in range(nbNeurones[n]):
                erreur.append(dfSigmoide(inner[n][j]) * (out[n][j] - y[j]))
            erreurCouche.append(erreur)
        elif n != 0:
            erreur = []
            for j in range(nbNeurones[n]):
                somme = 0
                for k in range(nbNeurones[n+1]):
                    somme = somme + poids[n][k][j] * erreurCouche[len(erreurCouche) - 1][k]
                erreur.append(somme * dfSigmoide(inner[n][j]))
            erreurCouche.append(erreur)
    print(erreurCouche)

    # mise a jour les poids
    for n in range(len(nbNeurones) - 1):
        for i in range(nbNeurones[n + 1]):
            for j in range(nbNeurones[n]):
                poids[n][i][j] = poids[n][i][j] + (-1) * coeffApprent * erreurCouche[len(erreurCouche) - 1 - n][i] * out[n][j]

def apprentissage(dataset):
    list_indice = []
    for i in range(nbClasse):
        for j in range(nbExApprent):
            list_indice.append(j + i * nbExParClasse)

    for i in range(nbApprent):
        j = random.choice(list_indice)
        propagation(dataset[j])
        retropropagation(j // nbExParClasse)




def evaluation(dataset):
    # Calcule et affiche la matrice de confusion et le taux de reco
	# =========== à compléter avec matrice de confusion ===========
    ok = ko = 0
    for i in range(len(dataset)):
        if i % nbExParClasse >= nbExApprent:  # partie test de la base
            propagation(dataset[i])
            sortie = out[len(nbNeurones) - 1]
            c = sortie.index(max(sortie))
            if c == i // nbExParClasse:
                ok += 1
            else:
                ko += 1
    print(ok)
    print(ok+ko)
    print("Taux reco : {:.3f} %".format(ok / (ok + ko)))


def main():
    global poids
    print("Début programme MLP")
    dataset = lectureFichierCSV()

    # Allocation et initialisation aléatoire des poids entre -0,05 et +0,05
    random.seed(1)
    poids = []
    for n in range(len(nbNeurones) - 1):  # couches
        m = []
        for i in range(nbNeurones[n + 1]):  # couche arrivée
            v = []
            for j in range(nbNeurones[n]):  # couche départ
                v.append(0.1 * random.random() - 0.05)
            m.append(v)
        poids.append(m)
    #print(poids)

    apprentissage(dataset)
    evaluation(dataset)

if __name__ == "__main__":
    main()
# --------------------------------- Fin MLP -----------------------------------