import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt

if __name__=="__main__": #Necessario para paralelizar o KMeans
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() #utliza o dataset cifar10 do keras

    sift = cv2.xfeatures2d.SIFT_create() #Utiliza SIFT para gerar descritores
    descritores = [] #lista de descritores
    print("Criando lista de descritores")

    #Extrai os descritores de todas as imagens do dataset de treinamento
    for i in np.arange(x_train.shape[0]):
        _,imgDesc = sift.detectAndCompute(x_train[i],None)
        if imgDesc is not None:
            for j in np.arange(imgDesc.shape[0]):
                descritores.append(imgDesc[j])

    descritores = np.array(descritores)
    print("Calculando centroides a partir dos descritores")

    #Aplica o KMeans na lista de descritores para encontrar 10 centroides correspondentes as categorias de imagens
    kmeans = KMeans(n_clusters=10,n_jobs=-1).fit(descritores)

    treino_hist_X=[]
    treino_hist_Y=[]
    print("Calculando histogramas para as imagens do treino")

    #Para cada descritor de uma imagem do conjunto de treinamento, encontra o centroide mais mais proximo
    #Isso retorna um vetor com o numero do centroide para cada descritor
    #Cria um histograma a partir desse vetor para cada imagem do conjunto de treinamento
    #Associa o label da imagem a esse histograma
    for i in np.arange(x_train.shape[0]):
         _,imgDesc = sift.detectAndCompute(x_train[i],None)
         if imgDesc is not None:
             kcent = kmeans.predict(imgDesc) #Calcula centroide pada cada descritor
             hist,_ = np.histogram(kcent,bins=[0,1,2,3,4,5,6,7,8,9,10],normed=True)
             treino_hist_X.append(hist)
             treino_hist_Y.append(y_train[i])


    treino_hist_X = np.array(treino_hist_X).reshape((-1,10))
    treino_hist_Y = np.array(treino_hist_Y).reshape((-1,))

    print("Treinando SVM")
    #Treina um classificador SVM com os pares histograma e label das imagens do conjunto de treinamento
    modSVM = SVC()
    modSVM.fit(treino_hist_X,treino_hist_Y.reshape((-1,)))

    matriz_conf_teste = np.zeros((10,10))

    #Aplica o classificador ao conjunto de teste
    #Extrai os descritores de cada imagem
    #Encontra o centroide mais proximo para cada decritor de cada imagem
    #cria histograma com o vetor de centroides de cada imagem
    #Preve a categoria da imagem com o classificador SVM
    #Adiciona erro ou acerto na previsoa na matriz de confusao 10x10
    #As linhas são as categorias verdadeiras e as colunas são as previstas pelo modelo SVM
    print("Avaliando modelo SVM com dados de teste")
    for i in np.arange(x_test.shape[0]):
         _,imgDesc = sift.detectAndCompute(x_test[i],None)
         if imgDesc is not None:
             kcent = kmeans.predict(imgDesc) #Calcula centroide pada cada descritor
             hist,_ = np.histogram(kcent,bins=[0,1,2,3,4,5,6,7,8,9,10],normed=True)
             ret = modSVM.predict(hist.reshape((1,-1)))[0]
             real = y_test[i,0]
             matriz_conf_teste[real,ret] = matriz_conf_teste[real,ret] + 1

    plt.pcolor(matriz_conf_teste,cmap="jet")
    plt.colorbar()
    plt.title("Matriz de confusão - conjunto de teste")


    print("Fim")


