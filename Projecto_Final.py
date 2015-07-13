# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as ss
import scipy.io.wavfile as wav

#Calculo de bk's
#num = nº de pontos/define a largura da curva
#cuts = frequencia de corte
#Pass = indica se vamos fazer um Passo baixo/alto/banda
def filtros(som, num, cuts, Pass):
    y = ss.firwin(num, cuts, pass_zero = Pass)
    
    return lfilter(y, som)
    
def lfilter(bk, x):
    t = ss.lfilter(bk, 1, x) #ak = 1, porque e um sistema FIR
    
    return t

def potencia(X):
    return (np.sum(X**2))/len(X)

def potencia_total(som):
    y1=filtros(som, 101, 1./5, True) #Passa_Baixo 
    y2=filtros(som, 101, [1./5, 2./5], False) #Passa_Banda
    y3=filtros(som, 101, [2./5, 3./5], False) #Passa_Banda
    y4=filtros(som, 101, [3./5, 4./5], False) #Passa_Banda
    y5=filtros(som, 101, 4./5, False) #Passa_Alto
    
    z1=potencia(y1)
    z2=potencia(y2)
    z3=potencia(y3)
    z4=potencia(y4)
    z5=potencia(y5)
    
    return np.array([z1,z2,z3,z4,z5])

#Vamos calcular um "ponto"
def Processar(ficheiro):
    #Ler os sons, dados
    Fs, x = wav.read(ficheiro)
    x = x.astype('float32')
           
    #Como os nossos ficheiros audio, tem 2 canais(esquerdo e direito), calculamos a media dos 2
    x1 = (x[:,0]+x[:,1])/2.0 
    
    #Vamos comecar pelos filtos, fazendo o passa-baixo, 3 passa-bandas, e por ultimo um passa-alto, para calcularmos os bk's
    #guardando os em varias variaveis.
    #Ja com os bk's calculados fazemos o lfilter.
    #Depois calculamos a potencia do resultado do lfilter.
    return potencia_total(x1)

    
def knn(treino, classe, teste, k):
    d = np.zeros(len(treino))
    for i in range(len(treino)):
        d[i] = np.sqrt((treino[i, 0] - teste[0])**2 + (treino[i, 1] - teste[1])**2 + (treino[i, 2] - teste[2])**2 + (treino[i, 3] - teste[3])**2 + (treino[i, 4] - teste[4])**2)
        
    ind = np.argsort(d)
    
    if sum(classe[ind][:k] == 1) >= sum(classe[ind][:k] == 2):     
        return ("O som é um assobio.")     
    
    else:     
        return ("O som é uma palma.")
    
##Sons que vao ser usados como templates(testes), para podermos identificar mais tarde novos sons 
assobios1 = Processar('/home/andre/Desktop/PDS/Projecto_Final/assobio1.wav')
assobios2 = Processar('/home/andre/Desktop/PDS/Projecto_Final/assobio2.wav')
assobios3 = Processar('/home/andre/Desktop/PDS/Projecto_Final/assobio3.wav')
assobios4 = Processar('/home/andre/Desktop/PDS/Projecto_Final/assobio4.wav')
assobios5 = Processar('/home/andre/Desktop/PDS/Projecto_Final/assobio5.wav')

palmas1 = Processar('/home/andre/Desktop/PDS/Projecto_Final/palmas1.wav')
palmas2 = Processar('/home/andre/Desktop/PDS/Projecto_Final/palmas2.wav')
palmas3 = Processar('/home/andre/Desktop/PDS/Projecto_Final/palmas3.wav')
palmas4 = Processar('/home/andre/Desktop/PDS/Projecto_Final/palmas4.wav')
palmas5 = Processar('/home/andre/Desktop/PDS/Projecto_Final/palmas5.wav')

#Som que vamos testar
teste = Processar('/home/andre/Desktop/PDS/Projecto_Final/teste.wav')

#Junção das variaveis numa só. (variavel classe)
#vstack cria um array vertical.Vamos usar isto para podermos criar um array duplo, com os assobios do lado esquerdo,
#e palmas do lado direito
assobios = np.vstack((assobios1, assobios2, assobios3, assobios4, assobios5))
palmas = np.vstack((palmas1, palmas2, palmas3, palmas4, palmas5))

classes = np.hstack((np.ones(len(assobios)), 2 * np.ones(len(palmas))))
treino = np.vstack((assobios, palmas))

#print(knn(treino,classes, teste, 6)) #teste sem trocas.O som de teste é um assobio

for i in range (len(treino)-1):
    teste = np.array(treino[i])
    np.delete(treino, i)
    print(knn(treino,classes, teste, 6))


