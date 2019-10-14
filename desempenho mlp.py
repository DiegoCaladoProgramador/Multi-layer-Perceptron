import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier

from os import listdir

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

def scores(clf_name, prediction, metodo, target_test, file, split_number, output):
    with open(output, 'at') as out_file:
        line = f"\"{file} , {clf_name} , {metodo} , Split # {split_number}\","
        line += f"{accuracy_score(target_test, prediction)},"
        line += f"{matthews_corrcoef(target_test, prediction)},"
        line += f"{f1_score(target_test, prediction,average='macro')},"
        line += f"{recall_score(target_test, prediction, average='macro')},"
        line += f"{precision_score(target_test, prediction, average='macro')}\n"
        out_file.writelines(line)
dir = 'features/'
output = 'output.csv'
with open(output, 'wt') as out_file: 
        out_file.writelines('\"Descrição\",\"Acurácia\",\"F1-Score\",\"Recall\",\"Precisão\",\"MCC\"\n')
    

names=[] # nome das colunas
for file in listdir(dir):
    names.clear()
    print(f"---{dir + file}---")
    with open(dir +file, 'rt') as in_file:
        for line in in_file:
            if line.startswith("@inputs"):
                for word in line.split(" "):
                    if word != '@inputs':
                        names.append(word.replace('\n', ''))
                names.append("classes")
            if line.startswith("@data"):
                break
    #data = pd.read_csv(dir + file, comment ='@', names=names)
    data = pd.read_csv(dir + file, comment = '@', header=None)
    encoder = LabelEncoder()
    data = data.apply(encoder.fit_transform)
    ultimaColuna = len(names) - 1
    
    ft = data.iloc[:, 0:ultimaColuna]
    tg = data.iloc[:,-1]
    vetor_epoca = [50,100,500]
   
    
    vetor_taxa_de_aprendizado= [0.01,0.001]
    vetor_nCamadasEscondidas = [(10,),(50,),(100,),(10,10),(50,10),(100,10)]
    
    print("entrando no for 5")
    for i in range(5):
        ft_train, ft_test, tg_train, tg_test = train_test_split(ft, tg,train_size=0.75, stratify =tg, random_state=i)
        ft_train, ft_valid, tg_train, tg_valid = train_test_split(ft_train, tg_train,train_size=0.9,stratify =tg_train,random_state=i+1000)
        
        s = StandardScaler()
        padr_ft_train = s.fit_transform(ft_train)
        padr_ft_test = s.transform(ft_test)
        
        n = Normalizer()
        norm_ft_train = n.fit_transform(ft_train)
        norm_ft_test = n.transform(ft_test)
        
        print("vetor_epoca")
        for epoca in vetor_epoca:
            print("vetor_taxa_de_aprendizado")
            print("x= ",epoca)
            for taxa_de_aprendizado in vetor_taxa_de_aprendizado:
                print("entrando no for vetor_ncamadasEscondidas")
                print("y",taxa_de_aprendizado)
                for nCamadasEscondidas in vetor_nCamadasEscondidas:
                    print("z",nCamadasEscondidas)
                    print(file)
                    print("iniciando MLP")
                    mlp = MLPClassifier(batch_size='auto',learning_rate_init=taxa_de_aprendizado, hidden_layer_sizes=(nCamadasEscondidas), max_iter=epoca)
                    features_r = mlp.fit(padr_ft_train, tg_train)
                    padr_prediction = mlp.predict(padr_ft_test) 
                    scores("MlP", padr_prediction, "Padronizado", tg_test, file, i, output)
                    
                    mlp = MLPClassifier(batch_size='auto',learning_rate_init=taxa_de_aprendizado, hidden_layer_sizes=(nCamadasEscondidas), max_iter=epoca)
                    features_r = mlp.fit(norm_ft_train, tg_train)
                    norm_prediction = mlp.predict(norm_ft_test) 
                    scores("MLP", norm_prediction, "Normalizado", tg_test, file, i, output)
                    print("saindo do MLP")
                    print("repeticao n: ", i,"de 5")
print("fim")