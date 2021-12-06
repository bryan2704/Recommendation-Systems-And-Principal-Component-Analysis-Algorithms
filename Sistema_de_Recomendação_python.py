import numpy as np
import copy as cp
import time
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat
from scipy.optimize import minimize
 
def Load():
    g=input("\nEndereço da pasta onde se encontram os dois arquivos com os dados (sem o nome dos mesmos): ")
    with open(g+"\dado3.txt") as f:
        T=f.readlines()
        T=[line.rstrip() for line in T]
    T=np.array(T).flatten()
    for a in T:
        a=str(a)
    g=g+"\dado2"
    dado2=loadmat(g)
    R=dado2.get('R')
    Y=dado2.get('Y')
    q=int(input("\nGostaria de Visualizar um plot com as distribuições de notas para cada filme? 1 para sim e 2 para não: "))
    if q==1:
        plt.figure(figsize=(8,16))
        plt.imshow(Y)
        plt.xlabel("Users")
        plt.ylabel("Movies")
        plt.show()
    aux_Y=np.zeros((Y.shape[0],Y.shape[1]))
    mean=np.zeros((Y.shape[0],1))
    for i in range(0,Y.shape[0]):
        mean[i]=np.sum(Y[i,:])/np.count_nonzero(R[i,:])
        aux_Y[i,R[i,:]==1] = Y[i,R[i,:]==1] - mean[i]
    Ynorm=aux_Y
    num_users=Y.shape[1]
    num_movies=Y.shape[0]
    num_features=int(input("\nQual você gostaria que fosse a dimensão de theta? "))
    theta=np.random.randn(num_users,num_features)
    X=np.random.randn(num_movies,num_features)
    params=np.append(X.flatten(),theta.flatten())
    return R,Ynorm,Y,T,X,theta,mean,params,num_users,num_movies,num_features

def Verification(T,Y,R):
    a=int(input("\nGostaria de verificar se um filme está na lista de avaliados? 1 para sim e 2 para não: "))
    if a==1:
        u=1
        while u==1:
            b=input("\nNome do filme que se quer buscar: ")
            e=cp.deepcopy(b).lower()
            for i in range(len(T)):
                d=cp.deepcopy(T[i]).lower()
                if d.find(e)!=-1:
                    print("\nSim, o filme %s, lançado em %s, está na lista com uma nota média de %.1f/5"%(b, d[-5:-1], np.sum(Y[i,:]*R[i,:])/np.sum(R[i,:])))
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    break
                elif i==len(T)-1:
                    print("\nNão, o filme %s não está na lista"%b)
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    break
    b=int(input("\nGostaria de verificar todos os filmes que saíram em um dado ano? 1 para sim e 2 para não: "))
    if b==1:
        u=1
        k=[]
        while u==1:
            b=int(input("\nAno que se quer verificar: "))
            for i in range(len(T)):
              d=cp.deepcopy(T[i])
              if d[-5]=='1':
                if b==int(d[-5:-1]):
                    if T[i][1]==" ":
                       k.append(d[2:-7])
                    elif T[i][2]==" ":
                       k.append(d[3:-7])
                    elif T[i][3]==" ":
                       k.append(d[4:-7])
                    elif T[i][4]==" ":
                       k.append(d[5:-7])
                if k!=[] and i==len(T)-1:
                    print("\nNo ano %d saíram os seguintes filmes:"%(b))
                    print("\n")
                    for j in k:
                        print("-----------------------")
                        print(j)
                    print("-----------------------")
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    k=[]
                    break
                elif i==len(T)-1 and k==[]:
                    print("\nNenhum filme da lista saiu em %d"%(b))
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    k=[]
                    break
    return             

def CostFunction(params,Ynorm,R,num_users,num_movies,num_features,beta,lamb):
    X=params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    predictions =  X @ Theta.T
    j=(predictions - Ynorm)
    J = 1/2 * np.sum((j**2) * R)
    X_grad = j*R @ Theta
    Theta_grad = (j*R).T @ X
    grad = np.append(X_grad.flatten(),Theta_grad.flatten())
    if beta==1:
         reg_X =  lamb/2 * np.sum(Theta**2)
         reg_Theta = lamb/2 *np.sum(X**2)
         reg_J = J + reg_X + reg_Theta
         reg_X_grad = X_grad + lamb*X
         reg_Theta_grad = Theta_grad + lamb*Theta
         reg_grad = np.append(reg_X_grad.flatten(),reg_Theta_grad.flatten())
         J_history.append(reg_J)
         return reg_J,reg_grad
    else:
        J_history.append(J)    
        return J, grad            
            
  
def Gradient(params,Ynorm,R,num_users,num_movies,num_features):
    alpha=int(input("\nNúmero de iterações desejado: "))
    c=int(input("\nGostaria de fazer a minimização através de gradiente descendente ou gradiente conjugado? 1 para descendente e 2 para conjugado: "))
    beta=int(input("\nGostaria que o custo e o gradiente obtidos fossem regularizados? 1 para sim e 2 para não: "))
    if beta==1:
        lamb=float(input("\nQual seria o valor de lambda? "))
    else:
        lamb=0
    if c==2:
        initial_time=time.time()
        result=minimize(CostFunction,params,args=(Ynorm,R,num_users,num_movies,num_features,beta,lamb), method='CG', jac=True, tol=None, callback=None, options={'maxiter':alpha,'disp':True , 'gtol':1e-4})
        end_time=time.time()
        h=int(input("\nGostaria de ver os resultados obtidos pela minimização através do gradiente conjugado? 1 para sim e 2 para não: "))
        if h==1:
            print("\n")
            print(result)
        return result.x, end_time-initial_time,(alpha,lamb,0,end_time-initial_time)
    if c==1:
        p=float(input("\nQual seria o valor da taxa de aprendizado? "))
        initial_time=time.time()
        result=GradientDescent(params,Ynorm,R,num_users,num_movies,num_features,p,alpha,beta,lamb)
        end_time=time.time()
        return result, end_time-initial_time,(alpha,lamb,p,end_time-initial_time)

def GradientDescent(params,Ynorm,R,num_users,num_movies,num_features,alpha,num_inter,beta,lamb):
    X=params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    for i in range(num_inter):
        params = np.append(X.flatten(),Theta.flatten())
        J,grad=CostFunction(params,Y,R,num_users,num_movies,num_features,beta,lamb)
        X_grad=grad[:num_movies*num_features].reshape(num_movies,num_features)
        Theta_grad=grad[num_movies*num_features:].reshape(num_users,num_features)
        X=X-(alpha * X_grad)
        Theta=Theta-(alpha * Theta_grad)
    params=np.append(X.flatten(),Theta.flatten())
    return params

def Result(params,total_time,J_history,mean,num_features,num_movies,num_users,T):
    X=params[:num_movies*num_features].reshape(num_movies,num_features)
    theta=params[num_movies*num_features:].reshape(num_users,num_features)
    result=X @ theta.T
    prediction=result[:,0][:,np.newaxis]+mean
    Z=int(input("Gostaria de visualizar a curva de custo por iteração? 1 para sim e 2 para não: "))
    if Z==1:
        q=list(range(len(J_history)))
        plt.plot(q, J_history)
        plt.xlabel('Número de Iterações')
        plt.ylabel('Custo')
        plt.show()
    df = pd.DataFrame(np.hstack((prediction,T[:,np.newaxis])))
    k=int(input("\nGostaria de buscar a nota predita pelo modelo de algum filme em específico? 1 para sim e 2 para não: "))
    if k==1:
        u=1
        while u==1:
            b=input("\nNome do filme que se quer buscar: ")
            e=cp.deepcopy(b).lower()
            for i in range(len(df[1])):
                d=cp.deepcopy(df[1][i]).lower()
                if d.find(e)!=-1:
                    print("\nO filme %s, lançado em %s, teve sua nota prevista de %.1f/5, enquanto através de média simples, sua nota foi %.1f/5"%(b, d[-5:-1], float(df[0][i]), mean[i,0]))
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    break
                elif i==len(T)-1:
                    print("\nInfelizmente o filme %s não está na lista"%b)
                    u=int(input("\nGostaria de uma nova consulta? 1 para sim e 2 para não: "))
                    break 
    df.sort_values(by=[0],ascending=False,inplace=True)
    df.reset_index(drop=True,inplace=True)
    t=int(input("\nGostaria de visualizar um 'top' dos melhores e piores filmes? 1 para sim e 2 para não: "))
    save=[]
    if t==1:
        g=int(input("\nQual seria o 'top' que se quer visualizar? "))
        print("\nMelhores recomendações para você:\n")
        for i in range(g):
            print("Nota predita",round(float(df[0][i]),1)," para o filme ",df[1][i])
            save.append((round(float(df[0][i]),1),df[1][i]))
        a=list(reversed(df[0]))
        b=list(reversed(df[1]))
        print("\n")
        print("Piores recomendações para você:\n")
        for j in range(g):
            print("Nota predita",round(float(a[j]),1)," para o filme",b[j])
            save.append((round(float(a[j]),1),b[j]))
    print("\nO tempo total de execução do modelo para predição das notas foi de %.1f segundos"%total_time)        
    return df,prediction,save

def Save(save,alpha,l,J_history,params,num_users,num_movies,num_features):
    X=params[:num_movies*num_features].reshape(num_movies,num_features)
    theta=params[num_movies*num_features:].reshape(num_users,num_features)
    g=int(input("Gostaria de salvar as principais informações deste programa em um arquivo txt e a matriz X e theta em arquivos CSV? 1 para sim e 2 para não: "))
    if g==1:
        o=(input("Endereço para salvar os arquivos (sem o nome dos mesmos): "))
        aux=cp.deepcopy(o)
        o=r"%s/Resultado_Recomendação.txt"%aux
        v=r"%s/Matriz_X.csv"%aux
        h=r"%s/Matriz_theta.csv"%aux
        f=open(o,'w+')
        f.write("Principais parâmetros e resultados do modelo:")
        f.write("\nPara a dimensão de colunas dos vetores theta, foi utilizado um valor de %d"%l)
        if alpha[2]==0:
            f.write("\nO método de minimização utilizado foi o de gradiente conjugado")
        else:
            f.write("\nO método de minimização utilizado foi o de gradiente descendente com uma taxa de aprendizado igual a %.5f"%alpha[2])
        f.write("\nO número total de iterações para o treino do modelo foi de %d"%len(J_history))
        if alpha[1]==0:
            f.write("\nNão houve regularização")
        else:
            f.write("\nHouve regularização e o valor de lambda utilizado foi %.5f"%alpha[1])
        f.write("\nO custo inicial foi de %.4f, e terminou com %.4f"%(J_history[0],J_history[-1]))
        f.write("\nO tempo total de execução do modelo foi de %.3f segundos"%alpha[3])
        f.write("\n")
        if save!=[]:
            f.write("\nAs melhores recomendações para o usuário foram as seguintes:")
            for i in range(0,int(len(save)/2)):
                f.write("\nFilme %s com nota %.1f"%(save[i][1],save[i][0]))
            f.write("\n")
            f.write("\nE as piores recomendações para o usuário foram as seguintes:")
            for j in range((int(len(save)/2)),len(save)):
                f.write("\nFilme %s com nota %.1f"%(save[j][1],save[j][0]))
        f.close()
        with open(v, 'w', encoding='UTF8', newline='') as f:
            writer=csv.writer(f)
            writer.writerows(X)
        with open(h, 'w', encoding='UTF8', newline='') as f:
            writer=csv.writer(f)
            writer.writerows(theta)
        print("\nSeus arquivos foram salvos no diretório %s com o nome de Resultado_recomendação.txt, Matriz_X.csv e Matriz_theta.csv :)"%aux)
        return

J_history=[]   
R,Ynorm,Y,T,X,theta,mean,params,num_users,num_movies,num_features=Load()
Verification(T,Y,R)
params,total_time,alpha=Gradient(params,Ynorm,R,num_users,num_movies,num_features)
df,prediction,save=Result(params,total_time,J_history,mean,num_features,num_movies,num_users,T)
Save(save,alpha,theta.shape[1],J_history,params,num_users,num_movies,num_features)