import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from scipy.io import loadmat
from random import randint
from numpy.linalg import svd

def Load(g):
    dado1=loadmat(g)
    X=dado1.get('X')
    return X 

def show_img(img_arr):
    pixels = img_arr.reshape((32, 32))  
    plt.imshow(pixels.T, cmap='gray')
    plt.xticks([])
    plt.yticks([])

def show_img_grid(data, n_imgs=100, grid_v=10, grid_h=10, img_size=15, randomic=True, inverse_order=False):
    fig = plt.figure(figsize=(img_size,img_size)) 
    fig.subplots_adjust(hspace=0, wspace=0)  

    for i in range(n_imgs):
        ax = fig.add_subplot(grid_v, grid_h, i+1)
        num = i
        if inverse_order:
            num = data.shape[1] - i - 1  
        if randomic:
            num = randint(0, len(data) - 1)  
        show_img(data[num])
    plt.show()
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X-mu)/sigma
    
    return X_norm, mu, sigma

def PCA(X):
    m,n=X.shape
    Sigma=(1/m)*(X.T @ X)
    U, S, V = svd(Sigma)
    
    return U, S, V

def eigenfaces_video(V, lim,y):
    y=y+"\eigen.gif"
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylim(0, 30)
    line, = ax.plot(0, 0)

    def animation_frame(i):
        img_arr = np.zeros(1024)
        for j in range(i):
            img_arr = img_arr + V[j, :]
        pixels = img_arr.reshape((32, 32)) 
        plt.imshow(pixels.T, cmap='gray')  
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Eigenface 1 - {i+1}')
        
    ax.invert_yaxis()  

    animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, lim, 1), interval=10)
    animation.save(y)
    
def project_pca(U, img_array, n, show=False):
    img_alt = U[:, :n] @ U[:, :n].T @ img_array.T  
    if show:
        show_img(img_alt)  
    
    return img_alt

def sbs_img_plot(img_array_1, img_array_2):
    fig = plt.figure(figsize=(25,25)) 
    fig.subplots_adjust(hspace=0, wspace=0)  

    ax = fig.add_subplot(5, 5, 1)
    show_img(img_array_1)
    plt.title('Reconstruída')
    
    ax = fig.add_subplot(5, 5, 2)
    show_img(img_array_2)
    plt.title('Original')
    plt.show()
def proj_evolution_video(U, img_array, lim, step,j):
    j=j+'\proj.gif'
    fig = plt.figure(figsize=(15,15)) 
    fig.subplots_adjust(hspace=10, wspace=0)  
    def animation_frame(i):
        '''
        Creates frame for animation
        '''
        ax = fig.add_subplot(1,2, 1)
        show_img(project_pca(U, img_array, i))
        plt.title(f'Reconstruída, {i} componentes principais', fontsize=20)
        

        ax = fig.add_subplot(1,2, 2)
        show_img(img_array)
        plt.title('Original', fontsize=20)
       

    animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,lim,step), interval=10)

    Path("images").mkdir(parents=True, exist_ok=True)
    animation.save(j)

g=input("\nEndereço da pasta onde se encontram os dados (sem o nome do arquivo): ")
g=g+'\dado1'
X=Load(g)
X_norm ,mu,sigma=feature_normalize(X)
a=int(input('Gostaria de visualizar uma amostra aleatória de imagens? 1 para sim e 2 para não: '))
if a==1:
    b=int(input('Qual o número de imagens desejado? '))
    show_img_grid(X, b, int(np.ceil(b**(1/2))), int(np.ceil(b**(1/2))), 15)
U,S,V=PCA(X_norm)
c=int(input('\nGostaria de ver as eigenfaces correspondentes aos n primeiros componentes principais? 1 para sim e 2 para não: '))    
if c==1:
    d=int(input('\nQual o valor de n? '))
    show_img_grid(V, d, int(np.ceil(d**(1/2))), int(np.ceil(d**(1/2))), 15, randomic=False)
    e=int(input('\nGostaria de ver as eigenfaces correspondentes aos n últimos componentes principais? 1 para sim e 2 para não: '))
    if e==1:
        show_img_grid(V, d, int(np.ceil(d**(1/2))), int(np.ceil(d**(1/2))), 15, randomic=False, inverse_order=True)
    f=int(input('\nGostaria de salvar o gif com as eigenfaces correspondentes aos n primeiros componentes principais somadas? 1 para sim e 2 para não: '))    
    if f==1:
        y=input('\nEndereço para se salvar o gif (Sem o nome do arquivo): ')
        eigenfaces_video(V, d,y)
        print('\nGif salvo com o nome de eigen.gif :)')
z=int(input("\nGostaria de visualizar as imagens reconstruídas e originais lado a lado? 1 para sim e 2 para não: "))
if z==1:
    h=int(input("\nQual seria o número de imagens? "))
    n=int(input("\nQual seria o número de componentes principais? "))
    for i in range(h):
        num = randint(0, len(X) - 1)
        sbs_img_plot(project_pca(U, X[num], n), X[num])
u=int(input("\nGostaria de salvar o gif com a evolução da reconstrução? 1 para sim e 2 para não: "))
if u==1:
    k=int(input("\nQual a quantidade de componentes principais? (Múltiplo de 5): "))
    j=input('\nEndereço para se salvar o gif (Sem o nome do arquivo): ')
    proj_evolution_video(U, X[randint(0, len(X) - 1)], k, 5,j)
    print('\nGif salvo com o nome de proj.gif :)')