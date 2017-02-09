import numpy as np
import pandas as pd
import collections
import powerlaw
from copy import copy
import os

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#parameters for straight exponentuially increasing N neutral model
Nmin=5000
Nmax=500000
t_steps=301
y_max=500
runs=50

mu=50./Nmin

for r in range(runs):

    #population sizes for the neutral_model at all time steps
    alpha=(1./t_steps)*np.log(Nmax/Nmin)
    t=np.asarray(range(t_steps))
    N_vec=np.round(Nmin*np.exp(alpha*t))


    tau = round(4/mu) # runin length
    T = tau + t_steps # total time_steps


    N=int(N_vec[0])#inital N value

    pop = np.asarray(range(N)) #inialise population
    base = N

    corpus=[]
    vocab=[]
    Z=pd.DataFrame(index=range(y_max),columns=range(t_steps-1))
    for t in range(int(T)):
        rnd = np.random.uniform(0, 1, N)

        select_vec = copy(pop)

        pop=np.asarray(np.zeros(N)) # start new empty population
        pop[rnd>=mu] = np.random.choice(select_vec, size=len(pop[rnd>=mu]), replace=True)
        pop[rnd<mu] = range(base,base+len(pop[rnd<mu]))

        base+= len(pop[rnd<mu]) # next trait to enter the population

        if t >= tau:

            N=int(N_vec[t-tau])

            #find the frequency of all discrete traits 
            C=collections.Counter(pop)
            C=C.most_common(len(C))
            MC=np.asarray([mc[1] for mc in C])


            #save populations for turn of each century
            if t-tau in [0,100,200,300]:
                path='data/neutral/'+str(r)+'/zipfs/'
                make_dir(path)
                pd.Series(MC).to_csv(path+'P_'+str(int(t-tau+1700))+'.csv')

            #store vocab and corpus size
            corpus.append(len(pop))
            vocab.append(len(MC))

            #create turnover matrix
            rank=np.asarray([mc[0] for mc in C])

            if t > tau:
                for y in range(y_max):
                    z=len(np.setdiff1d(rank[:y+1],rank_old[:y+1]))
                    Z.loc[y,t-tau-1]=z

            rank_old=copy(rank)

    #save Z,V and N series
    path='data/neutral/'+str(r)+'/'
    make_dir(path)
    pd.Series(corpus).to_csv(path+'corpus.csv') 
    pd.Series(vocab).to_csv(path+'vocab.csv')
    Z.to_csv(path+'Z.csv')

