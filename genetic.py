import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from copy import copy
import collections
import powerlaw
import random
import pickle

population_size=100# size of genetic population

#size of breeding population
top_breed=0.2

#mutation rate for children
mu=0.15

new_population_size=int(population_size*(1-top_breed))

#create the limits on the parameter values 
N_llim = 5000
N_ulim = 30000
Nmu_llim = 5
Nmu_ulim = 90
S0_llim = 1000
S0_ulim = 10000


#potential populations for the parameters to be drawn from
N=np.arange(N_llim,N_ulim,200)
Nmu=np.arange(Nmu_llim,Nmu_ulim,5)
S0=np.arange(S0_llim,S0_ulim,200)

var_num = {0 : N, 1 : Nmu, 2 : S0} # number of the variables

#target values for the model
heaps_exp = 0.55
heaps_int = 17
Z_decay = 0.011
Z_0 = [6,12,31]
zipfs=1.7
V_min = 300

#allowed error range for target values
heaps_expE=0.025
heaps_intE=3
Z_decayE=0.002
Z_0E= [2,3,4] 
zipfsE = 0.3

def fitness(pop):

    N=int(pop[0])
    Nmu=int(pop[1])
    S0=int(pop[2])

    alpha=0.028 #<- this should be taken from the data (we need to get it down)

    t_steps=301
    y_max=200
    mu=Nmu/float(N)
    t=np.asarray(range(t_steps))
    S_vec=np.round(S0*np.exp(alpha*t))

    score=0

    tau = round(4/mu) # runin length
    T = tau + t_steps # total time_steps

    pop = np.asarray(range(N)) #inialise population
    base = N

    #corpus=pd.Series(index=range(t_steps))
    #vocab=pd.Series(index=range(t_steps))

    corpus=[]
    vocab=[]
    Z=pd.DataFrame(index=range(y_max),columns=range(t_steps-1))

    history=np.asarray([])

    for t in range(int(T)):
        rnd = np.random.uniform(0, 1, N)

        select_vec = copy(pop)

        pop=np.asarray(np.zeros(N)) # start new empty population
        pop[rnd>=mu] = np.random.choice(select_vec, size=len(pop[rnd>=mu]), replace=True)
        pop[rnd<mu] = range(base,base+len(pop[rnd<mu]))

        base+= len(pop[rnd<mu]) # next trait to enter the population

        if t >= tau:

            history = np.append(history, pop)
            #find the random sample of the total population 
            S=S_vec[t-tau]
            popS=np.random.choice(history, size=S, replace=True)



            #find the frequency of all discrete traits 
            C=collections.Counter(popS)
            C=C.most_common(len(C))
            MC=np.asarray([mc[1] for mc in C])



            #store vocab and corpus size
            corpus.append(len(popS))
            vocab.append(len(MC))


            #create turnover matrix
            rank=np.asarray([mc[0] for mc in C])
            if t > tau:
                for y in range(y_max):
                    z=len(np.setdiff1d(rank[:y+1],rank_old[:y+1]))
                    Z.loc[y,t-tau-1]=float(z)


            rank_old=copy(rank)        

            #save populations for turn of each century
            if t-tau in [100,200,300]: 
                fit=powerlaw.Fit(MC,xmin=1.)
                #fig2=fit.plot_pdf(color=c,marker='s')
                #fit.power_law.plot_pdf(color=c, linewidth=2, ax=fig2)
                #plt.show()
                score+=int(abs(fit.power_law.alpha - zipfs) < zipfsE)

            #print the smallest vocab size
            if t == tau:
                Cll=collections.Counter(pop)
                Cll=Cll.most_common(len(Cll))
                MCll=np.asarray([mc[1] for mc in Cll])
                fit=powerlaw.Fit(MCll,xmin=1.)

                score+=int(len(MC) > V_min)




    corpus=np.asarray(corpus)
    vocab=np.asarray(vocab)

    #obtain heaps law exponent
    def func(t,a,b):
        return a*t**b

    popt, pcov = curve_fit(func, corpus, vocab,[0,1])  

    fit=func(corpus,popt[0],popt[1])

    heaps_expE=0.03
    heaps_intE=2

    score+=int(abs(popt[0]-heaps_int) < heaps_intE)
    score+=int(abs(popt[1] - heaps_exp) < heaps_expE)

    #obtain the exponential decay rate of turnover
    def func(t, a, beta):
        return a*np.exp(-beta*t)

    timestep=[49,99,199]
    for i, ze, z0 in zip(timestep,Z_0E,Z_0):

        z=Z.iloc[i,:].astype('float').values
        xdata=np.asarray(range(len(z)))

        popt, pcov = curve_fit(func, xdata, z,[0,0])

        fit=func(xdata,popt[0],popt[1])

        score+=int(abs(popt[0]-z0) < ze)

        score+=int(abs(popt[1]-Z_decay) < Z_decayE)

    return score/12.

def individual(N, Nmu, S0):
    return [np.random.choice(N), np.random.choice(Nmu), np.random.choice(S0)]

def population(N, Nmu, S0):
    return [individual(N,Nmu,S0) for x in xrange(population_size)]

def population_fitness(population,population_size):
    f=np.asarray([fitness(population[cnt]) for cnt in range(population_size)]) 
    return f 

def find_breeding_population(population,fit):
    #most fit individuals in population
    ii=np.argsort(fit)[::-1]
    top_pop=[population[i] for i in ii]
    
    fit_pop=fit[ii[:population_size*top_breed]]
    return [top_pop[i] for i in range(int(population_size*top_breed))], fit_pop

def mutate(child):
    #replace one random parameter in child
    ii=random.randint(0,2)
    V=var_num[ii]
    child[ii] = random.choice(V)
    return child

def create_child(breeding_pop):
    
    # take two random parents and produce a child 
    parents=random.sample(breeding_pop,2)
    child=np.zeros(3)
    
    mother=parents[0]
    father=parents[1]
    
    #take 2 random traits form the mother
    ind=np.random.choice(range(3),2,replace=False)
    m=[int(mother[i]) for i in ind]
    child[ind]=m
    
    #take the other from the father
    ind=np.setdiff1d(range(3),ind)
    child[ind]=int(father[ind])
    
    if random.random() < mu:
        child = mutate(child)
       
    return list(child)

def create_new_population(breeding_pop,new_population_size):
    return [list(create_child(breeding_pop)) for i in range(new_population_size)]

def define_new_populations(breeding_pop,new_pop,breeding_fit,new_fit):
    #define the new population and fittness' for the next step
    breeding_pop.extend(new_population)
    population=breeding_pop
    fit=np.append(breeding_fit,new_fit)
    
    return population,fit

def save_populations(fit_df,population_M,population,fit):
    
    #accumulate and save the populations and fitness scores
    fit_df.loc[i,:] = fit
    population_M.append(population)
    
    fit_df.to_csv('genetic_populations_bigalpha/fitness.csv')
    
    with open('genetic_populations_bigalpha/populations.pickle', 'wb') as handle:
        pickle.dump(population_M, handle)
        
    return



#gernate starting population and their fitnesses
population=population(N,Nmu,S0)
fit=population_fitness(population,population_size)


fit_df=pd.DataFrame(columns=range(population_size))
population_M=[]
for i in range(1000000):

    #find the fit population that will be used for breeding
    breeding_pop,breeding_fit=find_breeding_population(population,fit)

    #breed new population from the fit breeding population
    new_population=create_new_population(breeding_pop,new_population_size)

    #fitness of the new population
    new_fit=population_fitness(new_population,new_population_size)


    population,fit=define_new_populations(breeding_pop,new_population,breeding_fit,new_fit)
    
    save_populations(fit_df,population_M,population,fit)

    
