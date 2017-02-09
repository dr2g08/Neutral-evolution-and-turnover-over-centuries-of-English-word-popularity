import gzip
import csv
import string
import pickle

def csvreader(file):
    with gzip.open(file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t',quotechar='"')
        for row in reader:
            yield row

#list of barred punctuation and numbers
punnum=string.punctuation
punnum+=string.digits
punnum=punnum.replace("'", "")
punnum=punnum.replace("-", "")

#list of vowels
vowell =['a','e','i','o','u','A','E','I','O','U','y','Y']
            
            
d = {k: [] for k in range(1700,2001)}
turn = {k: [] for k in range(1700,2001)} # for the turnover profile


alphabet = string.ascii_lowercase
for letter in alphabet:

    fn = 'data/Ngrams/googlebooks-eng-all-1gram-20120701-' + letter + '.gz'
    f = csvreader(fn)

    for line in f:

        w = line[0]
        y = int(line[1])
        c = int(line[2])
        
        #only include word if it contains no banned puntuation or numbers
        info=[e for e in w if e in punnum]
        if len(info) == 0:
            
            #only include words that have vowells
            info=[e for e in w if e in vowell]
            if len(info) > 0:
                if y in range(1700,2001): # if the year is between 1700 and 2000
                    d[y].append(c)

                    if c > 500: # only add the word if its frequency is high enough
                        turn[y].append((w,c))
                        
                        
with open('data/counts.pickle', 'wb') as handle:
    pickle.dump(d, handle)
    
with open('data/words.pickle', 'wb') as handle:
    pickle.dump(turn, handle) 
