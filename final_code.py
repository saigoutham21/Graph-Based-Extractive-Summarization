
# coding: utf-8


import numpy as np
import PyPDF2
import sys

import matplotlib.pyplot as plt

import networkx as nx

from nltk.tokenize.punkt import PunktSentenceTokenizer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def readDoc():
    name = input('Please input a file name: ') 
    print('You have asked for the document {}'.format(name))

    if name.lower().endswith('.txt'):
        choice = 1
    elif name.lower().endswith('.pdf'):
        choice = 2
    else:
        choice = 3
        
    print(choice)
        
    if choice == 1:
        f = open(name, 'r')
        document = f.read()
        f.close()
            
     elif choice == 2:
        pdfFileObj = open(name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        document = pageObj.extractText()
        pdfFileObj.close()
    
    else:
        print('Failed to load a valid file')
        print('Returning an empty string')
        document = ''
    
    print(type(document))
    return document


def tokenize(document):
  
    doc_tokenizer = PunktSentenceTokenizer()
  
   sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


document = readDoc()
print('The length of the file is:', end=' ')
print(len(document))

sentences_list = tokenize(document)
print('The size of the list in Bytes is: {}'.format(sys.getsizeof(sentences_list)))

print('The size of the item 0 in Bytes is: {}'.format(sys.getsizeof(sentences_list[0])))



print(type(sentences_list))


print('The size of the list "sentences" is: {}'.format(len(sentences_list)))

for i in sentences_list:
    print(i)

cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)

cv_demo = CountVectorizer() 

text_demo = ["Ashish is good, you are bad", "I am not bad"] 
res_demo = cv_demo.fit_transform(text_demo)
print('Result demo array is {}'.format(res_demo.toarray()))


print('Feature list: {}'.format(cv_demo.get_feature_names()))

print('The data type of bow matrix {}'.format(type(cv_matrix)))
print('Shape of the matrix {}'.format(cv_matrix.get_shape))
print('Size of the matrix is: {}'.format(sys.getsizeof(cv_matrix)))
print(cv.get_feature_names())
print(cv_matrix.toarray())

normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())

print(normal_matrix.T.toarray)
res_graph = normal_matrix * normal_matrix.T

nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)
print('Number of edges {}'.format(nx_graph.number_of_edges()))
print('Number of vertices {}'.format(nx_graph.number_of_nodes()))

print('The memory used by the graph in Bytes is: {}'.format(sys.getsizeof(nx_graph)))

print(type(ranks))
print('The size used by the dictionary in Bytes is: {}'.format(sys.getsizeof(ranks)))

for i in ranks:
    print(i, ranks[i])

sentence_array = sorted(((ranks[i], s) for i, s in enumerate(sentences_list)), reverse=True)
sentence_array = np.asarray(sentence_array)


rank_max = float(sentence_array[0][0])
rank_min = float(sentence_array[len(sentence_array) - 1][0])


print(rank_max)
print(rank_min)



temp_array = []
if rank_max - rank_min == 0:
    temp_array.append(0)
    flag = 1

if flag != 1:
    for i in range(0, len(sentence_array)):
        temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))

print(len(temp_array))
threshold = (sum(temp_array) / len(temp_array)) + 0.2
sentence_list = []
if len(temp_array) > 1:
    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
                sentence_list.append(sentence_array[i][1])
else:
    sentence_list.append(sentence_array[0][1])


model = sentence_list

summary = " ".join(str(x) for x in sentence_list)
print(summary)

f = open('sum.txt', 'a+')

f.write('-------------------\n')
f.write(summary)
f.write('\n')

f.close

for lines in sentence_list:
    print(lines)

