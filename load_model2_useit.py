# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:32:05 2019


"""

from gensim import models
#import numpy as np
#import pymysql
#import pandas as pd
#import MeCab
#from progressbar import ProgressBar
#import time
#from pandas import Series,DataFrame
#from gensim import corpora,matutils
#from gensim.models import word2vec
#import math


if __name__ == '__main__':
    #model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
    
    ####### Load model ########
    #model_loaded = models.fasttext.FastText.load_fasttext_format('wiki.th.bin')
    model_loaded = models.doc2vec.Doc2Vec.load('model_dailynews_deepcut_doc2vec__1870')
    #model_loaded1 = models.doc2vec.Doc2Vec.load('model_dailynews1_eng')
    #model_loaded2 = models.doc2vec.Doc2Vec.load('model_dailynews1_mm')
    #print(model_loaded)
    #print(model_loaded1)
    
    
    ######### ! 1. USE TARGET WORD VECTOR --> Similar words ###########
    print("USE TARGET WORD VECTOR = ", model_loaded.most_similar(["เอเชี่ยนเกมส์"]))

    
    ########## Paragraph vector #############
    vec1 = model_loaded.docvecs['it_245']
    vec2 = model_loaded.docvecs['it_464']
    vec3 = model_loaded.docvecs['sports_1865']
    vec4 = model_loaded.docvecs['sports_782']
    vec5 = model_loaded.docvecs['sports_1463']
    vec6 = model_loaded.docvecs['sports_1830']
    vec7 = model_loaded.docvecs['it_876']
    vec8 = model_loaded.docvecs['it_622']
    vec9 = model_loaded.docvecs['it_1116']
    vec10 = model_loaded.docvecs['it_228']
    vec11 = model_loaded.docvecs['it_270']
    vec12 = model_loaded.docvecs['education_759']
    
       
    ###### ! 2. USE TARGET PARAGRAPH VECTOR --> Similar words ######
    tasu = (vec3)
    z = model_loaded.similar_by_vector(tasu, topn=20, restrict_vocab=None)
    print("USE TARGET PARAGRAPH VECTOR = ", z)
    
    
    ###### ! 3. USE FEATURE VECTOR --> Similar words #######
    #print(model_loaded.docvecs.most_similar(["sports_1865"])) #-- Find Similar article for Feature vector --#
    tasu1 = (vec3+vec4) 
    y = model_loaded.similar_by_vector(tasu1, topn=20, restrict_vocab=None)
    print("USE FEATURE VECTOR = ", y)
    
    
    
    