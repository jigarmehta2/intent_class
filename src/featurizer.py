import os,pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack # matrix ---- replacable

# tf-idf tfidf_vectorizer function
def featurizer(analyzer, n_gram, max_features,col,dat):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,  # adding log term
        strip_accents='unicode', ## stripping garbage
        analyzer=analyzer,  #word level / char level
        smooth_idf=True,  ## add 1 to avoid divide by 0
        token_pattern=r'\w{1,}',  # 1 and more length
        stop_words='english',
        ngram_range=n_gram,
        max_features=max_features,
        min_df=2)
    features=vectorizer.fit_transform(dat[col].values.astype('U')) # unicode formatting
    return vectorizer,features

   
def run_featurizer(df,train_eval_predict, model_dir, spacy="yes",tag="labelled" ):    
    print("\nRunning featurizer ...")

    if(train_eval_predict=='train'):
        # applied on cleaned utterance -- create word level features
        v1,word_features = featurizer(analyzer='word',n_gram=(1,2),max_features=35000,\
                                                col="sent",dat=df)
        # applied on cleaned utterance -- create char level features
        v2,char_features = featurizer(analyzer='char',n_gram=(1,5),max_features=35000,\
                                                col="sent",dat=df)  
        tfdif_vectors=[v1,v2]
        all_features = hstack([word_features,char_features])
        if(spacy=="yes"):
            #applied on extra spacy features -- create word level features
            v3,pos_features1 = featurizer(analyzer='word',n_gram=(1,2),max_features=12000,\
                                                     col="feat",dat=df)
            # applied on extra spacy features -- create char level features
            v4,pos_features2 = featurizer(analyzer='char',n_gram=(1,5),max_features=8000,\
                                                   col="feat",dat=df)  
            tfdif_vectors=[v1,v2,v3,v4]
            all_features = hstack([word_features,char_features,pos_features1,pos_features2])
        
        output = os.path.join(model_dir, "tfidf_pre_trained_vectors.pkl")
        print('Saving TF-IDF vectors to {}'.format(output)) 
        f=open(output, "wb")
        pickle.dump(tfdif_vectors,f)
        f.close()
        del output

    else:
        #load model artifacts from pickled object
        inp = os.path.join(model_dir, "tfidf_pre_trained_vectors.pkl")
        f=open(inp, "rb")
        tfidf_vectors=pickle.load(f)
        f.close()
        
        ## concatenate by columns raw_data features in one big scipy matrix
        v1=tfidf_vectors[0]; v2=tfidf_vectors[1]; 
        word_features=v1.transform(df.sent.values.astype('U'))
        char_features=v2.transform(df.sent.values.astype('U'))
        all_features = hstack([word_features,char_features])
            
        if(spacy=="yes"):
            v3=tfidf_vectors[2]; v4=tfidf_vectors[3]
            pos_features1=v3.transform(df.feat.values.astype('U'))  
            pos_features2=v4.transform(df.feat.values.astype('U'))
            all_features = hstack([word_features,char_features,pos_features1,pos_features2])
    print("Featurization process completed successfully")
    print(all_features.shape)
    
    if(train_eval_predict=='predict' or tag=="unlabelled"):
        return all_features
    else:
        labels=df[["display_example"]] 
        return all_features,labels
