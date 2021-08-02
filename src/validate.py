from preprocess import *
from featurizer import *
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report #evaluation
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#%matplotlib inline

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default="../models/")
    parser.add_argument('--train_eval_predict', type=str, default='eval')
    parser.add_argument('--use_social', type=str, default='no')
    parser.add_argument('--spacy_features', type=str, default='yes')
    parser.add_argument('--tag', type=str, default='labelled')
    
    args = parser.parse_args()
    
    colnames=['Sentence','Intent','display_example']
    input_file =  os.path.join(args.data_dir, args.test_file) 
    if(os.path.exists(input_file) == False):
        raise ValueError(('There are no files in {}.\n' +
                          'This indicates that the channel ({}) was incorrectly specified OR ,\n' +
                          'the data specification in S3 was incorrectly specified OR the role specified\n' +
                          'does not have permission to access the data.').format(input_file, "input_file"))
   

    df = pd.read_csv( input_file,error_bad_lines=False, warn_bad_lines=False,header=0)
    print("Input file dimension : ", df.shape)
    col="Sentence"
    pos=df.columns.get_loc(col)
   
    df[col]=df[[col]].astype(str)
    df=df[df[col].notnull()]
    df=df[df[col]!=""]
    #print(df.shape)
    
    if(args.tag=="labelled"):
        df.columns=colnames
       
        print("#Unique Intents : ", df.display_example.nunique())
        df=df[df.Sentence.notnull()]
        df=df[df.display_example.notnull()]
        
        if(args.use_social=="no"):  ##default value is no
            df=df[~df['Intent'].isin(['SENTENCE_FRAGMENTS',\
                                                        'INCOMPREHENSION','NO_INTENT','OUT_OF_SCOPE','YES','NO'])]
            df=df[~df.Intent.str.startswith('_')]
            df=df[~df.Intent.str.contains('SOCIAL')]
        print("Unique Intents after preprocessing : " + str(df.display_example.nunique()))
          # call preprocess 
        clean_data=run_preprocess( raw_data=df,spacy=args.spacy_features )
        test_features,y_test=run_featurizer(clean_data,train_eval_predict=args.train_eval_predict,\
                                            model_dir=args.model_dir,spacy=args.spacy_features,  tag=args.tag)
        print('\nTest labels dimension: {}'.format(y_test.shape))
        
    else:
        df1=df
        df=df.iloc[:,pos:pos+1]
        #print(df.head)
        #print(df.shape)
        df.columns=["Sentence"]
        clean_data=run_preprocess( raw_data=df,spacy=args.spacy_features )
        
        test_features=run_featurizer(clean_data,train_eval_predict=args.train_eval_predict,\
                                            model_dir=args.model_dir,spacy=args.spacy_features,  tag=args.tag)
    
    print('Test data feature dimension: {}'.format(test_features.shape))
    
    
    print("\nRunning Model Predictions ..")
    #load model artifacts from pickled object
    model_input = os.path.join(args.model_dir, "logistic_regression_model.pkl")
    f=open(model_input, "rb")
    model_obj=pickle.load(f)
    model=model_obj[0]
    #lookup=model_obj[1]
    f.close()

    # evaluate accuracy for 1 intent predicted on test set
    y_test_pred = pd.DataFrame(model.predict(test_features),columns=['display_pred'])
    y_test_pred=  y_test_pred[  y_test_pred.display_pred.notnull()]
    y_test_pred['Pred_conf_score']=list(np.max(model.predict_proba(test_features),axis=1))
    file="predicted_" +args.test_file
    
    if(args.tag!="labelled"):
        labels=list(model.classes_)
        intents=[]
        y_test_pred_mat=pd.DataFrame(np.argsort(model.predict_proba(test_features), axis=1)[:,-3:])
        score=pd.DataFrame(np.sort(model.predict_proba(test_features), axis=1)[:,-3:])
        score=score.apply(lambda x: round(x,3))
        y_test_pred_mat.columns=['Int3','Int2','Int1']
        score.columns=['Score3','Score2','Score1']

        for ind, row in y_test_pred_mat.iterrows():
            intents.append([labels[row['Int3']],labels[row['Int2']],labels[row['Int1']]])
        intents=pd.DataFrame(intents,columns=['Int3','Int2','Int1'])
        intents=intents[intents.notnull()]
        score=score[score.notnull()]
        col=df1.columns[0]
        df1['input_msg_len']=df1[col].str.split(" ").map(len)
        df1.reset_index(drop=True, inplace=True)
        intents.reset_index(drop=True, inplace=True)
        score.reset_index(drop=True, inplace=True)
        
        cols=[]
        cols.extend(df1.columns)
        cols.extend(intents.columns)
        cols.extend(score.columns)
        df1=pd.concat([ df1, intents,score],axis=1,ignore_index=True)
     
        df1.columns=cols
        
        #if not os.path.exists('../output'):
            #os.makedirs('../output')
        file="predicted_"+args.test_file+".csv"
        df1.to_csv(os.path.join("/Users/a656526/Documents/sec_classifier/sec_classifier/output",file),index=False)

    del file
    print("Model predictions Done")
    
    if(args.tag=="labelled"):
        print('\nUnseen Test set accuracy for 1 intent = ', round(accuracy_score(y_test_pred.display_pred,y_test.display_example),4)*100,"%")
        out=pd.DataFrame(classification_report(y_test.display_example,y_test_pred.display_pred , output_dict=True)).transpose()
        out.reset_index(inplace=True)
        out.to_csv("../output/model_validation_pr_report.csv",index=False)
        del out

    ###################### ----------------------------- #############################

    # evaluation for 3 intents

        labels=list(model.classes_)
        y_test_pred_score_mat = model.predict_proba(test_features)
        arr=[]
        y_test_pred_score_mat=pd.DataFrame(np.argsort(y_test_pred_score_mat, axis=1)[:,-3:])
        y_test_pred_score_mat.columns=['I3','I2','I1']
        for ind, row in y_test_pred_score_mat.iterrows():
            arr.append([labels[row['I1']],labels[row['I2']],labels[row['I3']]])
        arr=pd.DataFrame(arr)
        arr=arr[arr.notnull()]
        arr.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        clean_data.reset_index(drop=True, inplace=True)
        
        new=pd.concat([clean_data.Sentence,y_test,y_test_pred,arr],axis=1,ignore_index=True)

        new.columns=['Sentence','True_label','Pred_label','Pred_conf_score','Int1','Int2','Int3']
        new['flag_3_match']=((new.True_label==new.Int3) | (new.True_label==new.Int2) | (new.True_label==new.Int1))*1
        new['flag_1_match']=(new.True_label==new.Int1)*1  

        new['len_of_uttr']=new.Sentence.str.split(" ").map(len)
        new.loc[new.len_of_uttr>25,'len_of_uttr']=25
        new.to_csv("../output/predicted_model_results.csv",index=False)
        print('Unseen Test set accuracy for 3 intents = ',  round((sum(new['flag_3_match'])/new.shape[0])*100,2),"%")
        print("\nModel Validation Completed Successfully")
        #except:
        #    print("\nError occured in Model Validation process")
