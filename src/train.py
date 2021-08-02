from preprocess import *
from featurizer import *
import argparse
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default="../models/")
    parser.add_argument('--train_eval_predict', type=str, default='train')
    parser.add_argument('--use_social', type=str, default='no')
    parser.add_argument('--spacy_features', type=str, default='yes')
    
    args = parser.parse_args()
    
    colnames=['Sentence','Intent','ExternalId','display_example','Status','count']
    input_file =  os.path.join(args.data_dir, args.train_file) 
    if(os.path.exists(input_file) == False):
        raise ValueError(('There are no files in {}.\n' +
                          'This indicates that the channel ({}) was incorrectly specified OR ,\n' +
                          'the data specification in S3 was incorrectly specified OR the role specified\n' +
                          'does not have permission to access the data.').format(input_file, "input_file"))
   
    df = pd.read_csv( input_file,error_bad_lines=False, warn_bad_lines=False,header=0)
    #df=df.sample(frac=0.1)
    df.columns=colnames
    lookup=df[['display_example','Intent']]  
    lookup.drop_duplicates(inplace=True)
    lookup=lookup.set_index("display_example")['Intent'].to_dict()
    
    df=df[['Sentence','Intent','display_example']]    
    print("Input file dimension : ", df.shape)
    print("#Unique Intents : ", df.display_example.nunique())
    df=df[df.Sentence.notnull()]
    df=df[df.display_example.notnull()]
    
    if(args.use_social=="no"):  ##default value is no
        df=df[~df['Intent'].isin(['SENTENCE_FRAGMENTS',\
                                                    'INCOMPREHENSION','NO_INTENT','OUT_OF_SCOPE','YES','NO'])]
        df=df[~df.Intent.str.startswith('_')]
        df=df[~df.Intent.str.contains('SOCIAL')]
    
    df['word_len']=df.Sentence.str.split(" ").map(len)
    df=df[df.word_len>1]
    print("Unique Intents after preprocessing : " + str(df.display_example.nunique()))
       
    # call preprocess 
    clean_data=run_preprocess( raw_data=df,spacy=args.spacy_features)
    
    # call featurizer
    try:
        train_features,y_train=run_featurizer(clean_data,train_eval_predict=args.train_eval_predict,\
                                              model_dir=args.model_dir,spacy=args.spacy_features)
        
        print('\nTraining data feature dimension: {}'.format(train_features.shape))
        print('Training labels dimension: {}'.format(y_train.shape))
    
    except:
        print("\nError occured in TF-IDF featurization stage")
  
    ### model training
   
    try:
        y_train=np.ravel(y_train)
        print("\nTraining Logistic Regression Model for " + str( len(np.unique(y_train))) +  " Intents" )
        model = LogisticRegression(random_state=100,n_jobs=-1, max_iter=10,multi_class= 'ovr',C=10.0,\
                                        fit_intercept=False,dual=True,solver='liblinear')
        
        model.fit(train_features, y_train)
        model_output = os.path.join(args.model_dir, "logistic_regression_model.pkl")
        print('Saving trained logistic_regression model  to {}'.format(model_output)) 
        f=open(model_output, "wb")
        model_obj=[model,lookup]
       
        pickle.dump( model_obj,f)
        f.close()
        print("\nTraining Process Completed Successfully")
        del model
    except:
        print("\nError occured in  Model Training process")