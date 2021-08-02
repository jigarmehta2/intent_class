import json
from preprocess import *
from featurizer import *
import argparse
import warnings
warnings.filterwarnings("ignore")

def run_inference(payload,model_dir="../models/",train_eval_predict="predict",spacy_features="no"):

    out={
    "statusCode": 400,
    "data": [
            {
            "intent1": '',
            "score1": 0,
            "tag":""
            },
          {
            "intent2": '',
            "score2": 0,
            "tag":""
           },
          {
            "intent3": '',
            "score3": 0,
            "tag":""
           }],
    "message": ""
    }
    
    if(bool(json.loads(payload))==False):
        out['message']="Null string received as input"
    else:
        df=pd.DataFrame.from_dict(json.loads(payload), orient='index',columns=['Sentence']).reset_index(drop=True)
        df['word_len']=df.Sentence.str.strip().str.split(" ").map(len)
        df=df[df.word_len>2]
        if df.empty:
            out['message']="Less than 2 words passed to Fidelity NLU model"
   
   #call preprocess and featurizer 
    if(len( out['message'])==0):
        try:
            clean_data=run_preprocess( raw_data=df,spacy=spacy_features)  
            pred_features=run_featurizer(clean_data,train_eval_predict=train_eval_predict,\
                                                model_dir=model_dir,spacy=spacy_features)

            #print("\nRunning Model Evaluation on Test Data")
            #load model artifacts from pickled object
            print("\n")
            model_input = os.path.join(model_dir, "logistic_regression_model.pkl")
            f=open(model_input, "rb")
            model_obj=pickle.load(f)
            f.close()
            
            model=model_obj[0]
            lookup=model_obj[1]
            
            labels=list(model.classes_)
            intents=[]
            y_test_pred_mat=pd.DataFrame(np.argsort(model.predict_proba(pred_features), axis=1)[:,-3:])
            score=pd.DataFrame(np.sort(model.predict_proba(pred_features), axis=1)[:,-3:])
            score=score.apply(lambda x: round(x,3))
            y_test_pred_mat.columns=['Int3','Int2','Int1']
            score.columns=['Score3','Score2','Score1']

            for ind, row in y_test_pred_mat.iterrows():
                intents.append([labels[row['Int3']],labels[row['Int2']],labels[row['Int1']]])
            intents=pd.DataFrame(intents,columns=['Int3','Int2','Int1'])
            intents=intents[intents.notnull()]
            score=score[score.notnull()]
            if not intents.empty and not score.empty:
                out['statusCode']=200
                out['message']="Inference Success"
                out["data"]= [{
                    "intent1": intents.Int1[0],
                    "score1": score.Score1[0],
                    "tag":lookup[intents.Int1[0]]},
                    {
                    "intent2": intents.Int2[0],
                    "score2": score.Score2[0],
                    "tag":lookup[intents.Int2[0]]},
                    {
                    "intent3": intents.Int3[0],
                    "score3": score.Score3[0],
                    "tag":lookup[intents.Int3[0]]} ]
        except:
            out['message']="Error occured in Preprocess/TF-IDF/Inference stage"
  
    return json.dumps(out,indent=3)
