def clf(news):
    import joblib
    import pandas as pd
    import re
    import string
    loaded_model=joblib.load("LRmodel.pkl")
    vectorization = joblib.load("vectorization.pkl")

    def output_lable(n):
        if n == 0:
            return "Fake News"
        elif n == 1:
            return "Not A Fake News"
        
    def wordopt(text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W"," ",text) 
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)    
        return text

    
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = loaded_model.predict(new_xv_test)
    out=(output_lable(pred_LR[0]))
    return out
