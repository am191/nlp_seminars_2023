import spacy
import pandas as pd
from spacy.tokens import Doc

nlp = spacy.load("lv_udtb_latvian-2.11.0")
path = '../../../../data/udt211_test_token.txt' 

df = pd.read_csv(path, encoding='utf-8',sep='\t', header=None, names=['token'])

print(df.head())
print(df.count())
text = df['token'].values.tolist()
print(len(text))
doc = Doc(nlp.vocab, text) #preparing the test tokens; avoiding default tokenizer

def prediction(doc):
    predicted = pd.DataFrame()
    tokens = []
    pos1 = []
    morph = []

    preds = nlp(doc)
    for token in preds:
        tokens.append(token.text)
        pos1.append(token.pos_)
        morph.append(token.morph)

    print('prediction finished!')
    predicted['token'] = tokens
    predicted['pos_'] = pos1
    predicted['morph'] = morph

    return predicted

def print_txt(df, name):
    df.to_csv(f'../../../../results/{name}', header=False, index=False, sep='\t', mode='w',  encoding='utf-8')
    print('file saved!')

result = prediction(doc)
print_txt(result, 'udt211.txt')

    
