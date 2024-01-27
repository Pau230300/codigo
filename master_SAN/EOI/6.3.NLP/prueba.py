from nltk.tokenize import WordPunctTokenizer 
from nltk.tokenize import TreebankWordTokenizer 
texto = "Manolito consiguió hacerle la vida dificilísima (y supongo que del todo aburrida) a Mafalda. Pero ella, que odiaba la O.T.A.N., no se rindió."

WordPunctTokenizer().tokenize(texto)
TreebankWordTokenizer().tokenize(texto)


nltk.word_tokenize(texto)
import nltk
nltk.download()

from nltk.stem import WordNetLemmatizer

lematizer = WordNetLemmatizer()

lematizer.lemmatize('Manolito', pos='n')