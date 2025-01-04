import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor



class TextPreprocessor():
    def __init__(self,text):
        self.text = text
    
    def tokenize_sentence(self,text):
        sentences = sent_tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences
    
    def get_sentence_for_keyword(self,keywords):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in self.sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)
        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences
