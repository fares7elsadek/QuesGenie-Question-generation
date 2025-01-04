import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
import pke
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor


class TextPreprocessing():
    def __init__(self,text,num_words=10):
        self.text = text
        self.num_words = num_words
    
    def tokenize_sentences(self):
        sentences = sent_tokenize(self.text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences
    
    def get_keywords(self):
        text = self.text
        out=[]
        try:
            extractor = pke.unsupervised.YAKE()
            extractor.load_document(input=text,language='en')
            grammar = r"""
                    NP:
                        {<NOUN|PROPN>+}
                """
            extractor.ngram_selection(n=1)
            extractor.grammar_selection(grammar=grammar)
            extractor.candidate_selection(n=1)
            extractor.candidate_weighting(window=3,
                                        use_stems=False)
            keyphrases = extractor.get_n_best(n=30)
            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()

        return out[:self.num_words]
    

    def get_sentences_for_keyword(self):
        keywords = self.get_keywords()
        sentences = self.tokenize_sentences()
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=False)
            keyword_sentences[key] = values
        return keyword_sentences