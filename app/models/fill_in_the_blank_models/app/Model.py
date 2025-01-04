import string
import re
import string
import pke
from app.models.fill_in_the_blank_models.app.Preprocessing import *
from nltk.corpus import stopwords
import traceback


class FillInTheBlankModel():
    def __init__(self,text):
        self.text = text
        self.textpreprocessor = TextPreprocessor(text)
        self.sentences = self.textpreprocessor.tokenize_sentence(text)
    
    def get_noun_adj_verb(self):
        text = self.text
        output = []
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text,language='en')
            pos = {'VERB', 'ADJ', 'NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=30)
            for val in keyphrases:
                    output.append(val[0])
        except:
                output = []
                traceback.print_exc()
        return output
    
    def get_sentence_for_keyword(self):
        keywords = self.get_noun_adj_verb()
        sentences = self.sentences
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
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences
    
    def get_fill_in_the_blanks(self):
        mapping_sentences = self.get_sentence_for_keyword()
        output = {}
        blank_sentences = []
        processed = []
        keys=[]
        for key in mapping_sentences:
            if len(mapping_sentences[key])>0:
                sent = mapping_sentences[key][0]
                # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
                insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
                line = insensitive_sent.sub(' _________ ', sent)
                if (mapping_sentences[key][0] not in processed) and no_of_replacements<2:
                    show = {
                        "question":line,
                        "answer":key
                    }
                    blank_sentences.append(show)
                    processed.append(mapping_sentences[key][0])
                    keys.append(key)
        output["sentences"]=blank_sentences[:10]
        output["keys"]=keys[:10]
        return output