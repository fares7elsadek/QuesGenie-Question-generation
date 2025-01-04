import torch
import math
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from collections import namedtuple
from app.Modules.process import _create_features_from_records
import re
from nltk.corpus import wordnet as wn
from torch.nn.functional import softmax
from app.Preprocessing import *
import statistics
from statistics import mode
import re


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GlossSelectionRecord = namedtuple("GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])
MAX_SEQ_LENGTH = 128



class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()



class MatchingQuestions():
    def __init__(self,text,num_words=10):
        self.text = text
        self.model_dir = "app/Model/BERT-WSD"
        self.model = BertWSD.from_pretrained(self.model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        self.tokenizer.added_tokens_encoder['[TGT]'] = 100
        if '[TGT]' not in self.tokenizer.additional_special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.textPreprocessing = TextPreprocessing(text,num_words)
        self.mapping_keywords = self.textPreprocessing.get_sentences_for_keyword()
        self.model.to(device)
        self.model.eval()
    
    def get_sense(self,sent):
        re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
        if re_result is None:
            print("\nIncorrect input format. Please try again.")

        ambiguous_word = re_result.group(1).strip()
        results = dict()

        for i, synset in enumerate(set(wn.synsets(ambiguous_word))):
            results[synset] =  synset.definition()

        if len(results) ==0:
            return None

        sense_keys=[]
        definitions=[]
        for sense_key, definition in results.items():
            sense_keys.append(sense_key)
            definitions.append(definition)

        record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

        features = _create_features_from_records([record], MAX_SEQ_LENGTH, self.tokenizer,
                                                    cls_token=self.tokenizer.cls_token,
                                                    sep_token=self.tokenizer.sep_token,
                                                    cls_token_segment_id=1,
                                                    pad_token_segment_id=0,
                                                    disable_progress_bar=True)[0]

        with torch.no_grad():
            logits = torch.zeros(len(definitions), dtype=torch.double).to(device)
            for i, bert_input in list(enumerate(features)):
                logits[i] = self.model.ranking_linear(
                    self.model.bert(
                        input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(device),
                        attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(device),
                        token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(device)
                    )[1]
                )
            scores = softmax(logits, dim=0)

            preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))

        sense = preds[0][0]
        meaning = preds[0][1]
        return sense


    def get_synsets_for_word (self,word):
        return set(wn.synsets(word))
    

    def get_matching_questions(self):
        keyword_best_sense = {}
        for keyword in self.mapping_keywords:
            try:
                identified_synsets = self.get_synsets_for_word(keyword)
            except:
                continue
    
            top_3_sentences = self.mapping_keywords[keyword][:3]
            best_senses = []
            for sent in top_3_sentences:
                insensitive_keyword = re.compile(re.escape(keyword), re.IGNORECASE)
                modified_sentence = insensitive_keyword.sub(" [TGT] " + keyword + " [TGT] ", sent, count=1)
                modified_sentence = " ".join(modified_sentence.split())
                best_sense = self.get_sense(modified_sentence)
                if best_sense is not None:
                    best_senses.append(best_sense)
    
            if best_senses:
                try:
                    best_sense = mode(best_senses)
                    defn = best_sense.definition()
                    keyword_best_sense[keyword] = defn
                except statistics.StatisticsError:
                    # Skip keyword if no dominant sense is found
                    continue
    
        return keyword_best_sense
