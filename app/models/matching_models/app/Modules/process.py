import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from tqdm import tqdm
from collections import namedtuple

GlossSelectionRecord = namedtuple("GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])


def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [(gloss, 1 if i in record.targets else 0) for i, gloss in enumerate(record.glosses)]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label)
            )

        features.append(pairs)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()