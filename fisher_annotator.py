"""
Pre-processing and annotating Fisher transcripts using 
a SOTA joint parser and disfluency detector model. For 
a complete description of the model, please refer to 
the following paper:
https://www.aclweb.org/anthology/2020.acl-main.346.pdf


* DisfluencyTagger --> finds disfluency labels
* Parser --> finds constituency parse trees
* Annotate --> pre-processes transcripts for annotation

(c) Paria Jamshid Lou, 14th July 2020.
"""

import codecs
import fnmatch
import os
import sys
import re   
import torch
import pickle
import json

import disfluency.parse_nk as parse_nk
import disfluency.vocabulary as vocab

ABS_ROOT = os.getenv('MAIN_ROOT')
sys.path.insert(0,os.path.abspath(os.path.join(ABS_ROOT, "disfluency"))) # Remember to add this line to avoid "module no exist" error

# dir_path = os.path.dirname(os.path.realpath(__file__))
# parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# sys.path.insert(0, parent_dir_path)


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


class DisfluencyTagger:
    """
    This class is called when self.disfluency==True.    

    Returns:
        A transcript with disfluency labels:
            e.g. "she E she _ likes _ movies _"
            where "E" indicate that the previous 
            word is disfluent and "_" shows that 
            the previous word is fluent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    @staticmethod
    def fluent(tokens):
        leaves_tags = [t.replace(")","")+" _" for t in tokens if ")" in t]      
        return " ".join(leaves_tags)

    @staticmethod
    def disfluent(tokens):
        # remove first and last brackets
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        tags = []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if "(EDITED" in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
                
            elif ")" in tokens[pointer]:
                label = "E" if df_region else "_"  
                tags.append(
                    (tokens[pointer].replace(")", ""), label)
                    )                 
            if all(
                (close_bracket,
                open_bracket == close_bracket)
                ):
                open_bracket, close_bracket = 0, 0
                df_region = False            

            pointer += 1
        return " ".join(list(map(lambda t: " ".join(t), tags)))


class Parser(DisfluencyTagger):
    """
    Loads the pre-trained parser model to find silver parse trees     
   
    Returns:
        Parsed and disfluency labelled transcripts
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def torch_load(self):
        vocabulary = vocab.Vocabulary()
        if parse_nk.use_cuda:
            return torch.load(
                self.model
                )
        else:
            return torch.load(
                self.model, 
                map_location=lambda storage, 
                location: storage,
                )

    def run_parser(self, input_sentences, get_annotations=False):
        eval_batch_size = 1
        print("Loading model from {}...".format(self.model))
        assert self.model.endswith(".pt"), "Only pytorch savefiles supported"

        info = self.torch_load()
        assert "hparams" in info["spec"], "Older savefiles not supported"
        parser = parse_nk.NKChartParser.from_spec(
            info["spec"], 
            info["state_dict"],
        )
        # sentences = [sentence.split() for sentence in input_sentences]
        # sentences = [ y.split() for utt_id, y in input_sentences.items() if utt_id == "speakerIp16_B1a_003009001013-promptIp16_B1_en_35_248_105" ]
        # utt_ids = [ "speakerIp16_B1a_003009001013-promptIp16_B1_en_35_248_105" ]
        sentences = [ y.split() for _, y in input_sentences.items() ]
        utt_ids = [ utt_id for utt_id, _ in input_sentences.items() ]
        # Tags are not available when parsing from raw text, so use a dummy tag
        if "UNK" in parser.tag_vocab.indices:
            dummy_tag = "UNK"
        else:
            dummy_tag = parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            if get_annotations:
                # BUG: the length of vector dimension is not enough if encounter long utterances
                try:
                    predicted = parser.parse_batch(subbatch_sentences, get_annotations=True)
                except:
                    predicted = ([], [], [])
                all_predicted.append(predicted)
            else:
                predicted, _ = parser.parse_batch(subbatch_sentences, get_annotations=False)
                del _
                all_predicted.extend([p.convert() for p in predicted])

        if get_annotations: 
            return dict(zip(utt_ids, all_predicted))
        
        parse_trees, df_labels = [], []
        for tree in all_predicted:          
            linear_tree = tree.linearize()
            parse_trees.append(linear_tree)
            if self.disfluency:
                tokens = linear_tree.split()
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                else:
                    df_labels.append(self.disfluent(tokens))
                    
        return zip(utt_ids, parse_trees, df_labels)



class Annotate(Parser):   
    """
    Writes parsed and disfluency labelled transcripts into 
    *_parse.txt and *_dys.txt files, respectively.

    """ 
    def __init__(self, **kwargs):
        self.input_json = kwargs["input_json"]
        self.output_path = kwargs["output_path"] 
        self.output_pickle_path = kwargs["output_pickle_path"]
        self.model = kwargs["model"] 
        self.disfluency = kwargs["disfluency"] 
        self.save_pickle = kwargs["save_pickle"]
        self.get_annotations = kwargs["get_annotations"]

    def setup(self): 
        go = self.parse_sentences(
            get_annotations=self.get_annotations
        )

    def jsonLoad(self, input_json):
        with open(input_json) as json_file:
            return json.load(json_file)['utts'] # remove the utts layer

    def parse_sentences(self, get_annotations=False):
        
        dict_json = self.jsonLoad(self.input_json)
        segments = self.read_transcription(dict_json)
        # Loop over cleaned/pre-proceesed transcripts         
        gets = {segment[0]:[segment[1], segment[2]] for segment in segments}
        docs = {utt_id:segment[0] for utt_id, segment in gets.items()}
        utt_ids = list(docs.keys())
        tokens_combine_lists = {utt_id:segment[1] for utt_id, segment in gets.items()}
        embds = []

        if get_annotations:

            for utt_id, (v, b, w_i_l) in self.run_parser(docs, get_annotations=True).items():
                # Use dic here to meet the requirements of ESPNet toolkit
                if w_i_l:
                    _add = {
                        'word_index_list': w_i_l,
                        'vector_annotator_encoder': v.detach().cpu().clone().numpy().tolist(),
                        'vector_bert_last_hidden_layer': b.squeeze(0).detach().cpu().clone().numpy().tolist(),
                        'tokens_combine_lists': tokens_combine_lists[utt_id]
                    }
                else:
                    _add = {
                        'word_index_list': w_i_l,
                        'vector_annotator_encoder': [],
                        'vector_bert_last_hidden_layer': [],
                        'tokens_combine_lists': tokens_combine_lists[utt_id]
                    }
                embds.append(_add)
                del _add

            save_json = {
                'utts': dict(zip(utt_ids, embds))
            }

            # filter out the utt with problems
            problems = []
            for utt_id, d in save_json['utts'].items():
                if not d.get('word_index_list'):
                    problems.append(utt_id)

            new_filename = os.path.abspath(
                self.output_path
            )

            with open(new_filename+".problems", "w") as f:
                for utt_id in problems:
                    f.write("{}\n".format(utt_id))

            if self.save_pickle:

                assert self.output_pickle_path != None

                if not os.path.exists(self.output_pickle_path):
                    os.mkdir(self.output_pickle_path)

                save_pk_path = {}

                for utt_id, d in save_json['utts'].items():
                    
                    if utt_id in problems:
                        continue

                    save_pk = os.path.abspath(
                        os.path.join(
                            self.output_pickle_path,
                            utt_id+'.pk'
                        )
                    )
                    save_pk_path[utt_id] = {'save_path_pickle': save_pk}
                    pickleStore(d, save_pk)

                with open(new_filename, "w") as output_file:
                    json.dump({'utts':save_pk_path}, output_file, indent=4)
                
                return

            with open(new_filename, "w") as output_file:
                json.dump(save_json, output_file, indent=4)

        else:

            save_json = {
                'utts': {}
            }

            for (utt_id, parse_tree, df_label) in self.run_parser(docs, get_annotations=False):

                save_json['utts'][utt_id] = {
                    'pos_dis_labels': {
                        'parse_tree': parse_tree,
                        'df_label': self.process_edit_tagging(df_label)
                    }
                }

            # Write constituency parse trees and disfluency labels into files
            new_filename = os.path.abspath(
                self.output_path
            )
            with open(new_filename, "w") as output_file:
                json.dump(save_json, output_file, indent=4)

        return

    def process_edit_tagging(self, df_label):

        new_df_label = []
        df_label_ = df_label.split()
        for i, t in enumerate(df_label_[::2]):
            t = str(t)
            s = df_label_[i*2+1].upper()
            if s == 'E':
                new_df_label.append(t.upper())
            elif s == "_":
                new_df_label.append(t)

        return " ".join(new_df_label)

    def read_transcription(self, dict_json):
        for utt_id, data in dict_json.items():
            tokens_str = data.get("input")[1].get("stt")
            yield ( utt_id, *self.validate_transcription(tokens_str, utt_id) )

    @staticmethod
    def validate_transcription(label, utt_id):

        # the preprocess has already deal the special characters
        _replace = ["'s", "'m", "'d", "'ll", "n't", "'ve", "'re"]

        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace(";", "")
        label = label.replace("?", "")
        label = label.replace("!", "")
        label = label.replace(":", "")
        label = label.replace("\"", "")
        label = label.replace("'re", " 're")
        label = label.replace("'ve", " 've")
        label = label.replace("n't", " n't")
        label = label.replace("'ll", " 'll")
        label = label.replace("'d", " 'd")
        label = label.replace("'m", " 'm")
        label = label.replace("'s", " 's")
        label = label.strip()
        label = label.lower()

        label_list = ["[CLS]", *label.split(), "[SEP]"]

        r = []
        for i, t in enumerate(label_list):
            if t in _replace and label_list[i-1] not in _replace and i-1 >= 0:
                if r[-1][-1] == i-1:
                    r = r[:-1]
                r.append([i-1, i])
            else:
                r.append([i])

        # Check
        assert len(label_list) == len(list(set(list(sum(r, [])))))

        return (label, r) if label else None

