from HMM_add_Feature.Hmm import HiddenMarkovModel
from HMM_add_Feature.document import Document
from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from HMM_add_Feature.process_vlsp_data import ProcessDataVlsp
from utils.settings import DATA_MODEL_DIR
from os.path import join
import numpy as np
document = Document()
pdv = ProcessDataVlsp()

stop_word_path = join(DATA_MODEL_DIR, "stop_word/c_e_l_viettreebank.txt")
list_stop_word = Helper().load_stop_word(stop_word_path)
# path data
path_wiki = join(DATA_MODEL_DIR, "viwiki_data")
# path_data_train = join(DATA_MODEL_DIR, "vlsp/train")
path_vocab = join(DATA_MODEL_DIR, "wiki/vocab_wiki_punt_special.json")
file_feature_e_b = join(DATA_MODEL_DIR, "wiki/feature_independent/feature_basic_idp_punt_special_B_wiki.json")
file_feature_e_i = join(DATA_MODEL_DIR, "wiki/feature_independent/feature_basic_idp_punt_special_I_wiki.json")

#load data
helper = Helper()
# data = pdv.load_data_path(path_data_train)
data_wiki = helper.load_data_xml_path(path_wiki)
vocab_number = helper.loadfile_data_json(path_vocab)
syllables_vn = helper.load_dictionary("syllables_dictionary_1.txt")
punt = helper.load_punctuation()
result = []
list_folder = helper.GetFolder(path_wiki)

# covert data wiki
for folder in list_folder:
    list_file = helper.GetFiles(folder)
    print(folder)
    for file in list_file:
        list_doc = helper.load_data_xml(file)
        for doc in list_doc:
            doc_number = document.convert_doc_to_number_array(doc, vocab_number, syllables_vn, punt)
            if doc_number != []:
                result.extend(doc_number)
            else:
                continue
print(len(result))
# # print(data[1])
# for doc in data:
#         result.extend(pdv.convert_doc_to_number(doc, vocab_number, syllables_vn, punt, True))
# for i in result:
#     if i != []:
#         result_0.append(i)

# set model
states = [0, 1]
diction = Dictionary()
vocab_feature_e = diction.load_file_feature_e(file_feature_e_b, file_feature_e_i)
vocab_e = diction.covert_feature_to_array(vocab_feature_e, len(vocab_number), list_stop_word, False)
print(len(vocab_e))
vocab_t = diction.gen_feature_basic_t()
start_probabilities = [1, 0]
W_emission = np.random.rand(2, len(vocab_number))
print(W_emission.shape)
W_transition = np.array([[0.8, 0.4], [0.9, 0.1]], dtype=np.float64)
print(vocab_t)

hmm = HiddenMarkovModel(states, W_transition, W_emission, start_probabilities, vocab_e, vocab_t, vocab_number)
print(hmm.get_matrix_transition())
print(hmm.get_matrix_emission())
hmm.baum_welch_algorithm(result, 1)

# save model
hmm.save_model("model_basic_idp_punt_special_wiki.pickle")

print(hmm.get_matrix_emission())
print(hmm.get_matrix_transition())
