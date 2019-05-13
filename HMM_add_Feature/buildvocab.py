from HMM_add_Feature.Dictionary import Dictionary
from utils.settings import DATA_MODEL_DIR
from os.path import join
from HMM_add_Feature.Helper import Helper
import pickle

if __name__ == "__main__":

	path_stop_word = join(DATA_MODEL_DIR, "stop_word/c_e_l_viettreebank.txt")
	path_vlsp = join(DATA_MODEL_DIR, "vlsp/train")
	path_wiki = join(DATA_MODEL_DIR, "viwiki_data")
	path_syllable_vn = join(DATA_MODEL_DIR, "syllable_vn/syllables_dictionary_1.txt")
	dic = Dictionary()
	# dic.build_vocab(path_wiki, "vocab_wiki_punt_special_1.json", path_syllable_vn, "wiki", True)
	path_vocab = join(DATA_MODEL_DIR, "wiki/vocab_wiki_punt_special_1.json")
	vocab = Helper().loadfile_data_json(path_vocab)
	print(len(vocab))
	vocab_feature_basic = dic.gen_feature_basic_e(vocab)
	Helper.write_json(vocab_feature_basic[0], "feature_basic_idp_punt_special_B_wiki_1.json")
	Helper.write_json(vocab_feature_basic[1], "feature_basic_idp_punt_special_I_wiki_1.json")

	vocab_feature_enhance = dic.add_enhance_to_feature_e(vocab_feature_basic, path_stop_word)
	Helper.write_json(vocab_feature_enhance[0], "feature_enhance_idp_punt_special_B_wiki_1.json")
	Helper.write_json(vocab_feature_enhance[1], "feature_enhance_idp_punt_special_I_wiki_1.json")
	# # print(dic.gen_feature_basic_t())





