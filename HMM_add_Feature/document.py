from HMM_add_Feature.Helper import Helper
from HMM_add_Feature.Dictionary import Dictionary
from utils.settings import DATA_MODEL_DIR
from os.path import join
import re
helper = Helper()
class Document:
	"""class for process doc to list syllable and to number"""

	@staticmethod
	def detect_paragraph(doc):
		paragraphs = doc.splitlines()
		return paragraphs

	@staticmethod
	def split_sentences(paragraphs_array):
		array_pharagraph_with_sentence = []
		for paragraph in paragraphs_array:
			if paragraph:
				paragraph = paragraph.replace(". ", ".. ")
				sentences = paragraph.split(". ")
				for sentence in sentences:
					if sentence:
						sentences = Helper().clear_str(sentence)
						array_pharagraph_with_sentence.append(sentences)
		return array_pharagraph_with_sentence

	def convert_doc_to_number(self, doc, vocab_number, syllables_vn, punctuation, punt=False):
		"""
		function return list sentence, each sentence is list dict, each dict is a syllable
		:param doc:
		:param vocab_number:
		:param syllables_vn:
		:param punctuation:
		:param punt defaul = False sẽ không coi dấu câu là 1 âm tiết
		:return:
		"""
		list_pharagraph = self.detect_paragraph(doc)
		list_sentence = self.split_sentences(list_pharagraph)
		# syllables_appear = vocab_number.keys()
		list_sentence_convert_to_number = []
		for sentence in list_sentence:
			list_syllable_sentence = sentence.split()
			list_syllable_number_sentence = []
			for syllable in list_syllable_sentence:
				syllable_number = {}
				type_syllable = helper.check_type_syllable(syllable, syllables_vn, punctuation)
				if type_syllable == "VIETNAMESE_SYLLABLE" and syllable in vocab_number:
					syllable_number[syllable] = vocab_number.get(syllable)
				elif type_syllable == "PUNCT":
					if punt == False:
						syllable_number[syllable] = vocab_number.get(syllable)
				elif type_syllable == "VIETNAMESE_SYLLABLE" and syllable not in vocab_number:
					syllable_number[syllable] = vocab_number.get("FOREIGN_SYLLABLE")
				else:
					syllable_number[syllable] = vocab_number.get(type_syllable)
				list_syllable_number_sentence.append(syllable_number)
			list_sentence_convert_to_number.append(list_syllable_number_sentence)
		return list_sentence_convert_to_number

	def convert_doc_to_number_array(self, doc, vocab_number, syllables_vn, punctuation):
		"""
		covert doc to array number
		:param doc:
		:param vocab_number:
		:param syllables_vn:
		:param punctuation:
		:param punt:
		:return:
		"""
		list_pharagraph = self.detect_paragraph(doc)
		list_sentence = self.split_sentences(list_pharagraph)
		list_sentence_convert_to_number = []
		for sentence in list_sentence:
			list_syllable_sentence = sentence.split()
			list_syllable_number_sentence = []
			for syllable in list_syllable_sentence:
				if syllable in vocab_number:
					list_syllable_number_sentence.append(vocab_number.get(syllable))
				else:
					type_syllable = helper.check_type_syllable(syllable, syllables_vn, punctuation)
					if type_syllable == "VIETNAMESE_SYLLABLE" and syllable not in vocab_number:
						list_syllable_number_sentence.append(vocab_number.get("FOREIGN_SYLLABLE"))
					else:
						list_syllable_number_sentence.append(vocab_number.get(type_syllable))
			list_sentence_convert_to_number.append(list_syllable_number_sentence)
		return list_sentence_convert_to_number

	def test_sentence(self, doc):
		list_syllable_sentence = []
		list_pharagraph = self.detect_paragraph(doc)
		list_sentence = self.split_sentences(list_pharagraph)
		for sentence in list_sentence:
			list_syllable_sentence.append(sentence.split())
		return  list_syllable_sentence

	def convert_string_to_number(self, sentence):
		"""

		:param sentence
		"""
		strings = []
		index = []
		for syllable in sentence:
			strings.append(list(syllable.keys())[0])
			index.append(list(syllable.values())[0])
		return strings, index



if __name__ == "__main__":

	doc = helper.load_data_xml("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/data/viwiki_data/AA/wiki_00")
	vocab = helper.loadfile_data_json("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/data/wiki/vocab_wiki_punt_special.json")
	punt = helper.load_punctuation()
	syllable_vn = helper.load_dictionary("/home/trang/Downloads/job_rabiloo/Word_Tokenizer/data/syllable_vn/syllables_dictionary_1.txt")
	document = Document()
	doc_array = document.convert_doc_to_number_array(doc[1], vocab, syllable_vn, punt)
	print(doc_array)