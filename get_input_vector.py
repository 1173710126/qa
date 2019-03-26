import build_vocab_dic
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# 将questions字符串问题列表  转化为二维矩阵，即第一维是第i个问题， 第二维是第i个问题的第j个词语对应的数字
class Get_Input:
	def __init__(self, questions, max_sentence_len = 10):
		self.questions = questions
		self.max_sentence_len = max_sentence_len
	def build(self):
		texts = build_vocab_dic.build_vocab_dic('F:/NLP/SimpleQuestions_v2/annotated_fb_data_train.txt').build()


		tokenizer = Tokenizer(num_words = None)
		tokenizer.fit_on_texts(texts)                        # 导入文本建立词典
		questions = tokenizer.texts_to_sequences(self.questions)  # 将字符串列表 转换成矩阵， 即每个词汇变成一个数字
		
		questions = pad_sequences(questions, maxlen = self.max_sentence_len) # 默认向前0填充
		return questions