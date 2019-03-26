import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.layers import Input, Dense, Embedding
from keras.models import Model
import get_input_vector
import numpy as np

class QuestionEmbeddingLayer:
	def __init__(self,  questions, input_length=10, vocab_size=77167, embedd_dim=3000):
		self.questions = questions
		self.input_length = input_length
		self.vocab_size = vocab_size
		self.embedd_dim = embedd_dim	
	def build(self):	
		#weights = np.config('word2vec_100_dim.embeddings')
		questions_embedding = Embedding(input_dim = self.vocab_size, 
							  			output_dim = self.embedd_dim, 
							  			input_length = self.input_length)(self.questions) # weights.shape[1]
		return questions_embedding

'''
class AnswerEmbeddingLayer:
	answers = Input(shape = (answer_len,), dtype = 'int32') # kb_num是one_hot向量的维度， 每个知识库实体和关系都表示成one_hot，
	answers_embedding = Embedding(input_dim = kb_num, 
								  output_dim = weights.shape[1])(answers)

'''
#path_name = 
#training_set = pickle.load(open(os.path.join(path_name),'rb'))
if __name__ == '__main__':
	input_length = 10
	vocab_size = 7000
	embedd_dim = 300

	question_vector = Input(shape = (input_length,), dtype = 'int32')
	question_embedding = QuestionEmbeddingLayer(questions, input_length, vocab_size, embedd_dim).build()

	training_key_set =  ['how are you', 'you are welcome']# 把自然语言问题 先 转化为 词语列表， 再把每个词转化为one-hot向量
	input_data = get_input_vector.Get_Input(training_key_set).build()
	print(input_data.shape)
	'''
	model = Model(inputs=questions, outputs=questions_embedding)
	print(model.summary())
	model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])
	return model
	'''
	#model.fit(questions, answers, epochs = 50)
	#loss_and_metrics = model.evaluate(questions, answers)
	'''
	input_questions_dim = (len(questions), max_sentence_len, len(vocab_dic))
	questions_one_hot = np.zeros(input_questions_dim)
	for i, sample in enumerate(questions):
		sample = sample.split()
		#print("")
		for j, word in enumerate(sample):
			word = word.encode('utf-8')
			index = vocab_dic.get(word)
			#print(word, index)
			questions_one_hot[i, j, index] = 1        # 第i条问句， 对所有1<=j<=len(sample),第j个词汇的 下标为index的位置设为1
	'''
	#print(questions_one_hot, questions_one_hot.shape)