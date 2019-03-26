import os
# 将字典写入文件						
#if not os.path.exists('vocab_dic.txt'):	
#	os.makedirs('vocab_dic')
#file = open('vocab_dic', 'w')
#for word in dic:
#	file.write(str(word) + ' ')
#file.close()

# 将文件中的问题提取出来，然后加入列表，得到问题列表texts
class build_vocab_dic:
	def __init__(self, path):
		self.path = path
	def build(self):
		texts = []
		path = self.path

		file = open(path, 'rb')
		while True:
			line = file.readline().strip()
			if not line:
				break
			words = line.split()[3:]
			for i in range(len(words)):
				words[i] = words[i].decode('utf-8')
			question = ' '.join(words)
			texts.append(question)
		return texts 	

		                    
if __name__ == '__main__':

	vocab_dic = build_vocab_dic(path = 'F:/NLP/SimpleQuestions_v2/annotated_fb_data_train.txt').build()
	print('vocab_dic length:', len(vocab_dic))
	print(vocab_dic[0:10])
