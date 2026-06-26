from datasets import load_dataset
from transformers import AutoTokenizer
class P1:
	def __init__(self):
		# Load ultrafeedbackdataset
		self.dataset = load_dataset("trl-lib/ultrafeedback_binarized")
		# Use Qwen model as tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
		# Valid keys for dataset
		self.valid_keys = ("prompt", "chosen", "rejected")
		
	def load_data(self):
		train_data = self.dataset["train"]	
		# Return the data
		return train_data
	
	def cal_avg_length(self, type="chosen"):
		if type not in self.valid_keys:
			raise ValueError("Type must be in chosen or rejected")
		train_dataset = self.load_data()
		total_row = len(train_dataset)
		# Debug print
		print(f"Total length is {total_row}")
		length_list = []
		batch_size = 1000
		for i in range(0, total_row, batch_size):
			# Batch process to enhance the efficiency
			batch = train_dataset[i:i+batch_size]
			# Every conversation is prompt+content
			text_batch = [message[-1]["content"] for message in batch[type]]
			
			token_ids_batch = self.tokenizer(text_batch, add_special_tokens=False)["input_ids"]
			
			# Use tokenizer to calculate the token number
			lengths = [len(tokens) for tokens in token_ids_batch]
			length_list.extend(lengths)
		avg_length =  sum(length_list)/total_row
		return avg_length
	
	def print_data(self):
		data = self.load_data()
		data = data[0]
		print(data)
	def print_avg_length(self):
		chosen_avg_length = self.cal_avg_length(type="chosen")
		rejected_avg_length = self.cal_avg_length(type="rejected")
		print(f"Chosen average responses token lenght is {chosen_avg_length}. \nRejected average responses token length is {rejected_avg_length}.")
		
			
		

if __name__=='__main__':
	p1 = P1()
	# 1
	# p1.print_data()
	# 2
	p1.print_avg_length()
	
		


