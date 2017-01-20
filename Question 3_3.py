import sys
import random
from random import randint
def read_data(fname):
	labels=[]
	temp_data=[]
	try:
		with open(fname) as docs:
			for line in docs:
				line=line.split()
				temp_data.append(line[1:])
				labels.append(int(line[0]))
	except Exception:
		print "File Not Found, program will exit"
		exit()
	max_value=[int(item[-1].split(":")[0]) for item in temp_data]
	max_features=max(max_value)+1
	data=populate_data(temp_data,labels,max_features)
	return data,labels,max_features

def populate_data(data,labels,max_features):
	final_result=[]
	for row in data:
		temp_data=[0 for i in range(max_features)]
		for item in row:
			item=item.split(":")
			index=int(item[0])
			value=int(item[1])
			temp_data[index]=value
		final_result.append(temp_data)
	return final_result

def populate_weights(max_features):
	weights=[randint(-1,1) for i in range(max_features)]
	return weights

def perceptron(data,labels,weights,epoch,bias,margin,rate,mistakes):
	j=0
	x=0
	while j<epoch:
		for i in range(len(data)):
			dot_product=map(lambda x,y:x*y,weights,data[i])
			derived_label=reduce(lambda x,y:x+y,dot_product)+bias
			actual_label=labels[i]
			if derived_label*actual_label<=margin:
				mistakes+=1
				update=map(lambda x:x*actual_label*rate,data[i])
				weights=map(lambda x,y:x+y,weights,update)
				bias=bias+rate*actual_label
		j+=1
	return weights,mistakes,bias

def calculate_accuracy(weights,data,labels,bias,hits,miss):
	for i in range(len(data)):
		dot_product=map(lambda x,y:x*y,weights,data[i])
		derived_label=reduce(lambda x,y:x+y,dot_product)+bias
		actual_label=labels[i]
		if derived_label*actual_label>0:
			hits+=1
		else:
			miss+=1
	return float(hits)/(hits+miss)

def main():
	accuracies_final=[]
	file_name="data/a5a.train"
	try:
		margin=sys.argv[1]
		assert float(margin)<float(5) and float(margin)>=0,"Value between 0.0 to 5.0 required"
	except Exception:
		margin=0
		print "Margin Value Required Through Command Line between 0 to 5.0. Continuing with margin=0"
	epoch_list=[3,5]
	learning_rate=[1,0.1,0.01,0.001]
	data,labels,max_features=read_data(file_name)
	try:
		file_name=sys.argv[2]
	except Exception:
		print ("Filename for testing required from command line",
			   "Example: 'python prog_name margin_value a5a.train'.")
		print "Taking default input as 'a5a.test'"
		file_name="data/a5a.test"
	test_data,test_labels,max_test_feature=read_data(file_name)
	weight_initialize=populate_weights(max_features)
	initialized_bias=randint(-1,1)
	print initialized_bias, " This is randomly initialized bias"
	for epoch in epoch_list:
		total_data=[]
		for i in range(len(data)):
			total_data.append(data[i]+[labels[i]])
		random.shuffle(total_data)
		labels=[]
		labels=[i[-1] for i in total_data]
		data=[i[:-1] for i in total_data]
		for rate in learning_rate:
			bias=initialized_bias
			weights=weight_initialize
			final_weight_vector,mistakes,bias=perceptron(data,labels,weights,epoch,bias,float(margin),rate,0)
			if len(test_data[0])>len(data[0]):
				difference=len(test_data[0])-len(data[0])
				final_weight_vector=final_weight_vector+[0]*difference
			accuracy=calculate_accuracy(final_weight_vector,test_data,test_labels,bias,0,0)
			accuracies_final.append([accuracy,rate,epoch])
			print "Mistakes committed:", mistakes,"Accuracy found",accuracy*100,"for rate",rate,"with epoch ",epoch," and bias of the final vector",bias
			weights=[]
	result=max(accuracies_final)
	print "\n"
	print result[0],"max accuracy for learning rate",result[1],"and epoch",result[2]

main()






"""
******************3_1_1*****************************
1.File read data+label
2.Append each input -->bias+example
3.weight --> [1,0,0,..]
4.replace rate --->learning rate( given in assignment)
          t ->t_th example
****************************3_1_2******************
cross  validation
learning rate ,gamma --> discussions
C ---> discussions
"""
