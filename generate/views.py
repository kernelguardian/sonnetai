from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import render

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

train_on_gpu = torch.cuda.is_available()
module_dir = os.path.dirname(__file__)

def one_hot_encode(arr, n_labels):
	# Initialize the the encoded array
	one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
	# Fill the appropriate elements with ones
	one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
	# Finally reshape it to get back to the original array
	one_hot = one_hot.reshape((*arr.shape, n_labels))
	return one_hot

def predict(net, char, h=None, top_k=None):
		''' Given a character, predict the next character.
			Returns the predicted character and the hidden state.
		'''
		
		# tensor inputs
		x = np.array([[net.char2int[char]]])
		x = one_hot_encode(x, len(net.chars))
		inputs = torch.from_numpy(x)
		
		if(train_on_gpu):
			inputs = inputs.cuda()
		
		# detach hidden state from history
		h = tuple([each.data for each in h])
		# get the output of the model
		out, h = net(inputs, h)

		# get the character probabilities
		p = F.softmax(out, dim=1).data
		if(train_on_gpu):
			p = p.cpu() # move to cpu
		
		# get top characters
		if top_k is None:
			top_ch = np.arange(len(net.chars))
		else:
			p, top_ch = p.topk(top_k)
			top_ch = top_ch.numpy().squeeze()
		
		# select the likely next character with some element of randomness
		p = p.numpy().squeeze()
		char = np.random.choice(top_ch, p=p/p.sum())
		
		# return the encoded value of the predicted char and the hidden state
		return net.int2char[char], h	

def sample(net, size, prime='whose conduct others', top_k=None):
		
	if(train_on_gpu):
		net.cuda()
	else:
		net.cpu()
	
	net.eval() # eval mode
	
	# First off, run through the prime characters
	chars = [ch for ch in prime]
	h = net.init_hidden(1)
	for ch in prime:
		char, h = predict(net, ch, h, top_k=top_k)

	chars.append(char)
	
	# Now pass in the previous character and get a new one
	for ii in range(size):
		char, h = predict(net, chars[-1], h, top_k=top_k)
		chars.append(char)

	return ''.join(chars)        


model_path = os.path.join(module_dir, 'model_try.pt')


if train_on_gpu:
	neuralnetwork = torch.load(model_path)
else:
	neuralnetwork = torch.load(model_path, map_location=torch.device('cpu'))

# print(sample(neuralnetwork, 1000, prime='session', top_k=5))

def event(request):
	context = {}
	system = request.POST.get('line1', None)
	poem = sample(neuralnetwork, 1000, prime=system.lower(), top_k=5)

	poem = poem.replace('  ', ' ')
	words = poem.replace(' \'', '\'').split(' ')
	lis = ''
	words_per_line = 10
	number_of_lines = 14
	for i in range(0, number_of_lines*words_per_line, words_per_line):
		line = ''
		for e in words[0+i: words_per_line+i]:
			line = line + ' ' + e
		lis = lis + ' <br>' + line
	
	context['system'] = lis
	
	return render(request, 'index.html', context)

# Create your views here.
def generate(request):
	return render(request, 'index.html')

