import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch.autograd.variable import Variable
import torch.nn.functional as F
import time
import torch.optim as optim
import numpy as np
import torch._utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
'''
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
'''
import torchvision
import torchvision.transforms as transforms
import pdb
import glob
import random
import gc
#from torchsample.callbacks import ModelCheckpoint

random.seed(123)

whole_list = ['T0859','T0860','T0861','T0862','T0863','T0864','T0865','T0866','T0867','T0868','T0869','T0870','T0871','T0872','T0873','T0874','T0875','T0876','T0877','T0878','T0879','T0880','T0881','T0882','T0883']#,'T0884','T0885','T0886','T0887','T0888','T0889','T0890','T0891','T0892','T0893','T0894','T0895','T0896','T0897','T0898','T0899','T0900','T0901','T0902','T0903','T0904','T0905','T0906','T0907','T0909','T0910','T0911','T0912','T0913','T0914','T0915','T0917','T0918','T0920','T0921','T0922','T0923','T0928','T0929','T0941','T0942','T0943','T0944','T0945','T0946','T0947','T0948']

#print len(whole_list)
test_list = random.sample(whole_list, 20)
#print test_list
#['T0862', 'T0865', 'T0887', 'T0866', 'T0923', 'T0861', 'T0894', 'T0880', 'T0914', 'T0869', 'T0879', 'T0944', 'T0873', 'T0859', 'T0884', 'T0863', 'T0892', 'T0948', 'T0876', 'T0882']
train_list = list(set(whole_list) - set(test_list))
#print train_list
#['T0917', 'T0915', 'T0913', 'T0912', 'T0911', 'T0910', 'T0918', 'T0878', 'T0874', 'T0875', 'T0877', 'T0870', 'T0871', 'T0872', 'T0868', 'T0881', 'T0883', 'T0891', 'T0885', 'T0886', 'T0889', 'T0888', 'T0900', 'T0901', 'T0902', 'T0903', 'T0904', 'T0905', 'T0906', 'T0907', 'T0909', 'T0928', 'T0929', 'T0867', 'T0864', 'T0922', 'T0945', 'T0946', 'T0921', 'T0941', 'T0942', 'T0943', 'T0947', 'T0896', 'T0893', 'T0890', 'T0860', 'T0920', 'T0897', 'T0895', 'T0898', 'T0899']

#O = (W-K+2P)/S+1
def normal_init(m,mean,std):
	if isinstance(m,torch.nn.ConvTranspose3d) or isinstance(m,torch.nn.Conv3d):
		m.weight.data.normal_(mean,std)
		m.bias.data.zero_()


class SimpleCNN(torch.nn.Module):
	
	#Our batch shape for input x is (3, 32, 32)
	
	def __init__(self):
		super(SimpleCNN, self).__init__()
		
		#1
		#(57, 57, 57, 11)
		self.conv1 = torch.nn.Conv3d(11, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(0, 0, 0))
		#(55, 55, 55, 32)
		self.norm1 = torch.nn.BatchNorm3d(32)
		self.pool1 = torch.nn.MaxPool3d(kernel_size=(55,55,55), stride=(2,2,2), padding=0)
		#(27, 27, 27, 32)
		
		#33
		self.fc1 = torch.nn.Linear(32, 1)
		
	def weight_init(self, mean = 0, std = 0.01):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)
		
		
	def forward(self, x):
		
		#Computes the activation of the first convolution
		#Size changes from (3, 32, 32) to (18, 32, 32)
		#pdb.set_trace()
		x = self.pool1(F.relu(self.norm1(self.conv1(x))))
		
		x = self.fc1(x.squeeze())
		return(x)

		
#Test and validation loaders have constant batch sizes, so we can define them directly
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, sampler=test_sampler, num_workers=2)
#val_loader = torch.utils.data.DataLoader(train_set, batch_size=2, sampler=val_sampler, num_workers=2)

def trainNet(net, n_epochs, learning_rate):
	
	#Print all of the hyperparameters of the training iteration:
	print("===== HYPERPARAMETERS =====")
	#print("batch_size=", batch_size)
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	print("=" * 30)
	
	#load epoch 3 model
	net.load_state_dict(torch.load('model_progr/simple_regre100_32'))
	
	#loss = torch.nn.MarginRankingLoss(margin = 0.001)
	loss = torch.nn.L1Loss()
	#Create our loss and optimizer functions
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	
	#Time for printing
	training_start_time = time.time()
	
	num_batch = 1
	#Loop for n_epochs
	for epoch in range(n_epochs):
		print "epoch "+str(epoch+32)
		running_loss = 0.0
		start_time = time.time()
		total_train_loss = 0
		
		#for i in range(num_batch):
		#for files in glob.glob('batch/T0913*input*'):
		for train in train_list:
			#Get inputs
			#print files
			#print("loading batch " + str(i))
			for files in sorted(glob.glob('smaller_batch/'+train+'*input*')):
				print files
				#input1 = np.transpose(np.load('batch/T0913_input_batch'+str(num_batch+1)+'.npy'),(0,4,1,2,3))			
				
				#labels = np.load('batch/T0913_score_batch'+str(num_batch+1)+'.npy')
				input1 = np.transpose(np.load(files),(0,4,1,2,3))			
				
				labels = np.load(files.split('_input_')[0]+'_score_'+files.split('_input_')[1])
	
				#Wrap them in a Variable object
				input1, labels = Variable(torch.FloatTensor(input1).cuda()), Variable(torch.FloatTensor(labels).cuda())
				#Set the parameter gradients to zero
				optimizer.zero_grad()
				
				#Forward pass, backward pass, optimize
				output1 = net(input1)
				#output2 = net(input2)
				#pdb.set_trace()
				loss_size = loss(output1, labels)
				loss_size.backward()
				optimizer.step()
				
				#Print statistics
				#pdb.set_trace()
				
				running_loss += loss_size.item()
				total_train_loss += loss_size.item()
				del input1
				del labels
				gc.collect()
		#callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5)]
		
		torch.save(net.state_dict(),'/net/kihara/scratch/ding48/CASP12/model_progr/simple_regre100_'+str(epoch+32))
			
		print("total train loss: "+str(total_train_loss))
		print("finished training")
		#At the end of the epoch, do a pass on the validation set
		total_val_loss = 0
		num_val = 1
		#for i in range(num_val):
		#for vals in glob.glob('batch/T0887*input*'):
		for test in test_list:
			for vals in sorted(glob.glob('smaller_batch/'+test+'*input*')):
				print vals
				#val_input1 = np.transpose(np.load('batch/T0913_input_batch2.npy'),(0,4,1,2,3))
				#val_labels = np.load('batch/T0913_score_batch2.npy')
				val_input1 = np.transpose(np.load(vals),(0,4,1,2,3))
				val_labels = np.load(vals.split('_input_')[0]+'_score_'+vals.split('_input_')[1])
				
				#Wrap tensors in Variables
				val_input1,val_labels = Variable(torch.FloatTensor(val_input1).cuda()), Variable(torch.FloatTensor(val_labels).cuda())
				
				#Forward pass
				val_output1 = net(val_input1)
				print val_output1
				output1_mean = val_output1 - torch.mean(val_output1)
				label_mean = val_labels - torch.mean(val_labels)
				
				cost = torch.sum(output1_mean*label_mean)/(torch.sqrt(torch.sum(output1_mean ** 2)) * torch.sqrt(torch.sum(label_mean ** 2)))
				#print cost
				
				val_loss_size = loss(val_output1, val_labels)
				#print val_loss_size
				total_val_loss += val_loss_size.item()
				del val_input1
				del val_labels
				gc.collect()
		print("Validation loss = {:.2f}".format(total_val_loss))
		
		
		#callbacks = [ModelCheckpoint(file='/net/kihara/scratch/ding48/CASP12/model_progr/model_{epoch}_{total_train_loss}.pt', monitor='total_val_loss', save_best_only=False, max_checkpoints=3)]
		
	print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


CNN = SimpleCNN().to(device)
CNN.weight_init(mean=0, std=0.01)
CNN.cuda()
print(CNN)
trainNet(CNN, n_epochs=50, learning_rate=0.0001)
