# load and train model and save model
# load data to train model1 or model2

def train_model(model="model1"):
	if model == "model1":
		pass
		######Setting all the hyperparameters
		##You can change them if you want

		iters = 26
		num_latent = 8
		print_every = 5    #print after every 5 iterations
		model = Net(num_latent)

		device = ('cuda' if torch.cuda.is_available() else 'cpu')
		import torch.optim as optim
		optimizer = optim.Adam(model.parameters(), lr=1e-3)

		train(trainloader, iters, model, device, optimizer, print_every)
	else:
		pass