import torch
import torch.nn

class RNN(nn.Module):

	def __init__(self, seq_len, num_output,
				 hidden_size=64, num_layers=1):
		super(RNN, self).__init__()
		self.use_last = use_last
		self.hidden_size = hidden_size
		self.drop_en = nn.Dropout(p=0.6)

		# rnn module
		self.rnn = nn.GRU( input_size=seq_len, hidden_size=hidden_size,
							num_layers=num_layers, dropout=0.5,
							bidirectional=True)

		self.bn2 = nn.BatchNorm1d(hidden_size*2)
		self.fc = nn.Linear(hidden_size*2, num_output)

	def forward(self, x, seq_lengths):
		'''
		Args:
			x: (batch, time_step, input_size)
		Returns:
			num_output size
		'''

		output, ht = self.rnn(x, None)

		row_indices = torch.arange(0, x.size(0)).long()
		col_indices = seq_lengths - 1
		if next(self.parameters()).is_cuda:
			row_indices = row_indices.cuda()
			col_indices = col_indices.cuda()

		last_tensor=output[row_indices, col_indices, :]

		fc_input = self.bn2(last_tensor)
		out = self.fc(fc_input)
		return out

	def initHidden(self, N):
		return Variable(torch.randn(1, N, self.hidden_size))
