import torch
import numpy as numpy
from tqdm import tqdm


class BLNNAE(torch.nn.Module):
    ''' Classic neural network implementation which serves as a baseline NN model '''

    def __init__(self, d_in, d_hidden, d_out, bottleneck, activation_fn):
        super(BLNNAE, self).__init__()
        self.enc_layers = torch.nn.ModuleList()
        self.dec_layers = torch.nn.ModuleList()
        self.enc_nonlinearity = []
        self.dec_nonlinearity = []

        if activation_fn == 'Tanh':
            print('Using Tanh()...')
            nonlinear_fn = torch.nn.Tanh()
        elif activation_fn == 'ReLU':
            print('Using ReLU()...')
            nonlinear_fn = torch.nn.ReLU()

        self.enc_layers.append(torch.nn.Linear(d_in, d_hidden[0]))
        self.enc_nonlinearity.append(nonlinear_fn)

        for i in range(len(d_hidden) - 1):
            self.enc_layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i + 1]))
            self.enc_nonlinearity.append(nonlinear_fn)

        self.enc_last_layer = torch.nn.Linear(d_hidden[-1], bottleneck, bias=None)

        self.dec_layers.append(torch.nn.Linear(bottleneck, d_hidden[0]))
        self.dec_nonlinearity.append(nonlinear_fn)

        for i in range(len(d_hidden) - 1):
            self.dec_layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i + 1]))
            self.dec_nonlinearity.append(nonlinear_fn)

        self.dec_last_layer = torch.nn.Linear(d_hidden[-1], d_out, bias=None)


        print('encoder:',self.enc_layers)
        print('decoder:',self.dec_layers)

        for i in range(len(self.enc_layers)):
            torch.nn.init.orthogonal_(self.enc_layers[i].weight)

        torch.nn.init.orthogonal_(self.enc_last_layer.weight)

        for i in range(len(self.dec_layers)):
            torch.nn.init.orthogonal_(self.dec_layers[i].weight)

        torch.nn.init.orthogonal_(self.dec_last_layer.weight)
        
    def encoder(self, x):
        dict_layers = dict(zip(self.enc_layers, self.enc_nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.enc_last_layer(out)

    def decoder(self, x):
        dict_layers = dict(zip(self.dec_layers, self.dec_nonlinearity))
        for layer, nonlinear_transform in dict_layers.items():
            out = nonlinear_transform(layer(x))
            x = out
        return self.dec_last_layer(out)


    def forward(self, x):
        encoding = self.encoder(x)
        out = self.decoder(encoding)
        return out

    def time_derivative(self, x, t=None):
        return self.forward(x)

    def train(self, args, data, optim):

        # ----Training inputs-------
        x = torch.tensor(
            data['coords'], requires_grad=True, dtype=torch.float32)
        # ----Training labels-------
        dxdt = torch.Tensor(data['dcoords'])

        # ----Testing inputs-------
        test_x = torch.tensor(
            data['test_coords'], requires_grad=True, dtype=torch.float32)

        # ----Testing inputs-------
        test_dxdt = torch.Tensor(data['test_dcoords'])

        # number of batches
        no_batches = int(x.shape[0]/args.batch_size)

        L2_loss = torch.nn.MSELoss()

        stats = {'train_loss': [], 'test_loss': []}
        for epoch in tqdm(range(args.epochs), desc='Epochs', leave=True):
            train_loss_epoch = 0.0
            test_loss_epoch = 0.0
            for batch in tqdm(range(no_batches), desc='Batches', leave=True):

                optim.zero_grad()
                ixs = torch.randperm(x.shape[0])[:args.batch_size]
                dxdt_hat = self.time_derivative(x[ixs])
                dxdt_hat += args.input_noise * torch.randn(
                    *x[ixs].shape)  # add noise, maybe
                loss = L2_loss(dxdt[ixs], dxdt_hat)
                loss.backward()
                grad = torch.cat([p.grad.flatten()
                                  for p in self.parameters()]).clone()
                optim.step()

                train_loss_epoch += loss.item()

                test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
                test_dxdt_hat = self.time_derivative(test_x[test_ixs])
                test_dxdt_hat += args.input_noise * torch.randn(
                    *test_x[test_ixs].shape)  # add noise, maybe
                test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

                test_loss_epoch += test_loss.item()

            # logging
            stats['train_loss'].append(train_loss_epoch/no_batches)
            stats['test_loss'].append(test_loss_epoch/no_batches)
            if args.verbose:
                print(
                    "Epoch {}/{}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                    .format(epoch, args.epochs, train_loss_epoch/no_batches, test_loss_epoch/no_batches, grad @ grad,
                            grad.std()))

        return stats
    
