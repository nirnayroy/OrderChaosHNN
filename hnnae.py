import torch
import numpy as np
from ae import BLNNAE
from tqdm import tqdm


class HNNAE(torch.nn.Module):
    def __init__(self, d_in, baseline_model):
        super(HNNAE, self).__init__()
        self.baseline_model = baseline_model
        self.M = self.permutation_tensor(d_in)

    def forward(self, x):
        y = self.baseline_model(x)
        return y

    def encoder(self, x):
        return self.baseline_model.encoder(x)

    def time_derivative(self, x, t=None):
        F = self.forward(x)
        dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        vector_field = dF @ self.M.t()

        return vector_field

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])

        return M

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
