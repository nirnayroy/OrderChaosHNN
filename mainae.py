import torch
from torch import nn, optim
import numpy as np
from hnn import HNN
from  baseline_nn import BLNN
import argparse
from data import DynamicalSystem
from tqdm import tqdm
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--hamiltonian',
                        default='''(p1**2+p2**2)/(2)-((
        (1/(1+(exp(-(q1+2)/0.1))))-(1/(1+(exp(-(q1-2)/0.1)))))*(
        (1/(1+(exp(-(q2+2)/0.1))))-(1/(1+(exp(-(q2-2)/0.1))))))+1''',
                        type=str, help='Hamiltonian of the system')

    parser.add_argument('--input_dim', default=4,
                        type=int, help='Input dimension')
    parser.add_argument('--hidden_dim', nargs="*", default=[200, 200],
                        type=int, help='hidden layers dimension')
    parser.add_argument('--bottleneck',  default=4,
                        type=int, help='hidden layers dimension')
    parser.add_argument('--learn_rate', default=1e-03,
                        type=float, help='learning rate')
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batch size'),
    parser.add_argument('--input_noise', default=0.0,
                        type=float, help='noise strength added to the inputs')
    parser.add_argument('--epochs', default=2,
                        type=int, help='No. of training epochs')
    parser.add_argument('--integrator_scheme', default='RK45',
                        type=str, help='name of the integration scheme [RK4, RK45, Symplectic]')
    parser.add_argument('--activation_fn', default='Tanh', type=str,
                        help='which activation function to use [Tanh, ReLU]')
    parser.add_argument('--name', default='Square',
                        type=str, help='Name of the system')
    parser.add_argument('--model', default='baseline',
                        type=str, help='baseline or hamiltonian')
    parser.add_argument('--dsr', default=0.1, type=float,
                        help='data sampling rate')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Verbose output or not')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str,
                        help='where to save the trained model')
    parser.set_defaults(feature=True)

    return parser.parse_args()

def load_model(args, baseline=False):
    if baseline:
        output_dim = args.input_dim
        model = BLNN(args.input_dim, args.hidden_dim,output_dim,'ReLU')
        path = "/content/OrderChaosHNN/TrainedNetworks/Square_DSR_0.1_nlayers_2-orbits-baseline_integrator_RK45_epochs_2_BatchSize_512.tar"
    else:
        output_dim = 1
        nn_model = BLNN(args.input_dim, args.hidden_dim,output_dim,'ReLU')
        model = HNN(args.input_dim,baseline_model=nn_model)
        path = "/content/OrderChaosHNN/TrainedNetworks/Square_DSR_0.1_nlayers_2-orbits-hnn_integrator_RK45_epochs_2_BatchSize_512.tar"
    
    model.load_state_dict(torch.load(path),strict=False)
    return model

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential(
                      nn.Linear(4,200),
                      nn.ReLU(True),
                      nn.Linear(200,200),
                      nn.ReLU(True),
                      nn.Linear(200,4),
                      nn.ReLU(True)
            
                      )
        
        self.decoder=nn.Sequential(
                      nn.Linear(4,200),
                      nn.ReLU(True),
                      nn.Linear(200,200),
                      nn.ReLU(True),
                      nn.Linear(200, 4),
                      nn.Sigmoid(),
                      )
        
 
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
  


if __name__ == "__main__":

    args = get_args()
    
    args.save_dir = THIS_DIR + '/' + 'TrainedNetworks'
    args.name = args.name + '_DSR_' + str(args.dsr)
    base_model = load_model(args, baseline=True)
    hnn_model = load_model(args, baseline=False)

    print('model loaded!')

    state_symbols = ['q1', 'q2', 'p1', 'p2']
    tspan = [0, 1000]
    tpoints = int((1/args.dsr)*(tspan[1]))

    sys = DynamicalSystem(sys_hamiltonian=args.hamiltonian, tspan=tspan,
                          timesteps=tpoints, integrator=args.integrator_scheme, state_symbols=state_symbols, symplectic_order=4)

    data = sys.get_dataset(args.name, THIS_DIR)
    print('data loaded!')
    print('Hidden dimensions (excluding first and last layer) : {}'.format(
        args.hidden_dim))

    print('Training data size : {}'.format(data['coords'].shape))
    print('Testing data size : {}'.format(data['test_coords'].shape))

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

      
    ae=Autoencoder()
    criterion=nn.MSELoss()
    optimizer=optim.SGD(ae.parameters(),lr=0.01,weight_decay=1e-5)

    for epoch in tqdm(range(args.epochs), desc='Epochs', leave=True):
        for batch in tqdm(range(no_batches), desc='Batches', leave=True):

            optimizer.zero_grad()
            ixs = torch.randperm(x.shape[0])[:args.batch_size]
            dxdt_hat = hnn_model.time_derivative(x[ixs]).detach()

            #-----------------Forward Pass----------------------
            output=ae(x[ixs])
            loss=criterion(output,dxdt[ixs])
            #-----------------Backward Pass---------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    encoding = ae.encoder(test_x[ixs])
    print(encoding.shape)