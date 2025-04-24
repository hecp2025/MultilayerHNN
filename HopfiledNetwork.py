import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time

############################################################################################################
# Hopfield Network
############################################################################################################
class HopfieldNet(nn.Module):
    def __init__(self, num_neurons ,nonlin='tanh',device='cpu'):
        super(HopfieldNet, self).__init__()
        self.num_neurons = num_neurons
        self.device=device
        self.weights = torch.zeros((self.num_neurons, self.num_neurons)).to(self.device)
        self.hidden_state=torch.rand(num_neurons).to(self.device)-0.5
        self.state=torch.sign(self.hidden_state).to(self.device)
        self.Wr = nn.Linear(num_neurons, num_neurons, bias=False)
        self.var_matrix = torch.zeros((self.num_neurons, self.num_neurons)).to(self.device)
        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
        
    def update_state(self,sync=True,chnn=False,timestep=1e-3):
        if chnn==False:
            if sync == True:
                self.hidden_state = (self.weights @ self.state.t()).t()
                if self.binary==True:
                    self.state = torch.sign(self.hidden_state)
                else:
                    self.state=torch.tanh(self.hidden_state)
            elif sync == False:
                for i in np.random.permutation(list(range(self.num_neurons))):
                    self.hidden_state[i] = self.weights[i, :].t() @ self.state.t()
                    print(self.hidden_state[i].shape)
                    if self.binary==True:
                        self.state[i] = torch.sign(self.hidden_state[i])
                    else:
                        self.state[i]=torch.tanh(self.hidden_state[i])
        elif chnn==True:
            self.hidden_state = (self.weights @ self.state.t()).t()
            self.state += timestep*self.hidden_state
            self.state=torch.tanh(self.state)
        else:
            raise AttributeError('sync variable can take only boolean values or chnn take the wrong boolean values')
        return None

    def learn_patterns(self,patterns,rule, options):
        '''
        update the weight from different learning rules and learning options
        The learning rules are: Hebb, Storkey, Pesudo-inverse, Iterative Storkey.Oja, BCM, Foldiak, and Krotov
        '''
        self.patterns = patterns
        self.num_patterns = patterns.shape[0]
        if rule == "Hebb":
            self.weights = hebb_rule(self.num_neurons,patterns,self.weights,**options)  # Hebbian learning
        elif rule == "Storkey":
            self.weights = storkey(self.num_neurons,patterns,self.weights,**options)  # Storkey learning
        elif rule == "Pinv":
            self.weights = pinv(self.num_neurons,patterns,self.weights,**options)   # Pseudo-inverse learning
        elif rule == "IterativeStorkey":
            self.weights = iterative_storkey(self.num_neurons,patterns,self.weights,**options) # Iterative Storkey learning
        elif rule == "Gradient":
            self.weights = self.Wr.weight.data # Gradient learning
        elif rule == "Oja":
            self.weights=torch.randn(self.num_neurons,self.num_neurons).to(self.device)
            # self.weights = hebb_rule(self.num_neurons,patterns,self.weights,**options) # hebb learning for the initial weights setting
            self.weights = Oja_rule(self.num_neurons,patterns,self.weights,**options) # Oja learning
        elif rule == "Equilibrium":
            self.weights=torch.randn(self.num_neurons,self.num_neurons).to(self.device)
            # 将对角线元素设为零
            self.weights = self.weights - torch.diag(torch.diag(self.weights))
            # 使对称矩阵对角元素为零
            self.weights = (self.weights + self.weights.t()) / 20
            # self.weights = hebb_rule(self.num_neurons,patterns,self.weights,**options) # hebb learning for the initial weights setting
            self.weights = equilibrium_propagation(self.num_neurons,patterns,self.weights,**options)
        else:
            raise AttributeError('rule variable can take only Hebb, Storkey, Pinv, Iterative_Storkey, Oja, and Gradient')
        # Set the maximum absolute value of the weights to 1
        self.weights.fill_diagonal_(0)
        max_abs_value = torch.max(torch.abs(self.weights))
        self.weights = self.weights / max_abs_value

        return None

    
    def retrieve_pattern(self, initial_state, sync=True, chnn=False,binary=True, timestep=1e-3,max_iter=2000):
        self.state = initial_state.clone()
        self.binary=binary
        self.mid_state = []
        for i in range(max_iter):
            self.update_state(sync=sync, chnn=chnn, timestep=timestep)
            self.mid_state.append(self.state.clone())
            if torch.all(self.state == torch.sign(self.hidden_state)) and i>100:
                break
        if self.binary==True:
            self.state = torch.sign(self.state)
        return self.state

    def set_params(self):
        self.weights = torch.zeros((self.num_neurons, self.num_neurons)).to(self.device)
        return None

    def init_hidden(self, bsz):
        """Initializing sequence"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.num_neurons))

    def forward(self, prev):
        pred = self.nonlin(self.Wr(prev))
        return pred

    def get_energy(self, curr, prev):
        err = curr-self.forward(prev)
        energy = torch.sum(err**2)
        return energy
    
    #add memristor variation in the system
    def add_weight_variation(self,variation=0.0):
        self.weights -= self.var_matrix
        self.var_matrix = variation*torch.randn(self.num_neurons,self.num_neurons).to(self.device)
        self.weights += self.var_matrix
        self.weights.fill_diagonal_(0)
        return None
    
    # add stuck at fault in the system
    def add_mask(self,mask):
        self.weights*=mask
        return None
    
    def get_weights(self):
        return self.weights.detach().cpu().numpy()
    
############################################################################################################
# Learning rules for Hofield Network
############################################################################################################
def hebb_rule(num_neurons,patterns,weights):
    '''
    Hebbian learning rule
    '''
    for i in range(patterns.shape[0]):
        p = patterns[i, :]
        weights += torch.outer(p, p)
    weights /= num_neurons
    return weights

def storkey(num_neurons,patterns,weights):
    '''
    Storkey learning rule
    '''
    for i in range(patterns.shape[0]):
        p = patterns[i, :]
        h = torch.mv(weights, p)
        pre = torch.outer(h, p)
        post = torch.outer(p, h)
        weights += torch.outer(p, p)/num_neurons
        weights += torch.outer(h, h)/num_neurons
        weights -= torch.add(pre, post)/num_neurons
    return weights

def iterative_storkey(num_neurons,patterns,weights, num_iter=1):
    '''
    Iterative Storkey learning rule
    '''
    for _ in range(num_iter):
        for i in range(patterns.shape[0]):
            p = patterns[i, :]
            h = torch.mv(weights, p)
            pre = torch.outer(h, p)
            post = torch.outer(p, h)
            weights += torch.outer(p, p)/num_neurons
            weights += torch.outer(h, h)/num_neurons
            weights -= torch.add(pre, post)/num_neurons
    return weights

def pinv(num_neurons,patterns,weights):
    '''
    Pseudo-inverse learning rule
    '''
    C_matrix = 1/num_neurons*torch.mm(patterns, patterns.t())
    C_Inv = torch.linalg.pinv(C_matrix)
    weights = 1/num_neurons*(patterns.t() @ C_Inv @ patterns)
    return weights
  
def Oja_rule(num_neurons,patterns,weights,num_iter=100,lr=1e-4):
    '''
    Oja learning rule
    '''
    for iter in range(num_iter):
        Wprev = weights.clone()
        for i in range(patterns.shape[0]):
            x = patterns[i, :]
            y=x.view(-1,1) # convert to column vector
            x=x.view(1,-1) # convert to row vector
            weights += lr*(y*x-weights*(y**2))
        if torch.norm(Wprev - weights) < 1e-16:
            print('iteration=',iter)
            break
    print('iteration=',iter)
    return weights

def custom_sign(x, theta):
    return torch.where(x >= theta, torch.tensor(1.0), torch.where(x < -theta, torch.tensor(-1.0), torch.tensor(0.0)))

def retrieve_pattern(weights, patterns,beta=0,theta=0.5,max_iter=100):
    '''
    retrieve pattern
    '''
    state = patterns.clone()
    for i in range(max_iter):
        state_prev=state.clone()
        hidden_state=(weights @ state.t()).t()+beta*patterns
        state = custom_sign(hidden_state, theta)
        if torch.all(state == state_prev):
            break
    return state

def equilibrium_propagation(num_neurons,patterns,weights,num_iter=100,lr=0.01,beta=10):
    '''
    equilibrium propagation learning rule
    '''
    print('pattern shape',patterns.shape)
    for iter in range(num_iter):
        delta_weight=torch.zeros((num_neurons,num_neurons))
        for i in range(patterns.shape[0]):
            x = patterns[i, :]
            A_free=retrieve_pattern(weights, x,beta=0)
            A_nudge=retrieve_pattern(weights, x,beta=beta)
            delta_weight+=torch.outer(A_nudge,A_nudge)-torch.outer(A_free,A_free)
        weights += lr*delta_weight
    print('iteration=',iter)
    return weights

############################################################################################################
# Single Layer Network and Multi Layer Network
############################################################################################################


class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)
    
class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        return torch.where(inp > 0, torch.tensor(1.0), torch.tensor(0.0))

class SingleLayerHNN(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerHNN, self).__init__()

        single_layers = []
        single_layers.append(nn.Linear(input_size, input_size))
        single_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*single_layers)

    def forward(self, x):
        output = self.encoder(x)
        return output


    
class SingleLayerHNN_lambda(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerHNN_lambda, self).__init__()
        self.num_neuron=input_size
        self.linear = nn.Linear(input_size, input_size)
        self.mask = nn.Parameter(torch.ones(input_size) - torch.eye(input_size), requires_grad=False)  # Initialize the mask with zeros on the diagonal
        self.var_matrix = torch.zeros((self.num_neuron, self.num_neuron))
        self.noise_level = 0.0

    def add_noise(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

    def forward(self, x):
        if self.training:
            weight_noise=self.add_noise(self.linear.weight)
        else:
            weight_noise=self.linear.weight
        masked_weight = weight_noise * self.mask  # Apply the mask to the linear layer's weight
        x = torch.matmul(x, masked_weight.t()) + self.linear.bias
        x = torch.tanh(x)        
        return x

    def inference(self, x):
        masked_weight = self.linear.weight * self.mask  # Apply the mask to the linear layer's weight
        x = torch.matmul(x, masked_weight.t()) + self.linear.bias
        x = torch.tanh(x)
        return x
   
    def get_energy_sign(self, curr, prev):
        masked_weight = self.linear.weight * self.mask  # Apply the mask to the linear layer's weight
        prev = torch.matmul(prev, masked_weight.t()) + self.linear.bias
        err = curr - torch.sign(prev)
        energy = torch.sum(err**2)
        return energy

    def get_energy_tanh(self, curr, prev):
        masked_weight = self.linear.weight * self.mask  # Apply the mask to the linear layer's weight
        prev = torch.matmul(prev, masked_weight.t()) + self.linear.bias
        err = curr - torch.tanh(prev)
        energy = torch.sum(err**2)
        return energy

    #add stuck at fault in the system
    def set_mask(self, mask):
        self.mask.data = mask*self.mask # Set the mask to the provided tensor

    #add stuck at fault in the system
    def clear_mask(self):
        self.mask.data = torch.ones(self.num_neuron,self.num_neuron) # Set the mask to the provided tensor

    # add memristor variation in the system
    def add_variation(self,var):
        self.linear.weight.data -= self.var_matrix
        max_abs_value = torch.max(torch.abs(self.linear.weight.data))
        self.var_matrix = var*max_abs_value*torch.randn(self.linear.weight.shape)
        self.linear.weight.data += self.var_matrix
        return None


class SingleLayerHNN_beta(nn.Module):
    def __init__(self, input_size, device='cpu'):
        super(SingleLayerHNN_beta, self).__init__()
        self.num_neuron = input_size
        self.device = device  # 保存设备信息
        self.linear = nn.Linear(input_size, input_size).to(self.device)  # 将线性层移到设备上
        self.mask = nn.Parameter((torch.ones(input_size) - torch.eye(input_size)).to(self.device), requires_grad=False)  # 初始化mask，并将其移到设备上
        self.var_matrix = torch.zeros((self.num_neuron, self.num_neuron)).to(self.device)  # 初始化var_matrix，并将其移到设备上
        self.noise_level = 0.0

    def add_noise(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

    def forward(self, x):
        x = x.to(self.device)  # 确保输入张量在设备上
        if self.training:
            weight_noise = self.add_noise(self.linear.weight)
        else:
            weight_noise = self.linear.weight
        masked_weight = weight_noise * self.mask  # 应用mask
        x = torch.matmul(x, masked_weight.t()) + self.linear.bias
        x = torch.tanh(x)        
        return x

    def inference(self, x):
        x = x.to(self.device)  # 确保输入张量在设备上
        masked_weight = self.linear.weight * self.mask  # 应用mask
        x = torch.matmul(x, masked_weight.t()) + self.linear.bias
        x = torch.tanh(x)
        return x
   
    def get_energy_sign(self, curr, prev):
        curr = curr.to(self.device)  # 确保输入张量在设备上
        prev = prev.to(self.device)  # 确保输入张量在设备上
        masked_weight = self.linear.weight * self.mask  # 应用mask
        prev = torch.matmul(prev, masked_weight.t()) + self.linear.bias
        err = curr - torch.sign(prev)
        energy = torch.sum(err**2)
        return energy

    def get_energy_tanh(self, curr, prev):
        curr = curr.to(self.device)  # 确保输入张量在设备上
        prev = prev.to(self.device)  # 确保输入张量在设备上
        masked_weight = self.linear.weight * self.mask  # 应用mask
        prev = torch.matmul(prev, masked_weight.t()) + self.linear.bias
        err = curr - torch.tanh(prev)
        energy = torch.sum(err**2)
        return energy

    # 添加“stuck at fault”到系统中
    def set_mask(self, mask):
        mask = mask.to(self.device)  # 确保输入mask在设备上
        self.mask.data = mask * self.mask  # 设置mask为提供的张量

    # 清除“stuck at fault”
    def clear_mask(self):
        self.mask.data = torch.ones(self.num_neuron, self.num_neuron).to(self.device)  # 重置mask

    # 添加memristor变化到系统中
    def add_variation(self, var):
        self.linear.weight.data -= self.var_matrix
        max_abs_value = torch.max(torch.abs(self.linear.weight.data))
        self.var_matrix = var * max_abs_value * torch.randn(self.linear.weight.shape).to(self.device)  # 确保var_matrix在设备上
        self.linear.weight.data += self.var_matrix
        return None


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerHNN(nn.Module):
    def __init__(self, input_dim, encoding_dim, num_hidden_layers=1,nolinear='tanh'):
        super(MultiLayerHNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        if nolinear == 'tanh':
            self.nonlin = nn.Tanh()
        elif nolinear == 'relu':
            self.nonlin = nn.ReLU()

        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, encoding_dim))
        encoder_layers.append(self.nonlin)
        for _ in range(num_hidden_layers-1):
            encoder_layers.append(nn.Linear(encoding_dim, encoding_dim))
            encoder_layers.append(self.nonlin)
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        decoder_layers.append(nn.Linear(encoding_dim, input_dim))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def intermediate(self,x):
        intermediate = []
        for layer in self.encoder:
            x=layer(x)
            intermediate.append(x)
        return intermediate

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        return self.decoder(encoded)
    
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask_prob, device='cpu',seed=42):
        super(MaskedLinear, self).__init__(in_features, out_features)
        self.mask_prob = mask_prob
        self.device = device
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Generate mask and move to correct device
        self.mask = (torch.rand(out_features, in_features) > mask_prob).float().to(self.device)
        self.mask.requires_grad = False  # No gradient computation for the mask
        
    def forward(self, x):
        # Apply mask to the weight
        weight = self.weight * self.mask
        
        return F.linear(x, weight, self.bias)



class MultiLayerHNN_beta(nn.Module):
    def __init__(self, input_dim, encoding_dim, num_hidden_layers=1, nolinear='tanh', noise_level=0.0, mask_prob=0.0, device='cpu'):
        super(MultiLayerHNN_beta, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        self.device = device  # Set device

        # Select activation function
        if nolinear == 'tanh':
            self.nonlin = nn.Tanh()
        elif nolinear == 'relu':
            self.nonlin = nn.ReLU()

        # Construct encoder layers
        encoder_layers = []
        encoder_layers.append(MaskedLinear(input_dim, encoding_dim, self.mask_prob, device=self.device))
        encoder_layers.append(self.nonlin)
        for _ in range(num_hidden_layers - 1):
            encoder_layers.append(MaskedLinear(encoding_dim, encoding_dim, self.mask_prob, device=self.device))
            encoder_layers.append(self.nonlin)
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)

        # Construct decoder layers
        decoder_layers = []
        decoder_layers.append(MaskedLinear(encoding_dim, input_dim, self.mask_prob, device=self.device))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        if self.training:
            for layer in self.encoder + self.decoder:
                if isinstance(layer, nn.Linear):
                    noise_weight = torch.rand_like(layer.weight.data) * self.noise_level
                    x = layer(x) + torch.matmul(noise_weight, x.t()).t()
                else:
                    x = layer(x)
        else:
            for layer in self.encoder + self.decoder:
                x = layer(x)
        return x

    def add_variation(self, var):
        for layer in self.encoder + self.decoder:
            if isinstance(layer, nn.Linear):
                max_abs_value = torch.max(torch.abs(layer.weight.data))
                self.var_matrix = var * max_abs_value * torch.randn(layer.weight.shape).to(self.device)
                layer.weight.data += self.var_matrix
        return None

    def intermediate(self, x):
        intermediate = []
        x = x.to(self.device)  # Ensure input is on the correct device
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                noise_weight = torch.rand_like(layer.weight.data) * self.noise_level
                x = layer(x) + torch.matmul(noise_weight, x.t()).t()
            x = layer(x)
            intermediate.append(x)
        return intermediate

    def encode(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        return self.encoder(x)

    def decode(self, encoded):
        encoded = encoded.to(self.device)  # Ensure input is on the correct device
        return self.decoder(encoded)


class MultiLayerHNN_sigma(nn.Module):
    def __init__(self, input_dim, encoding_dim, num_hidden_layers=1, nolinear='tanh', noise_level=0.0, mask_prob=0.0, device='cpu'):
        super(MultiLayerHNN_sigma, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        self.device = device  # Set device

        # Select activation function
        if nolinear == 'tanh':
            self.nonlin = nn.Tanh()
        elif nolinear == 'relu':
            self.nonlin = nn.ReLU()

        # Construct encoder layers
        encoder_layers = []
        encoder_layers.append(MaskedLinear(input_dim, encoding_dim, self.mask_prob, device=self.device))
        encoder_layers.append(self.nonlin)

        # encoder_layers.append(MaskedLinear(encoding_dim, encoding_dim, self.mask_prob, device=self.device))
        # encoder_layers.append(self.nonlin)
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)

        # Construct decoder layers
        decoder_layers = []
        decoder_layers.append(MaskedLinear(encoding_dim, input_dim, self.mask_prob, device=self.device))
        decoder_layers.append(self.nonlin)  # Decoder final activation as Tanh
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        if self.training:
            # Apply noise to the weight of linear layers if in training mode
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    noise_weight = torch.rand_like(layer.weight.data) * self.noise_level
                    x = layer(x) + torch.matmul(noise_weight, x.t()).t()
                else:
                    x = layer(x)
            for layer in self.decoder:
                if isinstance(layer, nn.Linear):
                    noise_weight = torch.rand_like(layer.weight.data) * self.noise_level
                    x = layer(x) + torch.matmul(noise_weight, x.t()).t()
                else:
                    x = layer(x)
        else:
            # Forward pass without noise
            x = self.encoder(x)
            x = self.decoder(x)
        return x




############################################################################################################
#Modern Asymmetric Hopfield Network and Symmetric Hopfield Network
############################################################################################################

class ModernAsymmetricHopfieldNetwork(nn.Module):
    """
    MAHN. train() function is simply a placeholder since we don't really train these models
    """
    
    def __init__(self, input_size, sep='linear', beta=1):
        super(ModernAsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        self.sep = sep
        self.beta = beta
        self.var_matrix = torch.zeros((input_size, input_size))
    def forward(self, X, s):
        """
        X: stored memories, shape PxN
        s: query, shape (P-1)xN
        output: (P-1)xN matrix
        """
        _, N = X.shape
        if self.sep == 'exp':
            score = torch.exp(torch.matmul(s, X[:-1].t()))
        elif self.sep == 'softmax':
            score = F.softmax(self.beta * torch.matmul(s, X[:-1].t()), dim=1)
        elif self.sep == 'hebb':
            output=torch.matmul(self.W, s.t()).t()
            return output
        else:
            score = torch.matmul(s, X[:-1].t()) ** int(self.sep)
        output = torch.matmul(score, X[1:])
        
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

        return -1
    
    #add memristor variation in the system
    def add_weight_variation(self,variation=0.0):
        self.W -= self.var_matrix
        self.var_matrix = variation*torch.randn(self.W.shape[0],self.W.shape[1])
        self.W += self.var_matrix
        return None
    
    # add stuck at fault in the system
    def add_mask(self,mask):
        self.W*=mask
        return None

class ModernSymmetricHopfieldNetwork(nn.Module):
    """
    MAHN. train() function is simply a placeholder since we don't really train these models
    """
    
    def __init__(self, input_size, sep='linear', beta=1):
        super(ModernSymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        self.sep = sep
        self.beta = beta
        
    def forward(self, X, s):
        """
        X: stored memories, shape PxN
        s: query, shape PxN
        output: PxN matrix
        """
        _, N = X.shape
        if self.sep == 'exp':
            score = torch.exp(torch.matmul(s, X[0:].t()))
        elif self.sep == 'softmax':
            score = F.softmax(self.beta * torch.matmul(s, X[0:].t()), dim=1)
        elif self.sep == 'max':
            score=torch.zeros((s.shape[0],X.shape[0]))
            score = torch.argmax(torch.matmul(s, X[0:].t()),dim=1)
            score=torch.nn.functional.one_hot(score,num_classes=X.shape[0])
        else:
            score = torch.matmul(s, X[0:].t()) ** int(self.sep)
        score = score / torch.sum(score, dim=1).unsqueeze(1)
        output = torch.matmul(score, X[0:])
        
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Symmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[0:].T, X[0:]) / N

        return -1
    

    
############################################################################################################
# Predictive Coding Network
############################################################################################################

class SingleLayertPC(nn.Module):
    """Generic single layer tPC"""
    def __init__(self, input_size, nonlin='tanh'):
        super(SingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        elif nonlin == 'relu':
            self.nonlin = ReLU()
        else:
            raise ValueError("no such nonlinearity!")  
        self.input_size = input_size
        self.noise_level = 0.0
        self.var_matrix = torch.zeros((self.input_size, self.input_size))
        self.mask= nn.Parameter(torch.ones(input_size,input_size), requires_grad=False) 

    def init_hidden(self, bsz):
        """Initializing sequence"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.input_size))
    
    # def forward(self, prev):
    #     self.mask_weight = self.Wr.weight * self.mask
    #     pred=torch.matmul(self.mask_weight,self.nonlin(prev).t()).t()
    #     return pred

    def add_noise(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

    def forward(self, prev):
        if self.training:
            weight_noise=self.add_noise(self.Wr.weight)
        else:
            weight_noise=self.Wr.weight
        masked_weight = weight_noise * self.mask  # Apply the mask to the linear layer's weight
        pred = torch.matmul(masked_weight,self.nonlin(prev).t()).t()      
        return pred
    
    def update_errs(self, curr, prev):
        """
        curr: current observation
        prev: previous observation
        """
        pred = self.forward(prev)
        err = curr - pred
        return err
    
    def get_energy(self, curr, prev):
        err = self.update_errs(curr, prev)
        energy = torch.sum(err**2)
        return energy

    def add_variation(self,variation=0.0):
        self.Wr.weight.data -= self.var_matrix
        self.var_matrix = variation*torch.randn(self.Wr.weight.shape)
        self.Wr.weight.data += self.var_matrix
        return None
    
    def set_mask(self,mask):
        self.mask.data*=mask
        return None
    
class MultilayertPC(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, hidden_size, output_size, nonlin='tanh'):
        super(MultilayertPC, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
    
    def forward(self, prev_z):
        pred_z = self.Wr(self.nonlin(prev_z))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, bsz):
        """Initializing prev_z"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def update_errs(self, x, prev_z):
        pred_z, _ = self.forward(prev_z)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x
        return err_z, err_x
    
    def update_nodes(self, x, prev_z, inf_lr, update_x=False):
        err_z, err_x = self.update_errs(x, prev_z)
        delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
        self.z -= inf_lr * delta_z
        if update_x:
            delta_x = err_x
            x -= inf_lr * delta_x

    def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
        """prev_z should be set up outside the inference, from the previous timestep

        Args:
            train: determines whether we are at the training or inference stage
        
        After every time step, we change prev_z to self.z
        """
        with torch.no_grad():
            # initialize the current hidden state with a forward pass
            self.z, _ = self.forward(prev_z)

            # update the values nodes
            for i in range(inf_iters):
                self.update_nodes(x, prev_z, inf_lr, update_x)
                
    def update_grads(self, x, prev_z):
        """x: input at a particular timestep in stimulus
        
        Could add some sparse penalty to weights
        """
        err_z, err_x = self.update_errs(x, prev_z)
        self.hidden_loss = torch.sum(err_z**2)
        self.obs_loss = torch.sum(err_x**2)
        energy = self.hidden_loss + self.obs_loss
        return energy
    
