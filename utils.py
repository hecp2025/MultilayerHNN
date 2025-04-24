import torch
import torch.nn as nn
import time
import torchvision
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import numpy as np 
############################################################################################################
# utility functions
############################################################################################################

def to_np(x):
    return x.cpu().detach().numpy()

############################################################################################################
# data loading functions
############################################################################################################


def load_sequence_mnist(seed, seq_len, order=True, binary=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 28, 28))

    if order:
        # Loop through each digit class and randomly sample one image from each class
        for i in range(seq_len):
            indices = torch.where(mnist.targets == i)[0]
            idx = torch.randint(0, indices.size()[0], (1,))
            img, _ = mnist[indices[idx][0]]
            sequence[i] = (2*img-1).squeeze()

    else:
        # Sample `seq_len` random images from the MNIST dataset
        indices = torch.randint(0, len(mnist), (seq_len,))
        for i, idx in enumerate(indices):
            img, _ = mnist[idx]
            sequence[i] = (2*img-1).squeeze()

    if binary:
        sequence[sequence > 0.5] = 1
        sequence[sequence <= 0.5] = -1

    return sequence

def load_sequence_fashion_mnist(seed, seq_len, order=True, binary=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the Fashion MNIST dataset
    fashion_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of fashion items
    sequence = torch.zeros((seq_len, 28, 28))

    if order:
        # Loop through each fashion item class and randomly sample one image from each class
        for i in range(seq_len):
            indices = torch.where(fashion_mnist.targets == i)[0]
            idx = torch.randint(0, indices.size()[0], (1,))
            img, _ = fashion_mnist[indices[idx][0]]
            sequence[i] = (2*img-1).squeeze()

    else:
        # Sample `seq_len` random images from the Fashion MNIST dataset
        indices = torch.randint(0, len(fashion_mnist), (seq_len,))
        for i, idx in enumerate(indices):
            img, _ = fashion_mnist[idx]
            sequence[i] = (2*img-1).squeeze()

    if binary:
        sequence[sequence > 0.5] = 1
        sequence[sequence <= 0.5] = -1

    return sequence


def load_sequence_cifar(seed, seq_len,data_type='rgb'):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    # Define the luminance formula for converting RGB to grayscale

    if data_type=='gray':
        transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        # Initialize an empty tensor to store the sequence of digits
        sequence = torch.zeros((seq_len, 32, 32))
    elif data_type=='binary':
        transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        sequence = torch.zeros((seq_len, 32, 32))
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        # Initialize an empty tensor to store the sequence of digits
        sequence = torch.zeros((seq_len, 3, 32, 32))

    # Load the CIFAR10 dataset
    cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Sample `seq_len` random images from the cifar dataset
    indices = torch.randint(0, len(cifar), (seq_len,))
    for i, idx in enumerate(indices):
        img, _ = cifar[idx]
        sequence[i] = 2*((img>img.mean()).float())-1 if data_type=='binary' else (2*img-1)
    
    return sequence

def load_sequence_cifar100(seed, seq_len,data_type='rgb'):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    # Define the luminance formula for converting RGB to grayscale

    if data_type=='gray':
        transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        # Initialize an empty tensor to store the sequence of digits
        sequence = torch.zeros((seq_len, 32, 32))
    elif data_type=='binary':
        transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        sequence = torch.zeros((seq_len, 32, 32))
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        # Initialize an empty tensor to store the sequence of digits
        sequence = torch.zeros((seq_len, 3, 32, 32))

    # Load the CIFAR10 dataset
    cifar = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Sample `seq_len` random images from the cifar dataset
    indices = torch.randint(0, len(cifar), (seq_len,))
    for i, idx in enumerate(indices):
        img, _ = cifar[idx]
        sequence[i] = 2*((img>img.mean()).float())-1 if data_type=='binary' else (2*img-1)
    
    return sequence

def load_sequence_random(seed,seq_len,binary=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # 生成100个28x28的随机数张量，范围在0到1之间
    random_data = torch.rand(seq_len, 28, 28)

    # 将张量中的值二值化为0或1，可以使用阈值进行控制
    threshold = 0.5
    if binary:
        sequence = 2*(random_data > threshold).float()-1
    else:
        sequence=random_data
    return sequence

def load_random_dataset(number,dim,binary=True,seed=1):
    # set the random seed for reproducibility
    torch.manual_seed(seed)
    # generate random binary patterns
    random_data = torch.rand(number,dim)
    threshold = 0.5
    if binary:
        random_data = 2*(random_data > threshold).float()-1
    else:
        random_data = 2*random_data-1
    return random_data

def load_emnist_dataset(number,size,binary=True,seed=1):
    if size>24:
        transform = transforms.Compose([
        transforms.Resize(size),  # 将图像调整为8x8
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像 
        ])
    else:
        transform = transforms.Compose([
        transforms.CenterCrop(24),  # 将图像居中裁剪为24x24
        transforms.Resize(size),  # 将图像调整为8x8
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像 
        ])
    emnist_train = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True, transform=transform, download=True)
    num_samples = number  # 你想要处理的样本数量
    emnist_train_list = list(emnist_train)
    EMNIST_data=[]
    # 随机选择num_samples个样本
    random_samples = random.sample(emnist_train_list, num_samples)

    # 打印每个类别挑选的样本
    for i, (sample_image, sample_label) in enumerate(random_samples):
        sample_image = sample_image.squeeze().numpy().T
        EMNIST_data.append(sample_image.flatten())
    EMNIST_data=np.array(EMNIST_data)

    EMNIST_data_binary=[]   
    for i in range(len(EMNIST_data)):
        gray_image = EMNIST_data[i].reshape(size, size)
        mean_threshold = np.mean(gray_image)
        # 应用自适应阈值二值化
        binary_image = np.where(gray_image > mean_threshold, 1, -1)  # 这里简单地使用0.5作为阈值，您也可以根据需要调整
        EMNIST_data_binary.append(binary_image.flatten())
    EMNIST_data_binary=np.array(EMNIST_data_binary)
    if binary:
        return EMNIST_data_binary
    else:
        return EMNIST_data
    
    
def load_mnist_dataset(number,size,binary=True,seed=1):
    if size>24:
        transform = transforms.Compose([
        transforms.Resize(size),  # 将图像调整为8x8
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像 
        ])
    else:
        transform = transforms.Compose([
        transforms.CenterCrop(24),  # 将图像居中裁剪为24x24
        transforms.Resize(size),  # 将图像调整为8x8
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像 
        ])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    num_samples = number  # 你想要处理的样本数量
    mnist_train_list = list(mnist_train)
    MNIST_data=[]
    # 随机选择num_samples个样本
    random_samples = random.sample(mnist_train_list, num_samples)
    
    # 打印每个类别挑选的样本
    for i, (sample_image, sample_label) in enumerate(random_samples):
        sample_image = sample_image.squeeze().numpy().T
        MNIST_data.append(sample_image.flatten())
    MNIST_data=np.array(MNIST_data)
    
    MNIST_data_binary=[]
    for i in range(len(MNIST_data)):
        gray_image = MNIST_data[i].reshape(size, size)
        mean_threshold = np.mean(gray_image)
        # 应用自适应阈值二值化
        binary_image = np.where(gray_image > mean_threshold, 1, -1)
        MNIST_data_binary.append(binary_image.flatten())
    MNIST_data_binary=np.array(MNIST_data_binary)
    if binary:
        return MNIST_data_binary
    else:
        return MNIST_data
    
############################################################################################################
# noise functions
############################################################################################################

def add_gaussian_noise(image, sigma=0.1):
    # 将图像和噪声都移到相同的设备上，例如cuda:0或cpu
    device = image.device
    noisy_image = image.clone()
    noise = torch.randn(image.size(), device=device) * sigma
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # 将像素值截断在0和1之间
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.clone()
    salt_mask = torch.rand(image.shape) < salt_prob
    pepper_mask = torch.rand(image.shape) < pepper_prob
    noisy_image[salt_mask] = 1.0  # Set salt pixels to white
    noisy_image[pepper_mask] = -1.0  # Set pepper pixels to black
    return noisy_image

def generate_random_color_mask(mask_size):
    # 创建一个随机颜色的遮挡
    color_mask = torch.rand(3, mask_size[0], mask_size[1])  # 3通道的彩色遮挡
    return color_mask
    
def add_block_mask(image, mask_start, mask_size, mask_color):
    noisy_image = image.clone()
    x, y = mask_start
    w, h = mask_size
    noisy_image[:, x:x + w, y:y + h] = mask_color
    return noisy_image

def add_block_mask_binary(image, mask_start, mask_size, mask_color=-1):
    noisy_image = image.clone()
    x, y = mask_start
    w, h = mask_size
    noisy_image[x:x + w, y:y + h] = mask_color
    return noisy_image

def add_salt_and_pepper_noise_binary(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.clone()
    height, width = image.size(0), image.size(1)

    for i in range(height):
        for j in range(width):
            random_value = random.random()  # 生成随机值
            if random_value < salt_prob:
                noisy_image[i, j] = 1  # 设置为白色
            elif random_value > 1 - pepper_prob:
                noisy_image[i, j] = -1  # 设置为黑色

    return noisy_image

############################################################################################################
# training functions
############################################################################################################


def train_multilayer_mannul(hnn, optimizer, seq, learn_iters, device):
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev = hnn.init_hidden(1).to(device)
        batch_loss = 0
        for k in range(seq_len):
            x = seq[k]
            prev = x.clone().detach()
            optimizer.zero_grad()
            energy = hnn.get_energy(x, prev)
            energy.backward()
            optimizer.step()

            # add up the loss value at each time step
            epoch_loss += energy.item() / seq_len
        losses.append(epoch_loss/seq.shape[1])
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss/seq.shape[1]}')

    print(f'training HNN complete, time: {time.time() - start_time}')
    return losses


def train_multilayer_hnn(model,optimizer, seq, learn_iters, device):
    criterion = nn.MSELoss()
    seq_len = seq.shape[0]
    losses = []  # 存储损失值
    model.train()
    for epoch in range(learn_iters):
        # if epoch%200==0:
        #     lr=optimizer.param_groups[0]['lr']
        #     optimizer.param_groups[0]['lr']=lr*0.90
        epoch_loss = 0
        for k in range(seq_len):
            x=seq[k]
            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() /seq_len
        losses.append(epoch_loss)

        if epoch_loss<1e-7:
            break
    print(f'Epoch [{epoch}/{learn_iters}], Epoch_Loss: {epoch_loss}')
    return losses

def train_multilayer_batch(model,optimizer, seq, learn_iters, device):
    criterion = nn.MSELoss()
    seq_len = seq.shape[0]
    losses = []  # 存储损失值

    for epoch in range(learn_iters):
        # if epoch%200==0:
        #     lr=optimizer.param_groups[0]['lr']
        #     optimizer.param_groups[0]['lr']=lr*0.90
        epoch_loss = 0
        x=seq
        optimizer.zero_grad()
        output = model.forward(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        losses.append(epoch_loss)
        # if epoch%10==0:
        #     print(f'Epoch [{epoch}/{learn_iters}], Loss: {loss.item()}')
        if loss.item()<1e-8:
            print(f'Epoch [{epoch}/{learn_iters}], Loss: {loss.item()}')
            break
    print(f'Epoch [{epoch}/{learn_iters}], Loss: {loss.item()}')
    return losses


def train_multilayer_sequence(model,optimizer, seq, learn_iters, device):
    criterion = nn.MSELoss()
    seq_len = seq.shape[0]
    losses = []  # 存储损失值

    for epoch in range(learn_iters):
        prev = seq[0].clone().detach()
        epoch_loss = 0
        for k in range(1,seq_len+1):
            x=seq[k%(seq_len)]
            optimizer.zero_grad()
            output = model.forward(prev)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            prev = x.clone().detach()
            epoch_loss += loss.item()/seq_len 
        losses.append(epoch_loss)
        if loss.item()<1e-8:
            break
    return losses

import torch
import torch.nn as nn

def train_multilayer_sequence_batch(model, optimizer, seq, learn_iters, device):
    criterion = nn.MSELoss()
    seq_len = seq.shape[0]
    losses = []  # 用于存储损失值

    # 将序列扩展一位以实现循环
    inputs = seq
    targets = torch.cat((seq[1:], seq[:1]), dim=0)  # 目标为序列后移一位，最后一位对应第一位

    # 确保输入和目标在正确的设备上
    inputs = inputs.to(device)
    targets = targets.to(device)

    for epoch in range(learn_iters):
        # 模型训练模式
        model.train()

        optimizer.zero_grad()

        # 前向传播整个序列
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播并优化
        loss.backward()
        optimizer.step()

        # 记录损失
        losses.append(loss.item())

        # 提前停止条件
        if loss.item() < 1e-8:
            break

    return losses



def train_multilayer_hetero(model,optimizer, seq,seq_out, learn_iters, device):
    criterion = nn.MSELoss()
    seq_len = seq.shape[0]
    losses = []  # 存储损失值

    for epoch in range(learn_iters):
        epoch_loss = 0
        for k in range(0,seq_len):
            x=seq[k%(seq_len)]
            x_out=seq_out[k%(seq_len)]
            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, x_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/seq_len 
        losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{learn_iters}], Loss: {loss.item()}')
        if loss.item()<1e-8:
            break
    return losses

############################################################################################################
# Training Predictive Coding Network 
############################################################################################################

def train_singlelayer_tPC(pc, optimizer, seq, learn_iters, device):
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev = pc.init_hidden(1).to(device)
        batch_loss = 0
        for k in range(seq_len):
            x = seq[k]
            optimizer.zero_grad()
            energy = pc.get_energy(x, prev)
            energy.backward()
            optimizer.step()
            prev = x.clone().detach()

            # add up the loss value at each time step
            epoch_loss += energy.item() / (seq_len*seq.shape[1])
        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')
    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def train_multilayer_tPC(model, optimizer, seq, learn_iters, inf_iters, inf_lr, device):
    """
    Function to train multi layer tPC
    """
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev_z = model.init_hidden(1).to(device)
        for k in range(seq_len):
            x = seq[k].clone().detach()
            optimizer.zero_grad()
            model.inference(inf_iters, inf_lr, x, prev_z)
            energy = model.update_grads(x, prev_z)
            energy.backward()
            optimizer.step()
            prev_z = model.z.clone().detach()

            # add up the loss value at each time step
            epoch_loss += energy.item() / (seq_len*seq.shape[1])

        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')
        
    print(f'training PC complete, time: {time.time() - start_time}')
    return losses


############################################################################################################
# recall functions
############################################################################################################


def singlelayer_recall(model, seq, params):
    """recall function for Si
    
    seq: PxN patterns

    output: PxN recall of patterns
    """
    model.eval()
    seq_len, N = seq.shape
    recall = seq.clone().detach()
    mid_state=[]
    mid_state.append(recall.clone().detach())
    for i in range(params["recall_iters"]):
        recall[0:] = torch.sign(model(recall[0:])) if params["data_type"] == 'binary' else model(recall[0:])
        mid_state.append(recall.clone().detach())
    if 'mid_state'in params and params['mid_state']:
        return recall,mid_state
    else:
        return recall

def hn_recall(model, patterns, seq, params):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: PxN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = seq.clone().detach()
    # recall using true image at each step
    recall[0:] = torch.sign(model(patterns, seq[0:])) if params["data_type"] == 'binary' else model(patterns, seq[0:]) # PxN
    return recall

def singlelayer_sequence_recall(model, seq, device, params):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    model.eval()
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if params['query_type'] == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq[:-1])) if params['data_type'] == 'binary' else model(seq[:-1])
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            recall[k] = torch.sign(model(recall[k-1:k])) if params['data_type'] == 'binary' else model(recall[k-1:k]) # 1xN

    return recall

def singlelayer_hetero_recall(model, seq, device, params):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    model.eval()
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    # recall using true image at each step
    recall[:] = torch.sign(model(seq[:])) if params['data_type'] == 'binary' else model(seq[:])
    return recall

def multilayer_sequence_recall(model, seq, inf_iters, inf_lr, params, device):
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    prev_z = model.init_hidden(1).to(device)

    if params['query_type'] == 'online':
        # infer the latent state at each time step, given correct previous input
        for k in range(seq_len-1):
            x = seq[k].clone().detach()
            model.inference(inf_iters, inf_lr, x, prev_z)
            prev_z = model.z.clone().detach()
            _, pred_x = model(prev_z)
            recall[k+1] = pred_x

    elif params['query_type'] == 'offline':
        # only infer the latent of the cue, then forward pass
        x = seq[0].clone().detach()
        model.inference(inf_iters, inf_lr, x, prev_z)
        prev_z = model.z.clone().detach()

        # fast forward pass
        for k in range(1, seq_len):
            prev_z, pred_x = model(prev_z)
            recall[k] = pred_x

    return recall

def hn_sequence_recall(model, seq, device, params):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if params['query_type'] == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq, seq[:-1])) if params['data_type'] == 'binary' else model(seq, seq[:-1]) # (P-1)xN
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            # prev = torch.sign(model(seq, prev)) if binary else model(seq, prev) # 1xN
            recall[k] = torch.sign(model(seq, recall[k-1:k])) if params['data_type'] == 'binary' else model(seq, recall[k-1:k]) # 1xN

    return recall


############################################################################################################
# plotting functions
############################################################################################################

# local plotting functions for exploratory purposes
def plot_PC_loss(loss, seq_len, learn_iters,fig_path, data_type='continuous'):
    # plotting loss for tunning; temporary
    print("Final Loss: ", loss[-1])
    plt.figure()
    plt.plot(loss, label='squared error sum (PC))')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}_{data_type}')

def plot_HNN_loss(loss, seq_len, learn_iters,fig_path, data_type='continuous'):
    # plotting loss for tunning; temporary
    print("Final Loss: ", loss[-1])
    plt.figure()
    plt.plot(loss, label='squared error sum (HNN)')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}_{data_type}')

def plot_recalls(recall, model_name,fig_path, params):
    seq_len = recall.shape[0]
    num_per_row=params["plot_num_per_row"]
    num_row=math.ceil(seq_len/num_per_row)
    if seq_len<=num_per_row:
        num_per_row=seq_len
        num_row=1
    # Create a 2D array of subplots
    fig, axs = plt.subplots(num_row, num_per_row, figsize=(num_per_row, num_row))

    for j in range(seq_len):
        row, col = divmod(j, num_per_row)
        if seq_len==1:
            ax = axs
        else:
            ax = axs[col] if num_row == 1 else axs[row, col]
        if recall.shape[1]==784 or recall.shape[1]!=3072:
            n=int(math.sqrt(recall.shape[1]))
            if recall.shape[1]==392:
                n=28
            img=to_np(recall[j].reshape(n, math.ceil(recall.shape[1]/n)))
        else:
            img=to_np(recall[j].reshape((3, 32, 32)).permute(1, 2, 0))
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
    # 如果子图的数量不是num_per_row的倍数，将其余的子图删除
    for i in range(seq_len, num_per_row * num_row):
        row, col = divmod(i, num_per_row)
        fig.delaxes(axs[row, col])
    plt.suptitle(model_name,y=1.0,fontsize=10)
    # plt.subplots_adjust(top=0.85)  # 调整标题和子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    plt.savefig(fig_path + f'/{model_name}_len{seq_len}_data{params["data_type"]}_{params["data_set"]}_repeat{int(params["repeat"])}percen', dpi=150)

def plot_memory(x, seed,fig_path, params):
    seq_len = x.shape[0]
    num_per_row=params["plot_num_per_row"]
    num_row=math.ceil(seq_len/num_per_row)
    if seq_len<=num_per_row:
        num_per_row=seq_len
        num_row=1
    fig, axs = plt.subplots(num_row,num_per_row, figsize=(num_per_row, num_row))
    for j in range(seq_len):
        row, col = divmod(j, num_per_row)
        if seq_len==1:
            ax = axs
        else:
            ax = axs[col] if num_row == 1 else axs[row, col]
        if x.shape[1]==784 or x.shape[1]==1024 or x.shape[1]!=3072:
            n=int(math.sqrt(x.shape[1]))
            if x.shape[1]==392:
                n=28
            img=to_np(x[j].reshape(n, math.ceil(x.shape[1]/n)))
        else:
            img=to_np(x[j].reshape((3, 32, 32)).permute(1, 2, 0))
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
    # 如果子图的数量不是num_per_row的倍数，将其余的子图删除
    for i in range(seq_len, num_per_row * num_row):
        row, col = divmod(i, num_per_row)
        fig.delaxes(axs[row, col])
    plt.suptitle('memory',y=1.0,fontsize=10)
    # plt.subplots_adjust(top=0.85)  # 调整标题和子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    plt.savefig(fig_path + f'/memory_len{seq_len}_seed{seed}_data{params["data_type"]}_{params["data_set"]}_repeat{int(params["repeat"])}percen', dpi=150)

def plot_memory_noise(x, seed,fig_path, params):
    seq_len = x.shape[0]
    num_per_row=params["plot_num_per_row"]
    num_row=math.ceil(seq_len/num_per_row)
    if seq_len<=num_per_row:
        num_per_row=seq_len
        num_row=1
    fig, axs = plt.subplots(num_row,num_per_row, figsize=(num_per_row, num_row))
    for j in range(seq_len):
        row, col = divmod(j, num_per_row)
        if seq_len==1:
            ax = axs
        else:
            ax = axs[col] if num_row == 1 else axs[row, col]
        if x.shape[1]==784 or x.shape[1]!=3072:
            n=int(math.sqrt(x.shape[1]))
            if x.shape[1]==392:
                n=28
            img=to_np(x[j].reshape(n, math.ceil(x.shape[1]/n)))
        else:
            img=to_np(x[j].reshape((3, 32, 32)).permute(1, 2, 0))
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
    # 如果子图的数量不是num_per_row的倍数，将其余的子图删除
    for i in range(seq_len, num_per_row * num_row):
        row, col = divmod(i, num_per_row)
        fig.delaxes(axs[row, col])
    plt.suptitle('query with noise',y=1.0,fontsize=10)
    # plt.subplots_adjust(top=0.85)  # 调整标题和子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    # plt.tight_layout()
    plt.savefig(fig_path + f'/memory_query_len{seq_len}_seed{seed}_data{params["data_type"]}_{params["data_set"]}_repeat{int(params["repeat"])}percen', dpi=150)

def plot_weight(weights,name):
    vmax=np.abs(weights).max()
    vmin=-vmax
    plt.figure()
    plt.imshow(weights,vmax=vmax,vmin=vmin,cmap='seismic')
    plt.colorbar()
    plt.title(name+' weights')
    plt.show()
############################################################################################################
# Distance functions
############################################################################################################

def calculate_sim(x, y,distance_type='cos'):
    if distance_type=='cos':
        return  F.cosine_similarity(x, y, dim=1)
    elif distance_type=='l2':
        return torch.sqrt(torch.sum((x-y)**2,dim=1))
    elif distance_type=='hamming':
        # return torch.sum(x!=y,dim=1)/x.shape[1]
        return torch.sum(x!=y,dim=1)
    

############################################################################################################
# Mask Generation functions
############################################################################################################
def generate_mask(input_size, probability=None,seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if probability is not None:
        # Generate a mask with a specified probability of zeros
        mask = (torch.rand(input_size,input_size) > probability).float()
        # Set the diagonal elements to zero
        mask.fill_diagonal_(0)
    else:
        raise ValueError("'probability' must be provided.")
    return mask