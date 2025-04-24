import sys
sys.path.insert(0, './..')
import os
import numpy as np
import matplotlib.pyplot as plt
from HopfiledNetwork import *
from utils import *
import json
from skimage.filters import threshold_mean
from tqdm import tqdm
import configargparse
import pickle
from torchsummary import summary
global args

parser = configargparse.ArgParser()
parser.add('-c', '--config', required=False,is_config_file=True, help='config file')
parser.add_argument('--dimension', default=128, type=int, help='The Number/Dimension of Neurons')
parser.add_argument('--hidden', default=32, type=int, help='The Number of Neurons in Hidden Layer')
parser.add_argument('--interval', default=1, type=int, help='The interval of the number of stored patterns')
parser.add_argument('--max-pattern', default=128, type=int, help='The maximum number of stored patterns')
parser.add_argument('--min-pattern', default=1, type=int, help='The minimum number of stored patterns')
parser.add_argument('--train-eval', default='eval',type=str, help='train or eval')
parser.add_argument('--variation', default=0.0,type=float, help='memristor variation')
parser.add_argument('--stuck', default=0.0,type=float, help='stuck at fault rate')
parser.add_argument('--corruption', default=0.05, type=float, help='corruption rate')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--dataset', default='random', type=str, help='emnist or random')
parser.add_argument('--binary', default=1, type=int, help='binary or continuous')
parser.add_argument('--nonlin', default='tanh', type=str, help='nonlinearity')
parser.add_argument('--numlayer', default=1, type=int, help='hidden layer dimension')

def preprocessing(img):
    # Resize image
    # img = resize(img, (w, h), mode='reflect')
    w, h = img.shape

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1  # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten


def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def add_gaussian_noise_continuous(image, sigma=0.1):
    # 将图像和噪声都移到相同的设备上，例如cuda:0或cpu
    device = image.device
    noisy_image = image.clone()
    noise = torch.randn(image.size(), device=device) * sigma
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, -1, 1)  # 将像素值截断在0和1之间
    return noisy_image

# add block to the picture
def add_block_mask(image, block_size=4):
    noisy_image = np.copy(image)
    noisy_image = noisy_image.reshape(8, 8)
    noisy_image[:,:block_size] = -1
    noisy_image = noisy_image.reshape(64)
    return noisy_image


path = 'memristor_simulation'
result_path = os.path.join('./results/', path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

num_path = os.path.join('./results/', path, 'numerical')
if not os.path.exists(num_path):
    os.makedirs(num_path)

fig_path = os.path.join('./results/', path, 'fig')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

model_path = os.path.join('./results/', path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

def main(params):
    device = 'cuda'
    print(device)

    # command line inputs   
    num_dimen = params["dimensions"]
    interval=params["interval"]
    train_eval=params["train_eval"]
    mem_var=params["variation"]
    stuck_at_fault_rate=params["stuck"]
    corruption_rate=params["corruption"]
    hidden_dim=params["hidden"]


    #generate rabdom binary patterns
    n_random_patterns=10000
    random_data_list=torch.randint(-1,1,(n_random_patterns,num_dimen)).float()
    random_data_list=random_data_list*2+1
    random_data=random_data_list
    stored_data_list=random_data_list


    input_size=num_dimen
    encoder_dim=hidden_dim
    num_hidden_layers=params['numlayer']
    if params['nonlin']=='tanh':
        nonlin='tanh'
    else:
        nonlin='relu'
    learn_lr=3e-4
    learn_iters=60000
    distance_type='cos'
    params.update({
        'input_size':input_size,
        'encoder_dim':encoder_dim,
        'num_hidden_layers':num_hidden_layers,
        'nonlin':nonlin,
        'learn_lr':learn_lr,
        'learn_iters':learn_iters,
        'device':device,
        'recall_iters':100,
        'mid_state':False,
    })

    torch.manual_seed(params['seed'])

    #plot settings
    plot_weight_flag=True

    #memristor variation and stuck at fault
    num_simulations=20
    weight_decay=0.0
    mem_stuck_mask=torch.ones(input_size,input_size)

    if train_eval=='train': 
        num_pattern_list=np.arange(params['min_pattern'],params['max_pattern'],interval)

        stored_patterns=[]
        MultiHNN_weights={'encoder':[],'decoder':[]}
        
        num_load_data=60000
        if params['dataset']=='random':
            if params['data_type']=='binary':
                stored_data_list=load_random_dataset(num_load_data,num_dimen,binary=True)
            else:
                stored_data_list=load_random_dataset(num_load_data,num_dimen,binary=False)
        elif params['dataset']=='emnist':
            if params['data_type']=='binary':
                stored_data_list=torch.from_numpy(load_emnist_dataset(num_load_data,int(np.sqrt(num_dimen)),binary=True)).float()
            else:
                stored_data_list=torch.from_numpy(load_emnist_dataset(num_load_data,int(np.sqrt(num_dimen)),binary=False)).float()
        elif params['dataset']=='mnist':
            if params['data_type']=='binary':
                stored_data_list=torch.from_numpy(load_mnist_dataset(num_load_data,int(np.sqrt(num_dimen)),binary=True)).float()
            else:
                stored_data_list=torch.from_numpy(load_mnist_dataset(num_load_data,int(np.sqrt(num_dimen)),binary=False)).float()
        else:
            # error handling
            print("Invalid dataset")
            return
        
        for indice in tqdm(range(len(num_pattern_list))):
            num_pattern=num_pattern_list[indice]
            print(f"num_pattern: {num_pattern}")

            # 生成随机排列的索引
            random_indices = torch.randperm(stored_data_list.size(0))[:num_pattern]
            # 使用随机索引选择数据
            stored_data = stored_data_list[random_indices].to(device)

            # set the learning rate and the number of iterations
            if params['data_type']=='binary':
                print('binary')
                # if num_pattern > 256:
                #     learn_iters = params['learn_iters'] * 20
                #     learn_lr = params['learn_lr'] / 30
                # elif num_pattern > 128:
                #     learn_iters = params['learn_iters'] * 20
                #     learn_lr = params['learn_lr'] / 30
                # elif num_pattern > 64:
                #     learn_iters = params['learn_iters'] * 20
                #     learn_lr = params['learn_lr'] / 30
                # elif num_pattern > 32:
                #     learn_iters = params['learn_iters'] * 20
                #     learn_lr = params['learn_lr'] / 30
                # elif num_pattern > 16:
                #     learn_iters = params['learn_iters'] * 20
                #     learn_lr = params['learn_lr'] /30
                # else:
                #     learn_iters = params['learn_iters']*20
                #     learn_lr = params['learn_lr']/30
            else:
                print('continuous')
                # if num_pattern > 256:
                #     learn_iters = params['learn_iters'] * 4
                #     learn_lr = params['learn_lr'] / 8
                # elif num_pattern > 128:
                #     learn_iters = params['learn_iters'] * 16
                #     learn_lr = params['learn_lr'] / 8
                # elif num_pattern > 64:
                #     learn_iters = params['learn_iters'] * 16
                #     learn_lr = params['learn_lr'] / 8
                # elif num_pattern > 32:
                #     learn_iters = params['learn_iters'] * 8
                #     learn_lr = params['learn_lr'] / 8
                # elif num_pattern > 16:
                #     learn_iters = params['learn_iters'] * 4
                #     learn_lr = params['learn_lr']/4
                # else:
                #     learn_iters = params['learn_iters']*4
                #     learn_lr = params['learn_lr']/4
            # generate mask
            mem_stuck_mask=generate_mask(input_size,stuck_at_fault_rate,seed=1).to(device)

            # create the network
            multi_hnn=MultiLayerHNN_beta(input_dim=input_size,encoding_dim=encoder_dim,num_hidden_layers=num_hidden_layers,nolinear=nonlin,mask_prob=stuck_at_fault_rate,device=device).to(device)


            # train the network
            optimizer_multi = torch.optim.RMSprop(multi_hnn.parameters(), lr=learn_lr/2,weight_decay=weight_decay)
            # optimizer_multi = torch.optim.Adam(multi_hnn.parameters(), lr=learn_lr/2,weight_decay=weight_decay)
            # multi_hnn_losses=train_multilayer_hnn(multi_hnn, optimizer_multi, stored_data, 2*learn_iters, device)
            multi_hnn_losses=train_multilayer_batch(multi_hnn, optimizer_multi, stored_data, 2*learn_iters, device)
            #store the data
            stored_patterns.append(to_np(stored_data))
            for i,layer in enumerate(multi_hnn.encoder):
                if i%2==0:
                    MultiHNN_weights['encoder'].append(to_np(multi_hnn.encoder[i].weight))
            MultiHNN_weights['decoder'].append(to_np(multi_hnn.decoder[0].weight))


        # 保存数据
        if params['data_type']=='binary':
            with open(model_path + f'/Multilayer_capacity_{params["dataset"]}_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_simulation.pkl', 'wb') as f:
                pickle.dump({
                    'num_pattern_list': num_pattern_list,
                    'stored_patterns': stored_patterns,
                    'MultiHNN_weights': MultiHNN_weights
                }, f)
        else:  
            with open(model_path + f'/Multilayer_capacity_{params["dataset"]}_continuous_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_simulation.pkl', 'wb') as f:
                pickle.dump({
                    'num_pattern_list': num_pattern_list,
                    'stored_patterns': stored_patterns,
                    'MultiHNN_weights': MultiHNN_weights
                }, f)
    
    elif train_eval=='eval':
        if params['data_type']=='binary':
            with open(model_path + f'/Multilayer_capacity_{params["dataset"]}_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_simulation.pkl', 'rb') as f:
            # with open(model_path + f'AssociateMemory_capacity_random_256_simulation.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            with open(model_path + f'/Multilayer_capacity_{params["dataset"]}_continuous_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_simulation.pkl', 'rb') as f:
                data = pickle.load(f)

        # 访问加载的数据
        num_pattern_list = data['num_pattern_list']
        stored_patterns = data['stored_patterns']
        MultiHNN_weights = data['MultiHNN_weights']
        print(num_pattern_list)

        MultiHNN_similarity_list=[]
        print(num_pattern_list.shape)
        for indice in tqdm(range(len(num_pattern_list))):
            num_pattern=num_pattern_list[indice]
            stored_data=stored_patterns[indice]
            Encoder_weight=MultiHNN_weights['encoder']
            Decoder_weight=MultiHNN_weights['decoder'][indice]

  
            stored_data=torch.from_numpy(stored_data).float()
            with torch.no_grad():
                multi_hnn=MultiLayerHNN_beta(input_dim=input_size,encoding_dim=encoder_dim,num_hidden_layers=num_hidden_layers,nolinear=nonlin,device=device).to(device)
                            
                for i,layer in enumerate(multi_hnn.encoder):
                    if i%2==0:
                        multi_hnn.encoder[i].weight=torch.nn.Parameter(torch.from_numpy(Encoder_weight[num_hidden_layers*indice+int(i/2)]).float().to(device))
                multi_hnn.decoder[0].weight=torch.nn.Parameter(torch.from_numpy(Decoder_weight).float().to(device))

                for _ in range(num_simulations):
                    #corrupt the data
                    params['recall_iters']=100
                    if params['data_type']=='binary':
                        test_data=np.array([get_corrupted_input(d,corruption_rate) for d in stored_data])
                        test_data = torch.from_numpy(test_data).float().to(device)
                    else:
                        test_data=np.array([add_gaussian_noise_continuous(d,0.6) for d in stored_data])
                        test_data=test_data.reshape(-1,input_size)
                        test_data = torch.from_numpy(test_data).float().to(device)

                    # recall data
                    MultiHNN_recall=(singlelayer_recall(multi_hnn, test_data, params)).detach().cpu()

                    # calculate similarity
                    MultiHNN_similarity=torch.mean(calculate_sim(stored_data,MultiHNN_recall,distance_type=distance_type))

                    MultiHNN_similarity_list.append(to_np(MultiHNN_similarity))

        input_similarities=torch.mean(calculate_sim(stored_data,test_data.detach().cpu(),distance_type=distance_type))
        MultiHNN_similarity_list=np.array(MultiHNN_similarity_list).reshape(-1,num_simulations).T

        # 保存数据
        if params['data_type']=='binary':
            np.savez(num_path +f'/Multilayer_capacity_{params["dataset"]}_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_var={mem_var}_corr={corruption_rate}_simulation.npz',
                 num_pattern_list=num_pattern_list,MultiHNN_similarity_list=MultiHNN_similarity_list,input_similarities=input_similarities)
        
        else:
            np.savez(num_path +f'/Multilayer_capacity_{params["dataset"]}_continuous_{num_dimen}_hd={hidden_dim}_layer={num_hidden_layers}_stuck={stuck_at_fault_rate}_var={mem_var}_corr={corruption_rate}_simulation.npz',
                 num_pattern_list=num_pattern_list,MultiHNN_similarity_list=MultiHNN_similarity_list,input_similarities=input_similarities)      


if __name__ == "__main__":
    args = parser.parse_args()

    params = {
        "dimensions": args.dimension,
        "hidden": args.hidden,
        "interval": args.interval,
        "max_pattern": args.max_pattern,
        "min_pattern": args.min_pattern,
        "train_eval": args.train_eval,
        "variation": args.variation,
        "stuck": args.stuck,
        "corruption": args.corruption,
        "seed": args.seed,
        "dataset": args.dataset,   
        "nonlin": args.nonlin,
        "numlayer": args.numlayer,
    }
    if args.binary==0:
        params['data_type']= 'continuous'
    else:
        params['data_type']= 'binary'
    main(params)

