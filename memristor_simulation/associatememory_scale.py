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
global args

parser = configargparse.ArgParser()
parser.add('-c', '--config', required=False,is_config_file=True, help='config file')
parser.add_argument('--dimension', default=16, type=int, help='The Number/Dimension of Neurons')
parser.add_argument('--interval', default=1, type=int, help='The interval of the number of stored patterns')
parser.add_argument('--train-eval', default='eval',type=str, help='train or eval')
parser.add_argument('--variation', default=0.0,type=float, help='memristor variation')
parser.add_argument('--stuck', default=0.0,type=float, help='stuck at fault rate')
parser.add_argument('--corruption', default=0.05, type=float, help='corruption rate')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--dataset', default='random', type=str, help='emnist or random')
parser.add_argument('--binary', default=True, type=bool, help='binary or continuous')
parser.add_argument('--max-pattern', default=64, type=int, help='The maximum number of stored patterns')
parser.add_argument('--min-pattern', default=1, type=int, help='The minimum number of stored patterns')

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
    device = "cpu"
    
    # command line inputs   
    num_dimen = params["dimensions"]
    interval=params["interval"]
    train_eval=params["train_eval"]
    mem_var=params["variation"]
    stuck_at_fault_rate=params["stuck"]
    corruption_rate=params["corruption"]


    torch.random.manual_seed(params['seed'])

    #generate rabdom binary patterns

    input_size=num_dimen
    # learn_lr=0.03
    # learn_iters=400
    learn_lr=1e-3
    learn_iters=10000
    distance_type='cos'
    params.update({
        'input_size':input_size,
        'learn_lr':learn_lr,
        'learn_iters':learn_iters,
        'device':device,
        'recall_iters':100,
        'data_type':'binary',
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

        n_random_patterns=10000
        if params['dataset']=='random':
            if params['binary']:
                stored_data_list=load_random_dataset(n_random_patterns,num_dimen,binary=True)
            else:
                stored_data_list=load_random_dataset(n_random_patterns,num_dimen,binary=False)
        elif params['dataset']=='emnist':
            if params['binary']:
                stored_data_list=torch.from_numpy(load_emnist_dataset(n_random_patterns,int(np.sqrt(num_dimen)),binary=True)).float()
            else:
                stored_data_list=torch.from_numpy(load_emnist_dataset(n_random_patterns,int(np.sqrt(num_dimen)),binary=False)).float()
        elif params['dataset']=='mnist':
            if params['data_type']=='binary':
                stored_data_list=torch.from_numpy(load_mnist_dataset(n_random_patterns,int(np.sqrt(num_dimen)),binary=True)).float()
            else:
                stored_data_list=torch.from_numpy(load_mnist_dataset(n_random_patterns,int(np.sqrt(num_dimen)),binary=False)).float()
        else:
            # error handling
            print("Invalid dataset")
            return
        print(stored_data_list.shape)


        stored_patterns=[]
        Hebb_weights=[]
        Storkey_weights=[]
        Pinv_weights=[]
        Equilibrium_weights=[]
        HNN_weights=[]
        for indice in tqdm(range(len(num_pattern_list))):
            num_pattern=num_pattern_list[indice]
            print(f"num_pattern: {num_pattern}")
            # 生成随机排列的索引
            random_indices = torch.randperm(stored_data_list.size(0))[:num_pattern]
            # 使用随机索引选择数据
            stored_data = stored_data_list[random_indices]
            # # set the learning rate and the number of iterations
            # if num_pattern>32:
            #     learn_iters=params['learn_iters']*2
            #     learn_lr=params['learn_lr']/3
            # elif num_pattern>64:
            #     learn_iters=params['learn_iters']*4
            #     learn_lr=params['learn_lr']/9
            # elif num_pattern>128:
            #     learn_iters=params['learn_iters']*27
            #     learn_lr=params['learn_lr']/9
            # else:
            #     learn_iters=params['learn_iters']*4
            #     learn_lr=params['learn_lr']/3
            # generate mask
            mem_stuck_mask=generate_mask(input_size,stuck_at_fault_rate,seed=1)

            # create the network
            hnn=SingleLayerHNN_beta(input_size=input_size)
            Hebb = HopfieldNet(num_neurons=input_size)
            Storkey = HopfieldNet(num_neurons=input_size)
            Pinv=HopfieldNet(num_neurons=input_size)
            Equilibrium=HopfieldNet(num_neurons=input_size)

            # train the network
            Hebb.learn_patterns(stored_data, rule='Hebb',options={})
            Storkey.learn_patterns(stored_data, rule='Storkey',options={})     
            Pinv.learn_patterns(stored_data, rule='Pinv',options={})
            Equilibrium.learn_patterns(stored_data, rule='Equilibrium',options={})

            # optimizer_hnn = torch.optim.RMSprop(hnn.parameters(), lr=learn_lr,weight_decay=weight_decay)

            optimizer_hnn = torch.optim.Adam(hnn.parameters(), lr=learn_lr,weight_decay=weight_decay)
            hnn.set_mask(mask=mem_stuck_mask)
            # hnn_losses=train_multilayer_hnn(hnn, optimizer_hnn, stored_data, learn_iters, device)
            hnn_losses=train_multilayer_batch(hnn, optimizer_hnn, stored_data, learn_iters, device)
            Hebb.add_mask(mask=mem_stuck_mask)
            Storkey.add_mask(mask=mem_stuck_mask)
            Pinv.add_mask(mask=mem_stuck_mask)
            Equilibrium.add_mask(mask=mem_stuck_mask)


            stored_patterns.append(stored_data.numpy())
            Hebb_weights.append(Hebb.weights.detach().cpu().numpy())
            Storkey_weights.append(Storkey.weights.detach().cpu().numpy())
            Pinv_weights.append(Pinv.weights.detach().cpu().numpy())
            Equilibrium_weights.append(Equilibrium.weights.detach().cpu().numpy())
            HNN_weights.append(hnn.linear.weight.detach().cpu().numpy())

        # 保存数据
        with open(model_path + f'/AssociateMemory_capacity_{params['dataset']}_{num_dimen}_stuck={stuck_at_fault_rate}_simulation_0.pkl', 'wb') as f:
            pickle.dump({
                'num_pattern_list': num_pattern_list,
                'stored_patterns': stored_patterns,
                'Hebb_weights': Hebb_weights,
                'Storkey_weights': Storkey_weights,
                'Pinv_weights': Pinv_weights,
                'HNN_weights': HNN_weights,
                'Equilibrium_weights': Equilibrium_weights
            }, f)

    
    elif train_eval=='eval':

        with open(model_path + f'/AssociateMemory_capacity_{params['dataset']}_{num_dimen}_stuck={stuck_at_fault_rate}_simulation_0.pkl', 'rb') as f:
        # with open(model_path + f'AssociateMemory_capacity_random_256_simulation.pkl', 'rb') as f:
            data = pickle.load(f)

        # 访问加载的数据
        num_pattern_list = data['num_pattern_list']
        stored_patterns = data['stored_patterns']
        Hebb_weights = data['Hebb_weights']
        Storkey_weights = data['Storkey_weights']
        Pinv_weights = data['Pinv_weights']
        HNN_weights = data['HNN_weights']
        Equilibrium_weights = data['Equilibrium_weights']

        Hebb_similarity_list=[]
        Storkey_similarity_list=[]
        Pinv_similarity_list=[]
        HNN_similarity_list=[]
        Equilibrium_similarity_list=[]
        for indice in tqdm(range(len(num_pattern_list))):
            num_pattern=num_pattern_list[indice]
            stored_data=stored_patterns[indice]
            Hebb_weight=Hebb_weights[indice]
            Storkey_weight=Storkey_weights[indice]
            Pinv_weight=Pinv_weights[indice]
            Equilibrium_weight=Equilibrium_weights[indice]
            HNN_weight=HNN_weights[indice]

            stored_data=torch.from_numpy(stored_data).float()
            with torch.no_grad():
                hnn=SingleLayerHNN_lambda(input_size=input_size)
                Hebb = HopfieldNet(num_neurons=input_size)
                Storkey = HopfieldNet(num_neurons=input_size)
                Pinv=HopfieldNet(num_neurons=input_size)
                Equilibrium=HopfieldNet(num_neurons=input_size)

                Hebb.weights=torch.from_numpy(Hebb_weight).float()
                Storkey.weights=torch.from_numpy(Storkey_weight).float()
                Pinv.weights=torch.from_numpy(Pinv_weight).float()
                Equilibrium.weights=torch.from_numpy(Equilibrium_weight).float()
                hnn.linear.weight=torch.nn.Parameter(torch.from_numpy(HNN_weight).float())
                
                for _ in range(num_simulations):
                    #corrupt the data
                    params['recall_iters']=100
                    test_data=np.array([get_corrupted_input(d,corruption_rate) for d in stored_data])
                    test_data = torch.from_numpy(test_data).float()



                    hnn.add_variation(var=mem_var)
                    Hebb.add_weight_variation(variation=mem_var)
                    Storkey.add_weight_variation(variation=mem_var)
                    Pinv.add_weight_variation(variation=mem_var)
                    Equilibrium.add_weight_variation(variation=mem_var)


                    # recall data
                    HNN_recall = singlelayer_recall(hnn, test_data, params)
                    Hebb_recall=Hebb.retrieve_pattern(test_data, sync=True,chnn=False)
                    Storkey_recall=Storkey.retrieve_pattern(test_data, sync=True,chnn=False)
                    Pinv_recall=Pinv.retrieve_pattern(test_data, sync=True,chnn=False)
                    Equilibrium_recall=Equilibrium.retrieve_pattern(test_data, sync=True,chnn=False)

                    # calculate similarity
                    Hebb_similarity=torch.mean(calculate_sim(stored_data,Hebb_recall,distance_type=distance_type))
                    Storkey_similarity=torch.mean(calculate_sim(stored_data,Storkey_recall,distance_type=distance_type))
                    Pinv_similarity=torch.mean(calculate_sim(stored_data,Pinv_recall,distance_type=distance_type))
                    HNN_similarity=torch.mean(calculate_sim(stored_data,HNN_recall,distance_type=distance_type))
                    Equilibrium_similarity=torch.mean(calculate_sim(stored_data,Equilibrium_recall,distance_type=distance_type))
                

                    Hebb_similarity_list.append(to_np(Hebb_similarity))
                    Storkey_similarity_list.append(to_np(Storkey_similarity))
                    Pinv_similarity_list.append(to_np(Pinv_similarity))
                    HNN_similarity_list.append(to_np(HNN_similarity))
                    Equilibrium_similarity_list.append(to_np(Equilibrium_similarity))
        
        input_similarities=torch.mean(calculate_sim(stored_data,test_data,distance_type=distance_type))
        Hebb_similarity_list=np.array(Hebb_similarity_list).reshape(-1,num_simulations).T
        Storkey_similarity_list=np.array(Storkey_similarity_list).reshape(-1,num_simulations).T
        Pinv_similarity_list=np.array(Pinv_similarity_list).reshape(-1,num_simulations).T
        HNN_similarity_list=np.array(HNN_similarity_list).reshape(-1,num_simulations).T
        Equilibrium_similarity_list=np.array(Equilibrium_similarity_list).reshape(-1,num_simulations).T

        np.savez(num_path +f'/AssociateMemory_capacity_{params['dataset']}_{num_dimen}_stuck={stuck_at_fault_rate}_var={mem_var}_corr={corruption_rate}_simulation_0.npz',num_pattern_list=num_pattern_list,Hebb_similarity_list=Hebb_similarity_list,Storkey_similarity_list=Storkey_similarity_list,Pinv_similarity_list=Pinv_similarity_list,HNN_similarity_list=HNN_similarity_list,Equilibrium_similarity_list=Equilibrium_similarity_list,input_similarities=input_similarities)


if __name__ == "__main__":
    args = parser.parse_args()

    params = {
        "dimensions": args.dimension,
        "interval": args.interval,
        "train_eval": args.train_eval,
        "variation": args.variation,
        "stuck": args.stuck,
        "corruption": args.corruption,
        "seed": args.seed,
        "dataset": args.dataset,
        "binary": args.binary,
        'max_pattern': args.max_pattern,
        'min_pattern': args.min_pattern
    }
    main(params)

