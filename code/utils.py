import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn import metrics
import params
import torch.nn as nn
import torch.nn.init as init
from scipy.stats import pearsonr

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''

    if isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=0.0, std=1.0)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0.0, std=1.0)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=0.0, std=1.0)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data, mean=0.0, std=1.00)

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif layer_name.find("Linear") != -1:
        layer.weight.data.normal_(0.0, 0.10)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


def evaluate_regression_results(y_test, y_pred):

    if y_test.ndim == 1:
        y_test = y_test[:, np.newaxis]

    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    APE = np.abs(y_test - y_pred) / y_test
    MAPE = np.mean(APE)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    MAE = np.mean(np.abs(y_test - y_pred))


    R2 = metrics.r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_test[:,0], y_pred[:,0])
    r2 = corr ** 2
    #print('Root Mean Squared Error:', RMSE)
    #print('R squared:', R2)
    #print('Mean Absolute Percetage Error: ', MAPE)

    # scatter_plot(y_test, y_pred, RMSE, R2, save_name, save_scatter_image_path)

    return RMSE, R2, MAPE, r2, MAE