import torch

def load_weights_extra_large(weights_path):
    weights = torch.load(weights_path)
    avg2d = torch.nn.AvgPool2d(2, stride=2)
    avg2d1 = torch.nn.AvgPool2d((1, 2), stride=(1, 2))
    avg1d = torch.nn.AvgPool1d(2, stride=2)
    for dict in weights:
        if dict == 'module_list.13.batch_norm_13.weight':
            break
        w = weights[dict]
        if len(w.shape) == 4:
            if w.shape[1] == 3:
                w = w.permute(1, 2, 3, 0)
                w = avg2d1(w).permute(3, 0, 1, 2)
            elif w.shape[1] == 1024:
                w = w.permute(0, 2, 3, 1)
                w = avg2d1(w).permute(0, 3, 1, 2)
            else:
                w = w.permute(2, 3, 0, 1)
                w = avg2d(w).permute(2, 3, 0, 1)
        elif len(w.shape) != 0:
            w = avg1d(w.unsqueeze(0).unsqueeze(0)).squeeze()
        weights[dict] = w
    head_l = ['module_list.18.conv_18.weight', 'module_list.18.batch_norm_18.weight', 'module_list.18.batch_norm_18.bias',\
              'module_list.18.batch_norm_18.running_mean', 'module_list.18.batch_norm_18.running_var']
    for dict in head_l:
        w = weights[dict]
        if len(w.shape) == 4:
            w = w.permute(1, 2, 3, 0)
            w = avg2d1(w).permute(3, 0, 1, 2)
        else:
            w = avg1d(w.unsqueeze(0).unsqueeze(0)).squeeze()
        weights[dict] = w
    w = weights['module_list.21.conv_21.weight']
    w = w.permute(0, 2, 3, 1)
    w = avg2d1(w).permute(0, 3, 1, 2)
    weights['module_list.21.conv_21.weight'] = w
    return weights