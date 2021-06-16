from __future__ import print_function


from . import model_dict
dataset_names = [
    'mini-imagenet',
    'tiered-imagenet',
    'fc100',
    'cifarfs']

def create_model(name, n_cls, dataset='mini-imagenet', drop_rate=0.1):
    assert dataset in dataset_names
    """create model by name"""
    if dataset == 'mini-imagenet' or dataset == 'tiered-imagenet':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=drop_rate, dropblock_size=5, num_classes=n_cls)
        elif name.startswith('wrn'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'cifarfs' or dataset == 'fc100':
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=drop_rate, dropblock_size=2, num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
