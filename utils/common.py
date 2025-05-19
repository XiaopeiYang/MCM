import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import random
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
#eval with description
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp) 
#eval with description
def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
# def make_descriptor_sentence(descriptor):
#     return descriptor.replace('It', 'which').replace('.', ',')
    
def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor 

def load_gpt_descriptions(descriptor_fname, 
                          category_name_inclusion='prepend', 
                          apply_descriptor_modification=True, 
                          before_text='', 
                          between_text=', ', 
                          after_text='', 
                          classes_to_load=None):
    
    gpt_descriptions_unordered = load_json(descriptor_fname)
    unmodify_dict = {}
    
    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered
        
    if category_name_inclusion is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)
            
        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']
            
            word_to_add = wordify(k)
            
            if category_name_inclusion == 'append':
                build_descriptor_string = lambda item: f"{modify_descriptor(item, apply_descriptor_modification)}{between_text}{word_to_add}"
            elif category_name_inclusion == 'prepend':
                build_descriptor_string = lambda item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, apply_descriptor_modification)}{after_text}"
            else:
                build_descriptor_string = lambda item: modify_descriptor(item, apply_descriptor_modification)
            
            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
                
            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]
            
            if i == 0:
                print(f"\nExample description for class {k}: \"{gpt_descriptions[k][0]}\"\n")
                
    return gpt_descriptions, unmodify_dict

def get_test_labels(args, loader = None):
    if args.in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == "ImageNet10":
        if args.eval_with_description:
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='descriptions/ImageNet10_descriptors.json', classes_to_load=None)
            test_labels = gpt_descriptions
        else:
            test_labels = obtain_ImageNet10_classes()
        
    elif args.in_dataset == "ImageNet20":
        if args.eval_with_description:
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='descriptions/ImageNet20_descriptors.json', classes_to_load=None)
            test_labels = gpt_descriptions
        else:
            test_labels = obtain_ImageNet20_classes()
        
    elif args.in_dataset == "ImageNet100":
        if args.eval_with_description:
            print("Error: ImageNet100 does not contain descriptions. Cannot evaluate with descriptions.")
            sys.exit(1)
        else:
            test_labels = obtain_ImageNet100_classes()
    elif args.in_dataset in [ 'car196']:
        if args.eval_with_description:
            print("Error: car196 does not contain descriptions. Cannot evaluate with descriptions.")
            sys.exit(1)
        else:
            test_labels = loader.dataset.class_names_str
    elif args.in_dataset in ['bird200']:
        if args.eval_with_description:
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='/export/home/xyang/hiwi/MCM/descriptions/cub_descriptors.json', classes_to_load=None)
            test_labels = gpt_descriptions
        else:
            test_labels = loader.dataset.class_names_str
    elif args.in_dataset in ['food101']:
        if args.eval_with_description:
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='descriptions/food_descriptors.json', classes_to_load=None)
            test_labels = gpt_descriptions
        else:
            test_labels = loader.dataset.class_names_str
    elif args.in_dataset in ['pet37']:
        if args.eval_with_description:
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='descriptions/pets_descriptors.json', classes_to_load=None)
            test_labels = gpt_descriptions
        else:
            test_labels = loader.dataset.class_names_str
    elif args.in_dataset in ['lichen_in','lichen_out','fungi','manzanita_in','manzanita_out','bulrush_in','bulrush_out','wild_rye_in','wild_rye_out','wrasse_in','wrasse_out','lichen','manzanita','bulrush','wild_rye','wrasse']:
        if args.eval_with_description:
            cls_labels = loader.dataset.class_names_str
            gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='descriptions/inaturalist_descriptors.json', classes_to_load=cls_labels)
            test_labels = gpt_descriptions
        else:
            test_labels = loader.dataset.class_names_str
    #elif args.in_dataset in ['fungi']:
        #gpt_descriptions, unmodify_dict = load_gpt_descriptions(descriptor_fname='/export/home/xyang/hiwi/MCM/descriptions/idx_to_descriptions.json', classes_to_load=None)
        #test_labels = gpt_descriptions
    return test_labels



def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls


def obtain_ImageNet10_classes():

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()


def obtain_ImageNet20_classes():

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "eft",
                  "n02391049": "zebra", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()

def obtain_ImageNet100_classes():
    loc=os.path.join('data', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set

def get_num_cls(args):
    NUM_CLS_DICT = {
        'ImageNet10': 10,
        'ImageNet20': 20,
        'pet37': 37,
        'ImageNet100': 100, 
        'food101': 101, 
        'car196': 196,
        'bird200':200, 
        'ImageNet': 1000,
        'LC2500_lung': 3,
        'LC2500_colon': 2,
        'fungi': 6,
        'lichen_in': 4,
        #'lichen_out': 2,
        'manzanita_in': 4,
        'bulrush_in': 3,
        'wild_rye_in': 4 ,
        'wrasse_in': 4,
        'lichen': 6,
        'manzanita': 5,
        'bulrush': 5,
        'wild_rye': 5, 
        'wrasse': 5
    }
    n_cls = NUM_CLS_DICT[args.in_dataset]
    return n_cls


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # values, indices = input.topk(k, dim=1, largest=True, sorted=True)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def read_file(file_path, root='corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus


def calculate_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

