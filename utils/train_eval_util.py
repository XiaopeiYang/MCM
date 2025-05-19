
import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011
from dataloaders.iNaturelist import *



def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
                    "ViT-B/32":"openai/clip-vit-base-patch32",
                    "ViT-L/14":"openai/clip-vit-large-patch14"}
    args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    model =  CLIPModel.from_pretrained(args.ckpt)
    if args.model == 'CLIP-Linear':
        model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  # normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "car196":
        train_loader = torch.utils.data.DataLoader(StanfordCars(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "food101":
        train_loader = torch.utils.data.DataLoader(Food101(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "pet37":
        train_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="trainval", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "bird200":
        train_loader = torch.utils.data.DataLoader(Cub2011(root, train = True, transform=preprocess),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader

def get_all_labels(root_dir):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "car196":
        val_loader = torch.utils.data.DataLoader(StanfordCars(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "food101":
        val_loader = torch.utils.data.DataLoader(Food101(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "pet37":
        val_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="test", download=True, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "bird200":
        val_loader = torch.utils.data.DataLoader(Cub2011(root, train = False, transform=preprocess),
                    batch_size=args.batch_size, shuffle=False, **kwargs)
     
    elif args.in_dataset == "lichen_in":
        in_dataset, _ = random_split_in_out(
        root_dir=os.path.join(root, 'lichen'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # Adjust the number of "in" classes as needed
        seed=42,
        dataset_key="Lichen"
    )
        val_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "manzanita_in":
        # val_loader = torch.utils.data.DataLoader(Manzanita_in(os.path.join(root, 'manzanita'), transform=preprocess),
        # batch_size=args.batch_size, shuffle=False, **kwargs)
        in_dataset, _ = random_split_in_out(
        root_dir=os.path.join(root, 'manzanita'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # Adjust the number of "in" classes as needed
        seed=42,
        dataset_key="Manzanita"
    )
        val_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "bulrush_in":
        in_dataset, _ = random_split_in_out(
        root_dir=os.path.join(root, 'bulrush'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # Adjust the number of "in" classes as needed
        seed=42,
        dataset_key="Bulrush"
    )
        val_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "wild_rye_in":
        in_dataset, _ = random_split_in_out(
        root_dir=os.path.join(root, 'wild_rye'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # Adjust the number of "in" classes as needed
        seed=42,
        dataset_key="Wild Rye"
    )
        val_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "wrasse_in":
        in_dataset, _ = random_split_in_out(
        root_dir=os.path.join(root, 'wrasse'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # Adjust the number of "in" classes as needed
        seed=42,
        dataset_key="Wrasse"
    )
        val_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "lichen":
         # Define the dataset root directory
        root_dir_lichen = os.path.join(root, 'lichen')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_lichen)
    
        # Create a dataset that loads all classes with random label mapping
        full_dataset = iNaturalistDataset(
        root_dir=root_dir_lichen,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Lichen"
    )
    
    # Create the DataLoader using the full dataset
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    elif args.in_dataset == "manzanita":
         # Define the dataset root directory
        root_dir_manzanita = os.path.join(root, 'manzanita')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_manzanita)
    
        # Create a dataset that loads all classes with random label mapping
        full_dataset = iNaturalistDataset(
        root_dir=root_dir_manzanita,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Manzanita"
    )
    
    # Create the DataLoader using the full dataset
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    elif args.in_dataset == "bulrush":
        # val_loader = torch.utils.data.DataLoader(Bulrush(os.path.join(root, 'bulrush'), transform=preprocess),
        # batch_size=args.batch_size, shuffle=False, **kwargs)
        # Define the dataset root directory
        root_dir_bulrush = os.path.join(root, 'bulrush')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_bulrush)
    
        # Create a dataset that loads all classes with random label mapping
        full_dataset = iNaturalistDataset(
        root_dir=root_dir_bulrush,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Bulrush"
    )
    
    # Create the DataLoader using the full dataset
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    elif args.in_dataset == "wild_rye":
        root_dir_wild_rye = os.path.join(root, 'wild_rye')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_wild_rye)
    
        # Create a dataset that loads all classes with random label mapping
        full_dataset = iNaturalistDataset(
        root_dir=root_dir_wild_rye,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Wild Rye"
    )
    
    # Create the DataLoader using the full dataset
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    elif args.in_dataset == "wrasse":
        # Define the dataset root directory
        root_dir_wrasse = os.path.join(root, 'wrasse')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_wrasse)
    
        # Create a dataset that loads all classes with random label mapping
        full_dataset = iNaturalistDataset(
        root_dir=root_dir_wrasse,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Wrasse"
    )
    
    # Create the DataLoader using the full dataset
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'sun397/SUN397'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= '/export/compvis-nfs/group/datasets/Places365',transform=preprocess)  #os.path.join(root, 'Places')
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
        
    elif out_dataset == 'LC2500_lung':
        testsetout = torchvision.datasets.ImageFolder(os.path.join(args.root_dir, 'lung_colon_image_set', 'lung_image_sets'), transform=preprocess)
    elif out_dataset == 'LC2500_colon':
        testsetout = torchvision.datasets.ImageFolder(os.path.join(args.root_dir, 'lung_colon_image_set', 'colon_image_sets'), transform=preprocess)
    elif out_dataset == "lichen_out":
        #testsetout = Lichen_out(os.path.join(args.root_dir, 'lichen'), transform=preprocess)
        _, testsetout = random_split_in_out(
        root_dir=os.path.join(args.root_dir, 'lichen'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # The first 2 classes will be "in", the rest become "out"
        seed=42,
        dataset_key="Lichen"
    )
    elif out_dataset == "manzanita_out":
        #testsetout = Manzanita_out(os.path.join(args.root_dir, 'manzanita'), transform=preprocess)
        _, testsetout = random_split_in_out(
        root_dir=os.path.join(args.root_dir, 'manzanita'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # The first 2 classes will be "in", the rest become "out"
        seed=42,
        dataset_key="Manzanita"
    )
    elif out_dataset == "bulrush_out":
        # testsetout = Bulrush_out(os.path.join(args.root_dir, 'bulrush'), transform=preprocess)
        _, testsetout = random_split_in_out(
        root_dir=os.path.join(args.root_dir, 'bulrush'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # The first 2 classes will be "in", the rest become "out"
        seed=42,
        dataset_key="Bulrush"
    )
    elif out_dataset == "wild_rye_out":
        #testsetout = Wild_rye_out(os.path.join(args.root_dir, 'wild_rye'), transform=preprocess)
        _, testsetout = random_split_in_out(
        root_dir=os.path.join(args.root_dir, 'wild_rye'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # The first 2 classes will be "in", the rest become "out"
        seed=42,
        dataset_key="Wild Rye"
    )
    elif out_dataset == "wrasse_out":
        #testsetout = Wrasse_out(os.path.join(args.root_dir, 'wrasse'), transform=preprocess)
        _, testsetout = random_split_in_out(
        root_dir=os.path.join(args.root_dir, 'wrasse'),
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        num_in=args.n_cls,  # The first 2 classes will be "in", the rest become "out"
        seed=42,
        dataset_key="Wrasse"
    )
    elif out_dataset == "lichen":
        #testsetout = Lichen(os.path.join(args.root_dir, 'lichen'), transform=preprocess)
        root_dir_lichen = os.path.join(root, 'lichen')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_lichen)
    
        # Create a dataset that loads all classes with random label mapping
        testsetout = iNaturalistDataset(
        root_dir=root_dir_lichen,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Lichen"
    )
    elif out_dataset == "manzanita":
        #testsetout = Manzanita(os.path.join(args.root_dir, 'manzanita'), transform=preprocess)
        root_dir_manzanita = os.path.join(root, 'manzanita')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_manzanita)
    
        # Create a dataset that loads all classes with random label mapping
        testsetout = iNaturalistDataset(
        root_dir=root_dir_manzanita,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Manzanita"
    )
    elif out_dataset == "bulrush":
        root_dir_bulrush = os.path.join(root, 'bulrush')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_bulrush)
    
        # Create a dataset that loads all classes with random label mapping
        testsetout = iNaturalistDataset(
        root_dir=root_dir_bulrush,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Bulrush"
    )
    elif out_dataset == "wild_rye":
        #testsetout = Wild_rye(os.path.join(args.root_dir, 'wild_rye'), transform=preprocess)
        root_dir_wild_rye = os.path.join(root, 'wild_rye')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_wild_rye)
    
        # Create a dataset that loads all classes with random label mapping
        testsetout = iNaturalistDataset(
        root_dir=root_dir_wild_rye,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Wild Rye"
    )
    elif out_dataset == "wrasse":
        #testsetout = Wrasse(os.path.join(args.root_dir, 'wrasse'), transform=preprocess)
        root_dir_wrasse = os.path.join(root, 'wrasse')
        # Get all class folder names
        all_labels = get_all_labels(root_dir_wrasse)
    
        # Create a dataset that loads all classes with random label mapping
        testsetout = iNaturalistDataset(
        root_dir=root_dir_wrasse,
        label_names=all_labels,
        transform=preprocess,
        do_rescale=False,
        max_samples_per_class=100,
        dataset_key="Wrasse"
    )
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    
    
    
        
    
    return testloaderOut

