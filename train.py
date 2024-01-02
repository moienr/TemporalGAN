import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
from skimage import io
import os
from torchvision import datasets, transforms
import matplotlib
import os
import gc
import random
from datetime import date, datetime
import json
import pprint
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from temporalgan.gen_v1_1 import Generator as GeneratorV1_1
from temporalgan.gen_v1_2 import Generator as GeneratorV1_2
from temporalgan.gen_v1_3 import Generator as GeneratorV1_3
from temporalgan.gen_v1_4 import Generator as GeneratorV1_4
from temporalgan.gen_v1_5 import Generator as GeneratorV1_5
from temporalgan.gen_v1_6 import Generator as GeneratorV1_6
from temporalgan.gen_v1_7 import Generator as GeneratorV1_7
from temporalgan.disc_v2 import Discriminator as DiscriminatorV2
from temporalgan.disc_v1 import Discriminator as DiscriminatorV1
from eval_metrics.loss_function import WeightedL1Loss, reverse_map

from changedetection.utils import get_column_values
from dataset.data_loaders import *
from config import *
from eval_metrics import ssim
from eval_metrics.psnr import wpsnr
wssim = ssim.WSSIM(data_range=1.0)

from dataset.utils.utils import TextColors as TC
from dataset.utils.plot_utils import plot_s1s2_tensors, save_s1s2_tensors_plot

from train_utils import *
import argparse


parser = argparse.ArgumentParser(description='TemporalGAN Training')
parser.add_argument('--two_way_dataset', '-tw', action='store_true', help='Use two-way dataset')
parser.add_argument('--input_change_map', '-ic', action='store_true', help='Use input change map')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size')
parser.add_argument('--num_workers', '-nw', type=int, default=2, help='Number of workers')
parser.add_argument('--image_size', '-is', type=int, default=256, help='Image size')
parser.add_argument('--weighted_loss', '-wl', action='store_true', help='Use weighted loss')
parser.add_argument('--changed_l1_weight', '-clw', type=int, default=5, help='Changed L1 weight')
parser.add_argument('--num_epochs', '-ne', type=int, default=10, help='Number of epochs')
parser.add_argument('--load_model', '-lm', action='store_true', help='Load model')
parser.add_argument('--save_model', '-sm', action='store_true', help='Save model')
parser.add_argument('--save_model_every_epoch', '-sme', type=int, default=10, help='Save model every epoch')
parser.add_argument('--run_test_every_epoch', '-rte', type=int, default=1, help='Run test every epoch')
parser.add_argument('--save_example_plots', '-sep', action='store_true', help='Save example plots')
parser.add_argument('--examples_to_plot', '-etp', nargs='+', type=int, default=[1, 2], help='Examples to plot after training is done - if you want to plot both FTS and BTS, add the size of the dataset to the index of desired image.')
parser.add_argument('--random_seed', '-rs', type=int, default=None, help='Random seed - optional for reproducibility')
parser.add_argument('--gen_version', '-gv', type=str, default='1.3', help='Generator Version')

args = parser.parse_args()

TWO_WAY_DATASET = args.two_way_dataset
INPUT_CHANGE_MAP = args.input_change_map
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
IMAGE_SIZE = args.image_size
WEIGHTED_LOSS = args.weighted_loss
CHANGED_L1_WEIGHT = args.changed_l1_weight
NUM_EPOCHS = args.num_epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
SAVE_MODEL_EVERY_EPOCH = args.save_model_every_epoch
RUN_TEST_EVERY_EPOCH = args.run_test_every_epoch
SAVE_EXAMPLE_PLOTS = args.save_example_plots
EXAMPLES_TO_PLOT = args.examples_to_plot
RANDOM_SEED = args.random_seed
GEN_VERSION = args.gen_version

S2_INCHANNELS = 12 if INPUT_CHANGE_MAP else 6
S1_INCHANNELS = 7 if INPUT_CHANGE_MAP else 1


if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

hard_test_names = get_column_values(hard_test_csv_path, "name")

# Format the date and time
now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")
file_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
# print("Current Date and Time:", start_string)



transform = transforms.Compose([S2S1Normalize(),myToTensor()])
hard_test_dataset = Sen12DatasetHardTest(s1_t1_dir=s1_t1_dir_test,
                            s2_t1_dir=s2_t1_dir_test,
                            s1_t2_dir=s1_t2_dir_test,
                            s2_t2_dir=s2_t2_dir_test,
                            hard_test_names=hard_test_names,
                            transform=transform,
                            two_way=TWO_WAY_DATASET)
hard_test_loader = DataLoader(
        hard_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )



def train():
    disc = DiscriminatorV1(s2_in_channels=S2_INCHANNELS, s1_in_channels=S1_INCHANNELS).to(DEVICE)
#     gen = GeneratorV2(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64,pam_downsample=2).to(DEVICE)
    if GEN_VERSION == "1.1":
        gen = GeneratorV1_1(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64).to(DEVICE)
    elif GEN_VERSION == "1.2":
        gen = GeneratorV1_2(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64).to(DEVICE)
    elif GEN_VERSION == "1.3":
        gen = GeneratorV1_3(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64).to(DEVICE)
    elif GEN_VERSION == "1.5":
        gen = GeneratorV1_5(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64).to(DEVICE)
    elif GEN_VERSION == "1.6":
        gen = GeneratorV1_6(s2_in_channels=S2_INCHANNELS, s1_in_channels= S1_INCHANNELS, features=64).to(DEVICE)
    else:
        raise Exception("Wrong generator version!")
    
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    if WEIGHTED_LOSS:
        L1_LOSS = WeightedL1Loss(change_weight=CHANGED_L1_WEIGHT)
    else:
        L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )

    transform = transforms.Compose([S2S1Normalize(),myToTensor(dtype=torch.float32)])


    train_dataset = Sen12Dataset(s1_t1_dir=s1_t1_dir_train,
                                 s2_t1_dir=s2_t1_dir_train,
                                 s1_t2_dir=s1_t2_dir_train,
                                 s2_t2_dir=s2_t2_dir_train,
                                 transform=transform,
                                 two_way=TWO_WAY_DATASET)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
#     val_dataset = MapDataset(root_dir=VAL_DIR)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    validation_results = []
    
    for epoch in range(1, NUM_EPOCHS+1):
        print("\n\n" , end="")
        print(TC.BOLD_BAKGROUNDs.PURPLE, f"Epoch: {epoch}", TC.ENDC)
        print(TC.OKCYAN, "   Training:", TC.ENDC)
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen,
            L1_LOSS, BCE, g_scaler, d_scaler, weighted_loss= WEIGHTED_LOSS,
            cm_input=INPUT_CHANGE_MAP, grad_clip=False)
        print(TC.HIGH_INTENSITYs.YELLOW, "   Validation:", TC.ENDC)
        hard_eval_validation_all = eval_fn(gen, hard_test_loader, wssim, wpsnr, hard_test = True, loader_part="all", in_change_map=INPUT_CHANGE_MAP)
        validation_results.append(hard_eval_validation_all)
        
        

        if SAVE_MODEL and epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            save_checkpoint(epoch,gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(epoch,disc, opt_disc, filename=CHECKPOINT_DISC)

            if SAVE_EXAMPLE_PLOTS:
                for img_i in EXAMPLES_TO_PLOT:
                    save_some_examples(gen, train_dataset, epoch, folder="train_evaluation_plots",cm_input=INPUT_CHANGE_MAP, img_indx=img_i)

            
        gc.collect()
        torch.cuda.empty_cache()
            
    return gen, validation_results


def main():
    #matplotlib.use('Agg') # This refrains matplot lib form showing the plotted resualts below the cell
    gen_model, validation_results = train()

    plot_metrics(validation_results, save_path=f"results/RUN_{file_name}.png")
    
    transform = transforms.Compose([S2S1Normalize(),myToTensor()])
    test_dataset = Sen12Dataset(s1_t1_dir=s1_t1_dir_test,
                                s2_t1_dir=s2_t1_dir_test,
                                s1_t2_dir=s1_t2_dir_test,
                                s2_t2_dir=s2_t2_dir_test,
                                transform=transform,
                                two_way=TWO_WAY_DATASET,
                                binary_s1cm=False
                                )
    test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    print("\n\n")
    print(TC.BOLD_BAKGROUNDs.ORANGE, "Evaluating all dataset using soft changemaps.", TC.ENDC)
    print("Whole test dataset:")
    whole_test_eval_results_all = eval_fn(gen_model, test_loader, wssim, wpsnr, hard_test = True, in_change_map=INPUT_CHANGE_MAP)
    print("First half of test dataset:")
    whole_test_eval_results_first_half = eval_fn(gen_model, test_loader, wssim, wpsnr, hard_test = True,loader_part='first', in_change_map=INPUT_CHANGE_MAP)
    print("Second half of test dataset:")
    whole_test_eval_results_second_half = eval_fn(gen_model, test_loader, wssim, wpsnr, hard_test = True,loader_part='second', in_change_map=INPUT_CHANGE_MAP)

    print(TC.BOLD_BAKGROUNDs.ORANGE, "Evaluating selected hard test dataset using hard changemaps.", TC.ENDC)
    print("All hard test dataset:")
    hard_eval_results_all = eval_fn(gen_model, hard_test_loader, wssim, wpsnr, hard_test = True, loader_part="all", in_change_map=INPUT_CHANGE_MAP)
    print("First half of hard test dataset:")
    hard_eval_results_first_half = eval_fn(gen_model, hard_test_loader, wssim, wpsnr, hard_test = True, loader_part="first", in_change_map=INPUT_CHANGE_MAP)
    print("Second half of hard test dataset:")
    hard_eval_results_second_half = eval_fn(gen_model, hard_test_loader, wssim, wpsnr, hard_test = True, loader_part="second", in_change_map=INPUT_CHANGE_MAP)

    # Format the date and time
    now = datetime.now()
    finish_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Finish Date and Time:", finish_string)
    print("File Name:", file_name)
    
    
    log_json = {}
    log_json["Time"] = {"Start": start_string, "Finish": finish_string}
    log_json["GEN_VERSION"] = GEN_VERSION
    log_json["TWO_WAY_DATASET"] = TWO_WAY_DATASET
    log_json["INPUT_CHANGE_MAP"] = INPUT_CHANGE_MAP
    log_json["S2_INCHANNELS"] = S2_INCHANNELS
    log_json["S1_INCHANNELS"] = S1_INCHANNELS
    log_json["LEARNING_RATE"] = LEARNING_RATE
    log_json["BATCH_SIZE"] = BATCH_SIZE
    log_json["NUM_WORKERS"] = NUM_WORKERS
    log_json["IMAGE_SIZE"] = IMAGE_SIZE
    log_json["WEIGHTED_LOSS"] = WEIGHTED_LOSS
    log_json["L1_LAMBDA"] = L1_LAMBDA
    log_json["CHANGED_L1_WEIGHT"] = CHANGED_L1_WEIGHT
    log_json["NUM_EPOCHS"] = NUM_EPOCHS
    log_json["LOAD_MODEL"] = LOAD_MODEL
    log_json["SAVE_MODEL"] = SAVE_MODEL
    log_json["SAVE_MODEL_EVERY_EPOCH"] = SAVE_MODEL_EVERY_EPOCH
    log_json["SAVE_EXAMPLE_PLOTS"] = SAVE_EXAMPLE_PLOTS
    log_json["EXAMPLES_TO_PLOT"] = EXAMPLES_TO_PLOT
    log_json["CHECKPOINT_DISC"] = CHECKPOINT_DISC
    log_json["CHECKPOINT_GEN"] = CHECKPOINT_GEN
    log_json["RANDOM_SEED"] = RANDOM_SEED
    log_json["HardEval"] = {"Hard All": hard_eval_results_all,
                            "Hard First Half": hard_eval_results_first_half,
                            "Hard Second Half": hard_eval_results_second_half}

    log_json["FullEval"] = {"Full Test Dataset": whole_test_eval_results_all,
                            "First Half Test Dataset": whole_test_eval_results_first_half,
                            "Second Half Test Dataset": whole_test_eval_results_second_half}

    psnr_list, cw_psnr_list, rcwpsnr_list, ssim_list, cwssim_list, rcwssim_list = separate_lists(validation_results)
    log_json["Validation Lists"] = {"PSNR": psnr_list, "CWPSNR": cw_psnr_list, "RCWPSNR": rcwpsnr_list, "SSIM": ssim_list, "CWSSIM": cwssim_list, "RCWSSIM": rcwssim_list}

    with open(f"results/RUN_{file_name}.json", "w") as fp:
        json.dump(log_json, fp, indent=4)
        
    # Redeclare the test dataset with binary_s1cm = False for plotting
    transform = transforms.Compose([S2S1Normalize(),myToTensor()])
    test_dataset = Sen12Dataset(s1_t1_dir=s1_t1_dir_test,
                                s2_t1_dir=s2_t1_dir_test,
                                s1_t2_dir=s1_t2_dir_test,
                                s2_t2_dir=s2_t2_dir_test,
                                transform=transform,
                                two_way=TWO_WAY_DATASET,
                                binary_s1cm=False
                                )
    if EXAMPLES_TO_PLOT is not None:
        for indx in EXAMPLES_TO_PLOT:
            save_some_examples(gen_model, test_dataset, NUM_EPOCHS,
                    folder="test_evaluation_plots",cm_input=INPUT_CHANGE_MAP,
                    img_indx=indx, just_show = True,save_raw_images_folder="raw_imgs/", fig_size=(20,20))




if __name__ == "__main__":
    print("Running on:", DEVICE)
    main()
    print("Done!")