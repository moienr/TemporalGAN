import torch
import os
from dataset.data_loaders import *
from plot_utils import *
from config import *
from eval_metrics.ssim import WSSIM
from eval_metrics.psnr import wpsnr
from eval_metrics.loss_function import reverse_map
from changedetection.utils import get_binary_change_map
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cv2
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_some_examples(gen, val_dataset ,epoch, folder, cm_input, img_indx = 1, just_show = False, fig_size = (8,12), save_raw_images_folder = None):
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm  = val_dataset[img_indx]
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(DEVICE),s1t2.to(DEVICE),s2t1.to(DEVICE),s1t1.to(DEVICE),cm.to(DEVICE),rcm.to(DEVICE),s1cm.to(DEVICE)
    if cm_input:
        s2t2 = torch.cat((s2t2, cm), dim=0)
        s1t1 = torch.cat((s1t1, rcm), dim=0)
    
    if os.path.exists(folder) == False:
        os.mkdir(f"{folder}/")
    wssim = WSSIM(data_range=1.0)  
    gen.eval()
    with torch.no_grad():
        s1t2_generated = gen(s2t2.unsqueeze(0).to(torch.float32), s1t1.unsqueeze(0).to(torch.float32))
        
        s1cm_binary = get_binary_change_map(s1cm.to(torch.float32))  * 1.0

        s1cm_binary_reverse = torch.tensor(1.0) - s1cm_binary
        
        weighted_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm_binary.unsqueeze(0))
        normal_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)))
        reverse_weighted_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm_binary_reverse.unsqueeze(0))
         
        weighted_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm_binary.unsqueeze(0))
        normal_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)))
        reverse_weighted_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm_binary_reverse.unsqueeze(0))
        
        
        title = f"epoch:{epoch} -- image:{img_indx} \n\
            cwssim: {weighted_ssim:.3f} | ssim: {normal_ssim:.3f} | rcwssim: {reverse_weighted_ssim:.3f} \n\
            cwpsnr: {weighted_psnr:.3f} | psnr: {normal_psnr:.3f} | rcwpsnr: {reverse_weighted_psnr:.3f}"
        
        # s1t2 generated difference with real s1t2
        s1t2_generated_diff_w_s1t2 = torch.abs(s1t2_generated[0] - s1t2)
        # s1t2 generated difference with past s1t1
        if not cm_input:
            s1t2_generated_diff_w_s1t1 = torch.abs(s1t2_generated[0] - s1t1)
        else:
            s1t2_generated_diff_w_s1t1 = torch.abs(s1t2_generated[0] - s1t1[0].unsqueeze(0))
        
        s1t2_generated_change_highlited = s1t2_generated[0] * (s1cm_binary + 0.1)
            
        if cm_input:
            input_list = [s2t1, s2t2, torch.abs(cm), s1t1[0].unsqueeze(0),
                          s1t2,s1cm,s1t2_generated[0],s1t2_generated[0], s1cm_binary,
                          s1t2_generated_diff_w_s1t1, s1t2_generated_diff_w_s1t2,
                          s1t2_generated_change_highlited]
        else:
            input_list = [s2t1, s2t2, torch.abs(cm), s1t1,
                          s1t2,s1cm,s1t2_generated[0],s1t2_generated[0], s1cm_binary,
                          s1t2_generated_diff_w_s1t1, s1t2_generated_diff_w_s1t2,
                          s1t2_generated_change_highlited]
            
        save_s1s2_tensors_plot(input_list,
                               ["s2t1","s2t2","s2_change map",
                                "s1t1",  "s1t2","s1_change map",
                                "generated s1t2" ,"generated s1t2",
                                "s1_change map binary", "s1t2gen change from s1t1",
                                "s1t2gen change from s1t2", "s1t2gen cm highlited"],
                               n_rows=4,
                               n_cols=3,
                               filename=f"{folder}//epoc_{epoch}_img{img_indx}.jpg",
                               fig_size=fig_size,
                               title=title,
                               save_raw_images_folder=save_raw_images_folder,
                               img_indx=img_indx,
                               just_show= just_show)
    gen.train()




def plot_lcl_att_maps(gen, val_dataset ,epoch, folder, cm_input,
                  img_indx = 1,alpha_s1 = 0.5, alpha_s2 = 0.5, abs_atts = True, just_show = False, fig_size = (8,12)):
    
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm  = val_dataset[img_indx]
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(DEVICE),s1t2.to(DEVICE),s2t1.to(DEVICE),s1t1.to(DEVICE),cm.to(DEVICE),rcm.to(DEVICE),s1cm.to(DEVICE)
    if cm_input:
        s2t2 = torch.cat((s2t2, cm), dim=0)
        s1t1 = torch.cat((s1t1, rcm), dim=0)
    
    if os.path.exists(folder) == False:
        os.mkdir(f"{folder}/")
    wssim = WSSIM(data_range=1.0)  
    gen.eval()
    with torch.no_grad():
        s1t2_generated = gen(s2t2.unsqueeze(0).to(torch.float32), s1t1.unsqueeze(0).to(torch.float32))
        s1t2_generated = s1t2_generated.squeeze(0)
        try:
            s1_att_map = gen.glam4_s1.local_spatial_att.local_att_map
            s1_att_map = F.interpolate(s1_att_map, size=(256, 256), mode='bicubic', align_corners=True)
            s1_att_map = np.abs(s1_att_map[0, 0, :, :].detach().cpu().numpy()) if abs_atts else s1_att_map[0, 0, :, :].detach().cpu().numpy()
            
            s1_att_map = convert2uint8(normalize(s1_att_map))
            s2_att_map = gen.glam4_s2.local_spatial_att.local_att_map
            s2_att_map = F.interpolate(s2_att_map, size=(256, 256), mode='bicubic', align_corners=True)
            s2_att_map = np.abs(s2_att_map[0, 0, :, :].detach().cpu().numpy()) if abs_atts else s2_att_map[0, 0, :, :].detach().cpu().numpy()
            s2_att_map = convert2uint8(normalize(s2_att_map))
            
            
        except:
            raise Exception("No attention maps to plot")
        #print(s1_att_map.shape, np.min(s1_att_map), np.max(s1_att_map))

        s1_colormap = cv2.applyColorMap(s1_att_map, cv2.COLORMAP_JET)
        s2_colormap = cv2.applyColorMap(s2_att_map, cv2.COLORMAP_JET)
        # Color maps are in BGR format. But matplotlib uses RGB format.
        s1_colormap = cv2.cvtColor(s1_colormap, cv2.COLOR_BGR2RGB)
        s2_colormap = cv2.cvtColor(s2_colormap, cv2.COLOR_BGR2RGB)
        

        #print(s2_colormap.shape)
        
        s1t2_np = s1t2.permute(1,2,0).cpu().numpy()
        s1t2_np = convert2uint8(normalize(s1t2_np))
        s1t2_np = s1t2_np.repeat(3, axis=2)# repeat 3 times to combine with colormap
        
        # s1t2_generated_np = s1t2_generated.permute(1,2,0).cpu().numpy()
        # s1t2_generated_np = convert2uint8(normalize(s1t2_generated_np))
        # s1t2_generated_np = s1t2_generated_np.repeat(3, axis=2) # repeat 3 times to combine with colormap
        
        s2t2_np = s2t2.permute(1,2,0)[:,:,[2,1,0]].cpu().numpy()
        s2t2_np = convert2uint8(normalize(s2t2_np))
        
        
        #print(s2t2_np.shape, s1t2_np.shape)
        # # Stack RGB image and colormap
        # s2_stacked = np.stack((rgb_image, colormap), axis=-1)

        
        # Overlay attention map on RGB image
        s1t2_np_colormaped = cv2.addWeighted(s1t2_np, 1 - alpha_s1, s1_colormap, alpha_s1, 0)
        #s1t2_generated_np_colormaped = cv2.addWeighted(s1t2_generated_np, 1 - alpha, s1_colormap, alpha, 0)
        s2t2_np_colormaped = cv2.addWeighted(s2t2_np, 1 - alpha_s2, s2_colormap, alpha_s2, 0)
        

        #print(f"result shape {s2t2_np_colormaped.shape}, {s1t2_np_colormaped.shape}")
        
        s1_name = f"img{img_indx}_S1_lcl_ATT_ABS" if abs_atts else f"img{img_indx}_S1_lcl_ATT_REL"
        s2_name = f"img{img_indx}_S2_lcl_ATT_ABS" if abs_atts else f"img{img_indx}_S2_lcl_ATT_REL"
        
        gen.train() # set back to train mode
        plot_np_images([s2t2_np_colormaped, s1t2_np_colormaped],
                    [s2_name, s1_name],
                    folder=folder,
                    subplot_shape= (1,2), plot_name= "ATTENTION MAPS",
                    fig_size=fig_size, save_path=None)
        
        
def plot_glob_att_maps(gen, val_dataset ,epoch, folder, cm_input,
                  img_indx = 1,channel=0,alpha_s1 = 0.5, alpha_s2 = 0.5, abs_atts = False, just_show = False, fig_size = (8,12)):
    
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm  = val_dataset[img_indx]
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(DEVICE),s1t2.to(DEVICE),s2t1.to(DEVICE),s1t1.to(DEVICE),cm.to(DEVICE),rcm.to(DEVICE),s1cm.to(DEVICE)
    if cm_input:
        s2t2 = torch.cat((s2t2, cm), dim=0)
        s1t1 = torch.cat((s1t1, rcm), dim=0)
    
    if os.path.exists(folder) == False:
        os.mkdir(f"{folder}/")
    wssim = WSSIM(data_range=1.0)  
    gen.eval()
    with torch.no_grad():
        s1t2_generated = gen(s2t2.unsqueeze(0).to(torch.float32), s1t1.unsqueeze(0).to(torch.float32))
        s1t2_generated = s1t2_generated.squeeze(0)
        try:
            s1_att_map = gen.glam4_s1.global_spatial_att.att
            s1_att_map = F.interpolate(s1_att_map, size=(256, 256), mode='bicubic', align_corners=True)
            s1_att_map = np.abs(s1_att_map[0, channel, :, :].detach().cpu().numpy()) if abs_atts else s1_att_map[0, channel, :, :].detach().cpu().numpy()
            
            s1_att_map = convert2uint8(normalize(s1_att_map))
            s2_att_map = gen.glam4_s2.global_spatial_att.att
            s2_att_map = F.interpolate(s2_att_map, size=(256, 256), mode='bicubic', align_corners=True)
            s2_att_map = np.abs(s2_att_map[0, channel, :, :].detach().cpu().numpy()) if abs_atts else s2_att_map[0, channel, :, :].detach().cpu().numpy()
            s2_att_map = convert2uint8(normalize(s2_att_map))
            
            
        except:
            raise Exception("No attention maps to plot")
        #print(s1_att_map.shape, np.min(s1_att_map), np.max(s1_att_map))

        s1_colormap = cv2.applyColorMap(s1_att_map, cv2.COLORMAP_JET)
        s2_colormap = cv2.applyColorMap(s2_att_map, cv2.COLORMAP_JET)
        # Color maps are in BGR format. But matplotlib uses RGB format.
        s1_colormap = cv2.cvtColor(s1_colormap, cv2.COLOR_BGR2RGB)
        s2_colormap = cv2.cvtColor(s2_colormap, cv2.COLOR_BGR2RGB)
        

        #print(s2_colormap.shape)
        
        s1t2_np = s1t2.permute(1,2,0).cpu().numpy()
        s1t2_np = convert2uint8(normalize(s1t2_np))
        s1t2_np = s1t2_np.repeat(3, axis=2)# repeat 3 times to combine with colormap
        
        # s1t2_generated_np = s1t2_generated.permute(1,2,0).cpu().numpy()
        # s1t2_generated_np = convert2uint8(normalize(s1t2_generated_np))
        # s1t2_generated_np = s1t2_generated_np.repeat(3, axis=2) # repeat 3 times to combine with colormap
        
        s2t2_np = s2t2.permute(1,2,0)[:,:,[2,1,0]].cpu().numpy()
        s2t2_np = convert2uint8(normalize(s2t2_np))
        
        
        #print(s2t2_np.shape, s1t2_np.shape)
        # # Stack RGB image and colormap
        # s2_stacked = np.stack((rgb_image, colormap), axis=-1)

        
        # Overlay attention map on RGB image
        s1t2_np_colormaped = cv2.addWeighted(s1t2_np, 1 - alpha_s1, s1_colormap, alpha_s1, 0)
        #s1t2_generated_np_colormaped = cv2.addWeighted(s1t2_generated_np, 1 - alpha, s1_colormap, alpha, 0)
        s2t2_np_colormaped = cv2.addWeighted(s2t2_np, 1 - alpha_s2, s2_colormap, alpha_s2, 0)
        

        #print(f"result shape {s2t2_np_colormaped.shape}, {s1t2_np_colormaped.shape}")
        
        s1_name = f"img{img_indx}_chnl{channel}_S1_glob_ATT_ABS" if abs_atts else f"img{img_indx}_chnl{channel}_S1_glob_ATT_REL"
        s2_name = f"img{img_indx}_chnl{channel}_S2_glob_ATT_ABS" if abs_atts else f"img{img_indx}_chnl{channel}_S2_glob_ATT_REL"
        
        gen.train() # set back to train mode
        plot_np_images([s2t2_np_colormaped, s1t2_np_colormaped],
                    [s2_name, s1_name],
                    folder=folder,
                    subplot_shape= (1,2), plot_name= "ATTENTION MAPS",
                    fig_size=fig_size, save_path=None)

        

def save_checkpoint(epoc,model, optimizer, filename="my_checkpoint.pth.tar", folder = "checkpoints"):
    if os.path.exists(folder) == False:
        os.mkdir(f"{folder}/")
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filename =f"epoc{epoc}_" + filename  
    torch.save(checkpoint, folder + "/" + filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


from tqdm import tqdm
torch.backends.cudnn.benchmark = True

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, weighted_loss, cm_input,grad_clip=True):
    loop = tqdm(loader, leave=True)
    
    D_real_list = []   # Initialize empty list for D_real
    D_fake_list = []   # Initialize empty list for D_fake
    L1_list = []       # Initialize empty list for L1
    G_loss_list = []   # Initialize empty list for G_loss
    
    for idx, (s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm) in enumerate(loop):
        s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(DEVICE),s1t2.to(DEVICE),s2t1.to(DEVICE),s1t1.to(DEVICE),cm.to(DEVICE),rcm.to(DEVICE),s1cm.to(DEVICE)
        if cm_input:
            s2t2 = torch.cat((s2t2, cm), dim=1)
            s1t1 = torch.cat((s1t1, rcm), dim=1)
        # Train Discriminator
        with torch.cuda.amp.autocast():
            s1t2_fake = gen(s2t2, s1t1)
            D_real = disc(s2t2, s1t1, s1t2)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(s2t2, s1t1, s1t2_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        if grad_clip:
            torch.nn.utils.clip_grad_value_(disc.parameters(), clip_value=0.5)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(s2t2, s1t1, s1t2_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            if weighted_loss:
                L1 = l1_loss(s1t2_fake, s1t2, s1cm) * L1_LAMBDA
            else:
                L1 = l1_loss(s1t2_fake, s1t2) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        if grad_clip:
            torch.nn.utils.clip_grad_value_(gen.parameters(), clip_value=0.5)
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        D_real_list.append(torch.sigmoid(D_real).mean().item())
        D_fake_list.append(torch.sigmoid(D_fake).mean().item())
        L1_list.append(L1.item())
        G_loss_list.append(G_loss.item())
        
        if idx % 100 == 0 or idx == len(loader)-1:
            loop.set_postfix(
                D_real_mean = sum(D_real_list) / len(D_real_list),
                D_fake_mean = sum(D_fake_list) / len(D_fake_list),
                L1_mean = sum(L1_list) / len(L1_list),
                G_loss_mean = sum(G_loss_list) / len(G_loss_list),
            )


def get_tensor_ones_ratio(tensor: torch.Tensor):

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    ones_count = torch.sum(tensor)
    total_pixels = tensor.shape[-1] * tensor.shape[-2] * tensor.shape[-3]
    ratio = ones_count / total_pixels
    
    return ratio



def batch_eval_loop(loss_fn, labels: torch.Tensor, preds: torch.Tensor, weight_maps: torch.Tensor = None, hard_test = False):
    loss_list = []
    for l, p, w in zip(labels, preds, weight_maps):
        l, p, w = l.unsqueeze(0), p.unsqueeze(0), w.unsqueeze(0)
        loss = loss_fn((l, p), w)
        if hard_test:
            changed_ratio = get_tensor_ones_ratio(w)
        else:
            changed_ratio = torch.tensor(1.0)
        loss_list.append((loss, changed_ratio.item()))

    return loss_list


def weighted_mean(lst):
    numerator = 0
    denominator = 0
    for tup in lst:
        numerator += tup[0] * tup[1]
        denominator += tup[1]
    return numerator / denominator


def eval_fn(gen, loader, ssim, psnr, hard_test = False, loader_part = "all", in_change_map = False):
    """_summary_

    Args:
        gen (torch.nn.Module): Generator model
        loader (troch.utils.data.DataLoader): Data loader
        ssim (torch.nn.Module): ssim loss function
        psnr (torch.nn.Module): psnr loss function
        hard_test (bool, optional): Wether to use hard test or not. Defaults to False.
        loader_part (str, optional): Wether to use `all` of the loader or `first_half`  or `second_half` . Defaults to "all".

    Returns:
        dict: Dictionary of the evaluation metrics
    """
    gen.eval()
    loop = tqdm(loader, leave=True)
    weighted_ssim_list = []
    normal_ssim_list = []
    weighted_psnr_list = []
    normal_psnr_list = []
    reverse_weighted_ssim_list = []
    reverse_weighted_psnr_list = []
    
    for idx, (s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm) in enumerate(loop):
        if loader_part == "second" and idx < len(loader) // 2: # skip first half if second half is requested
            continue 
        s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(DEVICE),s1t2.to(DEVICE),s2t1.to(DEVICE),s1t1.to(DEVICE),cm.to(DEVICE),rcm.to(DEVICE),s1cm.to(DEVICE)
        s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm = s2t2.to(torch.float32),s1t2.to(torch.float32),s2t1.to(torch.float32),s1t1.to(torch.float32),cm.to(torch.float32),rcm.to(torch.float32),s1cm.to(torch.float32)
        if in_change_map:
            s2t2 = torch.cat((s2t2, cm), dim=1)
            s1t1 = torch.cat((s1t1, rcm), dim=1)
        preds = gen(s2t2, s1t1)
        
        s1cm_reversed = reverse_map(s1cm)
        
        # Rescaling from -1-1 to 0-1
        s1t2 = (s1t2 + 1) / 2
        preds = (preds + 1) / 2
        
        weighted_ssim = batch_eval_loop(ssim, s1t2, preds, s1cm, hard_test = hard_test)
        weighted_ssim_list = weighted_ssim_list + weighted_ssim # weighted_ssim is a list of tuples
        reverse_weighted_ssim = batch_eval_loop(ssim, s1t2, preds, s1cm_reversed, hard_test = hard_test)
        reverse_weighted_ssim_list = reverse_weighted_ssim_list + reverse_weighted_ssim # weighted_ssim is a list of tuples
        normal_ssim = ssim((s1t2, preds))
        normal_ssim_list.append(normal_ssim)
        
        weighted_psnr = batch_eval_loop(psnr, s1t2, preds, s1cm, hard_test = hard_test)
        weighted_psnr_list = weighted_psnr_list + weighted_psnr 
        reverse_weighted_psnr = batch_eval_loop(psnr, s1t2, preds, s1cm_reversed, hard_test = hard_test)
        reverse_weighted_psnr_list = reverse_weighted_psnr_list + reverse_weighted_psnr
        normal_psnr = psnr((s1t2, preds))
        normal_psnr_list.append(normal_psnr)


        if idx % 10 == 0:
            loop.set_postfix(
                wssim_mean = weighted_mean(weighted_ssim_list),
                ssim_mean = torch.tensor(normal_ssim_list).mean().item(),
                rwssim_mean = weighted_mean(reverse_weighted_ssim_list),
                wpsnr_mean = weighted_mean(weighted_psnr_list),
                psnr_mean = torch.tensor(normal_psnr_list).mean().item(),
                rwpsnr_mean = weighted_mean(reverse_weighted_psnr_list)
                
            )
        if loader_part == "first" and idx >= len(loader) // 2: # stop after first half if first half is requested
            break

    
    eval_dict = {"SSIM":{}, "PSNR":{}}
    
    eval_dict["SSIM"]["cwssim_mean"] = weighted_mean(weighted_ssim_list)
    eval_dict["SSIM"]["ssim_mean"] = torch.tensor(normal_ssim_list).mean().item()
    eval_dict["SSIM"]["rcwssim_mean"] = weighted_mean(reverse_weighted_ssim_list)
    eval_dict["PSNR"]["cwpsnr_mean"] = weighted_mean(weighted_psnr_list)
    eval_dict["PSNR"]["psnr_mean"] = torch.tensor(normal_psnr_list).mean().item()
    eval_dict["PSNR"]["rcwpsnr_mean"] = weighted_mean(reverse_weighted_psnr_list)
    
    return eval_dict

def separate_lists(dict_list):
    """
    Extracts PSNR and SSIM metrics from a list of dictionaries and returns them as separate lists.

    Args:
        dict_list (list[dict]): A list of dictionaries containing PSNR and SSIM metrics.

    Returns:
        tuple: A tuple containing six lists of floats each, representing the extracted PSNR and SSIM metrics.
    """
    psnr_list = []
    cw_psnr_list = []
    rcwpsnr_list = []
    ssim_list = []
    cwssim_list = []
    rcwssim_list = []

    for d in dict_list:
        psnr_list.append(d['PSNR']['psnr_mean'])
        cw_psnr_list.append(d['PSNR']['cwpsnr_mean'])
        rcwpsnr_list.append(d['PSNR']['rcwpsnr_mean'])
        ssim_list.append(d['SSIM']['ssim_mean'])
        cwssim_list.append(d['SSIM']['cwssim_mean'])
        rcwssim_list.append(d['SSIM']['rcwssim_mean'])
        
    return psnr_list, cw_psnr_list, rcwpsnr_list, ssim_list, cwssim_list, rcwssim_list



def plot_metrics(dict_list, save_path=None, colors = ['#33a9a5', '#3598db', '#f27085']):
    # First, create the six lists and the epochs list
    psnr_list, cw_psnr_list, rcwpsnr_list, ssim_list, cwssim_list, rcwssim_list = separate_lists(dict_list)
    epochs = list(range(len(psnr_list)))

    # Next, create a figure and four axis objects
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 5))
    

    # Plot the PSNRs on the first axis
    ax[0, 0].plot(epochs, psnr_list, label='PSNR', color=colors[0])
    ax[0, 0].plot(epochs, cw_psnr_list, label='CW-PSNR', color=colors[1])
    ax[0, 0].plot(epochs, rcwpsnr_list, label='RCW-PSNR', color = colors[2])
    ax[0, 0].legend(framealpha=1, frameon=True)
    ax[0, 0].set_ylabel('PSNR (dB)')
    ax[0, 0].set_title('PSNR Metrics')
    # Plot the SSIMs on the second axis
    ax[1, 0].plot(epochs, ssim_list, label='SSIM', color=colors[0])
    ax[1, 0].plot(epochs, cwssim_list, label='CW-SSIM', color=colors[1])
    ax[1, 0].plot(epochs, rcwssim_list, label='RCW-SSIM', color = colors[2])
    ax[1, 0].legend(framealpha=1, frameon=True)
    ax[1, 0].set_ylabel('SSIM')
    ax[1, 0].set_xlabel('Epochs')
    ax[1, 0].set_title('SSIM Metrics')
    # Add the new subplots for CWPSNR and CWSSIM
    ax[0, 1].plot(epochs, cw_psnr_list, label='CW-PSNR', color=colors[1])
    ax[0, 1].legend(framealpha=1, frameon=True)
    ax[0, 1].set_ylabel('CWPSNR (dB)')
    ax[0, 1].set_title('CWPSNR')

    ax[1, 1].plot(epochs, cwssim_list, label='CW-SSIM', color=colors[1])
    ax[1, 1].legend(framealpha=1, frameon=True)
    ax[1, 1].set_ylabel('CWSSIM')
    ax[1, 1].set_xlabel('Epochs')
    ax[1, 1].set_title('CWSSIM')

    # Set the x-axis tick locator to MaxNLocator with integer=True
    ax[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add horizontal grid lines to the plot
    ax[0, 0].yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[1, 0].yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[0, 1].yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[1, 1].yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Adjust the layout of the figure
    fig.tight_layout()


    # Save the plot to a file if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()














if __name__ == "__main__":
    
    transform = transforms.Compose([S2S1Normalize(),myToTensor()])

    print("Reading only S1 2021 train data...")
    s1s2_dataset = Sen12Dataset(s1_t1_dir="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s1_imgs\\train",
                                s2_t1_dir="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s2_imgs\\train",
                                s1_t2_dir="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s1_imgs\\train",
                                s2_t2_dir="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s2_imgs\\train",
                                transform=transform,
                                two_way=False)
    print("len(s1s2_dataset): ",len(s1s2_dataset))
    print("s1s2_dataset[0][0]shape: ",s1s2_dataset[0][1].shape)
    s2t2,s1t2,s2t1,s1t1,cm,rcm,s1cm =s1s2_dataset[4]
    save_s1s2_tensors_plot([s2t2,s1t2,s2t1,s1t1,torch.abs(cm),s1cm],
                           ["s2t2", "s1t2", "s2t1", "s1t1", "change map", "s1_change map"],
                           3,
                           2,
                           filename="test.png",
                           fig_size=(8,10))