import torch
import os
from dataset.data_loaders import *
from dataset.utils.plot_utils import *
from config import *
from eval_metrics.ssim import WSSIM
from eval_metrics.psnr import wpsnr
from eval_metrics.loss_function import reverse_map
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_some_examples(gen, val_dataset ,epoch, folder, cm_input, img_indx = 1):
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
        
        weighted_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm.unsqueeze(0).to(torch.float32))
        normal_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)))
        reverse_weighted_ssim = wssim((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), reverse_map(s1cm.unsqueeze(0).to(torch.float32)))
         
        weighted_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), s1cm.unsqueeze(0).to(torch.float32))
        normal_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)))
        reverse_weighted_psnr = wpsnr((s1t2.unsqueeze(0).to(torch.float32), s1t2_generated.to(torch.float32)), reverse_map(s1cm.unsqueeze(0).to(torch.float32)))
        
        
        title = f"epoch:{epoch} -- image:{img_indx} \n\
            cwssim: {weighted_ssim:.3f} | ssim: {normal_ssim:.3f} | rcwssim: {reverse_weighted_ssim:.3f} \n\
            cwpsnr: {weighted_psnr:.3f} | psnr: {normal_psnr:.3f} | rcwpsnr: {reverse_weighted_psnr:.3f}"
        
        input_list = [s2t1,s1t1,s2t2,s1t2,torch.abs(cm),s1cm,rcm,s1t2_generated[0]] if not cm_input else [s2t1,s1t1[0].unsqueeze(0),s2t2,s1t2,torch.abs(cm),s1cm,rcm,s1t2_generated[0]]
        save_s1s2_tensors_plot(input_list,
                               ["s2t1", "s1t1", "s2t2", "s1t2", "s2_change map", "s1_change map","reversed change map" ,"generated s1t2"],
                               n_rows=4,
                               n_cols=2,
                               filename=f"{folder}//epoc_{epoch}_img{img_indx}.png",
                               fig_size=(8,10),
                               title=title)
    gen.train()


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