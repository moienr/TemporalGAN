import matplotlib.pyplot as plt
import torch

def plot_s1s2_tensors(tensors, names, n_rows, n_cols):
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(tensors):
                break
            tensor = tensors[idx].to(torch.float32)
            if torch.min(tensor) < 0:
                tensor = (tensor + 1)/2
            name = names[idx] if names is not None else None
            if tensor.ndim > 2 and tensor.shape[0] > 1:
                axs[i][j].imshow(tensor[[3,2,1],:,:].permute(1,2,0).cpu().numpy())
                axs[i][j].set_title(name)
            else:
                axs[i][j].imshow(tensor[0].cpu().numpy())
                axs[i][j].set_title(name)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.show()
    
    
def save_s1s2_tensors_plot(tensors, names, n_rows, n_cols, filename, fig_size):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(tensors):
                break
            tensor = tensors[idx].to(torch.float32)
            if torch.min(tensor) < 0:
                tensor = (tensor + 1)/2
            name = names[idx] if names is not None else None
            if tensor.ndim > 2 and tensor.shape[0] > 1:
                axs[i][j].imshow(tensor[[3,2,1],:,:].permute(1,2,0).cpu().numpy())
                axs[i][j].set_title(name)
            else:
                axs[i][j].imshow(tensor[0].cpu().numpy())
                axs[i][j].set_title(name)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.savefig(filename)