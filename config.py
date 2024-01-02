import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L1_LAMBDA = 100

s1_t1_dir_train="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s1_imgs\\test"
s2_t1_dir_train="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s2_imgs\\test"
s1_t2_dir_train="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s1_imgs\\test"
s2_t2_dir_train="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s2_imgs\\test"
s1_t1_dir_test="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s1_imgs\\test"
s2_t1_dir_test="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2021\\s2_imgs\\test"
s1_t2_dir_test="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s1_imgs\\test"
s2_t2_dir_test="E:\\s1s2\\s1s2_patched_light\\s1s2_patched_light\\2019\\s2_imgs\\test"

hard_test_csv_path = "D:\\python\\TemporalGAN\\changedetection\\changed_pairs_extra_light.csv"