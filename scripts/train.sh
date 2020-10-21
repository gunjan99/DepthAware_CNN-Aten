python train.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--flip --scale --crop --colorjitter \
--depthconv \
--list '/media/jarvis/Windows/Users/gunja/Documents/Acads/3D_Object/NYU/Dataset/NYU_training.lst' \
--vallist '/media/jarvis/Windows/Users/gunja/Documents/Acads/3D_Object/NYU/Dataset/NYU_val.lst' \
--continue_train
