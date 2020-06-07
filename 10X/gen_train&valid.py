import os

save_path = 'data/custom/'
# label_path = 'data/custom/labels/'
# img_path = 'data/custom/images/'
label_path = 'data/custom/labels/'
img_path = 'data/custom/images/'

for root, dirs, labels in os.walk(label_path):
    break

i = 0
for label in labels:
    if i >= 10:
        i = 0
        with open(save_path + 'valid.txt', 'a') as file:
            file.write(img_path + label[:-4] + '.jpg\n')
    else:
        i += 1
        with open(save_path + 'train.txt', 'a') as file:
            file.write(img_path + label[:-4] + '.jpg\n')