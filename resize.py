from PIL import Image
import os
import _pickle as pickle


def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main():
    #splits = ['train', 'val']
    #for split in splits:
    folder = './training_data/VG_100K_2'


    #print('Start resizing %s images.'.format(split))
    image_files = os.listdir(folder)
    image_files = sorted(image_files)
    #print(image_files)
    num_images = len(image_files)
    num_train = num_images*0.8

    with open('./resized_training_data/not_list.pkl','rb') as f:
        not_list = pickle.load(f)


    for i, image_file in enumerate(image_files):
        img_num = image_file.split('.')[0]
        if int(img_num) in not_list:
            print(int(img_num))
            continue
        try:
            if i <= num_train:
                resized_folder = './resized_training_data/training_data'
                if not os.path.exists(resized_folder):
                    os.makedirs(resized_folder)
                with open(os.path.join(folder, image_file), 'r+b') as f:
                    with Image.open(f) as image:
                        image = resize_image(image)
                        image.save(os.path.join(resized_folder, image_file), image.format)
            else:
                resized_folder = './resized_training_data/test_data'
                if not os.path.exists(resized_folder):
                    os.makedirs(resized_folder)
                with open(os.path.join(folder, image_file), 'r+b') as f:
                    with Image.open(f) as image:
                        image = resize_image(image)
                        image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print('Resized images: %d/%d' %(i, num_images))
        except OSError as e:
            print(e)


if __name__ == '__main__':
    main()