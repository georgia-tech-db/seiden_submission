#### draw multiple figures with plt #####

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm


def draw_three_sets_of_images(images1, images2, images3):
    count = len(images1)
    columns = 3
    row_limit = min(10, count)
    w_size = 10
    h_size = 4 * row_limit
    fig = plt.figure(figsize=(w_size * 3, h_size))

    for i in range(1, columns * row_limit + 1):
        # img = np.random.randint(10, size=(h,w))
        fig.add_subplot(row_limit, columns, 3*i - 2)
        plt.imshow(images1[i - 1])
        fig.add_subplot(row_limit, columns, 3*i - 1)
        plt.imshow(images2[i - 1])
        fig.add_subplot(row_limit, columns, 3 * i)
        plt.imshow(images3[i - 1])

    plt.show()


def draw_two_sets_of_images(images1, images2, count):
    columns = 2
    row_limit = count

    w_size = 20
    h_size = 4 * row_limit
    fig = plt.figure(figsize=(w_size * 2, h_size))

    for i in range(1, columns * row_limit + 1):
        # img = np.random.randint(10, size=(h,w))
        fig.add_subplot(row_limit, columns, 2*i - 1)
        plt.imshow(images1[i - 1])
        fig.add_subplot(row_limit, columns, 2*i )
        plt.imshow(images2[i - 1])

    plt.show()


def draw_multiple_images_w_boxes(images, boxes):
    columns = 4
    images_num = len(images)
    row_limit = 100
    rows = min(int(images_num / columns), row_limit)
    print(f"Displaying {rows} rows")

    w_size = 20
    h_size = 4 * rows
    fig = plt.figure(figsize=(w_size, h_size))

    for i in range(1, columns*rows +1):
        #img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        new_image = draw_patches(images[i - 1], boxes[i - 1])

        plt.imshow(new_image)
    plt.show()



def draw_multiple_images(images):
    columns = min(5, len(images))
    images_num = len(images)
    row_limit = 100
    rows = min(row_limit, len(images) // columns)
    print(f"Displaying {rows} rows")

    w_size = 10 * columns
    h_size = 10 * rows
    fig = plt.figure(figsize=(w_size, h_size))

    for i in range(1, columns*rows +1):
        #img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        if images[i-1].ndim == 2:
            plt.imshow(images[i - 1], cmap = 'gray')
        else:
            plt.imshow(images[i - 1])
        plt.axis('off')

    plt.show()


def generate_video(images, save_dir, fps, frame_width, frame_height, code = 'XVID', bw = False):
    fourcc = cv2.VideoWriter_fourcc(*code)

    if bw:
        out = cv2.VideoWriter(save_dir, fourcc, fps, (frame_width, frame_height), 0)
    else:
        out = cv2.VideoWriter(save_dir, fourcc, fps, (frame_width, frame_height))

    print(out)
    for frame in tqdm(images):
        out.write(frame)

    out.release()

    return


def generate_annotation(labels, save_dir):
    import os

    dir_name = os.path.dirname(save_dir)
    os.makedirs(dir_name, exist_ok = True)
    if os.path.isdir(dir_name):
        with open(save_dir, "w") as save_file:
            print('we have opened the save directory...')
            for i, label in enumerate(labels):
                label_string = f"{i},{label}\n"
                save_file.write(label_string)

        print(f"Saved annotation to {save_dir}")
    else:
        print(f"{dir_name} does not exist, could not generate a save file")





#### drawing patches

# img: img to draw the patches on
# patches: list of rectangle points to draw
def draw_patches(img, patches, labels = None, format='ml'):
    import cv2
    new_img = np.copy(img)
    if labels is not None:
        ## we assume this is used for 'car', 'bus', 'others', 'van'
        colors = {'car': (0,0,255),'bus': (255, 0, 0),'others': (0, 255, 0),'van': (255, 255, 0)}
    else:
        color = (0,0,255)
    if format == 'cv':
        if patches is not None:
            for i in range(len(patches)):
                if labels:
                    color = colors[labels[i]]
                cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                              (int(patches[i][0] + patches[i][2]), int(patches[i][1] + patches[i][3])), color, 2)

    if format == 'ml':
        if patches is not None:
            for i in range(len(patches)):
                if labels:
                    color = colors[labels[i]]
                cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                              (int(patches[i][2]), int(patches[i][3])), color, 2)

    return new_img


### creating dataLoader object for deprecated
def createTorchDataLoader(images, batch_size = 16, shuffle = False):
    import torch
    dataset = PytorchDataset(transform = PytorchTransform())
    dataset.set_images(images)
    dataset.set_target_images(None)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class PytorchDataset(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """
    def __init__(self, transform=None):

        self.transform = transform

        self.root = ""
        self.image_width = -1
        self.image_height = -1
        self.X_train = None
        self.y_train = None

    def __getitem__(self, index):
        """
        Function that is called when doing enumeration / using dataloader
        :param index:
        :return:
        """
        return self.pull_item(index)

    def set_images(self, images):
        self.X_train = images

    def set_target_images(self, target_images):
        self.y_train = target_images

    def provide_args(self):
        return None

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img = self.X_train[index]
        args = self.provide_args()
        if self.transform is not None:
            ## we expect this code to run only if target_trans
            img = self.transform(img, args)

        if self.y_train is not None:
            target = self.y_train[index]

        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(target)


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        return self.X_train[index] ## note this images is BRG


    def __len__(self):
        return len(self.X_train)


class PytorchTransform:
    """
        BaseTransform will perform the following operations:
        1. Avg -- convert the image to grayscale
        2. Normalize -- arrange all pixel values to be between 0 and 1
        3. Resize -- resize the images to fit the network specifications
    """
    def __init__(self, size):
        self.size = size

    def transform(self, image, args):
        mean, std = args
        ### resize, normalize
        size = self.size
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x /= std
        x = x.astype(np.float32)
        y = np.copy(x)
        y = np.mean(y, axis=2)

        ## we need a transform for the output as well
        ## make the base_transform return 2 different objects

        return x, y

    def __call__(self, image, args):
        return self.transform(image, args)

