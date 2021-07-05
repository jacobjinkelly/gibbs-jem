"""
Code to generate toy attribute dataset
"""
import torch
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF


class MNISTWithAttributes(torch.utils.data.Dataset):
    def __init__(self, root, train=True, img_size=64, transform=None):
        self.mnist = datasets.MNIST(root, train=train, transform=None, download=True)

        self.mnist_digs = [0, 1, 2, 3]
        # shitty way of doing logical_or in case old torch
        digs_ind = torch.zeros_like(self.mnist.targets)
        for mnist_dig in self.mnist_digs:
            digs_ind += (self.mnist.targets == mnist_dig).long()
        digs_ind = digs_ind == 1

        self.mnist.data = self.mnist.data[digs_ind]
        self.mnist.targets = self.mnist.targets[digs_ind]

        self.img_size = img_size
        # self.sizes = {'small': .2,
        #               'medium': .3,
        #               'large': .4,
        #               'huge': .5,
        #               }
        self.rotations = [0, 90, 180, 270]
        self.transform = transform
        self.quadrants = {
            'ul': (0., 0.),
            'ur': (0., .5),
            'll': (.5, 0.),
            'lr': (.5, .5)
        }
        self.bg_colors = {

            # 'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'orange': (255, 153, 51),
            'pink': (255, 51, 153),
            'purple': (127, 0, 255),
        }

        self.fg_colors = {
            'yellow': (255, 255, 102),
            'grey': (128, 128, 128),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
        }

        self.n_att = len(self.mnist_digs) + \
                     len(self.rotations) + \
                     len(self.quadrants) + \
                     len(self.bg_colors) + \
                     len(self.fg_colors)

    @staticmethod
    def _ind_of(d, v):
        """
        Return the index of a value in a dictionary.
        """
        return sorted(list(d.keys())).index(v)

    @staticmethod
    def _color_tensor(color, colors):
        return (torch.tensor(colors[color])[:, None, None]) / 255.

    def _generate_image(self, ind=None, quadrant=None, rotation=None, digit_color=None, bg_color=None):
        """
        Given an index into the dataset, generate an image and its attributes.
        """
        ind = np.random.randint(0, len(self.mnist)) if ind is None else ind
        im, label = self.mnist[ind]

        rotation = int(np.random.choice(self.rotations)) if rotation is None else rotation
        rot_ind = self.rotations.index(rotation)

        quadrant = np.random.choice(list(self.quadrants.keys())) if quadrant is None else quadrant
        quad_ind = MNISTWithAttributes._ind_of(self.quadrants, quadrant)

        digit_color = np.random.choice(list(self.fg_colors.keys())) if digit_color is None else digit_color
        dc_ind = MNISTWithAttributes._ind_of(self.fg_colors, digit_color)

        bg_color = np.random.choice(list(self.bg_colors.keys())) if bg_color is None else bg_color
        bc_ind = MNISTWithAttributes._ind_of(self.bg_colors, bg_color)

        size_px = int(0.5 * self.img_size)
        im_resize = transforms.Resize(size_px)(im)
        im_rotate = transformsF.rotate(im_resize, rotation)
        im_tensor = transforms.ToTensor()(im_rotate)

        color_im = MNISTWithAttributes._color_tensor(digit_color, self.fg_colors) * im_tensor

        out_tensor = torch.zeros((3, self.img_size, self.img_size))
        slack = int(self.img_size / 2) - size_px  # always 0, not varying sizes
        start_x, start_y = self.quadrants[quadrant]
        start_x, start_y = int(start_x * self.img_size), int(start_y * self.img_size)

        placement = [np.random.randint(0, slack + 1), np.random.randint(0, slack + 1)]
        start_x += placement[0]
        start_y += placement[1]

        out_tensor[:, start_x: start_x + color_im.size(1), start_y: start_y + color_im.size(2)] = color_im
        color_bg = MNISTWithAttributes._color_tensor(bg_color, self.bg_colors)
        is_black = ((out_tensor == 0.).float().sum(0) == 3.).float()[None, :, :]

        color_tensor = out_tensor * (1. - is_black) + color_bg * is_black

        att = torch.tensor([label, rot_ind, quad_ind, dc_ind, bc_ind])

        if self.transform is not None:
            color_tensor = self.transform(color_tensor)

        return color_tensor, att

    def __getitem__(self, index):
        return self._generate_image(ind=index)

    def __len__(self):
        return len(self.mnist)


if __name__ == "__main__":
    MNISTWithAttributes("../data", img_size=64)
