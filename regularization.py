import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Batch_Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)
        bs = img.size(0)
        c = img.size(1)

        mask = np.ones((bs, h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h, size=bs)
            x = np.random.randint(w, size=bs)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            for i in range(bs):
                mask[i, y1[i]: y2[i], x1[i]: x2[i]] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).cuda()
        img = img * mask

        return img


# normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# randomCrop = RandomCrop(32, padding=4)


# class Rotation(object):
#
#     def __init__(self, train_mode=True):
#         self.train_mode = train_mode
#
#     def __call__(self, img):
def Rotation(img):
        # if self.train_mode and np.random.uniform() < 0.5:
        #     x_orig = np.copy(x_orig)[:, ::-1]
        # else:
        x_orig = torch.clone(img)
        #
        # if self.train_mode:
        #     x_orig = Image.fromarray(x_orig)
        #     x_orig = randomCrop(32, padding=4)(x_orig)
        #     x_orig = np.asarray(x_orig)

        x_rot0 = torch.clone(x_orig)
        x_rot90 = torch.rot90(x_orig.clone(), k=1, dims=(2, 3)).clone()
        x_rot180 = torch.rot90(x_orig.clone(), k=2, dims=(2, 3)).clone()
        x_rot270 = torch.rot90(x_orig.clone(), k=3, dims=(2, 3)).clone()
        # print(x_rot0.shape, x_rot90.shape, x_rot180.shape)
        # possible_translations = list(itertools.product([0, 8, -8], [0, 8, -8]))
        # num_possible_translations = len(possible_translations)
        # tx, ty = possible_translations[random.randint(0, num_possible_translations - 1)]
        # tx_target = {0: 0, 8: 1, -8: 2}[tx]
        # ty_target = {0: 0, 8: 1, -8: 2}[ty]
        # x_tf_trans = cv2f.affine(np.asarray(x_orig).copy(), 0, (tx, ty), 1, 0, interpolation=cv2.INTER_CUBIC,
        #                          mode=cv2.BORDER_REFLECT_101)
        inputs_ = torch.cat((x_rot0, x_rot90, x_rot180, x_rot270), 0)
        # inputs_ = torch.FloatTensor(inputs_)
        # return (trnF.to_tensor(x_tf_0),
        #         trnF.to_tensor(x_tf_90),
        #         trnF.to_tensor(x_tf_180),
        #         trnF.to_tensor(x_tf_270)), torch.tensor(classifier_target)
        return inputs_
    # torch.tensor(tx_target), \
    # torch.tensor(ty_target), \
    # normalize(trnF.to_tensor(x_tf_trans)), \


def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
