import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    def __init__(self, prompt_size=3, image_size=(448, 224)):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size_0 = image_size[0] - pad_size*2
        self.base_size_1 = image_size[1] - pad_size*2
        #self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size[1]]))
        #self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size[1]]))
        #self.pad_left = nn.Parameter(torch.randn([1, 3, image_size[0] - pad_size*2, pad_size]))
        #self.pad_right = nn.Parameter(torch.randn([1, 3, image_size[0] - pad_size*2, pad_size]))

        self.pad_up = nn.Parameter(torch.zeros([1, 3, pad_size, image_size[1]]))
        self.pad_down = nn.Parameter(torch.zeros([1, 3, pad_size, image_size[1]]))
        self.pad_left = nn.Parameter(torch.zeros([1, 3, image_size[0] - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.zeros([1, 3, image_size[0] - pad_size*2, pad_size]))
        self.max_value = (1.0 - 0.4) / 0.2

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size_0, self.base_size_1).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        prompt = torch.clamp(prompt, max=self.max_value)
        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)