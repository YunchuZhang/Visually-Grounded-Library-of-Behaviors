import numpy as np
import torch

class DoublePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        np.random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.images = []

    def fetch(self):
        return self.embeds, self.images

    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full

    def update(self, embeds, images):
        # embeds is B x ... x C
        # images is B x ... x 3

        for embed, image in zip(embeds, images):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)
            # add to the back
            self.embeds.append(embed)
            self.images.append(image)
        return self.embeds, self.images


class DoublePoolMoc():
    def __init__(self, pool_size, requires_images=False):
        self.pool_size = pool_size
        self.requires_images = requires_images
        np.random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            if self.requires_images:
                self.images = []

    def fetch(self, num=None):
        emb_array = torch.stack(self.embeds)
        if self.requires_images:
            img_array = torch.stack(self.images)
        if num is not None:
            # if there are not that many elements just return however many there are
            if len(self.embeds) < num:
                if self.requires_images:
                    return emb_array, img_array
                else:
                    return emb_array
            else:
                idxs = np.random.randint(len(self.embeds), size=num)
                if self.requires_images:
                    return emb_array[idxs], img_array[idxs]
                else:
                    return emb_array[idxs]
        else:
            if self.requires_images:
                return emb_array, img_array
            else:
                return emb_array  # all the stacked embeddings are returned


    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full

    def __len__(self):
        return len(self.embeds)

    def update(self, embeds, images=None):
        # embeds is B x ... x C
        # images is B x ... x 3
        if images is not None:
            for embed, image in zip(embeds, images):
                if self.num < self.pool_size:
                    # the pool is not full, so let's add this in
                    self.num = self.num + 1
                else:
                    # the pool is full
                    # pop from the front
                    self.embeds.pop(0)
                    self.images.pop(0)
                # add to the back
                self.embeds.append(embed)
                self.images.append(image)
        else:
            for embed in embeds:
                if self.num < self.pool_size:
                    # the pool is not full, so let's add this in
                    self.num = self.num + 1
                else:
                    # the pool is full
                    # pop from the front
                    self.embeds.pop(0)
                # add to the back
                self.embeds.append(embed)

if __name__ == '__main__':
    d = DoublePoolMoc(pool_size=10000)
    for i in range(100):
        x = np.random.randn(32, 100)
        d.update(x)

    # so I will check each of the operations and see if it is fine
    # operation update is done
    print(len(d.embeds))
    assert len(d.embeds) == 3200
    for e in d.embeds:
        assert len(e) == 100

    # fetch operation
    sampled_embeds = d.fetch(num=100)
    print(len(sampled_embeds))
    assert len(sampled_embeds) == 100

    # check if you sample more than you have what does it say
    num = 3300
    sampled_embeds = d.fetch(num)
    print(len(sampled_embeds))
    assert len(sampled_embeds) == 3200

    x = np.random.randn(6800, 100)
    d.update(x)
    print(d.is_full())

    # next I want to add more stuff and it should use the pop thing
    # generate some random data
    y = np.random.randn(32, 100)
    d.update(y)
    # now the last 32 elements of the embed should be same as that of y
    z = np.asarray(d.embeds[-32:])
    print(np.equal(y, z).all())
    assert np.equal(y, z).all(), "I just appended you to the list"
