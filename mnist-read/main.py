from os.path import join as pathjoin
import gzip
from struct import unpack
from PIL import Image
import numpy as np


def open_idx_images(images_archive):
    with gzip.open(images_archive) as byte_stream:
        _, total_count, width, height = unpack('>IIII', byte_stream.read(4 * 4))

        stream_len = total_count * width * height
        return (np.array(unpack('>{}B'.format(stream_len), byte_stream.read(stream_len)), dtype=np.ubyte)
                  .reshape(total_count, width, height))


def open_idx_labels(labels_archive):
    with gzip.open(labels_archive) as byte_stream:
        _, total_count = unpack('>II', byte_stream.read(2 * 4))

        return unpack('>{}B'.format(total_count), byte_stream.read(total_count))


if __name__ == '__main__':
    mnist_path = r'./mnist/'
    test_imgs = pathjoin(mnist_path, 't10k-images-idx3-ubyte.gz')
    test_labels = pathjoin(mnist_path, 't10k-labels-idx1-ubyte.gz')

    images = open_idx_images(test_imgs)
    labels = open_idx_labels(test_labels)

    for img, label in zip(images[:10], labels[:10]):
        pimg = Image.fromarray(img)
        pimg.show(title=label)
