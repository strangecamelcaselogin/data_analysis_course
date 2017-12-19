from PIL import Image

from common import MNIST


if __name__ == '__main__':
    mnist_train = MNIST('../../data/mnist').load_test()
    images, labels = mnist_train.data, mnist_train.target
    for img, label in zip(images[:10], labels[:10]):
        pimg = Image.fromarray(img)
        pimg.show(title=label)
