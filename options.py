import optparse


def TrainOptions():
    parser = optparse.OptionParser()

    parser.add_option('--batch_size', default=1024, help='number of imgs in a batch')
    parser.add_option('--EPOCH', default=60, help='total epochs')
    parser.add_option('--train_data_path', default='../Dataset/MNIST/mnist_train/', help='train_img_path')
    parser.add_option('--model_save_path', default='checkpoint/', help='model_save_path')
    parser.add_option('--model_load_path', default='checkpoint/04.04 20 29/', help='model_load_path')

    options, args = parser.parse_args()

    return options, args

def TestOptions():
    parser = optparse.OptionParser()

    parser.add_option('--batch_size', default=1024, help='number of imgs in a batch')
    parser.add_option('--test_data_path', default='../Dataset/MNIST/mnist_test/', help='test_img_path')
    parser.add_option('--model_load_path', default='checkpoint/04.04 14 04/', help='model_load_path')
    parser.add_option('--model_load_name', default='Epoch10.pth', help='model')

    options, args = parser.parse_args()

    return options, args