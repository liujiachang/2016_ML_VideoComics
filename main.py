from train import *
from test import *

to_test = False # 设置为true进行测试
to_train = True # 设置为true进行训练

if __name__ == '__main__':
    if to_train:
        train()
    if to_test:
        test()

