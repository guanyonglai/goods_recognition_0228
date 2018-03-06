lkdsaf;lsadjf;lsdajf;klsdjf;klsdj为射门不能添加gogoods_recogintion文件夹下面的全部文件啊:
2018.03.06 1107希望今天可以添加成功
公司resnet网址：https://pan.baidu.com/s/1zn0lEe0SBJxQTTMmod3hjw/info/refs

caffe生成prototxt测试
# -*- coding: UTF-8 -*-
import caffe                                                      #导入caffe包

caffe_root = "/home/Jack-Cui/caffe-master/my-caffe-project/"      #my-caffe-project目录
train_lmdb = caffe_root + "img_train.lmdb"                        #train.lmdb文件的位置
mean_file = caffe_root + "mean.binaryproto"                     #均值文件的位置

#网络规范
net = caffe.NetSpec()
#第一层Data层
net.data, net.label = caffe.layers.Data(source = train_lmdb, backend = caffe.params.Data.LMDB, batch_size = 64, ntop=2,
                                        transform_param = dict(crop_size = 40,mean_file = mean_file,mirror = True))
#第二层Convolution层
net.conv1 = caffe.layers.Convolution(net.data, num_output=20, kernel_size=5,weight_filler={"type": "xavier"},
                                bias_filler={"type": "constant"})

print (str(net.to_proto()))
