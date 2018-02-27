#coding:utf-8
import os
import numpy as np
import cv2
from numpy import linalg as LA
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from detection_and_localization.detector import init_detector,det

import pdb

wd = '/mnt/disk1/gyl/new_retail_0226/goods_recognition/data/'

if __name__ == "__main__":
    image_name = 'P80121-174446.jpg'
    im_path = os.path.join(wd,image_name)

    image = cv2.imread(im_path)

    cfg_file_name = 'yolo-voc-608'
    weights_file_name = 'yolo-voc-608_40000'
    # --init detector
    net, meta = init_detector(model_cfg_name=cfg_file_name, model_weights_name=weights_file_name)
    # --detect and localization
    res = det(im_path,net,meta,conf_thres=0.2) #[cls,conf,x,y,w,h]

    if len(res)==0:
        print('No goods detected!')

    # pdb.set_trace()

    # --recognition
    input_shape = [224, 224, 3]
    model = VGG16(weights = 'imagenet', input_shape = (input_shape[0], input_shape[1], input_shape[2]), pooling = 'max', include_top = False)

    def extract_feat(img, model):
        #pdb.set_trace()
        input_shape = (224, 224, 3)
        #img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        #pdb.set_trace()

        img=cv2.resize(img,(input_shape[0], input_shape[1]),interpolation=cv2.INTER_CUBIC)
        img = keras.preprocessing.image.img_to_array(img)
        #img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])#也许，这就是归一化把
        return norm_feat

    #载入各产品类的名称,即groundtruth图片名
    name_path='/mnt/disk1/gyl/new_retail_0226/goods_name/'
    #number 1
    baihujiao_name = open(name_path+'baihujiao.xml').read()
    baihujiao_name=baihujiao_name.split()
    baihujiao_name_num=len(baihujiao_name)
    #number 2
    baishiguan_name = open(name_path+'baishiguan.xml').read()
    baishiguan_name=baishiguan_name.split()
    baishiguan_name_num=len(baishiguan_name)
    #number 3
    baishiping_name = open(name_path+'baishiping.xml').read()
    baishiping_name=baishiping_name.split()
    baishiping_name_num=len(baishiping_name)
    #number 4
    fenda_name = open(name_path+'fenda.xml').read()
    fenda_name=fenda_name.split()
    fenda_name_num=len(fenda_name)
    #number 5
    guolicheng_name = open(name_path+'guolicheng.xml').read()
    guolicheng_name=guolicheng_name.split()
    guolicheng_name_num=len(guolicheng_name)
    #number 6
    hongshaoniurou_name = open(name_path+'hongshaoniurou.xml').read()
    hongshaoniurou_name=hongshaoniurou_name.split()
    hongshaoniurou_name_num=len(hongshaoniurou_name)
    #number 7
    laotansuancaidai_name = open(name_path+'laotansuancaidai.xml').read()
    laotansuancaidai_name=laotansuancaidai_name.split()
    laotansuancaidai_name_num=len(laotansuancaidai_name)
    #number 8
    laotansuancaihe_name = open(name_path+'laotansuancaihe.xml').read()
    laotansuancaihe_name=laotansuancaihe_name.split()
    laotansuancaihe_name_num=len(laotansuancaihe_name)
    #number 9
    nongfushanquan_name = open(name_path+'nongfushanquan.xml').read()
    nongfushanquan_name=nongfushanquan_name.split()
    nongfushanquan_name_num=len(nongfushanquan_name)
    #number 10
    pijiujin_name = open(name_path+'pijiujin.xml').read()
    pijiujin_name=pijiujin_name.split()
    pijiujin_name_num=len(pijiujin_name)
    #number 11
    pijiuyin_name = open(name_path+'pijiuyin.xml').read()
    pijiuyin_name=pijiuyin_name.split()
    pijiuyin_name_num=len(pijiuyin_name)
    #number 12
    shupiandai_name = open(name_path+'shupiandai.xml').read()
    shupiandai_name=shupiandai_name.split()
    shupiandai_name_num=len(shupiandai_name)
    #number 13
    shupiantong_name = open(name_path+'shupiantong.xml').read()
    shupiantong_name=shupiantong_name.split()
    shupiantong_name_num=len(shupiantong_name)
    #number 14
    tudouya_name = open(name_path+'tudouya.xml').read()
    tudouya_name=tudouya_name.split()
    tudouya_name_num=len(tudouya_name)
    #number 15
    xianggudunji_name = open(name_path+'xianggudunji.xml').read()
    xianggudunji_name=xianggudunji_name.split()
    xianggudunji_name_num=len(xianggudunji_name)
    #number 16
    xuebi_name = open(name_path+'xuebi.xml').read()
    xuebi_name=xuebi_name.split()
    xuebi_name_num=len(xuebi_name)


    #读取数据集的所有图片名
    all_db_name = open('/mnt/disk1/gyl/new_retail_0226/goods16_name.xml').read()
    all_db_name=all_db_name.split()
    # read ku image's feature
    a = np.loadtxt('/mnt/disk1/gyl/new_retail_0226/goods16_data_2018_02_24_1603.xml')
    feats = a.reshape(215,512)
    Topk = 1
    roi_count = 0

    image_size=image.shape
    h=image_size[0]
    w=image_size[1]
    #for控制商品大图image中检测出的所有商品个数，每一个商品自成一个query_image
    for roi_num in range(0,len(res)):
    #读取query图片.这里的query_image用的就是商品检测的基于 x,y,w,h 的roi区域

        #image_size=image.shape
        #h=image_size[0]
        #w=image_size[1]
        #pdb.set_trace()
        image_cx = int(res[roi_num][2]*w)
        image_cy = int(res[roi_num][3]*h)
        image_w = int(res[roi_num][4]*w)
        image_h = int(res[roi_num][5]*h)
        #变换为roi的左上角坐标image_tf
        image_tfx = image_cx - int(image_w/2)
        image_tfy = image_cy - int(image_h/2)
        #越界判断
        if(image_tfx < 0):
            image_tfx = 0
        if(image_tfy < 0):
            image_tfy = 0
        if((image_tfx + image_w)>w):
            break
        if((image_tfy + image_h)>h):
            break
        query_image = image[image_tfy:image_tfy+image_h,image_tfx:image_tfx+image_w]
        #pdb.set_trace()
        queryVec = extract_feat(query_image,model)

        scores = np.dot(queryVec, feats.T)
        scores = np.array(scores)
        rank_ID = np.argsort(-scores)  #从大到小逆序排序  rank_ID[0]存的就是得分最高图片的序号
        rank_score = scores[rank_ID]   #已经把相似度按从大到小的顺序排列了，rank_score[0]就是最高的得分，即最相似的图片

        top_similar_image=all_db_name[rank_ID[0]] #找到相似图片在文档中对应的图片名
    #    resultImg = cv2.imread('D:\\Text_image\\Product_Retrieval\\Goods_db\\'+top_similar_image+'.jpg')
    #    cv2.imshow("resultImg",resultImg)
    #    cv2.waitKey(0)
        print(' -> ',roi_num, top_similar_image,' most similar picture:','%-5d'%(rank_ID[0]+1),'scores:',rank_score[0])

	    #得分大于0.78才认为是同类商品,下面进行商品归类
        if(rank_score[0]>0.70):
            for k in range(0,Topk):
                #number 1
                for j in range(0,baihujiao_name_num):
	            if(all_db_name[rank_ID[k]]==baihujiao_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'baihujiao',(image_tfx,image_tfy),font,0.8,(0,175,255),1)

	        #number 2
	        for j in range(0,baishiguan_name_num):
		    if(all_db_name[rank_ID[k]]==baishiguan_name[j]):
		        font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
		        cv2.putText(image,'baishiguan',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 3
	        for j in range(0,baishiping_name_num):
	            if(all_db_name[rank_ID[k]]==baishiping_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'baishiping',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 4
	        for j in range(0,fenda_name_num):
	            if(all_db_name[rank_ID[k]]==fenda_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
           	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'fenda',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 5
	        for j in range(0,guolicheng_name_num):
	            if(all_db_name[rank_ID[k]]==guolicheng_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
            	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'guolicheng',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 6
	        for j in range(0,hongshaoniurou_name_num):
	            if(all_db_name[rank_ID[k]]==hongshaoniurou_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'hongshaoniurou',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 7
	        for j in range(0,laotansuancaidai_name_num):
	            if(all_db_name[rank_ID[k]]==laotansuancaidai_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'laotansuancaidai',(image_tfx,image_tfy),font,1.5,(0,20,255),2)

	        #number 8
	        for j in range(0,laotansuancaihe_name_num):
	            if(all_db_name[rank_ID[k]]==laotansuancaihe_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'laotansuancaihe',(image_tfx,image_tfy),font,1.5,(0,20,255),2)

	        #number 9
	        for j in range(0,nongfushanquan_name_num):
	            if(all_db_name[rank_ID[k]]==nongfushanquan_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'nongfushanquan',(image_tfx,image_tfy),font,0.8,(0,140,255),1)

	        #number 10
	        for j in range(0,pijiujin_name_num):
	            if(all_db_name[rank_ID[k]]==pijiujin_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'pijiujin',(image_tfx,image_tfy),font,1.5,(0,20,255),2)
			#print("ok - jinse guanyonglai")
	        #number 11
	        for j in range(0,pijiuyin_name_num):
	            if(all_db_name[rank_ID[k]]==pijiuyin_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'pijiuyin',(image_tfx,image_tfy),font,1.5,(0,20,255),2)
			#print("ok - yinse")
	        #number 12
	        for j in range(0,shupiandai_name_num):
	            if(all_db_name[rank_ID[k]]==shupiandai_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'shupiandai',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 13
	        for j in range(0,shupiantong_name_num):
	            if(all_db_name[rank_ID[k]]==shupiantong_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'shupiantong',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 14
	        for j in range(0,tudouya_name_num):
	            if(all_db_name[rank_ID[k]]==tudouya_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'tudouya',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 15
	        for j in range(0,xianggudunji_name_num):
	            if(all_db_name[rank_ID[k]]==xianggudunji_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'xianggudunji',(image_tfx,image_tfy),font,0.8,(0,20,255),1)

	        #number 16
	        for j in range(0,xuebi_name_num):
	            if(all_db_name[rank_ID[k]]==xuebi_name[j]):
			font=cv2.FONT_HERSHEY_SIMPLEX
             	        cv2.rectangle(image, (image_tfx, image_tfy), (image_tfx+image_w, image_tfy+image_h), (0,255,0),2)
			cv2.putText(image,'xuebi',(image_tfx,image_tfy),font,0.4,(0,20,255),1)
    cv2.imwrite('/mnt/disk1/gyl/'+image_name,image)
    print("imwrite picture to /mnt/disk1/gyl/result_reco.jpg is OK!")
    # --plot bb on image





