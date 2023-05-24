import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import cv2
from shutil import copyfile
import pickle
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from xception import xception
#from rep_VGGNET import *
#from repvgg import *
#from repvgg_plus import *
#from models import build_model
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
from models import densenet121
import sys
from efficientnet_pytorch import EfficientNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def fundus_test():
    args = parser.parse_args()
    

    dataset_subdir = [#'external_ZEH_1019',
                       # 'external_PHONE_1019',
                      # 'External_NOC',
                      # 'External_ZEH',
                      #   'train',
                      #  'Hangzhou',
                      #  'val',
                      #   'External_original_all_lowmobel'
                      # 'train1_lowquality',
                      # 'val1_lowquality',
                      # 'val1',
                      # 'test1_original',
                      # 'test1_original_lowquality',
                      # 'test1_lowquality',
                     ]
    model_names = [
                'densenet121_c2',
                #'densenet201',
                # 'inception_v3',
                # 'RepVGG_B2g4',
                # 'Transform_large',
                # 'Transform_base',
                #'resnet50_cbam'
                #'alexnet'
                ]
    for index_name in range(0,len(model_names)):
        args.arch = model_names[index_name]
        for i in range(0,len(dataset_subdir)):
            dataset_dir = '' + dataset_subdir[i]
            resultset_dir = './result/' + dataset_subdir[i]
            args, model, val_transforms = load_modle_trained(args)
            mk_result_dir(args, resultset_dir)
            fundus_test_exec(args, model, val_transforms, dataset_dir,resultset_dir)
            # fundus_test_exec_external(args, model, val_transforms, dataset_dir, resultset_dir)


def load_modle_trained(args):

    normalize = transforms.Normalize(mean = [0.22754873, 0.1638334, 0.09096901], 
                                     std = [0.28501758, 0.21946926, 0.14383657])
  
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    print("=> loading checkpoint### ", args.arch)
    if args.arch.find('alexnet') != -1:
        pre_name = './alexnet'
    elif args.arch.find('inception_v3') != -1:
        pre_name = './inception_v3'
    elif args.arch.find('densenet121_c2') != -1:
        pre_name = './densenet121_0.5'
    elif args.arch.find('densenet201') != -1:
        pre_name = './densenet201'
    elif args.arch.find('inceptionresnetv2') != -1:
        pre_name = './inceptionresnetv2'
    elif args.arch.find('xception') != -1:
        pre_name = './xception'
    elif args.arch.find('RepVGG_B2g4') != -1:
        pre_name = './RepVGG_B2g4'
    elif args.arch.find('Transform_large') != -1:
        pre_name = './Transform_large'
    elif args.arch.find('Transform_base') != -1:
        pre_name = './Transform_base'
    elif args.arch.find('resnet50_cbam') != -1:
        pre_name = './resnet50_cbam'
    else:
        print('### please check the args.arch###')
        exit(-1)
    PATH = pre_name + '_model_best_acc.pth.tar'
    # PATH = './densenet_nodatanocost_best.ckpt'

    # PATH = pre_name + '_checkpoint.pth.tar'

    if args.arch.find('alexnet') != -1:
        model = models.__dict__[args.arch](pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 4)
    elif args.arch.find('inception_v3') != -1:
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        num_auxftrs = model.AuxLogits.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
        model.AuxLogits.fc = nn.Linear(num_auxftrs, 4)
        model.aux_logits = False
    elif args.arch.find('densenet121_c2') != -1:
        model = densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
        model.classifier2 = nn.Linear(num_ftrs, 3)
    elif args.arch.find('densenet121') != -1:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3)
    elif args.arch.find('densenet201') != -1:
        model = models.densenet201(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 4)
    elif args.arch.find('inceptionresnetv2') != -1:
        # model = inceptionresnetv2(pretrained=True)
        model = InceptionResnetV2(num_classes=4)
        # num_ftrs = model.last_linear.in_features
        # model.classifier = nn.Linear(num_ftrs, 4)

    else:
        print('### please check the args.arch for load model in testing###')
        exit(-1)

    print(model)
    if args.arch.find('alexnet') == -1:
        model = torch.nn.DataParallel(model).cuda()  #for modles trained by multi GPUs: densenet inception_v3 resnet50
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    if args.arch.find('alexnet') != -1:
        model = torch.nn.DataParallel(model).cuda()   #for models trained by single GPU: Alexnet
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    print('best_epoch , best_acc1 ,best_loss are: ' ,start_epoch   , best_acc1, best_loss)
    return args, model, val_transforms
def mk_result_dir(args,testdata_dir='./data/val1'):
    testdatadir = testdata_dir
    model_name = args.arch
    result_dir = testdatadir + '/' + model_name
    grade1_grade2 = 'group1_group2'
    grade1_grade3 = 'group1_group3'
    grade2_grade3 = 'group2_group3'
    grade2_grade1 = 'group2_group1'
    grade3_grade1 = 'group3_group1'
    grade3_grade2 = 'group3_group2'

    # grade1_grade1 = 'normal_normal'
    # grade2_grade2 = 'keratitis_keratitis'
    # grade3_grade3 = 'other_other'

    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
        os.makedirs(result_dir + '/' + grade1_grade2)
        os.makedirs(result_dir + '/' + grade1_grade3)
        os.makedirs(result_dir + '/' + grade2_grade3)
        os.makedirs(result_dir + '/' + grade2_grade1)
        os.makedirs(result_dir + '/' + grade3_grade1)
        os.makedirs(result_dir + '/' + grade3_grade2)

        # os.makedirs(result_dir + '/' + grade1_grade1)
        # os.makedirs(result_dir + '/' + grade2_grade2)
        # os.makedirs(result_dir + '/' + grade3_grade3)

def fundus_test_exec(args,model,val_transforms,testdata_dir='./data/val1',resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg','.JPEG','.BMP','.tif','.TIF','.png','.PNG')
    grade1_grade2 = 'group1_group2'
    grade1_grade3 = 'group1_group3'
    grade2_grade3 = 'group2_group3'
    grade2_grade1 = 'group2_group1'
    grade3_grade1 = 'group3_group1'
    grade3_grade2 = 'group3_group2'
    grade1_grade1 = 'group1'
    grade2_grade2 = 'group2'
    grade3_grade3 = 'group3'

    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        list_grade1_grade2=[grade1_grade2]
        list_grade1_grade3=[grade1_grade3]
        grade1_2=[grade1_grade2]
        grade1_3=[grade1_grade3]
        grade1_1=[grade1_grade1]
        root = testdatadir + '/Group1'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
            _,ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]


            grade1_num = grade1_num + 1
            #print(grade1_num)
            if pred_0 == 0:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
            elif pred_0 == 1:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/group1_group2' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/group1_group3' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num)

        grade2_grade1_num = 0
        grade2_grade3_num = 0
        grade2_grade2_num = 0
        grade2_num = 0
        list_grade2_grade1=[grade2_grade1]
        list_grade2_grade3=[grade2_grade3]
        grade2_1=[grade2_grade1]
        grade2_3=[grade2_grade3]
        grade2_2=[grade2_grade2]
        root = testdatadir + '/Group2'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            # img_PIL.show()  # 原始图片
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            # image = cv2.imread(os.path.join(root, img))  # image = image.unsqueeze(0) # PIL_image = Image.fromarray(image)
            _,ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade2_num = grade2_num + 1
            if pred_0 == 0:
                # print('location to ok')
                grade2_grade1_num = grade2_grade1_num + 1
                list_grade2_grade1.append(img)
                grade2_1.append(prob_list)
                file_new_1 = desdatadir + '/group2_group1' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                # print('location to quality')
                grade2_grade3_num = grade2_grade3_num + 1
                list_grade2_grade3.append(img)
                grade2_3.append(prob_list)
                file_new_1 = desdatadir + '/group2_group3' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('location to location')
                grade2_grade2_num = grade2_grade2_num + 1
                grade2_2.append(prob_list)
        print(grade2_grade1_num, grade2_grade2_num, grade2_grade3_num)

        grade3_grade1_num = 0
        grade3_grade3_num = 0
        grade3_grade2_num = 0
        grade3_num = 0
        list_grade3_grade1=[grade3_grade1]
        list_grade3_grade2=[grade3_grade2]
        grade3_1=[grade3_grade1]
        grade3_2=[grade3_grade2]
        grade3_3=[grade3_grade3]
        root = testdatadir + '/Group3'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            _,ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade3_num = grade3_num + 1
            if pred_0 == 0:
                # print('quality to ok')
                grade3_grade1_num = grade3_grade1_num + 1
                list_grade3_grade1.append(img)
                grade3_1.append(prob_list)
                file_new_1 = desdatadir + '/group3_group1' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                # print('quality  to location')
                grade3_grade2_num = grade3_grade2_num + 1
                list_grade3_grade2.append(img)
                grade3_2.append(prob_list)
                file_new_1 = desdatadir + '/group3_group2' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality  to quality')
                grade3_grade3_num = grade3_grade3_num + 1
                grade3_3.append(prob_list)
        print(grade3_grade1_num, grade3_grade2_num, grade3_grade3_num)


    confusion_matrix = [ [grade1_grade1_num, grade1_grade2_num, grade1_grade3_num],
                         [grade2_grade1_num, grade2_grade2_num, grade2_grade3_num],
                         [grade3_grade1_num, grade3_grade2_num, grade3_grade3_num]]
    print('confusion_matrix:')
    print (confusion_matrix)

    result_confusion_file = args.arch + '_1.txt'
    result_pro_file =  args.arch + '_2.txt'
    result_value_bin = args.arch + '_3.txt'


    with open(desdatadir + '/' + result_confusion_file, "w") as file_object:
        for i in confusion_matrix:
            file_object.writelines(str(i) + '\n')
        file_object.writelines('ERROR_images\n')
        for i in list_grade1_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade3:
            file_object.writelines(str(i) + '\n')

        for i in list_grade2_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade3:
            file_object.writelines(str(i) + '\n')

        for i in list_grade3_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade2:
            file_object.writelines(str(i) + '\n')
        file_object.close()

    with open(desdatadir + '/' + result_pro_file, "w") as file_object:
        for i in grade1_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade2_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade3_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        file_object.close()

    with open(desdatadir + '/' + result_value_bin, "wb") as file_object:
        pickle.dump(confusion_matrix, file_object)  # 顺序存入变量
        pickle.dump(grade1_1, file_object)
        pickle.dump(grade1_2, file_object)
        pickle.dump(grade1_3, file_object)
        pickle.dump(grade2_1, file_object)
        pickle.dump(grade2_2, file_object)
        pickle.dump(grade2_3, file_object)
        pickle.dump(grade3_1, file_object)
        pickle.dump(grade3_2, file_object)
        pickle.dump(grade3_3, file_object)
        file_object.close()

def fundus_test_exec_external(args,model,val_transforms,testdata_dir='./data/val1',resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg','.JPEG','.BMP','.tif','.TIF','.png','.PNG')
    grade1_grade2 = 'normal_keratitis'
    grade1_grade3 = 'normal_other'
    grade2_grade3 = 'keratitis_other'
    grade2_grade1 = 'keratitis_normal'
    grade3_grade1 = 'other_normal'
    grade3_grade2 = 'other_keratitis'

    grade1_grade1 = 'normal_normal'
    grade2_grade2 = 'keratitis_keratitis'
    grade3_grade3 = 'other_other'

    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        list_grade1_grade2=[grade1_grade2]
        list_grade1_grade3=[grade1_grade3]
        grade1_2=[grade1_grade2]
        grade1_3=[grade1_grade3]
        grade1_1=[grade1_grade1]
        # root = testdatadir + '/normal'
        root = testdatadir
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]


            grade1_num = grade1_num + 1
            print(grade1_num)
            if pred_0 == 1:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
                file_new_1 = desdatadir + '/normal_normal' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 0:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/normal_keratitis' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/normal_other' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num)

if __name__ == '__main__':
    #main()
    fundus_test()

