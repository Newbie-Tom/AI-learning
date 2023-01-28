#根据传入的文件true_label和predict_label来求模型预测的精度、召回率和F1值，另外给出微观和宏观取值。
'''
acc = (TP+TN)/(TP+FP+FN+TN)
perceion =TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = 2*perction*Recall/(perction+Recall)
'''

import os
import sys
import json
sys.path.insert(0,os.getcwd())
import torch
from PIL import Image
from torchvision import transforms
from model_v2 import MobileNetV2
import numpy as np
from tqdm import tqdm

def get_info(Json_path):
    with open(Json_path, encoding='utf-8') as f:
        class_indict = json.load(f)
    names = []
    indexs = []
    for data in class_indict:
        names.append(class_indict[data])
        indexs.append(int(data))
    return names, indexs

def getTrueLabel(dataSetPath,jsonPath):
    classes, indexs = get_info(jsonPath)

    txtFile = open('trueLabel.txt', 'w')
    classesName = os.listdir(dataSetPath)
    for name in classesName:
        if name not in classes:
            continue
        clsId = indexs[classes.index(name)]
        images_path = os.path.join(dataSetPath, name)
        images_name = os.listdir(images_path)
        for photo_name in images_name:
            # os.path.splitext是分离文件名和扩展名
            _, postfix = os.path.splitext(photo_name)
            if postfix not in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                continue
            txtFile.write(str(clsId) + ' ' + '%s' % (os.path.join(name)))
            txtFile.write('\n')
    trueLabel = txtFile.name
    txtFile.close()
    return trueLabel

def getPredictLabel(dataSetPath,jsonPath):
    with open(jsonPath, encoding='utf-8') as f:
        class_indict = json.load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # create model
    model = MobileNetV2(num_classes=23).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    txtFile = open('predictLabel.txt','w')
    for root, dirs, files in os.walk(dataSetPath):
        for file in files:
            img = Image.open(root + '/' + str(file))
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).item()
                txtFile.write(str(predict_cla) + ' ' + class_indict[str(predict_cla)])
                txtFile.write('\n')
    predictLabel = txtFile.name
    txtFile.close()
    return predictLabel

def getLabelData(file_dir):
    '''
    模型的预测生成相应的label文件，以及真实类标文件，根据文件读取并加载所有label
    1、参数说明：
        file_dir：加载的文件地址。
        文件内数据格式：每行包含两列，第一列为编号1,2，...，第二列为预测或实际的类标签名称。两列以空格为分隔符。
        需要生成两个文件，一个是预测，一个是实际类标，必须保证一一对应，个数一致
    2、返回值：
        返回文件中每一行的label列表，例如['true','false','false',...,'true']
    '''
    labels = []
    with open(file_dir, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            labels.append(i.strip().split(' ')[1])
    return labels


def getLabel2idx(labels):
    '''
    获取所有类标
    返回值：label2idx字典，key表示类名称，value表示编号0,1,2...
    '''
    label2idx = dict()
    for i in labels:
        if i not in label2idx:
            label2idx[i] = len(label2idx)
    return label2idx

def buildConfusionMatrix(predict_file, true_file):
    '''
    针对实际类标和预测类标，生成对应的矩阵。
    矩阵横坐标表示实际的类标，纵坐标表示预测的类标
    矩阵的元素(m1,m2)表示类标m1被预测为m2的个数。
    所有元素的数字的和即为测试集样本数，对角线元素和为被预测正确的个数，其余则为预测错误。
    返回值：返回这个矩阵numpy
    '''
    true_labels = getLabelData(true_file)
    predict_labels = getLabelData(predict_file)
    # label2idx = getLabel2idx(true_labels)
    label2idx = getLabel2idx(true_labels)
    confMatrix = np.zeros([len(label2idx), len(label2idx)], dtype=np.int32)
    for i in range(len(true_labels)):
        true_labels_idx = label2idx[true_labels[i]]
        predict_labels_idx = label2idx[predict_labels[i]]
        confMatrix[true_labels_idx][predict_labels_idx] += 1
    return confMatrix, label2idx


def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    perceion =TP/(TP+FP)    横坐标
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    Recall = TP/(TP+FN)
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)

def main():

    dataSetPath = '/home/tom/AI_project/deep-learning-for-image-processing/data_set/fruit_data/test'
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    trueFileName = getTrueLabel(dataSetPath,json_path)
    trueLabelPath = os.path.join(os.path.dirname(json_path),trueFileName)
    predictFileName = getPredictLabel(dataSetPath,json_path)
    predictLabelPath = os.path.join(os.path.dirname(json_path), predictFileName)
    '''
    该为主函数，可将该函数导入自己项目模块中
    打印精度、召回率、F1值的格式可自行设计
    '''
    # 读取文件并转化为混淆矩阵,并返回label2idx
    confMatrix, label2idx = buildConfusionMatrix(trueLabelPath, predictLabelPath)
    total_sum = confMatrix.sum()
    all_prediction = calculate_all_prediction(confMatrix)
    label_prediction = []
    label_recall = []
    print('total_sum=', total_sum, ',label_num=', len(label2idx), '\n')
    for i in label2idx:
        print('  ', i)
    print('  ')
    for i in label2idx:
        print(i, end=' ')
        label_prediction.append(calculate_label_prediction(confMatrix, label2idx[i]))
        label_recall.append(calculate_label_recall(confMatrix, label2idx[i]))
        for j in label2idx:
            labelidx_i = label2idx[i]
            label2idx_j = label2idx[j]
            print('  ', confMatrix[labelidx_i][label2idx_j], end=' ')
        print('\n')

    print('prediction(accuracy)=', all_prediction, '%')
    print('individual result\n')
    for ei, i in enumerate(label2idx):
        print(ei, '\t', i, '\t', 'prediction=', label_prediction[ei], '%,\trecall=', label_recall[ei], '%,\tf1=',
              calculate_f1(label_prediction[ei], label_recall[ei]))
    p = round(np.array(label_prediction).sum() / len(label_prediction), 2)
    r = round(np.array(label_recall).sum() / len(label_prediction), 2)
    print('MACRO-averaged:\nprediction=', p, '%,recall=', r, '%,f1=', calculate_f1(p, r))

if __name__ == "__main__":
    main()