import os
import sys
import json
import math
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils import read_split_data, train_one_epoch, evaluate, read_split_data2, outPut_rankLabel, ouputINFO, writeInInfo, writeINFO, read_split_data3

from model.fuisonNet import FusionNet
from my_dataset import MyDataSet
from utils_metric import ALL_metricCompute


import argparse

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    #     "val": transforms.Compose([transforms.Resize((224, 224)),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   ])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"])

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data2(args.Train_path, args.Val_path, args.GT_path)
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data2(args.Train_path, args.Val_path, args.GT_path)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data3(args.Train_path, args.Val_path, args.GT_path)

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              matrix_Shuffle=args.MatrixShuffle)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"],
                            matrix_Shuffle=0)


    train_num = len(train_dataset)
    val_num = len(val_dataset)



    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn2)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # print("{}".format())

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # model_name = "vgg16"
    # net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    # 网络定义
    backbone_type = "unet"
    net = FusionNet(input_channel=5, backbone_type=backbone_type)
    net.to(device)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # best_acc = 0.0
    # save_path = './{}Net.pth'.format(model_name)
    best_sor = 0
    best_tt = 0
    best_sorEpoch = 0
    best_ttEpoch = 0

    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        mean_loss = torch.zeros(1).to(device)

        for step, data in enumerate(train_bar):
            blood_input, relative_input, labels, img_nameList = data
            optimizer.zero_grad()
            outputs = net(blood_input.to(device), relative_input.to(device))


            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)


        scheduler.step()


        tags = ["train_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss.item(), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

        # validate
        net.eval()
        val_mean_loss = torch.zeros(1).to(device)
        # acc = 0.0  # accumulate accurate number / epoch
        GT_pathList = []
        Pred_NameList = []
        Pred_ValueList = []


        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            # for val_data in val_bar:
            for step, val_data in enumerate(val_bar):
                blood_input, relative_input, val_labels, img_nameList = val_data
                outputs = net(blood_input.to(device), relative_input.to(device))
                # predict_y = torch.max(outputs, dim=1)[1]
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                if args.if_store == True:
                    save_predResult = os.path.join(args.save_path, "pred_results")
                    if not os.path.exists(save_predResult):  # 判在文件夹如果不存在则创建为文件夹
                        os.makedirs(save_predResult)
                    writeInInfo(save_predResult, cur_imgName, Pred_NameOrder, Pred_ValueOrder)

                loss = loss_function(outputs, val_labels.to(device))
                val_mean_loss = (val_mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

                # print(outputs.shape)
                # path_List 用于计算 指标
                for i in range(0, len(img_nameList)):
                    # cur_path = path_list[i]
                    cur_pred = outputs[i]
                    cur_pred = cur_pred.detach()
                    cur_imgName = img_nameList[i]
                    cur_imgGTPath = os.path.join(args.GT_path, cur_imgName + ".txt")
                    Pred_NameOrder, Pred_ValueOrder = outPut_rankLabel(cur_pred, cur_imgGTPath)

                    GT_pathList.append(cur_imgGTPath)
                    Pred_NameList.append(Pred_NameOrder)
                    Pred_ValueList.append(Pred_ValueOrder)
                    # 将预测排序以及GT排序转化为数字排序， 计算指标
                    # print("{}".format())

        sor, tt, str_list = ALL_metricCompute(GT_pathList, Pred_NameList, Pred_ValueList)
        # 一轮所有的test都预测完了，再算指标

        # 写loss
        tags = ["val_loss"]
        tb_writer.add_scalar(tags[0], val_mean_loss, epoch)


        # val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f sor: %.3f tt: %.3f' %
              (epoch + 1, running_loss / train_steps, val_mean_loss, sor, tt))

        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path)

        # sor最佳的保存一下
        # tt最佳的保存一下
        if not os.path.exists(args.save_path):  # 判在文件夹如果不存在则创建为文件夹
            os.makedirs(args.save_path)

        if sor > best_sor:
            save_bestSOR = os.path.join(args.save_path, "best_sor"  + ".pth")
            best_sor = sor
            best_sorEpoch = epoch
            best_sorInfo = str_list
            torch.save(net.state_dict(), save_bestSOR)

        if tt > best_tt:
            save_bestTT = os.path.join(args.save_path, "best_tt"  + ".pth")
            best_tt = tt
            best_ttEpoch = epoch
            best_ttInfo = str_list
            torch.save(net.state_dict(), save_bestTT)


    print('Finished Training')
    print("Best SOR IN epoch {} : {}".format(best_sorEpoch, best_sor))
    save_metricINFO = os.path.join(args.save_path, "metric" + ".txt")

    ouputINFO(best_sorInfo)
    writeINFO(best_sorInfo, best_sorEpoch, save_metricINFO)
    print("Best tt IN epoch {} : {}".format(best_ttEpoch, best_tt))
    ouputINFO(best_ttInfo)
    writeINFO(best_ttInfo, best_ttEpoch, save_metricINFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--Train_path', type=str,
    #                     default=r"D:\A_CatRank\test_newWholeINS\train_file\eps_new_REL_intrSim08_intraThe02\360Cat_NewDataset\whole_testResults")
    # parser.add_argument('--Train_path', type=str,
    #                     default=r"D:\A_CatRank\test_newWholeINS\train_file")
    parser.add_argument('--Train_path', type=str,
                                             default=r"D:\A_CatRank\test_newWholeINS\train_file\eps")
    parser.add_argument('--Val_path', type=str,
                        default=r"D:\A_CatRank\test_newWholeINS\New_data_test\eps_new_REL_intrSim07_intraThe02_FS_3_1\360Cat_NewDataset\whole_testResults")
    parser.add_argument('--GT_path', type=str,
                        default=r"E:\final_project\360-test-557\NEW_ANNA\whole_allMask\Z_FinalDataset\GT_COCO_catRank")
    # parser.add_argument('--save_path', type=str,
    #                     default=r"./weights_new_new/atsal/atsal_new_REL_intrSim09_intraThe05_FS_5_1_aug2_norm_depth3_Fusion2")
    # E:\final_project\weights\Vit

    parser.add_argument('--save_path', type=str,
                        default=r"E:\final_project\weights\Vit/eps/eps_new_REL_intrSim07_intraThe02_FS_3_1_aug1_norm_depth3_Fusion2")

    # MatrixShuffle
    parser.add_argument('--MatrixShuffle', type=int,
                        default=1)

    parser.add_argument('--if_store', type=bool,
                        default=False)
    parser.add_argument('--save_predResult', type=str,
                        default=False)

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
