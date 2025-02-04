import train_dataset  # train
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import ResNet10_diffea
import ProtoNet
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import random
import copy
import warnings

warnings.filterwarnings("ignore", category=Warning)


# 领域分类器
class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)  # 两个类别：源领域和目标领域

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# 反转层用于领域对抗训练
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, target_loader, model, Siamese_model, Siamese_model2, head, loss_fn, optimizer, domain_classifier, domain_optimizer, params, iter_cnt, expect_iter):
    model.train()
    top1 = utils.AverageMeter()
    total_loss = 0
    softmax = nn.Softmax(dim=1)

    for i, (x, target_x) in enumerate(zip(train_loader, target_loader)):
        optimizer.zero_grad()
        domain_optimizer.zero_grad()  # 重置领域分类器的优化器

        x_96 = torch.stack(x[2:8]).cuda()  # (6,way,shot+query,3,96,96)
        x_224 = torch.stack(x[8:]).cuda()  # (1,way,shot+query,3,224,224)
        support_set_anchor = x_224[0, :, :params.n_support, :, :, :]  # (way,shot,3,224,224)
        query_set_anchor = x_224[0, :, params.n_support:, :, :, :]  # (way,query,3,224,224)
        query_set_aug_96 = x_96[:, :, params.n_support:, :, :, :]  # (6,way,query,3,96,96)
        temp_224 = torch.cat((support_set_anchor, query_set_anchor), 1)  # (way,shot+query,3,224,224)
        temp_224 = temp_224.contiguous().view(params.n_way * (params.n_support + params.n_query), 3, 224,
                                              224)  # (way*(shot+query),3,224,224)

        temp_224_res = model(temp_224)  # (way*(shot+query),512)
        temp_224_diffea = model_diffea(temp_224)

        # 四层残差快
        temp_224_res = temp_224_res.view(params.n_way, params.n_support + params.n_query, 512)  # (way,shot+query,512)
        support_set_anchor = temp_224_res[:, :params.n_support, :]  # (way,shot,512)
        support_set_anchor = torch.mean(support_set_anchor, 1)  # (way, 512)
        query_set_anchor = temp_224_res[:, params.n_support:, :]  # (way,query,512)
        query_set_anchor = query_set_anchor.contiguous().view(params.n_way * params.n_query, 512)  # (way*query,512)

        # 一层残差块
        temp_224_diffea = temp_224_diffea.view(params.n_way, params.n_support + params.n_query, 64)  # (way,shot+query,64)
        support_set_anchor_diffea = temp_224_diffea[:, :params.n_support, :]  # (way,shot,64)
        support_set_anchor_diffea = torch.mean(support_set_anchor_diffea, 1)  # (way, 64)
        query_set_anchor_diffea = temp_224_diffea[:, params.n_support:, :]  # (way,query,64)
        query_set_anchor_diffea = query_set_anchor_diffea.contiguous().view(params.n_way * params.n_query, 64)  # (way*query,64)
        support_set_anchor_diffea = F.normalize(support_set_anchor_diffea, p=2, dim=1)
        query_set_anchor_diffea = F.normalize(query_set_anchor_diffea, p=2, dim=1)
        # Support set anchor fusion
        fused_support_set_anchor = torch.cat([support_set_anchor, support_set_anchor_diffea], dim=-1)  # (way, 512 + 64)

        # Query set anchor fusion
        fused_query_set_anchor = torch.cat([query_set_anchor, query_set_anchor_diffea], dim=-1)  # (way*query, 512 + 64)

        pred_query_set_fused = head(fused_support_set_anchor, fused_query_set_anchor)

        query_set_aug_96 = query_set_aug_96.contiguous().view(6 * params.n_way * params.n_query, 3, 96, 96)  # (6*way*query,3,96,96)
        with torch.no_grad():
            query_set_aug_96_512 = Siamese_model(query_set_aug_96)  # (6*way*shot+6*way*query,512)
            query_set_aug_96_64 = Siamese_model2(query_set_aug_96)
        query_set_aug_96_512 = query_set_aug_96_512.contiguous().view(6 * params.n_way * params.n_query, 512)  # (6*5*15, 512)
        query_set_aug_96_64 = query_set_aug_96_64.contiguous().view(6 * params.n_way * params.n_query, 64)  # (6*5*15, 64)
        query_set_aug_96 = torch.cat([query_set_aug_96_512, query_set_aug_96_64], dim=-1)  # (6*way*query, 512 + 64)

        pred_query_set = head(fused_support_set_anchor, query_set_aug_96)  # (6*5*15,5)

        pred_query_set_aug = pred_query_set.contiguous().view(6, params.n_way * params.n_query, params.n_way)  # (6,75,5)

        # pred_query_set_anchor = pred_query_set[0]  # (75,5)
        # pred_query_set_aug = pred_query_set[1:]  # (6,75,5)

        query_set_y = torch.from_numpy(np.repeat(range(params.n_way), params.n_query)).cuda()

        ce_loss = loss_fn(pred_query_set_fused, query_set_y)

        epsilon = 1e-10

        pred_query_set_anchor = softmax(pred_query_set_fused)
        pred_query_set_aug = pred_query_set_aug.contiguous().view(6 * params.n_way * params.n_query, params.n_way)
        pred_query_set_aug = softmax(pred_query_set_aug)
        pred_query_set_anchor = torch.cat([pred_query_set_anchor for _ in range(6)], dim=0)
        pred_query_set_aug = torch.clamp(pred_query_set_aug, min=epsilon)
        pred_query_set_anchor = torch.clamp(pred_query_set_anchor, min=epsilon)
        self_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_aug ** (-pred_query_set_anchor)), dim=1))

        pred_query_set_global = pred_query_set_fused  # (75,5)
        pred_query_set_global = pred_query_set_global.view(params.n_way, params.n_query, params.n_way)
        rand_id_global = np.random.permutation(params.n_query)
        pred_query_set_global = pred_query_set_global[:, rand_id_global[0], :]  # (way,way)
        pred_query_set_global = softmax(pred_query_set_global)  # (way,way)
        pred_query_set_global = pred_query_set_global.unsqueeze(0)  # (1,5,5)
        pred_query_set_global = pred_query_set_global.expand(6, params.n_way, params.n_way)  # (6,5,5)
        pred_query_set_global = pred_query_set_global.contiguous().view(6 * params.n_way, params.n_way)  # (6*way,way)

        rand_id_local_sample = np.random.permutation(params.n_query)
        pred_query_set_local = pred_query_set_aug.view(6, params.n_way, params.n_query, params.n_way)
        pred_query_set_local = pred_query_set_local[:, :, rand_id_local_sample[0], :]  # (6,way,way)
        pred_query_set_local = pred_query_set_local.contiguous().view(6 * params.n_way, params.n_way)  # (6*way,way)

        pred_query_set_local = torch.clamp(pred_query_set_local, min=epsilon)
        pred_query_set_global = torch.clamp(pred_query_set_global, min=epsilon)
        cross_image_loss = torch.mean(torch.sum(torch.log(pred_query_set_local ** (-pred_query_set_global)), dim=1))


        # 领域分类器损失
        p = float(iter_cnt) / expect_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_domain_labels = torch.zeros(fused_support_set_anchor.size(0)).cuda()  # 假设源领域为0

        concat_images_list = [fused_support_set_anchor, ]
        concat_domains_list = [source_domain_labels, ]
        # print('fused_support_set_anchor', fused_support_set_anchor.shape)

        target_support = torch.stack(target_x[8:]).cuda()
        target_support = target_support[0, :, :params.n_support, :, :, :]  # (way,shot,3,224,224)
        target_support11 = target_support.contiguous().view(params.n_way * params.n_support, 3, 224, 224)
        target_support = model(target_support11)
        target_support = target_support.view(params.n_way, params.n_support, 512)
        target_support = target_support[:, :params.n_support, :]  # (way,shot,512)
        target_support = torch.mean(target_support, 1)

        # 一层残差块
        target_support_diffea = model_diffea(target_support11)
        target_support_diffea = target_support_diffea.view(params.n_way, params.n_support, 64)  # (way,shot,64)
        target_support_diffea = target_support_diffea[:, :params.n_support, :]  # (way,shot,64)
        target_support_diffea = torch.mean(target_support_diffea, 1)  # (way, 64)
        target_support_diffea = F.normalize(target_support_diffea, p=2, dim=1)
        # Support set anchor fusion
        target_support = torch.cat([target_support, target_support_diffea], dim=-1)  # (way, 512 + 64)
        # print('target_support',target_support.shape)

        # print(target_support.shape)
        target_domain_labels = torch.ones(target_support.size(0)).cuda()
        # print('target_domain_labels', target_domain_labels.shape)

        concat_images_list.append(target_support)
        concat_domains_list.append(target_domain_labels)

        concat_images = torch.cat(concat_images_list)
        concat_domains = torch.cat(concat_domains_list)
        # print('concat_images', concat_images.shape)
        # print('concat_domains', concat_domains.shape)

        domain_preds = domain_classifier(ReverseLayerF.apply(concat_images, alpha))  # 使用反转层
        domain_loss = loss_fn(domain_preds, concat_domains.long())

        # 总损失
        loss = ce_loss + self_image_loss * params.lamba1 + cross_image_loss * params.lamba2 + domain_loss * params.lamba3
        # print(loss)

        _, predicted = torch.max(pred_query_set_fused.data, 1)
        correct = predicted.eq(query_set_y.data).cpu().sum()
        top1.update(correct.item() * 100 / (query_set_y.size(0) + 0.0), query_set_y.size(0))
        loss.backward()
        optimizer.step()
        domain_optimizer.step()  # 更新领域分类器

        total_loss += loss.item()

        iter_cnt += 1  # Update iteration count

    avg_loss = total_loss / float(i + 1)
    return avg_loss, top1.avg, iter_cnt


if __name__ == '__main__':
    params = parse_args_eposide_train()
    setup_seed(params.seed)

    datamgr_train = train_dataset.Eposide_DataManager(data_path=params.source_data_path,
                                                      num_class=params.train_num_class,
                                                      n_way=params.n_way,
                                                      n_support=params.n_support,
                                                      n_query=params.n_query,
                                                      n_eposide=params.train_n_eposide)
    train_loader = datamgr_train.get_data_loader()

    # 加载目标域数据
    datamgr_target = train_dataset.Eposide_DataManager(data_path=params.current_data_path,
                                                       num_class=params.current_class,
                                                       n_way=params.n_way,
                                                       n_support=params.n_support,
                                                       n_query=params.n_query,
                                                       n_eposide=params.train_n_eposide)
    target_loader = datamgr_target.get_data_loader()

    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims,
                                          list_of_stride=params.list_of_stride,
                                          list_of_dilated_rate=params.list_of_dilated_rate)

    model_diffea = ResNet10_diffea.ResNet(list_of_out_dims=params.list_of_out_dims,
                                          list_of_stride=params.list_of_stride,
                                          list_of_dilated_rate=params.list_of_dilated_rate)

    head = ProtoNet.ProtoNet()

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state, strict=False)
    Siamese_model = copy.deepcopy(model)
    Siamese_model2 = copy.deepcopy(model_diffea)
    model = model.cuda()
    model_diffea = model_diffea.cuda()
    Siamese_model = Siamese_model.cuda()
    Siamese_model2 = Siamese_model2.cuda()
    head = head.cuda()

    domain_classifier = DomainClassifier(input_dim=576, hidden_dim=256).cuda()
    domain_optimizer = torch.optim.Adam(domain_classifier.parameters(), lr=params.lr)

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=params.lr)

    # Initialize iter_cnt
    iter_cnt = 0
    expect_iter = params.epoch * len(train_loader)  # 预期总的迭代次数

    print(f"Lambda values: lamba1={params.lamba1}, lamba2={params.lamba2}, lamba3={params.lamba3}")

    for epoch in range(params.epoch):
        train_loss, train_acc, iter_cnt = train(train_loader, target_loader, model, Siamese_model, Siamese_model2, head, loss_fn, optimizer, domain_classifier,
                                      domain_optimizer, params, iter_cnt, expect_iter)
        print('train:', epoch + 1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc)

    print('iter_cnt:', iter_cnt)
    print('expect_iter:', expect_iter)

    outfile = os.path.join(params.save_dir, '{:d}_epilocalfuseDANN_ISIC.tar'.format(epoch + 1))
    torch.save({
        'epoch': epoch + 1,
        'state_model': model.state_dict(),
        'state_Siamese_model': Siamese_model.state_dict(),
        'state_domain_classifier': domain_classifier.state_dict()},
        outfile)
