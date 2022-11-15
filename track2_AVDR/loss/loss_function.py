#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import logging

from .loss_ce import BatchCalCE
from .loss_mse import BatchCalMSE
from .loss_snr import BatchCalSNR
from .loss_combine_mse_snr_ce import BatchCalCEplusSNRplusMSE

logging.getLogger('matplotlib.font_manager').disabled = True


class LossFunctionWorker(object):
    def __init__(self, log_type=None, logger=None):
        super(LossFunctionWorker, self).__init__()
        self.log_level = log_type
        self.logger = logger
        self._build_map()

    def __call__(self, loss_type, loss_setting, **other_params):
        loss_function = self.init_loss(loss_type=loss_type, **loss_setting)
        return loss_function

    def _build_map(self):
        self.name2loss = {
            'mse': BatchCalMSE,
            'ce': BatchCalCE,
            'snr': BatchCalSNR, 
            'ce+snr+mse': BatchCalCEplusSNRplusMSE
        }

    def init_loss(self, loss_type, **loss_setting):
        write_log(content='Using loss: {}'.format(loss_type), level=self.log_level, logger=self.logger)
        loss_function = self.name2loss[loss_type](**loss_setting)
        return loss_function


def write_log(content, logger=None, level=None, **other_params):
    """
    write log
    :param content: content need to write
    :param level: level of content or None
    :param logger: False or logger
    :param other_params: reserved interface
    :return: None
    """
    if not logger:
        pass
    elif logger == 'print':
        print(content)
    elif isinstance(logger, logging.Logger):
        if not level:
            pass
        else:
            assert level in ['debug', 'info', 'warning', 'error', 'critical'], 'unknown level'
            getattr(logger, level)(content)
    else:
        raise NotImplementedError('unknown logger')
    return None


# class BatchCalCE(nn.Module):
#     def __init__(self, mode='hard', skip_softmax=False, ignore_idx=None, reduction='mean', **other_params):
#         super(BatchCalCE, self).__init__()
#         self.ignore_idx = ignore_idx
#         self.mode = mode
#         self.reduction = reduction
#
#     def forward(self, net_output, label, length=None):
#         net_output = net_output.view(-1, net_output.size(-1))
#         if self.mode == 'hard':
#             label = label.view(-1, )
#             ce_loss = nf.cross_entropy(
#                 input=net_output, target=label, weight=None, size_average=None, ignore_index=self.ignore_idx,
#                 reduce=None, reduction=self.reduction)
#             ce_value = ce_loss.numpy() if self.reduction == 'none' else ce_loss.item()
#             pred = nf.softmax(net_output, dim=-1).argmax(dim=-1)
#             num_correct = (pred == label).sum().float().item()
#             num_all = (label != self.ignore_idx).sum().float().item()
#             acc_value = num_correct / num_all
#         elif self.mode == 'soft':
#             label = label.view(-1, label.size(-1))
#             ce_loss_items = torch.sum(-label * nf.log_softmax(net_output, dim=-1), dim=-1)
#             weight_items = torch.sum(label, dim=-1)
#             weighted_ce_loss_items = weight_items * ce_loss_items
#             if self.reduction == 'none':
#                 ce_loss = weighted_ce_loss_items
#                 ce_value = ce_loss.numpy()
#             elif self.reduction == 'mean':
#                 ce_loss = torch.mean(weighted_ce_loss_items)
#                 ce_value = ce_loss.item()
#             elif self.reduction == 'sum':
#                 ce_loss = torch.sum(weighted_ce_loss_items)
#                 ce_value = ce_loss.item()
#             else:
#                 raise ValueError('unknown reduction')
#             predict = nf.softmax(net_output, dim=-1).argmax(dim=-1)
#             correct = label.argmax(dim=-1)
#             num_correct = (weight_items * (predict == correct).float()).sum().item()
#             acc_value = num_correct / weight_items.sum().item()
#         else:
#             raise NotImplementedError('unknown mode')
#         return ce_loss, torch.tensor([ce_value, acc_value])


# def batch_cal_ce_torch(net_outputs, labels, ignore_idx=38, **params):
#     net_output, label = net_outputs[0], labels[0]
#     # print('net_output', net_output.size())
#     # print('label', label.size())
#     net_output = net_output.view(-1, net_output.size(-1))
#     label = label.view(-1, )
#     # print('net_output', net_output.size())
#     # print('label', label.size())
#     ce_loss = nf.cross_entropy(
#         input=net_output, target=label, weight=None, size_average=None, ignore_index=ignore_idx, reduce=None,
#         reduction='mean')
#     ce_value = ce_loss.item()
#     pred = nf.softmax(net_output, dim=-1).argmax(dim=-1)
#     num_correct = (pred == label).sum().float().item()
#     num_all = (label != ignore_idx).sum().float().item()
#     acc_value = num_correct / num_all
#     return ce_loss, [ce_value, acc_value]
#
#
# def batch_cal_ce_soft_label_torch(net_outputs, labels, reduction='mean', **other_params):
#     net_output, label = net_outputs[0], labels[0]
#     # print('net_output', net_output.size())
#     # print('label', label.size())
#     net_output = net_output.view(-1, net_output.size(-1))
#     label = label.view(-1, label.size(-1))
#     ce_loss_items = torch.sum(-label * nf.log_softmax(net_output, dim=-1), dim=-1)
#     weight_items = torch.sum(label, dim=-1)
#     weighted_ce_loss_items = weight_items*ce_loss_items
#     if reduction == 'none':
#         ce_loss = weighted_ce_loss_items
#         ce_value = ce_loss.numpy()
#     elif reduction == 'mean':
#         ce_loss = torch.mean(weighted_ce_loss_items)
#         ce_value = ce_loss.item()
#     elif reduction == 'sum':
#         ce_loss = torch.sum(weighted_ce_loss_items)
#         ce_value = ce_loss.item()
#     else:
#         raise ValueError('unknown reduction')
#     predict = nf.softmax(net_output, dim=-1).argmax(dim=-1)
#     correct = label.argmax(dim=-1)
#     num_correct = (weight_items*(predict == correct).float()).sum().item()
#     acc_value = num_correct / weight_items.sum().item()
#     return ce_loss, [ce_value, acc_value]
#
#
# def batch_cal_ce_numpy(predict_inputs, true_inputs, reduction='mean', **other_params):
#     predict_embeddings = predict_inputs[0]
#     assert isinstance(predict_embeddings, np.ndarray), 'unknown predict_embeddings'
#     predict_embeddings = np.reshape(a=predict_embeddings, newshape=(-1, predict_embeddings.shape[-1]))
#     sample_num, class_num = predict_embeddings.shape
#     predicts_probability = ss.softmax(x=predict_embeddings, axis=-1)
#     predicts_token = np.argmax(a=predicts_probability, axis=-1)
#     weight = other_params.get('weight', np.ones(shape=(sample_num,)))
#     trues_posteriori = true_inputs[0]
#     if isinstance(trues_posteriori, int):
#         assert sample_num == 1, 'unmatched sample_num'
#         trues_posteriori = np.array([trues_posteriori], dtype='int')
#     elif isinstance(trues_posteriori, np.ndarray):
#         if len(trues_posteriori.shape) == 1 and trues_posteriori.shape[0] == sample_num:
#             pass
#         else:
#             trues_posteriori = np.reshape(a=trues_posteriori, newshape=(-1, trues_posteriori.shape[-1]))
#             assert sample_num == trues_posteriori.shape[0], 'unmatched sample_num'
#             assert class_num == trues_posteriori.shape[0], 'unmatched class_num'
#     else:
#         raise ValueError('unknown trues_posteriori')
#     if len(trues_posteriori.shape) == 1:
#         trues_probability = np.zeros(shape=(sample_num, class_num))
#         trues_token = -1 * np.ones(shape=(trues_posteriori.shape[0],), dtype='int')
#         for sample_idx in range(sample_num):
#             if -1 < trues_posteriori[sample_idx] < class_num:
#                 trues_probability[sample_idx, trues_posteriori[sample_idx]] = 1.
#                 trues_token[sample_idx] = trues_posteriori[sample_idx]
#             else:
#                 weight[sample_idx] = 0
#     else:
#         trues_probability = trues_posteriori
#         trues_token = np.argmax(a=trues_probability, axis=1)
#         trues_token[np.sum(a=trues_probability, axis=1) == 0.] = -1
#         weight[np.sum(a=trues_probability, axis=1) == 0.] = 0
#     ce_loss_items = np.sum(a=-trues_probability * np.log(predicts_probability), axis=-1)*weight
#     if reduction == 'none':
#         ce_value = ce_loss_items
#     elif reduction == 'mean':
#         ce_value = np.sum(a=ce_loss_items) / np.sum(weight)
#     elif reduction == 'sum':
#         ce_value = np.sum(a=ce_loss_items)
#     else:
#         raise ValueError('unknown reduction')
#     num_correct = np.sum(a=(predicts_token == trues_token))
#     acc_value = num_correct / np.sum(weight)
#     return ce_value, acc_value
#
#
#
# def batch_cal_dual_mse_torch(net_outputs, labels, mode='error', alpha=0.5, **params):
#     net_output0 = [net_outputs[0]]
#     net_output1 = [net_outputs[1]]
#     mse_loss0 = batch_cal_mse_torch(net_outputs=net_output0, labels=labels, mode=mode)
#     mse_loss1 = batch_cal_mse_torch(net_outputs=net_output1, labels=labels, mode=mode)
#     return alpha*mse_loss0+(1.-alpha)*mse_loss1
#
#
# def batch_cal_mae_torch(net_outputs, labels, mode, **params):
#     net_output = net_outputs[0]
#     label = labels[0]
#     if len(labels) == 2:
#         length = labels[1]
#         mask = label.new_ones(label.size())
#         for i in range(mask.size(0)):
#             mask[i, length[i]:, :] = 0
#         net_output = mask*net_output
#         mae_sum = nf.l1_loss(input=net_output, target=label, size_average=None, reduce=None, reduction='sum')
#         if mode == 'error':
#             mae_num = length.sum().float()
#         elif mode == 'correct':
#             mae_num = (mask == 1).sum().float()
#         else:
#             raise ValueError('unknown mode')
#         return mae_sum / mae_num
#     else:
#         return nf.l1_loss(input=net_output, target=label, size_average=None, reduce=None, reduction='mean')
#
#
# def batch_cal_cos_torch(net_outputs, labels, **params):
#     net_output = net_outputs[0]
#     label = labels[0]
#     cos_item = nf.cosine_similarity(x1=net_output, x2=label, dim=-1, eps=1e-16)
#     if len(labels) == 2:
#         length = labels[1]
#         mask = label.new_ones(cos_item.size())
#         for i in range(mask.size(0)):
#             mask[i, length[i]:] = 0
#         cos_item = mask * cos_item
#         cos_num = length.sum().float()
#         cos_sum = cos_item.sum()
#         return cos_sum / cos_num
#     else:
#         return cos_item.mean()
#
#
# def batch_cal_mae_cos_torch(net_outputs, labels, mode, alpha, **params):
#     mae_loss = batch_cal_mae_torch(net_outputs=net_outputs, labels=labels, mode=mode)
#     cos_loss = batch_cal_cos_torch(net_outputs=net_outputs, labels=labels)
#     return mae_loss+alpha*cos_loss


# def batch_cal_multi_level_ce_torch(net_outputs, labels, ignore_idx, spec_idx, alpha, beta, **params):
#     labels = labels[0]
#     labels = labels.view(-1, )
#     net_outputs = net_outputs[0]
#     net_outputs = net_outputs.view(-1, net_outputs.size(2))
#     net_outputs = net_outputs[labels != ignore_idx]
#     labels = labels[labels != ignore_idx]
#     net_output1 = torch.cat([net_outputs[:, :spec_idx].max(dim=1, keepdim=True)[0],
#                              net_outputs[:, spec_idx:].max(dim=1, keepdim=True)[0]], dim=1)
#     label1 = (labels == spec_idx).long()
#     net_output2 = net_outputs[labels != spec_idx]
#     net_output2 = net_output2[:, :spec_idx]
#     label2 = labels[labels != spec_idx]
#     ce1 = nf.cross_entropy(input=net_output1, target=label1, weight=None, size_average=None,
#                            ignore_index=-100, reduce=None, reduction='mean')
#     # import pdb
#     # pdb.set_trace()
#     ce2 = nf.cross_entropy(input=net_output2, target=label2, weight=None, size_average=None,
#                            ignore_index=-100, reduce=None, reduction='mean')
#     # print(net_output2)
#     # print(label2)
#     # print(ce2)
#     return alpha*ce1+beta*ce2
if __name__ == '__main__':
    from data.data_io import safe_load
    loss_id = 'loss_mcegm'
    all_loss_setting = safe_load(file='../backup/exp_yml_20210714/0_loss.yml')
    loss_generator = LossFunctionWorker(logger='print')
    loss_function = loss_generator(**all_loss_setting[loss_id])
    # predicted_mask, mixture_wave, clean_wave(, length)
    input_x = [torch.ones(3, 200, 201), torch.rand(3, 32000), torch.rand(3, 32000), torch.tensor([30000, 31000, 32000])]
    output_y = loss_function(*input_x)
    print(output_y)
