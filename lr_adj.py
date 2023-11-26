# -*- coding = utf-8 -*-

# @time:2023/5/11 16:32

# Author:Cui

def update_learning_rate(optimizer_G):
    niter_decay = 50
    old_lr = optimizer_G.state_dict()['param_groups'][0]['lr']
    lrd = old_lr / niter_decay

    lr = old_lr - lrd
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    print('update learning rate: %f -> %f' % (old_lr, lr))
