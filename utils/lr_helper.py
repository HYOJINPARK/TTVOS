class CycleScheduler(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, initial=0.1, cycle_len=5, ep_cycle=50, ep_max=100):
        super(CycleScheduler, self).__init__()

        self.min_lr = initial# minimum learning rate
        self.m = cycle_len
        self.ep_cycle = ep_cycle
        self.ep_max = ep_max
        self.poly_start = initial
        self.step = initial/ self.ep_cycle
        print('Using Cyclic LR Scheduler with warm restarts and poly step'
              + str(self.step))

    def get_lr(self, epoch):
        if epoch==0:
            current_lr = self.min_lr
        elif 0< epoch and epoch <= self.ep_cycle:
            counter = (epoch-1) % self.m
            current_lr = round((self.min_lr * self.m) - (counter * self.min_lr), 5)
        else:

            current_lr = round(self.poly_start - (epoch-self.ep_cycle )*self.step, 8)

            # current_lr = round(self.poly_start * (1 - (epoch-self.ep_cycle) / (self.ep_max-self.ep_cycle)) ** 0.9, 8)

        return current_lr


class WarmupPoly(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, init_lr, total_iter, warmup_ratio=0.05, poly_pow = 0.98):
        super(WarmupPoly, self).__init__()
        self.init_lr = init_lr
        self.total_iter = total_iter
        self.warmup_ep = int(warmup_ratio*total_iter)
        print("warup unitl " + str(self.warmup_ep))
        self.poly_pow = poly_pow

    def get_lr(self, epoch):
        #
        if epoch < self.warmup_ep:
            curr_lr =  self.init_lr*pow((((epoch) / self.warmup_ep)), self.poly_pow)
        else:
            # curr_lr = self.init_lr*pow((1 - ((epoch- self.warmup_ep)  / (self.total_ep-self.warmup_ep))), self.poly_pow)
            curr_lr = self.init_lr*((1 - ((epoch- self.warmup_ep)  / (self.total_iter-self.warmup_ep)))** self.poly_pow)
        # baseLR * ((1 - (ite / float(maxEpochs * step_each_epoch))) ** power)
        return curr_lr
