import torch
import math
import numpy as np
from torch.autograd.function import InplaceFunction

class Grads_Ops(InplaceFunction):
    
    @classmethod
    def forward(cls, ctx, input, p=0.0, steps=0, update_steps=0, hadle_stategy='origin', training=False, target=None, inplace=False):
        
        if p < 0 or p > 1:
            raise ValueError("A subnetwork probability of DPSDense has to be between 0 and 1,"
                             " but got {}".format(p))
        
        ctx.p = p    
        ctx.training = training
        ctx.cur_steps = steps
        ctx.hadle_stategy=hadle_stategy
        ctx.update_steps = update_steps
        
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        if ctx.p==0.0 or not ctx.training:
            return output

        if steps == 1:
            cls.steps = 0
            cls.grads_mask = []
            cls.noise_mask = []

        elif math.ceil(steps/update_steps)%2 == 1:
            cls.steps += 1
        
        elif math.ceil(steps/update_steps)%2 == 0 and (steps-1)%update_steps == 0:
            cls.steps += 1
            assert input.size() == cls.grads_mask[-cls.steps].size()          
            index_list = cls.grads_mask[-cls.steps].view(-1).to(input.device)
            sorted_num,_ = torch.sort(index_list,descending=True)
            tar_num = torch.index_select(sorted_num,-1,torch.tensor([math.ceil(index_list.size()[0]*(1-ctx.p))-1]).to(input.device))
            ctx.noise = torch.where((cls.grads_mask[-cls.steps])>=tar_num,1,0)
            cls.noise_mask[-cls.steps] = ctx.noise.clone()
                                            
        else:
            cls.steps += 1 
            assert input.size() == cls.grads_mask[-cls.steps].size()            
            ctx.noise = cls.noise_mask[-cls.steps]
        
        if ctx.hadle_stategy == 'origin':
            return output
        
        return output
        
    @classmethod
    def backward(cls, ctx, grad_output, epsilon=1e3):
        if ctx.p > 0 and ctx.training:
            if ctx.cur_steps == 1:
                cls.grads_mask.append(grad_output.clone()**2/epsilon)
                cls.noise_mask.append(grad_output.clone()**2/epsilon)
            elif math.ceil(ctx.cur_steps/ctx.update_steps)%2 == 1 and (ctx.cur_steps-1) % ctx.update_steps == 0:
                cls.grads_mask[-cls.steps] = grad_output.clone()**2/epsilon
                cls.steps -= 1
            elif math.ceil(ctx.cur_steps/ctx.update_steps)%2 == 0:
                cls.steps -= 1 
            else:
                cls.grads_mask[-cls.steps] += grad_output.clone()**2/epsilon
                cls.steps -= 1
            
            if math.ceil(ctx.cur_steps/ctx.update_steps)%2 == 0:
                return grad_output * ctx.noise, None, None, None, None, None, None, None
            else:
                return grad_output, None, None, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None, None

def grads_ops(input, p=0.0, steps=1, update_steps=1,hadle_stategy='origin', training=False, target=None, inplace=False):
    return Grads_Ops.apply(input, p, steps, update_steps, hadle_stategy, training, target, inplace)