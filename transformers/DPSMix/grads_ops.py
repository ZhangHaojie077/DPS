import imp
import torch
import math
import numpy as np
import logging
from torch.autograd.function import InplaceFunction
logger = logging.getLogger(__name__)

class Grads_Ops(InplaceFunction): 
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, steps=0, update_steps=0, training=False, inplace=False):
        

        if p < 0 or p > 1:
            raise ValueError("A subnetwork probability of DPSMix has to be between 0 and 1,"
                             " but got {}".format(p))
        
        ctx.p = p    
        ctx.training = training
        ctx.cur_steps = steps
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
            cls.accum_mask = []

            ctx.noise = cls._make_noise(input)
            if len(ctx.noise.size()) == 1:
                ctx.noise.bernoulli_(1 - ctx.p)
            else:
                ctx.noise[0].bernoulli_(1 - ctx.p)
                ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size())-1)))
            ctx.noise.expand_as(input)

        elif math.ceil(steps/update_steps)%2 == 1:
            cls.steps += 1

            ctx.noise = cls._make_noise(input)
            if len(ctx.noise.size()) == 1:
                ctx.noise.bernoulli_(1 - ctx.p)
            else:
                ctx.noise[0].bernoulli_(1 - ctx.p)
                ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size())-1)))
            ctx.noise.expand_as(input)
        
        elif math.ceil(steps/update_steps)%2 == 0 and (steps-1)%update_steps == 0:
            cls.steps += 1
            assert input.size() == cls.grads_mask[-cls.steps].size()
            accum_mask = torch.where(cls.accum_mask[-cls.steps]>0,cls.accum_mask[-cls.steps],1)  
            if update_steps<50:
                grads_norm = (cls.grads_mask[-cls.steps]/accum_mask) 
            else:
                grads_norm = (cls.grads_mask[-cls.steps]/accum_mask) * torch.exp(-accum_mask/update_steps) 
            index_list = (grads_norm).view(-1).to(input.device)
            sorted_num,_ = torch.sort(index_list,descending=True)
            tar_num = torch.index_select(sorted_num,-1,torch.tensor([math.ceil(index_list.size()[0]*(1-ctx.p))-1]).to(input.device))
            ctx.noise = torch.where((grads_norm)>=tar_num,1,0)
            cls.noise_mask[-cls.steps] = ctx.noise.clone()
                                            
        else:
            cls.steps += 1 
            assert input.size() == cls.grads_mask[-cls.steps].size()            
            ctx.noise = cls.noise_mask[-cls.steps]
        
        
        output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        
        return output
        
    @classmethod
    def backward(cls, ctx, grad_output, epsilon=1e3):
        if ctx.p > 0 and ctx.training:
            if ctx.cur_steps == 1:
                cls.grads_mask.append((grad_output.clone()**2)*ctx.noise/epsilon)
                cls.noise_mask.append((grad_output.clone()**2)*ctx.noise/epsilon)
                cls.accum_mask.append(ctx.noise.long())
            elif ctx.cur_steps != 1 and math.ceil(ctx.cur_steps/ctx.update_steps)%2 == 1 and (ctx.cur_steps-1) % ctx.update_steps == 0:
                cls.grads_mask[-cls.steps] = (grad_output.clone()**2)*ctx.noise/epsilon
                cls.accum_mask[-cls.steps] = ctx.noise.long()
                cls.steps -= 1
            elif math.ceil(ctx.cur_steps/ctx.update_steps)%2 == 0:
                cls.steps -= 1 
            else:
                cls.grads_mask[-cls.steps] += (grad_output.clone()**2)*ctx.noise/epsilon
                cls.accum_mask[-cls.steps] += ctx.noise.long()
                cls.steps -= 1

            return grad_output * ctx.noise, None, None, None, None, None, None
            
        else:
            return grad_output, None, None, None, None, None, None

def grads_ops(input, target=None, p=0.0, steps=1, update_steps=1, training=False, inplace=False):
    return Grads_Ops.apply(input, target, p, steps, update_steps, training, inplace)