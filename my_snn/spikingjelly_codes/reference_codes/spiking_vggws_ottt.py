import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from spikingjelly.activation_based  import functional, neuron, layer

__all__ = [
    'OTTTSpikingVGG',
    'ottt_spiking_vggws', 
    'ottt_spiking_vgg11','ottt_spiking_vgg11_ws',
    'ottt_spiking_vgg13','ottt_spiking_vgg13_ws',
    'ottt_spiking_vgg16','ottt_spiking_vgg16_ws',
    'ottt_spiking_vgg19','ottt_spiking_vgg19_ws',
]

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
    
# 'S': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512]
# model = OTTTSpikingVGG(cfg=cfgs[cfg], weight_standardization=True넣긴함, spiking_neuron=spiking_neuron, **kwargs)
class OTTTSpikingVGG(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, light_classifier=True, drop_rate=0., **kwargs):
        super(OTTTSpikingVGG, self).__init__()

        print("cfg:", cfg)
        print("weight_standardization:", weight_standardization)
        print("num_classes:", num_classes)
        print("init_weights:", init_weights)
        print("spiking_neuron:", spiking_neuron)
        print("light_classifier:", light_classifier)
        print("drop_rate:", drop_rate)
        print("Contents of **kwargs:", kwargs)





        self.fc_hw = kwargs.get('fc_hw', 1)
        print('self.fc_hw:', self.fc_hw)
        if weight_standardization:
            ws_scale = 2.74
            print('yes ws')
        else:
            ws_scale = 1.
            print('no ws')
        self.neuron = spiking_neuron
        
        self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization,
                                         neuron=spiking_neuron, drop_rate=0., **kwargs)
        if light_classifier:
            self.classifier = layer.OTTTSequential(
                layer.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw)),
                layer.Flatten(1),
                layer.Linear(512*(self.fc_hw**2), num_classes),
            )
        else:
            Linear = layer.WSLinear if weight_standardization else layer.Linear
            self.classifier = layer.OTTTSequential(
                layer.AdaptiveAvgPool2d((7, 7)),
                layer.Flatten(1),
                Linear(512 * 7 * 7, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                Scale(ws_scale),
                # TODO check multi-gpu condition
                layer.Dropout(),
                Linear(4096, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                Scale(ws_scale),
                # TODO check multi-gpu condition
                layer.Dropout(),
                layer.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # print('classifier output size:', x.shape)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: callable = None, drop_rate=0., **kwargs):
        layers = []
        in_channels = 3
        Conv2d = layer.WSConv2d if weight_standardization else layer.Conv2d
        for v in cfg:
            if v == 'M':
                print('maxpool')
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                print('avepool')
                layers += [layer.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, SIZE_CHECK_post_conv(), neuron(**deepcopy(kwargs)), SIZE_CHECK_post_neuron()]
                if weight_standardization:
                    layers += [Scale(2.74)]
                in_channels = v
                if drop_rate > 0.:
                    # TODO check multi-gpu condition
                    layers += [layer.Dropout(drop_rate)]
        return layer.OTTTSequential(*layers)




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    # 'S': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'S': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512],
    # 'S': [64, 128, 'A', 256, 256],
    # 'S': [64, 128, 'A', 256, 256, 'A', 512, 512,],
}

# _spiking_vgg('vggws', 'S', True, spiking_neuron, light_classifier=True, **kwargs)

def _spiking_vgg(arch, cfg, weight_standardization, spiking_neuron: callable = None, **kwargs):
    model = OTTTSpikingVGG(cfg=cfgs[cfg], weight_standardization=weight_standardization, spiking_neuron=spiking_neuron, **kwargs)
    return model

#  model = vggmodel.ottt_spiking_vggws(num_classes=10, spiking_neuron=neuron.OTTTLIFNode)

def ottt_spiking_vggws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG (sWS), model used in 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vggws', 'S', True, spiking_neuron, light_classifier=True, **kwargs)




def ottt_spiking_vgg11(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg11', 'A', False, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg11_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg11_ws', 'A', True, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg13(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-13
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg13', 'B', False, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg13_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg13_ws', 'B', True, spiking_neuron, light_classifier=False, **kwargs)




def ottt_spiking_vgg16(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg16', 'D', False, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg16_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16 with weight standardization
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg16_ws', 'D', True, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg19(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-19
        :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg19', 'E', False, spiking_neuron, light_classifier=False, **kwargs)



def ottt_spiking_vgg19_ws(spiking_neuron: callable = neuron.OTTTLIFNode, **kwargs):
    """
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with weight standardization
    :rtype: torch.nn.Module
    """

    return _spiking_vgg('vgg19_ws', 'E', True, spiking_neuron, light_classifier=False, **kwargs)





######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
class LIF_layer(nn.Module):
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on):
        super(LIF_layer, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on

    def forward(self, input_current):
        v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init
        post_spike = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float, requires_grad=False) 
        # i와 v와 post_spike size는 여기서 다 같음: [Time, Batch, Channel, Height, Width] 
        Time = v.shape[0]
        for t in range(Time):
            # leaky하고 input_current 더하고 fire하고 reset까지 (backward직접처리)
            post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t], 
                                            self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 
        return post_spike 



class LIF_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_current_one_time, v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on):
        v_one_time = v_one_time * v_decay + input_current_one_time # leak + pre-synaptic current integrate
        spike = (v_one_time >= v_threshold).float() #fire
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        else:
            pass

        if (BPTT_on == True):
            BPTT_on = 1
        else:
            BPTT_on = 0

        ctx.save_for_backward(v_one_time, torch.tensor([v_decay], requires_grad=False), 
                            torch.tensor([v_threshold], requires_grad=False), 
                            torch.tensor([v_reset], requires_grad=False), 
                            torch.tensor([sg_width], requires_grad=False),
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([BPTT_on], requires_grad=False)) # save before reset
        
        if (v_reset >= 0 and v_reset < 10000): # soft reset
            v_one_time = (v_one_time - spike * v_threshold) # reset
            # v_one_time = (v_one_time - spike * v_threshold).clamp_min(0) # reset # 0미만으로는 안 가게 하려면 이 줄 on
        elif (v_reset >= 10000 and v_reset < 20000): # hard reset 
            v_reset -= 10000 
            v_one_time = v_one_time * (1 - spike) + v_reset * spike # reset
            v_reset += 10000
        return spike, v_one_time

    @staticmethod
    def backward(ctx, grad_output_spike, grad_output_v):
        v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on = ctx.saved_tensors
        v_decay=v_decay.item()
        v_threshold=v_threshold.item()
        v_reset=v_reset.item()
        sg_width=sg_width.item()
        surrogate=surrogate.item()
        BPTT_on=BPTT_on.item()

        grad_input_current = grad_output_spike.clone()
        if BPTT_on == 1:
            grad_input_v = grad_output_v.clone() # not used

        ################ select one of the following surrogate gradient functions ################
        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            sig = torch.sigmoid((v_one_time - v_threshold))
            grad_input_current *= 4*sig*(1-sig)
            # grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input_current *= ((v_one_time - v_threshold).abs() < sg_width/2).float() / sg_width

        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_input_current[(v_one_time - v_threshold).abs() > sg_width/2] = 0
        else: 
            assert False, 'surrogate doesn\'t exist'
        ###########################################################################################

        ## if BPTT_on == 1, then second return value is not None
        if (BPTT_on == 1):
            grad_input_v = v_decay * grad_input_v 
        else:
            grad_input_v = None 
        
        return grad_input_current, grad_input_v, None, None, None, None, None, None
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
    
    
class SIZE_CHECK_post_conv(nn.Module):
    def __init__(self):
        super(SIZE_CHECK_post_conv, self).__init__()

    def forward(self, x):
        # print('conv output size:', x.shape)
        return x
    
class SIZE_CHECK_post_neuron(nn.Module):
    def __init__(self):
        super(SIZE_CHECK_post_neuron, self).__init__()

    def forward(self, x):
        # print('neuron output size:', x.shape)
        return x
    
class SIZE_CHECK_post_fc(nn.Module):
    def __init__(self):
        super(SIZE_CHECK_post_fc, self).__init__()

    def forward(self, x):
        # print('fc output size:', x.shape)
        return x
    




##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_CONV, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time dimension {Time} should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)
        output_current = []
        
        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0],requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0],requires_grad=False)
        for t in range(Time):
            # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past

            # output_current[t]= SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) 
            output_current.append( SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) )
            
            spike_past = spike_now
            # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))

        output_current = torch.stack(output_current, dim=0)
        return output_current

class SYNAPSE_CONV_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias, stride=1, padding=1):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias, torch.tensor([stride], requires_grad=False), torch.tensor([padding], requires_grad=False))
        return F.conv2d(spike_one_time, weight, bias=bias, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output_current):
        spike_one_time, spike_now, weight, bias, stride, padding = ctx.saved_tensors
        stride=stride.item()
        padding=padding.item()
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()


        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = F.conv_transpose2d(grad_output_current_clone, weight, stride=stride, padding=padding)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.nn.grad.conv2d_weight(spike_now, weight.shape, grad_output_current_clone,
                                                    stride=stride, padding=padding)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum((0, -1, -2))


        # print('grad_input_spike_conv', grad_input_spike)
        # print('grad_weight_conv', grad_weight)
        # print('grad_bias_conv', grad_bias)
        # print('grad_input_spike_conv', ctx.needs_input_grad[0])
        # print('grad_weight_conv', ctx.needs_input_grad[2])
        # print('grad_bias_conv', ctx.needs_input_grad[3])

        return grad_input_spike, None, grad_weight, grad_bias, None, None
   
class SYNAPSE_FC(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))
        # Xavier 초기화
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

        # nn.init.normal_(m.weight, 0, 0.01)
        # nn.init.constant_(m.bias, 0)
        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time({Time}) dimension should be same as TIME({self.TIME})'
        Batch = spike.shape[1] 

        # output_current = torch.zeros(Time, Batch, self.out_features, device=spike.device)
        output_current = []

        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)

        for t in range(Time):
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past
            # output_current[t]= SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) 
            output_current.append( SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) )
            
            spike_past = spike_now

        output_current = torch.stack(output_current, dim=0)
        return output_current 
    



class SYNAPSE_FC_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias)
        return F.linear(spike_one_time, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output_current):
        #############밑에부터 수정해라#######
        spike_one_time, spike_now, weight, bias = ctx.saved_tensors
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()

        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = grad_output_current_clone @ weight
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output_current_clone.t() @ spike_now
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum(0)

        # print('grad_input_spike_FC', grad_input_spike)
        # print('grad_weight_FC', grad_weight)
        # print('grad_bias_FC', grad_bias)
        # print('grad_input_spike_FC', ctx.needs_input_grad[0])
        # print('grad_weight_FC', ctx.needs_input_grad[2])
        # print('grad_bias_FC', ctx.needs_input_grad[3])
        
        return grad_input_spike, None, grad_weight, grad_bias

##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################