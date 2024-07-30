
import random

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *

def seed_assign(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
# mapping = {0: 'Hand Clapping',1: 'Right Hand Wave',2: 'Left Hand Wave',3: 'Right Arm CW',4: 'Right Arm CCW',5: 'Left Arm CW',6: 'Left Arm CCW',7: 'Arm Roll',8: 'Air Drums',9: 'Air Guitar',10: 'Other'}
def dvs_visualization(inputs, labels, TIME, BATCH, my_seed):
    seed_assign(seed = my_seed)
    what_input = random.randint(0, BATCH - 1)
    inputs_for_view = inputs.permute(1, 0, 2, 3, 4)
    for i in range(TIME):
        # 예시 데이터 생성
        data1 = inputs_for_view[what_input][i][0].numpy()  # torch tensor를 numpy 배열로 변환
        data2 = inputs_for_view[what_input][i][1].numpy()  # torch tensor를 numpy 배열로 변환

        # 데이터 플로팅
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열의 subplot 생성

        # 첫 번째 subplot에 데이터1 플로팅
        im1 = axs[0].imshow(data1, cmap='viridis', interpolation='nearest')
        axs[0].set_title(f'Channel 0\nLabel: {labels[what_input]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        axs[0].grid(False)
        fig.colorbar(im1, ax=axs[0])  # Color bar 추가

        # 두 번째 subplot에 데이터2 플로팅
        im2 = axs[1].imshow(data2, cmap='viridis', interpolation='nearest')
        axs[1].set_title(f'Channel 1\nLabel: {labels[what_input]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[1].set_xlabel('X axis')
        axs[1].set_ylabel('Y axis')
        axs[1].grid(False)
        fig.colorbar(im2, ax=axs[1])  # Color bar 추가

        plt.tight_layout()  # subplot 간 간격 조정
        plt.show()
    sys.exit("정상 종료")

########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################


##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################      
class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8, OTTT_sWS_on = False, first_conv = False):
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

        self.OTTT_sWS_on = OTTT_sWS_on
        self.first_conv = first_conv 

        if (self.OTTT_sWS_on == True):
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time dimension {Time} should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        WS_weight = self.weight
        if (self.OTTT_sWS_on == True):
            fan_in = np.prod(self.weight.shape[1:])
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
            WS_weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
            WS_weight = WS_weight * self.gain



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
            
            if (self.first_conv == True):  
                print('first_conv 확인했냐???\n')
                spike_now = spike[t].detach()
            output_current.append( SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, WS_weight, self.bias, self.stride, self.padding) )
            
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
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.constant_(self.bias, 0)

        # ottt
        nn.init.normal_(self.weight, 0, 0.01)
        nn.init.constant_(self.bias, 0)

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

##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################






## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################
class VGG(nn.Module):

    def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3,
                 lif_layer_v_threshold=0.5, lif_layer_v_decay=0.25, lif_layer_sg_width=1.0):
        super(VGG, self).__init__()

        self.features, out_c = make_layers_nda(cfg, batch_norm, in_c, lif_layer_v_threshold, lif_layer_v_decay, lif_layer_sg_width)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(out_c, num_classes)),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #정규화
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def add_dimension(self, x):
        return add_dimention(x, self.T)
    
    
    def forward(self, x):
        # print('1    ',x.size())
        x = self.add_dim(x) if len(x.shape) == 4 else x
        # print('2      ',x.size())

        x = self.features(x)
        # print('3        ',x.size())
        x = self.avgpool(x)
        # print('4     ',x.size())
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        # print('5      ',x.size())
        x = self.classifier(x)
        # print('6        ',x.size())
        
        x = x.mean(axis=1)
        # x = x.sum(axis=1)
        return x


def make_layers_nda(cfg, batch_norm=True, in_c=3, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
    layers = []
    in_channels = in_c
    i = 0
    for v in cfg:
        # avgpool이면 H,W절반, conv면 H,W유지.  
        # print('i', i, 'v', v)
        i+=1
        if v == 'P':
            layers += [SpikeModule(nn.AvgPool2d(kernel_size=2, stride=2))]
        elif v == 'M':
            layers += [SpikeModule(nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            
            layers += [SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))]
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
            #                                 out_channels=v, 
            #                                 kernel_size=3, 
            #                                 stride=1, 
            #                                 padding=1, 
            #                                 trace_const1=1, 
            #                                 trace_const2=lif_layer_v_decay,
            #                                 TIME=10)]
            # layers += [DimChanger_for_change_0_1()]
            
            
            

            if batch_norm:
                layers += [tdBatchNorm(v)]
            else:
                pass


            layers += [LIFSpike(lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [LIF_layer(v_init=0, 
            #             v_decay=lif_layer_v_decay, 
            #             v_threshold=lif_layer_v_threshold, 
            #             v_reset=0, 
            #             sg_width=lif_layer_sg_width,
            #             surrogate='rough_rectangle',
            #             BPTT_on=True)]
            # layers += [DimChanger_for_change_0_1()]
            

            in_channels = v

    return nn.Sequential(*layers), in_channels



class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        # print('x_seq',x_seq.size())
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        # print('y_shape',y_shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        # print('y_seq',y_seq.size())
        y_shape.extend(y_seq.shape[1:])
        # print('y_shape',y_shape)

        # print('y_seq.view(y_shape)',y_seq.view(y_shape).size())
        return y_seq.view(y_shape)


class SpikeModule(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.ann_module = module

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        a = x.shape
        out = self.ann_module(x.reshape(B * T, *spatial_dims))
        b = out.shape
        
        BT, *spatial_dims = out.shape
        out = out.view(B , T, *spatial_dims).contiguous() # 요소들을 정렬시켜줌.
        return out


def fire_function(gamma):
    class ZIF(torch.autograd.Function): # zero is firing
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).float()
            # gradient를 위해 input을 저장하는 코드인듯 ㅇㅇ
            # 예의주시해봐
            ctx.save_for_backward(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            # forward에서 저장해놨던 input가져오는거임
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).float() / gamma
            # 사각형 형태의 surrogate gradient임.
            # 1/2 0    ----
            # -1/2 0   |  |
            # 1/2 1    ----
            # -1/2 1
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
        super(LIFSpike, self).__init__()
        self.thresh = lif_layer_v_threshold
        self.tau = lif_layer_v_decay
        self.gamma = lif_layer_sg_width

    def forward(self, x):
        mem = torch.zeros_like(x[:, 0])

        spikes = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...] #걍 인덱스별로 각각 덧셈
            spike = fire_function(self.gamma)(mem - self.thresh)
            mem = (1 - spike) * mem #spike나감과 동시에 reset
            spikes.append(spike)

        # print('spikes size',spikes.size())
        # print('torch.stack(spikes,dim=1)', torch.stack(spikes, dim=1).size())
            
        # print('xsize22222!!',torch.stack(spikes, dim=1).size())
        
        return torch.stack(spikes, dim=1)



#     tensor.clone()	새롭게 할당	계산 그래프에 계속 상주
# tensor.detach()	공유해서 사용	계산 그래프에 상주하지 않음
# tensor.clone().detach()	새롭게 할당	계산 그래프에 상주하지 않음

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    # T= 10 시계열 데이터 추가
    return x

## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################