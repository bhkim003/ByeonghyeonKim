
import torch
import torch.nn as nn

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *

import modules.neuron as neuron

class SSBH_DimChanger_for_fc(nn.Module):
    def __init__(self):
        super(SSBH_DimChanger_for_fc, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
class SSBH_DimChanger_for_conv1(nn.Module):
    def __init__(self, out_channels):
        super(SSBH_DimChanger_for_conv1, self).__init__()
        self.out_channels = out_channels
    def forward(self, x):
        x = x.view(x.size(0), self.out_channels, -1)
        return x
class SSBH_DimChanger_for_unsuqeeze(nn.Module):
    def __init__(self, dim=1):
        super(SSBH_DimChanger_for_unsuqeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x
class SSBH_DimChanger_for_suqeeze(nn.Module):
    def __init__(self, dim=1):
        super(SSBH_DimChanger_for_suqeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        x = x.squeeze(self.dim)
        return x
class SSBH_size_detector(nn.Module):
    def __init__(self):
        super(SSBH_size_detector, self).__init__()

    def forward(self, x):
        print(x.size())
        return x
class SSBH_L2NormLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(SSBH_L2NormLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        # return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        x = x / (norm + self.eps)
        return x
    
class SSBH_DimChanger_one_two(nn.Module):
    def __init__(self):
        super(SSBH_DimChanger_one_two, self).__init__()

    def forward(self, x):
        dims = list(range(x.dim()))  # 전체 차원의 인덱스 리스트
        dims[0], dims[1] = dims[1], dims[0]  # 첫 번째와 두 번째 차원 교체
        return x.permute(*dims)
    
class SSBH_DimChanger_for_one_two_coupling(nn.Module):
    def __init__(self, TIME):
        super(SSBH_DimChanger_for_one_two_coupling, self).__init__()
        self.TIME = TIME

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        assert T == self.TIME
        x = x.reshape(T * B, *spatial_dims)
        return x   
class SSBH_DimChanger_for_one_two_decoupling(nn.Module):
    def __init__(self, TIME):
        super(SSBH_DimChanger_for_one_two_decoupling, self).__init__()
        self.TIME = TIME

    def forward(self, x):
        TB, *spatial_dims = x.shape
        T = self.TIME 
        B = TB // T
        assert TB == T*B
        x = x.reshape(T, B, *spatial_dims)
        return x
class SSBH_activation_watcher(nn.Module):
    def __init__(self):
        super(SSBH_activation_watcher, self).__init__()

    def forward(self, x):
        print(x.size())
        print(x)
        return x









# Autoencoder 모델 정의
class SAE_fc_only(nn.Module):
    def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], in_channels=1, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False):
        super(SAE_fc_only, self).__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.in_channels = in_channels
        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.need_bias = need_bias
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last

        assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.in_channels]

        self.encoder = []
        self.decoder = []

        past_channel = self.in_channels

        # self.encoder.append(SSBH_size_detector())
        self.encoder += [SSBH_DimChanger_one_two()]
        if self.lif_add_at_first:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
        # self.encoder.append(SSBH_size_detector())
        for en_i in range(len(self.encoder_ch)):
            # self.encoder += [SSBH_size_detector()]
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
            # self.encoder += [SYNAPSE_FC_BPTT(in_features=past_channel,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                 out_features=self.encoder_ch[en_i], 
            #                 trace_const1=self.synapse_fc_trace_const1,  #BPTT에선 안 씀
            #                 trace_const2=self.synapse_fc_trace_const2, #BPTT에선 안 씀
            #                 TIME=self.TIME)]
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_L2NormLayer()] 
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
        
        # self.encoder.append(SSBH_size_detector())

        self.encoder += [SSBH_DimChanger_one_two()]
        self.encoder = nn.Sequential(*self.encoder)


        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            # self.decoder += [SSBH_size_detector()]
            self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
            # self.decoder += [SYNAPSE_FC_BPTT(in_features=past_channel,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                 out_features=self.decoder_ch[de_i], 
            #                 trace_const1=self.synapse_fc_trace_const1,  #BPTT에선 안 씀
            #                 trace_const2=self.synapse_fc_trace_const2, #BPTT에선 안 씀
            #                 TIME=self.TIME)]
            self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

            if self.lif_add_at_last == True:
                self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
                                                v_decay=self.v_decay, 
                                                v_threshold=self.v_threshold, 
                                                v_reset=self.v_reset, 
                                                sg_width=self.sg_width,
                                                surrogate=self.surrogate,
                                                BPTT_on=self.BPTT_on)]
            else:
                if de_i != len(self.decoder_ch)-1:
                    self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
                                                    v_decay=self.v_decay, 
                                                    v_threshold=self.v_threshold, 
                                                    v_reset=self.v_reset, 
                                                    sg_width=self.sg_width,
                                                    surrogate=self.surrogate,
                                                    BPTT_on=self.BPTT_on)]
                    
            # self.decoder.append(SSBH_size_detector())
            past_channel = self.decoder_ch[de_i]
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    











# Autoencoder 모델 정의
class SAE_conv1(nn.Module):
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False):
        super(SAE_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last


        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        # self.encoder += [SSBH_size_detector()]
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_L2NormLayer()] 
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
        # self.encoder += [SSBH_activation_watcher()]

        self.encoder += [SSBH_DimChanger_one_two()]
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)


        self.length_save = self.length_save[::-1]

        print('conv length',self.length_save)

        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_one_two()]
        
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
                                        v_decay=self.v_decay, 
                                        v_threshold=self.v_threshold, 
                                        v_reset=self.v_reset, 
                                        sg_width=self.sg_width,
                                        surrogate=self.surrogate,
                                        BPTT_on=self.BPTT_on)]
        
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))
        self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # self.decoder.append(SSBH_size_detector())
        
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels # 1

            # self.decoder.append(SSBH_size_detector())
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

            if self.lif_add_at_last == True:
                self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
                                                v_decay=self.v_decay, 
                                                v_threshold=self.v_threshold, 
                                                v_reset=self.v_reset, 
                                                sg_width=self.sg_width,
                                                surrogate=self.surrogate,
                                                BPTT_on=self.BPTT_on)]
            else: 
                if de_i != len(self.decoder_ch)-1:
                    self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
                                                    v_decay=self.v_decay, 
                                                    v_threshold=self.v_threshold, 
                                                    v_reset=self.v_reset, 
                                                    sg_width=self.sg_width,
                                                    surrogate=self.surrogate,
                                                    BPTT_on=self.BPTT_on)]
        # self.decoder.append(SSBH_size_detector())
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=2))
        
        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    














# Autoencoder 모델 정의
class Autoencoder_only_FC(nn.Module):
    def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], n_sample=50, need_bias=False,
                 l2norm_bridge=True):
        super(Autoencoder_only_FC, self).__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.n_sample = n_sample
        self.need_bias = need_bias
        self.l2norm_bridge = l2norm_bridge

        assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.n_sample]
        
        
        self.encoder = []
        past_channel = self.n_sample
        for en_i in range(len(self.encoder_ch)):
            self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
            if en_i != len(self.encoder_ch)-1:
                self.encoder += [nn.ReLU()]
            past_channel = self.encoder_ch[en_i]
        
        # 노말라이즈 안 할 거면 빼
        if self.l2norm_bridge:
            self.encoder.append(SSBH_L2NormLayer())
        # else:
        #     self.encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        past_channel = self.encoder_ch[-1]
        for de_i in range(len(self.decoder_ch)):
            self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
            if de_i != len(self.decoder_ch)-1:
                self.decoder += [nn.ReLU()]
            past_channel = self.decoder_ch[de_i]

        self.decoder = nn.Sequential(*self.decoder)

        # Xavier 초기화 적용
        self.init_type_fc = "uniform"
        # self._initialize_weights() # 일단 이거 주석처리.

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def _initialize_weights(self):
        """
        다양한 초기화 방법을 선택할 수 있도록 옵션 제공
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type_fc == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_fc}")












# Autoencoder 모델 정의 (1D Convolution 기반)
class Autoencoder_conv1(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, need_bias = False,
                l2norm_bridge=True):
        super(Autoencoder_conv1, self).__init__()
        assert input_channels == 1
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.l2norm_bridge = l2norm_bridge
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)
        

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 1))
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder.append(nn.ReLU())

        # self.encoder.append(SSBH_size_detector())
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        # self.encoder.append(SSBH_size_detector())

        # 노말라이즈 안 할 거면 빼
        if self.l2norm_bridge:
            self.encoder.append(SSBH_L2NormLayer())
        # else:
        #     self.encoder.append(nn.ReLU())

        # self.encoder.append(SSBH_size_detector())

        self.encoder = nn.Sequential(*self.encoder)

        self.length_save = self.length_save[::-1]
        # Decoder
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels

            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))

            if de_i != len(self.decoder_ch)-1:
                self.decoder.append(nn.ReLU())
            
        
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1))
        self.decoder = nn.Sequential(*self.decoder)
        
        # Xavier 초기화 적용
        # self._initialize_weights()  # 일단 이거 주석처리.

    def forward(self, x):
        # Encoder
        x = self.encoder(x)  # Conv1d를 통해 압축
        # Decoder
        x = self.decoder(x)  # Transposed Conv1d를 통해 복원
        return x


    def _initialize_weights(self):
        """
        다양한 초기화 방법을 선택할 수 있도록 옵션 제공
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if self.init_type_conv == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_conv}")
            elif isinstance(m, nn.Linear):
                if self.init_type_fc == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_fc}")
