"""
        spec(200, 300) --> transformer --> alexnet --> 128 ----------------------------
    /                                                   |                              \ 
wav -   mfcc(40, 300)  --> transformer --> lstm -----> 128 ---------------------------------------> 384 --> 256 --> 128 --> 4
    \                                                   | X                            /
        wav2vec(149,768) --> transformer ---------> (256, 256) --> (256) --> 128 ------

"""


"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import  Wav2Vec2Model
from models.ser_spec import SER_AlexNet
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
import os
from models.transformers_encoder.transformer import TransformerEncoder as TransformerEncoder_traditional
from models.transformers_encoder_RoPE.transformer import TransformerEncoder as TransformerEncoder_RoPE
from models.transformers_encoder_RoPE_Fib.transformer import TransformerEncoder as TransformerEncoder_RoPE_Fib
from models.transformers_encoder_APE.transformer import TransformerEncoder as TransformerEncoder_APE
import torchvision
"""
        spec(200, 300) --> transformer --> ViT --> 128 ----------------------------
    /                                                   |                              \ 
wav -   mfcc(40, 300)  --> transformer --> ViT -----> 128 ---------------------------------------> 384 --> 256 --> 128 --> 4
    \                                                   | X                            /
        wav2vec(149,768) --> transformer ---------> (256, 256) --> (256) --> 128 ------

"""

"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import  Wav2Vec2Model
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
import os
from models.transformers_encoder.transformer import TransformerEncoder as TransformerEncoder_traditional
from models.transformers_encoder_RoPE.transformer import TransformerEncoder as TransformerEncoder_RoPE
from models.transformers_encoder_RoPE_Fib.transformer import TransformerEncoder as TransformerEncoder_RoPE_Fib
from models.transformers_encoder_APE.transformer import TransformerEncoder as TransformerEncoder_APE

# __all__ = ['Ser_Model']
class Ser_Model(nn.Module):
    def __init__(self, ratio=None):
        super(Ser_Model, self).__init__()
        
        SPEC_PIC_SIZE = (256, 384)
        SPEC_PATCH_SIZE = (16, 16)
        MFCC_PIC_SIZE = (300, 40)
        MFCC_PATCH_SIZE = (20, 10)
        WAV2VEC_PIC_SIZE = (149, 768)
        WAV2VEC_PATCH_SIZE = (16, 16)
        
        LSTM_MFCC_HIDDEN_SIZE = 256

        # 确定设备
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.device = device
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)
        
        # ViT模型的配置
        vit_config = ViTConfig(hidden_size=384, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        
        # 初始化ViT模型
        self.vit_model_spec = ViTModel(vit_config)
        self.vit_model_mfcc = ViTModel(vit_config)
        self.vit_model_wav = ViTModel(vit_config)
        
        self.wav2vec2_feature_extractor = Wav2Vec2Model.from_pretrained("/home/h666/Zq/2025ICASSP/baseline/CA-MSER/features_extraction/pretrained_model/wav2vec2-base-960h").feature_extractor
        
        self.vision_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(75648, 128)
        
        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(75648, 128)
        
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(75648, 128)
    
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        torch.cuda.empty_cache()
        
        audio_wav = self.wav2vec2_feature_extractor(audio_wav) # [batch, 512, 156]
        
        audio_mfcc = audio_mfcc.unsqueeze(1)  # Add a channel dimension
        audio_mfcc = audio_mfcc.repeat(1, 3, 1, 1)  # Repeat to get 3 channels
        
        audio_wav = audio_wav.unsqueeze(1)  # Add a channel dimension
        audio_wav = audio_wav.repeat(1, 3, 1, 1)  # Repeat to get 3 channels
        
        audio_spec_input = self.vision_transform(audio_spec)
        audio_mfcc_input = self.vision_transform(audio_mfcc)
        audio_wav_input = self.vision_transform(audio_wav)
        
        spec_output = self.vit_model_spec(audio_spec_input).last_hidden_state
        mfcc_output = self.vit_model_mfcc(audio_mfcc_input).last_hidden_state
        wav_output = self.vit_model_wav(audio_wav_input).last_hidden_state
        
        spec_output = spec_output.view(spec_output.size(0), -1) # Flatten the output
        mfcc_output = mfcc_output.view(mfcc_output.size(0), -1) # Flatten the output
        wav_output = wav_output.view(wav_output.size(0), -1) # Flatten the output

        spec_output = self.post_spec_dropout(spec_output)
        audio_spec_p = F.relu(self.post_spec_layer(spec_output), inplace=False)
        
        mfcc_output = self.post_mfcc_dropout(mfcc_output)
        audio_mfcc_p = F.relu(self.post_mfcc_layer(mfcc_output), inplace=False)
        
        wav_output = self.post_wav_dropout(wav_output)
        audio_wav_p = F.relu(self.post_wav_layer(wav_output), inplace=False)
        
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] 
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
        output = {
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        
        return output
    
    def save_pic_wav2vec(self, audio_wav, save_path, device):
        audio_wav = torch.tensor(audio_wav).unsqueeze(0).to(device)  # [1, 48000]
        
        with torch.no_grad():
            audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state  # [batch, 149, 768]
        
        audio_wav_pic = audio_wav.squeeze(0).cpu().detach().numpy()  # [149, 768]
        
        min_val, max_val = np.min(audio_wav_pic), np.max(audio_wav_pic)
        audio_wav_pic_normalized = (audio_wav_pic - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
        
        plt.imsave(save_path, audio_wav_pic_normalized, cmap='jet')
        
        img = plt.imread(save_path)
        img = img[:,:,:3]  # 移除图像的 alpha 通道（如果存在）
        
        grayscale_img = np.dot(img, [0.2989, 0.5870, 0.1140])
        
        single_channel_img = np.array(grayscale_img, dtype=np.float32)
        
        single_channel_img = single_channel_img / 255 * 6 - 3
        
        return single_channel_img
    
    def save_pic_wav2vec_speedup(self, audio_wav, save_path, device):
        try:
            audio_wav = torch.tensor(audio_wav).unsqueeze(0).to(device)  # [1, 48000]
            
            with torch.no_grad():
                audio_wav = self.wav2vec2_model(audio_wav).last_hidden_state  # [batch, 149, 768]
            
            audio_wav_pic = audio_wav.squeeze(0).cpu().detach().numpy()  # [149, 768]
            
            min_val, max_val = np.min(audio_wav_pic), np.max(audio_wav_pic)
            audio_wav_pic_normalized = (audio_wav_pic - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
            
            plt.imsave(save_path, audio_wav_pic_normalized, cmap='jet')
            
            img = plt.imread(save_path)
            img = img[:,:,:3]  # 移除图像的 alpha 通道（如果存在）
            
            grayscale_img = np.dot(img, [0.2989, 0.5870, 0.1140])
            
            single_channel_img = np.array(grayscale_img, dtype=np.float32)
            
            single_channel_img = single_channel_img / 255 * 6 - 3
            
            return single_channel_img
        
        except Exception as e:
            print("Error occurred while processing audio:", e)
