"""
        spec(200, 300) --> transformer --> 128 ---------------------------------
    /                                               |                              \
wav -   mfcc(40, 300)  --> transformer --> 128 ---------------------------------   ------> 384 --> 256 --> 128 --> 4
    \                                               | X                            /
        wav2vec(149,768) --> transformer --> (432, 256) --> (256) --> 128 ------


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
# from models.transformers_encoder.transformer import TransformerEncoder as TransformerEncoder_traditional
# from models.transformers_encoder_RoPE_2D.transformer import TransformerEncoder as TransformerEncoder_RoPE_2D
from models.transformers_encoder_RoPE_Fib.transformer import TransformerEncoder as TransformerEncoder_RoPE_Fib

# __all__ = ['Ser_Model']
class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        SPEC_PIC_SIZE = (256, 384)
        SPEC_PATCH_SIZE = (16, 16)
        MFCC_PIC_SIZE = (40, 300)
        MFCC_PATCH_SIZE = (10, 20)
        WAV2VEC_PIC_SIZE = (149, 768)
        WAV2VEC_PATCH_SIZE = (16, 16)
        
        # CNN for Spectrogram
        self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(SPEC_PIC_SIZE[0] // SPEC_PATCH_SIZE[0] * SPEC_PIC_SIZE[1] // SPEC_PATCH_SIZE[1] * SPEC_PATCH_SIZE[0] * SPEC_PATCH_SIZE[1], 128)
        
        # LSTM for MFCC        
        self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(MFCC_PIC_SIZE[0] // MFCC_PATCH_SIZE[0] * MFCC_PIC_SIZE[1] // MFCC_PATCH_SIZE[1] * MFCC_PATCH_SIZE[0] * MFCC_PATCH_SIZE[1], 128)
        
        # Spectrogram + MFCC  
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, WAV2VEC_PIC_SIZE[0] // WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PIC_SIZE[1] // WAV2VEC_PATCH_SIZE[1])
                        
        # WAV2VEC 2.0
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/h666/Zq/2025ICASSP/baseline/CA-MSER/features_extraction/pretrained_model/wav2vec2-base-960h")

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PATCH_SIZE[1], 128) 
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        # self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)
        
        # 确定设备
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        
        self.transformer_encoder_spec = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=device)
        self.transformer_encoder_mfcc = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=device)
        self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=device)
        
        # self.transformer_post_spec_dropout = nn.Dropout(p=0.1)
        # self.transformer_post_spec_layer = nn.Linear(98304, 128)
    
    # spec(transformer +  alexnet) + mfcc(transformer +  lstm)
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        # print("audio_spec.shape: ", audio_spec.shape)
        # print("audio_mfcc.shape: ", audio_mfcc.shape)
        # print("audio_wav.shape: ", audio_wav.shape)
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        # # >>> 尝试保存为图片——开始
        # # 保存图片的代码
        # path_prefix = "/home/h666/Zq/2025ICASSP/pic/mfcc/"
        # os.makedirs(path_prefix, exist_ok=True)

        # for i in range(len(audio_mfcc)):
        #     # 将 [300, 40] 转换为 [40, 300] 以便保存
        #     image = audio_mfcc[i].cpu().numpy().T
        #     plt.imsave(path_prefix + str(i) + ".png", image, cmap='jet')

        # # 读取图片并放回原来的位置
        # for i in range(len(audio_mfcc)):
        #     # 读取图片
        #     image = plt.imread(path_prefix + str(i) + ".png")
        #     # 将 [40, 300] 转换为 [300, 40]
        #     audio_mfcc[i] = torch.from_numpy(image.T[0]).to(audio_mfcc.device)
        # # <<< 尝试保存为图片——结束
        
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # spectrogram - SER_CNN
        # # >>> 尝试保存为图片——开始
        # # 1、保存为图片
        # path_prefix = "/home/h666/Zq/2025ICASSP/pic/spec/"
        # os.makedirs(path_prefix, exist_ok=True)

        # for i in range(len(audio_spec)):
        #     # 将 [3, 256, 384] 转换为 [256, 384, 3]
        #     image = audio_spec[i].cpu().numpy().transpose(1, 2, 0)
        #     image = (image - image.min()) / (image.max() - image.min())
        #     plt.imsave(path_prefix + str(i) + ".png", image, cmap='jet')

        # # 2、读取图片
        # # 读取图片并放回原来的位置
        # for i in range(len(audio_spec)):
        #     # 读取图片
        #     image = plt.imread(path_prefix + str(i) + ".png")
        #     # 将 [256, 384, 3] 转换为 [3, 256, 384]
        #     audio_spec[i] = torch.from_numpy(image.transpose(2, 0, 1)[:3]).to(audio_spec.device)
        # # <<< 尝试保存为图片——结束
        
        # NOTE
        # 尝试嵌入transformer -- spec
        # (batch_size, in_channels, height, width)
        # (batch_size, 3, 256, 384) --> (batch_size, 384, 256)
        audio_spec = self.transformer_encoder_spec(audio_spec) # [batch, 384, 256]
        # NOTE
        
        audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 98304]  
        audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 98304]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  
        
        
        # audio -- MFCC with BiLSTM
        # NOTE
        # 尝试嵌入transformer -- spec
        # (batch_size, 300, 40) --> (batch_size, 1, 40, 300) --> (batch_size, 60, 200)
        audio_mfcc = audio_mfcc.permute(0, 2, 1)
        audio_mfcc = torch.unsqueeze(audio_mfcc, 1)
        audio_mfcc = self.transformer_encoder_mfcc(audio_mfcc)
        audio_mfcc = torch.flatten(audio_mfcc, 1) # [batch, 12000]
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc) # [batch, 12000]
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]
        # NOTE
        
        
        # audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 300, 512]  
        # # + audio_mfcc = self.att(audio_mfcc)
        # audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 153600]  
        # audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 153600]  
        # audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  
        

        # FOR WAV2VEC2.0 WEIGHTS 
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1) # [batch, 256] 
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)# [batch, 256] 
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=False)# [batch, 432] 
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)# [batch, 1, 432] 
        # + audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)

        # wav2vec 2.0 
        # # NOTE
        audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        # # NOTE
        
        # >>> 尝试保存为图片——开始
        # # 多线程  —— 效果不好
        # # 设置路径前缀
        # path_prefix = "/home/h666/Zq/2025ICASSP/pic/wav/"
        # os.makedirs(path_prefix, exist_ok=True)

        # # 保存图片的函数
        # def save_image(i):
        #     try:
        #         image = audio_wav[i].detach().cpu().numpy().T  # 将 [149, 768] 转换为 [768, 149]
        #         plt.imsave(path_prefix + str(i) + ".png", image, cmap='jet')
        #     except Exception as e:
        #         print(f"Error saving image {i}: {e}")

        # # 多线程保存图片 
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(save_image, range(len(audio_wav)))

        # weights = np.array([0.2989, 0.5870, 0.1140])

        # # 读取图片并放回原来位置的函数
        # def read_image(i):
        #     try:
        #         image = plt.imread(path_prefix + str(i) + ".png")
        #         audio_wav[i] = torch.from_numpy(np.tensordot(weights, image.T[:3], axes=(0, 0))).to(audio_wav.device)
        #         # print("hello")
        #     except Exception as e:
        #         print(f"Error reading image {i}: {e}")

        # # 多线程读取图片
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(read_image, range(len(audio_wav)))
        # <<< 尝试保存为图片——结束
        
        
        # NOTE
        # 尝试嵌入transformer -- wav2vec
        # (batch_size, 149, 768) --> (batch_size, 1, 149, 768) --> (batch_size, 432, 256)
        audio_wav = torch.unsqueeze(audio_wav, 1)
        audio_wav = self.transformer_encoder_wav2vec(audio_wav)
        # NOTE
        
        
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 256] 
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 256] 
        #audio_wav = torch.mean(audio_wav, dim=1)
        
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 256] 
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False) # [batch, 128] 
        
        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] 
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
        
        ## combine()
        # audio_att = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1)  # [batch, 256] 
        # audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 256] 
        # audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        # audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        # audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        # output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
  
        output = {
            # 'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        

        return output
    
    # NOTE
    def save_pic_wav2vec(self, audio_wav, save_path, device):
        # 将 audio_wav 从 [48000] 转换为 [1, 48000]
        audio_wav = torch.tensor(audio_wav).unsqueeze(0).to(device)  # [1, 48000]
        
        with torch.no_grad():
            # 使用 wav2vec2 模型提取特征，将其转换为 [batch, 149, 768]
            audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state  # [batch, 149, 768]
        
        # 将张量转换为 numpy 数组，并移除 batch 维度 [149, 768]
        audio_wav_pic = audio_wav.squeeze(0).cpu().detach().numpy()  # [149, 768]
        
        # 在保存为图像之前对张量进行归一化处理
        min_val, max_val = np.min(audio_wav_pic), np.max(audio_wav_pic)
        audio_wav_pic_normalized = (audio_wav_pic - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
        
        # 保存归一化后的图像，使用 jet 颜色映射
        plt.imsave(save_path, audio_wav_pic_normalized, cmap='jet')
        
        # 读取保存的图像
        img = plt.imread(save_path)
        img = img[:,:,:3]  # 移除图像的 alpha 通道（如果存在）
        
        # 使用 jet 颜色映射表将 RGB 图像转换为灰度图像
        grayscale_img = np.dot(img, [0.2989, 0.5870, 0.1140])
        
        # 通过反归一化恢复到原始的数据范围，但是最大最小应该不可知
        # grayscale_img = grayscale_img * (max_val - min_val) + min_val
        
        # 转换回 numpy 数组，并确保类型为 float32
        single_channel_img = np.array(grayscale_img, dtype=np.float32)
        
        # 把结果弄到-3 ～ 3之间
        single_channel_img = single_channel_img / 255 * 6 - 3
        
        return single_channel_img
    
    
    def save_pic_wav2vec_speedup(self, audio_wav, save_path, device):
        try:
            # 将 audio_wav 从 [48000] 转换为 [1, 48000]
            audio_wav = torch.tensor(audio_wav).unsqueeze(0).to(device)  # [1, 48000]
            
            with torch.no_grad():
                # 使用 wav2vec2 模型提取特征，将其转换为 [batch, 149, 768]
                audio_wav = self.wav2vec2_model(audio_wav).last_hidden_state  # [batch, 149, 768]
            
            # 如果想要只保存图片，不修改原数据，可以取消注释以下代码，返回的就是一个tensor类型，同时要注意修改限制线程数目，防止显存爆炸
            # return audio_wav.squeeze(0)
            
            # 将张量转换为 numpy 数组，并移除 batch 维度 [149, 768]
            audio_wav_pic = audio_wav.squeeze(0).cpu().detach().numpy()  # [149, 768]
            
            # 在保存为图像之前对张量进行归一化处理
            min_val, max_val = np.min(audio_wav_pic), np.max(audio_wav_pic)
            audio_wav_pic_normalized = (audio_wav_pic - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
            
            # 保存归一化后的图像，使用 jet 颜色映射
            plt.imsave(save_path, audio_wav_pic_normalized, cmap='jet')
            
            # 读取保存的图像
            img = plt.imread(save_path)
            img = img[:,:,:3]  # 移除图像的 alpha 通道（如果存在）
            
            # 使用 jet 颜色映射表将 RGB 图像转换为灰度图像
            grayscale_img = np.dot(img, [0.2989, 0.5870, 0.1140])
            
            # 通过反归一化恢复到原始的数据范围，但是最大最小应该不可知
            # grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img))
            # grayscale_img = grayscale_img * (max_val - min_val) + min_val
            
            # 转换回 numpy 数组，并确保类型为 float32
            single_channel_img = np.array(grayscale_img, dtype=np.float32)
            
            
            
            # 把结果弄到-3 ～ 3之间
            single_channel_img = single_channel_img / 255 * 6 - 3
            
            return single_channel_img
        
        except Exception as e:
            print("Error occurred while processing audio:", e)