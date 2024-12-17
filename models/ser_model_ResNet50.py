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
        
        # # CNN for Spectrogram
        # self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)
        
        # self.post_spec_dropout = nn.Dropout(p=0.1)
        # self.post_spec_layer = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        
        # # LSTM for MFCC        
        # # self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True
        # self.lstm_mfcc = nn.LSTM(input_size=MFCC_PATCH_SIZE[0] * MFCC_PATCH_SIZE[1], hidden_size=LSTM_MFCC_HIDDEN_SIZE, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        # self.post_mfcc_dropout = nn.Dropout(p=0.1)
        # # self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        # self.post_mfcc_layer = nn.Linear(MFCC_PIC_SIZE[0] // MFCC_PATCH_SIZE[0] * MFCC_PIC_SIZE[1] // MFCC_PATCH_SIZE[1] * 2 * LSTM_MFCC_HIDDEN_SIZE, 128)
        
        # # Spectrogram + MFCC  
        # self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        # # self.post_spec_mfcc_att_layer = nn.Linear(256, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        # self.post_spec_mfcc_att_layer = nn.Linear(256, WAV2VEC_PIC_SIZE[0] // WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PIC_SIZE[1] // WAV2VEC_PATCH_SIZE[1])
                        
        # # WAV2VEC 2.0
        # self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/h666/Zq/2025ICASSP/baseline/CA-MSER/features_extraction/pretrained_model/wav2vec2-base-960h")

        # self.post_wav_dropout = nn.Dropout(p=0.1)
        # # self.post_wav_layer = nn.Linear(768, 128) # 512 for 1 and 768 for 2
        # self.post_wav_layer = nn.Linear(WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PATCH_SIZE[1], 128)
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        # self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)
        
        # self.transformer_encoder_spec = TransformerEncoder_traditional(embed_dim=256, num_heads=4, layers=1)
        # self.transformer_encoder_mfcc = TransformerEncoder_traditional(embed_dim=40, num_heads=4, layers=1)
        # self.transformer_encoder_wav2vec = TransformerEncoder_traditional(embed_dim=768, num_heads=4, layers=1)
        
        # self.transformer_encoder_spec = TransformerEncoder_RoPE_Fib(embed_dim=256, num_heads=4, layers=1)
        # self.transformer_encoder_mfcc = TransformerEncoder_RoPE_Fib(embed_dim=40, num_heads=4, layers=1)
        # self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_Fib(embed_dim=768, num_heads=4, layers=1)
        
        # 确定设备
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        self.device = device
        
        # Fib
        # self.transformer_encoder_spec = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=device, ratio=ratio)
        # self.transformer_encoder_mfcc = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        # self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_Fib(num_heads=1, layers=1, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        # APE
        # self.transformer_encoder_spec = TransformerEncoder_APE(num_heads=1, layers=1, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=device, ratio=ratio)
        # self.transformer_encoder_mfcc = TransformerEncoder_APE(num_heads=1, layers=1, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        # self.transformer_encoder_wav2vec = TransformerEncoder_APE(num_heads=1, layers=1, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        # RoPE
        # self.transformer_encoder_spec = TransformerEncoder_RoPE(num_heads=1, layers=1, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=device, ratio=ratio)
        # self.transformer_encoder_mfcc = TransformerEncoder_RoPE(num_heads=1, layers=1, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        # self.transformer_encoder_wav2vec = TransformerEncoder_RoPE(num_heads=1, layers=1, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=device, ratio=ratio)
        
        # self.transformer_post_spec_dropout = nn.Dropout(p=0.1)
        # self.transformer_post_spec_layer = nn.Linear(98304, 128)
        
        
        # # 冻结掉wav2vec的所有参数，探究输出元素的梯度影响
        # for param in self.wav2vec2_model.parameters():
        #     param.requires_grad = False
        
        
        self.wav2vec2_feature_extractor = Wav2Vec2Model.from_pretrained("/home/h666/Zq/2025ICASSP/baseline/CA-MSER/features_extraction/pretrained_model/wav2vec2-base-960h").feature_extractor
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.eval()
        self.base_model_feature_extractor = nn.Sequential(*list(self.base_model.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(2048, 128)
        
        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(2048, 128)
        
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(2048, 128)
    
    # spec(transformer +  alexnet) + mfcc(transformer +  lstm)
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [batch, 48000]
        
        
        self.base_model.eval()
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # # NOTE
        # # 尝试嵌入transformer -- spec
        # audio_spec = self.transformer_encoder_spec(audio_spec) # [batch, 3, 256, 384] --> [batch, 384, 256]
        # audio_spec = torch.unsqueeze(audio_spec, 1) # [batch, 1, 384, 256]
        # audio_spec = audio_spec.repeat(1, 3, 1, 1)   # [batch, 3, 384, 256]
        # # NOTE
        
        # audio_spec, output_spec_t = self.alexnet_model(audio_spec) # [batch, 256, 6, 6], []
        # audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1) # [batch, 256, 36]  
        
        # audio_spec = self.transformer_encoder_spec(audio_spec) # [batch, 256, 384]
        # audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 98304]
        # audio_spec_d = self.transformer_post_spec_dropout(audio_spec_) # [batch, 98304]
        # audio_spec_p = F.relu(self.transformer_post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]
        
        
        # audio -- MFCC with BiLSTM
        # # NOTE
        # # 尝试嵌入transformer -- mfcc
        # audio_mfcc = audio_mfcc.unsqueeze(1) # [batch, 1, 300, 40]
        # audio_mfcc = self.transformer_encoder_mfcc(audio_mfcc) # [batch, 60, 200]
        
        
        # # NOTE
        # audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 60, 512]  
        
        # audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 9216]  
        # audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 9216]  
        # audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  
        
        #+ audio_mfcc = self.att(audio_mfcc)
        # audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 30720]  
        # audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 30720]  
        # audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  
        

        # FOR WAV2VEC2.0 WEIGHTS 
        # spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1) # [batch, 256] 
        # audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)# [batch, 256] 
        # audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=False)# [batch, 432] 
        # audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)# [batch, 1, 432] 
        #+ audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)

        # wav2vec 2.0 
        # # NOTE
        # audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        # # NOTE
        
        # >>> 尝试保存为图片——开始
        # # 单线程
        # path_prefix = "/home/h666/Zq/2025ICASSP/pic/wav/"
        # os.makedirs(path_prefix, exist_ok=True)

        # for i in range(len(audio_wav)):
        #     # 将 [149, 768] 转换为 [768, 149] 以便保存
        #     image = audio_wav[i].detach().cpu().numpy().T
        #     plt.imsave(path_prefix + str(i) + ".png", image, cmap='jet')

        # weights = np.array([0.2989, 0.5870, 0.1140])
        
        # # 读取图片并放回原来的位置
        # for i in range(len(audio_wav)):
        #     # 读取图片
        #     image = plt.imread(path_prefix + str(i) + ".png")
        #     # 将 [768, 149] 转换为 [149, 768]
        #     audio_wav[i] = torch.from_numpy(np.tensordot(weights, image.T[:3], axes=(0, 0))).to(audio_wav.device)
            
            
        # 多线程  —— 效果不好
        # 设置路径前缀
        # path_prefix = "/home/h666/Zq/2025ICASSP/pic/wav/"
        # os.makedirs(path_prefix, exist_ok=True)

        # # 保存图片的函数
        # def save_image(i):
        #     try:
        #         # image = audio_wav[i].detach().cpu().numpy()
        #         tmp = audio_wav[i].clone()
        #         image = tmp.detach().cpu().numpy()
        #         plt.imsave(path_prefix + str(i) + ".png", image, cmap='jet')
        #     except Exception as e:
        #         print(f"Error saving image {i}: {e}")

        # # 多线程保存图片 
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(save_image, range(len(audio_wav)))

        # weights = np.array([0.2989, 0.5870, 0.1140])

        # tmp_audio_wav = np.zeros(audio_wav.shape)
        # # 读取图片并放回原来位置的函数
        # def read_image(i):
        #     try:
        #         image = plt.imread(path_prefix + str(i) + ".png")
        #         # audio_wav[i] = torch.from_numpy(np.tensordot(weights, image.T[:3], axes=(0, 0)))
        #         # audio_wav[i] = torch.from_numpy(image[:, :, 0] * 6 - 2.5)
        #         tmp_audio_wav[i] = np.tensordot(weights, image.T[:3], axes=(0, 0)).T
                
        #         # print("hello")
        #     except Exception as e:
        #         print(f"Error reading image {i}: {e}")

        # # 多线程读取图片
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(read_image, range(len(audio_wav)))
            
        # # audio_wav = audio_wav.to(self.device)
        
        # # print("before: ", audio_wav[0])
        
        # audio_wav.data.copy_(torch.from_numpy(tmp_audio_wav).to(audio_wav.device))
        
        # print()
        # print("latter: ", audio_wav[0])
        
        
        
        # XXX 后悔标记
        
        # 首先，我们需要将 tensor 的值归一化到 [0, 1] 之间
        # audio_wav = (audio_wav - audio_wav.min()) / (audio_wav.max() - audio_wav.min())
        # audio_wav = F.normalize(audio_wav, p=2, dim=(1,2))
        # audio_wav -= torch.tensor(0, device=audio_wav.device)
        

        # # 定义颜色映射的查找表，这里使用 jet 颜色映射的 256 色阶
        # cmap = plt.get_cmap('jet')
        # jet_colormap = torch.tensor(cmap(range(256))[:, :3], dtype=torch.float32, device=audio_wav.device)  # [256, 3]

        # # print(jet_colormap)

        # # 将归一化后的值放大到 0-255 的范围
        # audio_wav = (audio_wav * 255).long()

        # # 利用查找表映射颜色
        # audio_wav = F.embedding(audio_wav, jet_colormap)  # [batch, 149, 768, 3]

        # # 调整维度以匹配图像数据格式
        # audio_wav = audio_wav.permute(0, 3, 1, 2) # [batch, 3, 149, 768]

        
        
        
        
        
        # TODO: 回来测试一下下面这个命令对最终结果的影响
        # audio_wav_numpy =  audio_wav.detach().cpu().numpy()
        # audio_wav = torch.from_numpy(audio_wav_numpy).to(self.device)
        
        
        # # 使用 with torch.no_grad(): 测试梯度的影响
        # with torch.no_grad():
        #     tmp = audio_wav.clone()
        #     audio_wav_numpy = tmp.detach().cpu().numpy()
        #     audio_wav = torch.from_numpy(audio_wav_numpy).to(self.device)
        # # 重新进入梯度计算模式，继续后续操作
        # audio_wav = audio_wav + 0  # 这一步不会改变值，但会确保重新启用计算图
        
        
        # 尝试使用 data.copy_ 函数，有效果！！！
        # tmp = audio_wav.clone()
        # audio_wav.data.copy_(torch.from_numpy(tmp.detach().cpu().numpy()))
        
        
        torch.cuda.empty_cache()
        
        # <<< 尝试保存为图片——结束
        
        # NOTE
        # 尝试嵌入transformer -- wav2vec
        # audio_wav = audio_wav.unsqueeze(1) # [batch, 1, 149, 768]
        # audio_wav = self.transformer_encoder_wav2vec(audio_wav) # [batch, 432, 256]
        
        # NOTE
        
        # audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 256] 
        # audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 256] 
        #audio_wav = torch.mean(audio_wav, dim=1)
        
        # audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 256] 
        # audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False) # [batch, 128] 
        
        
        
        audio_wav = self.wav2vec2_feature_extractor(audio_wav) # [batch, 512, 156]
        
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [batch, 512, 156]
        
        # Transform audio_mfcc to match the input requirements of VGG16
        audio_mfcc = audio_mfcc.unsqueeze(1)  # Add a channel dimension
        audio_mfcc = audio_mfcc.repeat(1, 3, 1, 1)  # Repeat to get 3 channels
        
        # Transform audio_wav to match the input requirements of VGG16
        audio_wav = audio_wav.unsqueeze(1)  # Add a channel dimension
        audio_wav = audio_wav.repeat(1, 3, 1, 1)  # Repeat to get 3 channels
        
        
        audio_spec_input = self.vision_transform(audio_spec)
        audio_mfcc_input = self.vision_transform(audio_mfcc)
        audio_wav_input = self.vision_transform(audio_wav)
        
        
        spec_output = self.base_model_feature_extractor(audio_spec_input)
        mfcc_output = self.base_model_feature_extractor(audio_mfcc_input)
        wav_output = self.base_model_feature_extractor(audio_wav_input)
        
        
        # Apply global average pooling
        spec_output = self.global_avg_pool(spec_output)
        mfcc_output = self.global_avg_pool(mfcc_output)
        wav_output = self.global_avg_pool(wav_output)
        
        
        spec_output = spec_output.view(spec_output.size(0), -1) # Flatten the output
        mfcc_output = mfcc_output.view(mfcc_output.size(0), -1) # Flatten the output
        wav_output = wav_output.view(wav_output.size(0), -1) # Flatten the output

        spec_output = self.post_spec_dropout(spec_output)
        audio_spec_p = F.relu(self.post_spec_layer(spec_output), inplace=False)
        
        mfcc_output = self.post_mfcc_dropout(mfcc_output)
        audio_mfcc_p = F.relu(self.post_mfcc_layer(mfcc_output), inplace=False)
        
        wav_output = self.post_wav_dropout(wav_output)
        audio_wav_p = F.relu(self.post_wav_layer(wav_output), inplace=False)
        
        
        
        
        
        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] 
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
        
        # # combine()
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