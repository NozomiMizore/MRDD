#from https://github.com/LGAI-Research/L-Verse
import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    # 生成一个从0到half_dim-1的等差数列，并乘以-emb，然后取指数
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.type_as(timesteps)
    # 计算sin和cos嵌入
    # 先将timesteps张量扩展到 [N, 1] 的形状，然后与 emb 相乘，结果为 [N, half_dim]
    emb = timesteps.float()[:, None] * emb[None, :]
    # 拼接正弦和余弦嵌入，得到的张量形状为 [N, embedding_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果 embedding_dim 是奇数，则在最后一列进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_name='batch'):
    if norm_name == 'group': # 将通道划分为32个组，组内进行归一化
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return torch.nn.BatchNorm2d(in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
                                        
    def forward(self, x):
        # 使用最近邻插值方法上采样x
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode="nearest")
        if self.with_conv:
            x = self.conv(x)         
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
            # 在输入stride为4时创建第二个卷积层，进一步下采样
            if stride == 4:
                self.conv_2 = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)                
        self.stride = stride

    def forward(self, x):
        if self.with_conv:
            # 在右方和下方填充常数0
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            if self.stride == 4: 
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv_2(x)                
        else: # 使用平均池化进行下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # 标准化层1对输入特征进行标准化
        self.norm1 = Normalize(in_channels)
        # 卷积层1: 通道数变为out_channels，其他维度尺寸保持不变
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # temb_proj线性层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        # 标准化层2    
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        # 卷积层2：各维度尺寸保持不变
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            # 两种conv_shortcut都将通道数改为out_channels，并保持其他维度尺寸不变
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        # 第一个标准化和激活
        h = self.norm1(h)
        h = nonlinearity(h)
        # 第一个卷积层
        h = self.conv1(h)

        # 激活并线性映射时间步嵌入后加入到h中
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        # 第二个标准化和激活
        h = self.norm2(h)
        h = nonlinearity(h)
        # dropout
        h = self.dropout(h)
        # 第二个卷积层
        h = self.conv2(h)

        # 如果输入通道数与输出通道数不同，使用conv_shortcut调整x的通道数
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        # 残差连接
        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape

        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        # 计算注意力权重矩阵
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5)) # 缩放
        w_ = torch.nn.functional.softmax(w_, dim=2) # 在k的hw维度上进行softmax

        # attend to values
        v = v.reshape(b,c,h*w)   # b, c, hw
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)
        # 1×1卷积层进行投影
        h_ = self.proj_out(h_)
        # 残差连接
        return x+h_

class Encoder(nn.Module):
    def __init__(self, *, hidden_dim, in_channels, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, 
                 resolution, z_channels, double_z=True, use_attn=False, **ignore_kwargs):
        """
        z_channels: 潜在空间的通道数
        double_z: 是否输出双倍通道
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks # 每个分辨率层的resnet块数
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_attn = use_attn
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.hidden_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution # 当前分辨率
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList() # 包含不同分辨率的下采样块
        # 为每个分辨率层创建块
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            if self.use_attn:
                attn = nn.ModuleList()
            block_in = hidden_dim*in_ch_mult[i_level] # 当前层的输入通道数
            block_out = hidden_dim*ch_mult[i_level] # 当前层的输出通道数
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out  # 更新输入通道数
                if self.use_attn:
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            if self.use_attn:
                down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv) # 下采样层
                curr_res = curr_res // 2 # 更新当前分辨率
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        if self.use_attn:                               
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if self.use_attn:
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        if self.use_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        latent = self.norm_out(h)
        latent = nonlinearity(latent)
        latent = self.conv_out(latent)
        return latent


class Decoder(nn.Module):
    def __init__(self, *, hidden_dim, out_channels, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, use_attn=False, **ignorekwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end # 是否直接返回中间特征
        self.use_attn = use_attn
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = hidden_dim*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        if self.use_attn:                                       
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            if self.use_attn:        
                attn = nn.ModuleList()
            block_out = hidden_dim*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if self.use_attn:                
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            if self.use_attn:           
                up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        if self.use_attn:    
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if self.use_attn:    
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


if __name__ == '__main__':
    pass
    x = torch.rand(10, 1, 32, 32)
    encoder = Encoder(hidden_dim=8, in_channels=1, z_channels=10, ch_mult=[1, 2, 4],
                      num_res_blocks=2, resolution=2, use_attn=False, attn_resolutions=None,
                      double_z=False)
    decoder = Decoder(hidden_dim=8, out_channels=1, in_channels=100, z_channels=11, ch_mult=[1, 2, 4],
                      num_res_blocks=2, resolution=2, use_attn=False, attn_resolutions=None,
                      double_z=False)
    to_dist = nn.Linear(10*8*8, 200)
    to_decoder_input = nn.Linear(100+10, 11*8*8)
    latent = encoder(x)
    print(latent.shape) # [10, 10, 8, 8]
    latent = torch.flatten(latent, start_dim=1)
    print(latent.shape) # [10, 640]
    z = to_dist(latent)
    print(z.shape) # [10, 200]
    mu, logvar = torch.split(z, 100, dim=1) # 均值和对数方差
    print(mu.shape) # [10, 100]
    print(logvar.shape) # [10, 100]
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) # 从标准正态分布中采样的噪声
    new_z = eps * std + mu # 线性变换得到所需分布
    print(new_z.shape) # [10, 100]
    y = torch.rand(10, 10)
    new_z = torch.cat([new_z, y], dim=1)
    print(new_z.shape) # [10, 110]
    new_z = to_decoder_input(new_z).view(-1, 11, 8, 8)
    print(new_z.shape) # [10, 11, 8, 8]
    x_recon = decoder(new_z)
    print(x_recon.shape) # [10, 1, 32, 32]
    # print(latent.shape, z.shape, x_recon.shape)