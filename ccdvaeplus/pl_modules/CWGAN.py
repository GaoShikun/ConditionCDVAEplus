import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn import init
from ccdvaeplus.pl_modules.condition_block import ScalarEmbedding, comp_embedding


class Generator(nn.Module):
    def __init__(self, dim=[256, 128, 50], condition_prop=False, condition_formula=True):

        super(Generator, self).__init__()
        # 初始化
        self.in_dim = dim[0]

        if condition_prop:
            self.emb_dim = dim[2]
            self.label_embedding = ScalarEmbedding(
                prop_name='formation_energy_per_atom',
                batch_norm=False,
                no_expansion=False,
                n_basis=50,
                start=-2,
                stop=2,
                trainable_gaussians=False,
                no_mlp=True,
            )
        if condition_formula:
            self.emb_dim = dim[1]
            self.label_embedding  = comp_embedding(self.emb_dim)


        self.model = nn.Sequential(
            # input noise+label
            self.block(self.in_dim+self.emb_dim,512,normalize=False),
            self.block(512, 512),
            self.block(512,1024),
            self.block(1024, 1024),
            #np.prod 输入a是数组，返回指定轴上的乘积，不指定轴默认是所有元素的乘积
            self.block(1024, 512),
            nn.Linear(512, self.in_dim),
            # nn.Tanh(),
        )

        # self.output_adjust = nn.Linear(self.in_dim, self.in_dim)
        # init.xavier_uniform_(self.output_adjust.weight)
        # init.zeros_(self.output_adjust.bias)

    def block(self,in_channels,out_channels,normalize=True):
        layers = []
        layers.append(nn.Linear(in_channels,out_channels))
        if normalize:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append((nn.ReLU()))
        return nn.Sequential(*layers)

    def forward(self,noise,labels):
        # temp2 = labels # 256
        # temp = self.label_embedding(labels) # torch.Size([256, 10])
        input_x = torch.cat((self.label_embedding(labels), noise), -1) # torch.Size([256, 110])
        output_latent = self.model(input_x)
        # output_latent = torch.cat((self.label_embedding(labels), output_latent), -1) # torch.Size([256, 110])

        # output_latent = self.output_adjust(output_latent)
        return output_latent

class Discriminator(nn.Module):
    def __init__(self, dim=[256, 128, 50], condition_prop=False, condition_formula=True):
        '''
        :param latent vector:(batch_size,z_dim)
        :param label:(batch_size,label_dim)
        '''
        super(Discriminator, self).__init__()

        self.in_dim = dim[0]
        if condition_prop:
            self.emb_dim = dim[2]
            self.label_embedding = ScalarEmbedding(
                prop_name='formation_energy_per_atom',
                batch_norm=False,
                no_expansion=False,
                n_basis=50,
                start=-2,
                stop=2,
                trainable_gaussians=False,
                no_mlp=True,
            )

        if condition_formula:
            self.emb_dim = dim[1]
            self.label_embedding = comp_embedding(self.emb_dim)

        self.model = nn.Sequential(
            # input latent+label
            nn.Linear(self.in_dim+self.emb_dim,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1),
        )

    def forward(self, latent, labels):
        input_x = torch.cat((latent, self.label_embedding(labels)), -1)
        output_class = self.model(input_x)
        return output_class

#test
if __name__ == '__main__':
    loss = nn.MSELoss()

    Gen = Generator()
    Dis = Discriminator()

    z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (16, 256))))  #
    g_labels = Variable(torch.randn(16, 1).cuda())  #

    valid = Variable(torch.cuda.FloatTensor(16, 1).fill_(1.0), requires_grad=False)  # torch.Size([64, 1])
    fake = Variable(torch.cuda.FloatTensor(16, 1).fill_(0.0), requires_grad=False)  # torch.Size([64, 1])
    # optim_G.zero_grad()
    # g_img = Generator(z, g_labels)
    # g_class = discriminator(g_img, g_labels)
    # g_loss = loss(g_class, valid)
    # g_loss.backward()
    # optim_G.step()
    # optim_D.zero_grad()
    # d_real_class = discriminator(batch_img, batch_label)
    # d_real_loss = loss(d_real_class, valid)
    # # g_img.detach() 不要忘记detach，不然会报错的，因为前面backward已经释放了
    # d_fake_class = discriminator(g_img.detach(), g_labels)
    # d_fake_loss = loss(d_fake_class, fake)
    # all_loss = (d_fake_loss + d_real_loss) / 2
    # all_loss.backward()
    # optim_D.step()

