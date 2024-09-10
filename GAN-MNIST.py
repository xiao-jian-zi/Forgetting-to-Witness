import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class cGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_upsample = nn.Sequential(nn.Embedding(10,50)
                                           ,nn.Linear(50,49)
                                           ,nn.ReLU(True)
                                           )
        self.noise_upsample = nn.Sequential(nn.Linear(100,6272)
                                           ,nn.LeakyReLU(0.2,True)
                                           )
        self.main = nn.Sequential(nn.ConvTranspose2d(129,128, kernel_size=4, stride=2, padding=1)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.ConvTranspose2d(128,128,4,2,1)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.Conv2d(128,1,kernel_size=3,padding=1)
                                  ,nn.Sigmoid()
                                 )
    
    def forward(self,label,noise):
        label = self.label_upsample(label)
        label = label.view(-1,1,7,7)
        
        noise = self.noise_upsample(noise)
        noise = noise.view(-1,128,7,7)
        
        inputs = torch.cat((noise,label),dim=1)
        
        fakedata = self.main(inputs)
        
        return fakedata

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
trans_mnist = transforms.Compose([
                                  transforms.ToTensor() 
                                  # ,transforms.Normalize((0.1307,), (0.3081,))
])
mnist_dataset_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans_mnist)
split1_indices = list(range(0,10000))
train_subset1 = torch.utils.data.Subset(mnist_dataset_train, split1_indices)

trainloader = torch.utils.data.DataLoader(mnist_dataset_train, batch_size=100, shuffle=True, num_workers=2)

latent_dim = 100  # 潜在空间维度
num_classes = 10  # MNIST数据集中的类别数

generator = cGAN_gen()
generator.to(device)

discriminator = Discriminator()
discriminator.to(device)

classifier = CNNMnist()
classifier.load_state_dict(torch.load('./classifier_path.pth'))
classifier.to(device)
classifier.eval()

criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.BCELoss().to(device)
# criterion2 = nn.BCEWithLogitsLoss().to(device)
# optimizer_G = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0003,weight_decay=0.0001)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0003, weight_decay=0.0001)
from torch.optim import lr_scheduler
scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=25, gamma=0.5)
scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=25, gamma=0.5)

# Train
def train():
    loss_list = []
    num_epochs = 100
    batch_size = 100
    for epoch in tqdm(range(num_epochs),desc='training',unit='epoch'):
        discriminator.train()
        generator.train()
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # train discriminator
            real_labels_D = torch.full((labels.shape[0],), 1., device=device)
            fake_labels_D = torch.full((labels.shape[0],), 0., device=device)
            real_labels_D = real_labels_D.unsqueeze(1)
            fake_labels_D = fake_labels_D.unsqueeze(1)
            z_D = torch.randn(batch_size, latent_dim, device=device)# 生成随机潜在向量和标签
            random_labels_D = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images_D = generator(random_labels_D, z_D)# 生成器生成图像
            fake_output_D = discriminator(fake_images_D)
            optimizer_D.zero_grad()
            real_output_D = discriminator(inputs)
            d_loss = 0.5*criterion2(real_output_D,real_labels_D) + 0.5*criterion2(fake_output_D,fake_labels_D)
            d_loss.backward()
            optimizer_D.step()
            
            # train generator
            real_labels_G = torch.full((labels.shape[0],), 1., device=device)
            real_labels_G = real_labels_G.unsqueeze(1)
            optimizer_G.zero_grad()
            z_G = torch.randn(batch_size, latent_dim, device=device)# 生成随机潜在向量和标签
            random_labels_G = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images_G = generator(random_labels_G, z_G)# 生成器生成图像
            fake_pred_G = classifier(fake_images_G)
            fake_output_G = discriminator(fake_images_G)
            g_loss = 0.5*criterion1(fake_pred_G, random_labels_G) + 0.5*criterion2(fake_output_G,real_labels_G)
            g_loss.backward()
            optimizer_G.step()
            
            
            
        print(g_loss.item(),d_loss.item())  
        loss_list.append([g_loss,d_loss])
        scheduler_D.step()
        scheduler_G.step()
    with open("generator_save_path.pkl", "wb") as file:
        pickle.dump(generator, file)

def show_results(generator, num_images=10):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        # random_labels = torch.randint(0, num_classes, (num_images,), device=device)
        random_labels = torch.arange(num_images,device=device)%10
        fake_images = generator(random_labels, z)
    print(fake_images)
    grid = vutils.make_grid(fake_images, nrow=5, normalize=True)
    
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    train()