import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {d: i for i, d in enumerate(sorted(self.class_dirs))}
        self.images = []
        self.labels = []

        for class_dir in self.class_dirs:
            class_path = os.path.join(self.root_dir, class_dir)
            for image_name in os.listdir(class_path):
                if image_name.endswith('.pgm'):
                    self.images.append(os.path.join(class_path, image_name))
                    self.labels.append(self.class_to_idx[class_dir])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 归一化
])

dataset = FaceDataset(root_dir='./data', transform=transform)
test_list = []
train_list = []

for data in dataset:
    if data[1] < 4 :
        train_list.append(data)
print(len(train_list))

trainloader = DataLoader(dataset, batch_size=40, shuffle=True)



class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super(SimpleCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  
        self.tanh = nn.Tanh()  
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(32 * 32 * 32, 512) 
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = x.view(-1, 32 * 32 * 32) 
        x = self.dropout(torch.relu(self.fc1(x))) 
        x = self.fc2(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 64, 64
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 32, 32
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 32, 32
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 16, 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),  # batch, 128, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 128, 8, 8
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 1024),
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
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_upsample = nn.Sequential(nn.Embedding(40,1*4*4)
                                           # ,nn.Linear(50,1*8*8)
                                           ,nn.ReLU(True)
                                           )
        self.noise_upsample = nn.Sequential(nn.Linear(100,99*4*4)
                                           ,nn.LeakyReLU(0.2,True)
                                           )
        self.main = nn.Sequential(nn.ConvTranspose2d(100,512, kernel_size=4, stride=2, padding=1)# 512 8 8 
                                 ,nn.BatchNorm2d(512)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.ConvTranspose2d(512,256, kernel_size=4, stride=2, padding=1)# 256 16 16
                                 ,nn.BatchNorm2d(256)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1)# 128 32 32
                                 ,nn.BatchNorm2d(128)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1)# 64 64 64
                                 ,nn.BatchNorm2d(64)
                                 ,nn.LeakyReLU(0.2,True)
                                 ,nn.ConvTranspose2d(64,1, kernel_size=4, stride=2, padding=1)# 1 128 128
                                 ,nn.AvgPool2d(2, stride=2)
                                 ,nn.Tanh()
                                 )
    
    def forward(self,label,noise):
        label = self.label_upsample(label)
        label = label.view(-1,1,4,4)
        
        noise = self.noise_upsample(noise)
        noise = noise.view(-1,99,4,4)
        
        inputs = torch.cat((noise,label),dim=1)
        
        fakedata = self.main(inputs)
        
        return fakedata

classifier = SimpleCNNClassifier(num_classes=40).cuda()

classifier.load_state_dict(torch.load('./classifier_path.pth'))
classifier.to(device)
classifier.eval()

discriminator = Discriminator()
discriminator.to(device)

generator = Generator()
generator.to(device)

latent_dim = 100  
num_classes = 40  

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
# 训练GAN网络
def train():
    loss_list = []
    num_epochs = 400
    batch_size = 40
    for epoch in tqdm(range(num_epochs),desc='training',unit='epoch'):
        discriminator.train()
        generator.train()
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            for _ in range(5):
                #  train discriminator
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
            
            for _ in range(5):
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
        if (epoch+1) % 100 ==0 :
            with open('./AT-all-'+str(epoch)+'.pkl', "wb") as file:
                pickle.dump(generator, file)
        # scheduler_D.step()
        # scheduler_G.step()

if __name__ == '__main__':
    train()