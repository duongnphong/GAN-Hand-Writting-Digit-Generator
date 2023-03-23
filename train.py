from model import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import LoadData
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
L_RATE = 0.0002
BATCH_SIZE = 128

NetG = Generator().to(DEVICE)
NetD = Discriminator().to(DEVICE)

optimG = torch.optim.Adam(NetG.parameters(), L_RATE)
optimD = torch.optim.Adam(NetD.parameters(), L_RATE)

train_data = LoadData()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    for idx, (input_gauss, img) in enumerate(train_loader):
        input_gauss = input_gauss.to(DEVICE)
        img = img.to(DEVICE)

        # Train NetD
        NetD.train()
        NetG.eval()
        optimD.zero_grad()

        outD_real = NetD(img)
        labelD_real = torch.ones([img.size(0), 1]).to(DEVICE)
        lossD_real = criterion(outD_real, labelD_real)

        outD_fake = NetD(NetG(input_gauss))
        labelD_fake = torch.zeros([img.size(0), 1]).to(DEVICE)
        lossD_fake = criterion(outD_fake, labelD_fake)

        lossD = lossD_fake + lossD_real
        lossD.backward()
        optimD.step()

        T.ToPILImage()(torch.reshape((NetG(input_gauss)[0] * 0.5) + 0.5, (28, 28))).save("test.png")
        T.ToPILImage()(torch.reshape((NetG(input_gauss)[100] * 0.5) + 0.5, (28, 28))).save("test1.png")

        # Train NetG
        NetG.train()
        NetD.eval()
        optimG.zero_grad()

        outD_fake = NetD(NetG(input_gauss))
        outD_real = NetD(img)

        lossG = criterion(outD_fake, labelD_real)
        lossG.backward()
        optimG.step()
    
    if epoch % 1 == 0:
        print(f"Epoch: {epoch + 1} | LossD : {lossD} | LossG : {lossG}")

    torch.save(NetG.state_dict(), "model.pth")




        




