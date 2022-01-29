import torch
import torch.optim as optim
from data_load import *
from unet_model import *

epochs = 20
learning_rate = 0.001
min_loss = np.Inf

net = UNet(1, 1, bilinear=True)
net.cuda()

device = torch.device('cuda')

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = DiceLoss()

for epoch in range(1, epochs+1):
    train_loss = 0.0 
    net.train()

    count = 0
    for scan, mask in train_loader:
        scan, mask = scan.cuda(), mask.cuda()
        optimizer.zero_grad()
        output = net(scan)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*scan.size(0)
        count += 1
        print(f"Epoch {epoch} / Batch {count}")

    train_loss /= len(train_loader.sampler)

    valid_loss = 0
    stop_signal = 0
    net.eval()

    for scan, mask in valid_loader:
        scan, mask = scan.cuda(), mask.cuda()
        output = net(scan)
        loss = criterion(output, mask)

        valid_loss += loss.item()*scan.size(0)

    
    valid_loss = valid_loss/len(valid_loader.dataset)
    print('Valid Loss: {:.6f}\n'.format(valid_loss))

    if valid_loss < min_loss:
        torch.save(net.state_dict(), 'Brain Tumour Segmentation\\model.pt')
        print("Improvement. Model Saved")
        min_loss = valid_loss
        stop_signal = 0
    
    else:
        stop_signal += 1

    if (stop_signal == 5):
        break
