import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import Net


IMG_FOLDER = 'data/data/'

# Transformatin on images
transform = transforms.Compose([transforms.Resize((15,15)),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder(IMG_FOLDER, transform = transform)

# split the data into train and test
train_set, test_set = torch.utils.data.random_split(dataset, [int(.8 * len(dataset)), int(.2 * len(dataset))])

# use train and test loader
trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

net = Net()

# optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Trainig loop
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = 'model.pth'
torch.save(net.state_dict(), PATH)