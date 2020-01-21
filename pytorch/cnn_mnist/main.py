import network
import torch
from torch.utils import data
import torch.optim as optim
from torchvision import datasets, transforms

if __name__=="__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False)

    data_loader_train = data.DataLoader(dataset=data_train,
                                                    batch_size=64,
                                                    shuffle=True)

    data_loader_test = data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   shuffle=True)

    network = network.Network()
    print(network)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        network = network.to(device)

    for epoch in range(10):
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        for i, data in enumerate(data_loader_train):
            count += 1
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print('i %d', i)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_acc += torch.sum(predicted == labels)
            if i % 200 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / count, running_acc / count))
                running_loss = 0.0
                running_acc = 0.0
                count = 0
        print('[%d] loss: %.3f, accuracy: %.3f' %
              (epoch + 1, running_loss / count, running_acc / count))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader_test:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


