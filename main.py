
from segment_anything import SamPredictor, sam_model_registry, utils

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import glob
from utils import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('PyCharm')


    # Loading the model based on checkpoint
    sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
    predictor = SamPredictor(sam)
    transform = transforms.Compose([
        transforms.Resize((576, 576)),  # Resize to the size a model expects
        transforms.ToTensor()])

    #transform = transforms.Compose(
     #   [#transforms.ToPILImage(),
         #transforms.Resize((576, 576)),
      #   transforms.ToTensor(),
       #  transforms.Normalize((0.5,), (0.5,))
        # ])

    train_images = glob.glob("LE_cand1_largebox_coco/tiles/*jpg")
    train_masks = glob.glob("LE_cand1_largebox_coco/annotations/*000.png")
    print(train_masks)
    print(train_images)

    train_data = CustomDataset(train_images, train_masks, transform)
    dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

    """# Load custom dataset
    dataset = CustomDataset(root_dir='', mask_dir='<path_to_masks>', transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Load custom dataset
    dataset = CustomDataset(root_dir="LE_cand1_largebox_coco/tiles_and_annotations",
                        transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    """
    print(dataloader)
    for i,(data, masks) in enumerate(dataloader):
        print('Batch Index: {}'.format(i))
        print('Shape of data item 1: {}; shape of data item 2: {}'.format(data.shape, masks.shape))
            # i1, i2 = i1.to('cuda:0'), i2.to('cuda:0')
        print('Device of data item 1: {}; device of data item 2: {}\n'.format(data.device, masks.device))
        #print(data1, data2)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Fine-tuning the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.model.to(device)
    predictor.model.train()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(predictor.model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i,(data, masks) in enumerate(dataloader):
            print(data)
            inputs = [{"image":d[j], "masks":masks[j], "original_size":d[j].shape} for j,d in enumerate(data)]
            #inputs = inputs.to(device)
            ## print(inputs)
            #print(inputs.shape)

            optimizer.zero_grad()
            print(inputs)
            outputs = predictor.model(inputs, multimask_output=1)
            outmasks = torch.Tensor(np.array([outputs[i]["masks"]
                                              for i in range(len(outputs))])).long()
            inmasks = torch.Tensor(np.array([inputs[i]["masks"]
                                             for i in range(len(inputs))])).long()
            loss = criterion(outmasks, inmasks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if 1: #i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')