# python libraties
import os, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
data_dir = "~/Downloads/skin-cancer-mnist-ham10000/"
all_image_path = glob(os.path.join(data_dir, "*", "*.jpg"))
imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
}
lesion_type_dict = {
    "nv": "Melanocytic nevi",
    "mel": "dermatofibroma",
    "bkl": "Benign keratosis-like lesions ",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

norm_mean = [0.763038, 0.54564667, 0.57004464]
norm_std = [0.14092727, 0.15261286, 0.1699712]

df_original = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
df_original["path"] = df_original["image_id"].map(imageid_path_dict.get)
df_original["cell_type"] = df_original["dx"].map(lesion_type_dict.get)
df_original["cell_type_idx"] = pd.Categorical(df_original["cell_type"]).codes
print(df_original.head())

# this will tell us how many images are associated with each lesion_id
df_undup = df_original.groupby("lesion_id").count()
# now we filter out lesion_id's that have only one image associated with it
df_undup = df_undup[df_undup["image_id"] == 1]
df_undup.reset_index(inplace=True)
df_undup.head()

# here we identify lesion_id's that have duplicate images and those that have only one image.
def get_duplicates(x):
    unique_list = list(df_undup["lesion_id"])
    if x in unique_list:
        return "unduplicated"
    else:
        return "duplicated"


# create a new colum that is a copy of the lesion_id column
df_original["duplicates"] = df_original["lesion_id"]
# apply the function to this new column
df_original["duplicates"] = df_original["duplicates"].apply(get_duplicates)
df_original.head()

print(df_original["duplicates"].value_counts())

# now we filter out images that don't have duplicates
df_undup = df_original[df_original["duplicates"] == "unduplicated"]
df_undup.shape

# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
y = df_undup["cell_type_idx"]
_, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
df_val.shape

df_val["cell_type_idx"].value_counts()

# This set will be df_original excluding all rows that are in the val set
# This function identifies if an image is part of the train or val set.
def get_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val["image_id"])
    if str(x) in val_list:
        return "val"
    else:
        return "train"


# identify train and val rows
# create a new colum that is a copy of the image_id column
df_original["train_or_val"] = df_original["image_id"]
# apply the function to this new column
df_original["train_or_val"] = df_original["train_or_val"].apply(get_val_rows)
# filter out train rows
df_train = df_original[df_original["train_or_val"] == "train"]
print(len(df_train))
print(len(df_val))

print(df_train["cell_type_idx"].value_counts())
print(df_val["cell_type"].value_counts())

# Copy fewer class to balance the number of 7 classes
data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
for i in range(7):
    if data_aug_rate[i]:
        df_train = df_train.append(
            [df_train.loc[df_train["cell_type_idx"] == i, :]] * (data_aug_rate[i] - 1),
            ignore_index=True,
        )
df_train["cell_type"].value_counts()

# # We can split the test set again in a validation set and a true test set:
# df_val, df_test = train_test_split(df_val, test_size=0.5)
df_train = df_train.reset_index()
df_val = df_val.reset_index()
# df_test = df_test.reset_index()

# feature_extract is a boolean that defines if we are finetuning or feature extracting.
# If feature_extract = False, the model is finetuned and all model parameters are updated.
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


# resnet,vgg,densenet,inception
model_name = "densenet"
num_classes = 7
feature_extract = False
# Initialize the model for this run
model_ft, input_size = initialize_model(
    model_name, num_classes, feature_extract, use_pretrained=True
)
# Define the device:
device = torch.device("cuda:0")
# Put the model on the device:
model = model_ft.to(device)

train_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)
# define the transformation of the val images.
val_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = HAM10000(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
# Same for the validation set:
validation_set = HAM10000(df_val, transform=train_transform)
val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

# we use Adam optimizer, use cross entropy loss as our loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


total_loss_train, total_acc_train = [], []


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.size(0), 'label shape',labels.size(0))
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print(
                "[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]"
                % (epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg)
            )
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)

            val_loss.update(criterion(outputs, labels).item())

    print("------------------------------------------------------------")
    print(
        "[epoch %d], [val loss %.5f], [val acc %.5f]"
        % (epoch, val_loss.avg, val_acc.avg)
    )
    print("------------------------------------------------------------")
    return val_loss.avg, val_acc.avg


epoch_num = 10
best_val_acc = 0
total_loss_val, total_acc_val = [], []
for epoch in range(1, epoch_num + 1):
    loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        print("*****************************************************")
        print(
            "best record: [epoch %d], [val loss %.5f], [val acc %.5f]"
            % (epoch, loss_val, acc_val)
        )
        print("*****************************************************")

fig = plt.figure(num=2)
fig1 = fig.add_subplot(2, 1, 1)
fig2 = fig.add_subplot(2, 1, 2)
fig1.plot(total_loss_train, label="training loss")
fig1.plot(total_acc_train, label="training accuracy")
fig2.plot(total_loss_val, label="validation loss")
fig2.plot(total_acc_val, label="validation accuracy")
plt.legend()
plt.show()
