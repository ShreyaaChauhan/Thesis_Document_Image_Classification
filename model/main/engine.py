import numpy as np
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import os 
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report, roc_auc_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import random
import config as cfg
import shutil
import matplotlib.pyplot as plt
from data_setup import load_complete_val_dataset
plt.style.use("fivethirtyeight")

class Engine:
    def __init__(self, model, loss_fn, optimizer,classNames, val_transforms):
        self.model = model 
        self.loss_fn = loss_fn 
        self.optimizer = optimizer 
        self.device = cfg.DEVICE
        self.ckpt_interval = cfg.CKPT_INTERVAL
        self.model.to(self.device)
        self.parent_folder_path = os.path.abspath(os.path.dirname(__file__))
        self.train_loader = None 
        self.val_loader = None 
        self.writer = None
        self.is_load_checkpoint = False
        self.total_epochs = 0
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.epochs = cfg.EPOCHS
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
        self.threshold = 0.5
        self.val_transforms = val_transforms
        self.classNames = classNames
        
    def to(self,device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = cfg.DEVICE
            print(
                f"Couldn't send it to {device}, \
                    sending it to {self.device} instead.",
            )
            self.model.to(self.device)
    
    def set_loader(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def set_tensorboard(self, name , folder = "run"):
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")
    
    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # calculating classification accuracy
            # y_pred_class = torch.argmax(torch.softmax(yhat, dim=1), dim=1)
            # acc = (y_pred_class == y).sum().item() / len(yhat)
            #acc = self.accuracy( yhat, y)
            _, out = torch.max(yhat, dim=1)
            acc = torch.tensor(torch.sum(out == y).item())
            total_elements = len(y)
            # if str(self.loss_fn) =='BCEWithLogitsLoss()':
            #     y_hat = ((yhat>0.0)==y).float()
            #     acc = y_hat.mean()*100
            # elif str(self.loss_fn) == 'BCELoss()' or str(self.loss_fn) == 'CrossEntropyLoss()' :
            #     y_hat = (yhat>=self.threshold).float()
            #     acc = (y_hat==y).float().mean()*100
            return loss.item(), acc, total_elements
        return perform_train_step
    
    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()
            yhat= self.model(x)
            loss = self.loss_fn(yhat, y)
            #y_pred_class = yhat.argmax(dim=1)
            #acc = (y_pred_class==y).sum().item()/len(yhat)
            # acc = self.accuracy(yhat, y)
            _, out = torch.max(yhat, dim=1)
            acc = torch.tensor(torch.sum(out == y).item())
            total_elements = len(y)
            # if str(self.loss_fn) =='BCEWithLogitsLoss()':
            #     y_hat = ((yhat>0.0)==y).float()
            #     acc = (y_hat).mean()*100
                
            # elif str(self.loss_fn) == 'BCELoss()':
            #     y_hat = (yhat>=self.threshold).float()
            #     acc = (y_hat==y).float().mean()*100
            return loss.item(), acc, total_elements
        return perform_val_step
    
    def _mini_batch(self, validation = False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step
        else:
            data_loader = self.train_loader
            step_fn = self.train_step
        
        if data_loader is None:
            return None 
        
        mini_batch_loss = []
        mini_batch_accuracy = []
        total_elements_in_batch = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            batch_loss, batch_accuracy, total_elements = step_fn(x_batch, y_batch)
            mini_batch_accuracy.append(int(batch_accuracy.item()))
            mini_batch_loss.append(batch_loss)
            total_elements_in_batch.append(total_elements)
        #print(mini_batch_accuracy, "mini_batch_accuracy")
        #print(sum(mini_batch_accuracy), "sum mini_batch_accuracy")
        #print(len(data_loader.dataset) ,"dataloader length")
        #accuracy = np.mean(mini_batch_accuracy)
        loss = np.mean(mini_batch_loss)
        #accuracy = sum(mini_batch_accuracy)/len(data_loader.dataset)
        accuracy = sum(mini_batch_accuracy)/sum(total_elements_in_batch)
        #no_elements = sum(total_elements_in_batch)
        #print(no_elements, "no_elements")
        return loss, accuracy
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass
    
    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        if not self.is_load_checkpoint:
            try:
                shutil.rmtree(
                    os.path.join(
                        self.parent_folder_path,
                        "checkpoints",
                    ),
                )
            except Exception as e:
                print(e)
        for epoch in range(self.total_epochs, n_epochs):
            self.total_epochs += 1
            loss, accuracy = self._mini_batch(validation=False)
            self.train_loss.append(loss)
            self.train_accuracy.append(accuracy)
            
            with torch.no_grad():
                val_loss, val_accuracy = self._mini_batch(validation=True)
                self.val_loss.append(val_loss)
                self.val_accuracy.append(val_accuracy)
            if epoch % self.ckpt_interval == 0:
                self.save_checkpoint(epoch)
            print(
            f'Epoch: {epoch+1} | '
            f'train_loss: {loss:.4f} | '
            f'train_acc: {accuracy:.4f} | '
            f'test_loss: {val_loss:.4f} | '
            f'test_acc: {val_accuracy:.4f}',
            )
            if self.writer:
                scalars = {"training": loss}
                if val_loss is not None:
                    scalars.update({"validation": val_loss})
                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch,
                )
        
        # y__hat = torch.empty((16,1), dtype=torch.float32)
        # Y = torch.empty((16,1), dtype=torch.float32)
        # for step, (x, y) in enumerate(val_data_loader):
        #     y_hat = torch.as_tensor(self.predict(x)>=self.threshold).float()
        #     y__hat = torch.cat((y__hat, y_hat), 0)
        #     Y = torch.cat((Y, y), 0)
        # y__hat = y__hat[16:].float().numpy()
        # Y =  Y[16:].float().numpy()
        self.plot_confusion_matrix()
        # print("classification report is \n",classification_report(Y, y__hat))
        #self.roc_auc_score_multiclass(Y, y__hat)
        
        self.save_checkpoint(epoch, LATEST=True)
        if self.writer:
            self.writer.close()
    
    def accuracy(self, pred, label):
        _, out = torch.max(pred, dim=1)
        return torch.tensor(torch.sum(out == label).item()/len(pred))
    
    def save_checkpoint(self, epoch, LATEST: bool = False):
        ckpt_dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "checkpoints",)
        os.makedirs(ckpt_dir_path, exist_ok=True)
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.train_loss,
            "val_loss": self.val_loss,
            "accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy
        }
        checkpoint_name = f"epoch_{epoch}.pth" if not LATEST else "latest.pth"
        filename = os.path.join(ckpt_dir_path, checkpoint_name)
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, filename):
        self.is_load_checkpoint = True
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.train_loss = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]
        self.train_accuracy = checkpoint["accuracy"]
        self.val_accuracy = checkpoint["val_accuracy"]
        self.model.train()
        
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()
    
    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_loss, label="Training Loss", c="b")
        if self.val_loader:
            plt.plot(self.val_loss, label="Validation Loss", c="r")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.parent_folder_path, "plot_losses"))
    
    def plot_accuracy(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_accuracy, label="Training Accuracy", c="b")
        if self.val_loader:
            plt.plot(self.val_accuracy, label="Validation Accuracy", c="r")

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.parent_folder_path, "plot_accuracy"))
    def print(self):
        print(self.parent_folder_path)
    
    def count_parameters(self):
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )  # noqa
     
    def plot_confusion_matrix(self):
        data_loader = self.val_loader
        y_pred = []
        y_label = []
        correct = 0
        with torch.no_grad():
            for X_test, y_test in data_loader:
                X_test = X_test.to(cfg.DEVICE)  
                y_test = y_test.to(cfg.DEVICE)
                y_hat = self.model(X_test)
                predicted = torch.max(y_hat,1)[1]
                y_pred +=  predicted.tolist()
                y_label += y_test.tolist()
        for i, (y_hat, y) in enumerate(zip(y_pred, y_label)):
            if y_hat==y:
                correct +=1
        print(f'Test accuracy: {correct}/{len(y_hat)} = {correct*100/len(y_hat):7.3f}%')
        cm = confusion_matrix(y_pred,y_label)
        print("classification report is \n",classification_report(y_pred,y_label))
        df_cm = pd.DataFrame(cm, index=self.classNames, columns=self.classNames)
        fig = plt.figure(figsize = (9,6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        # plt.show()
        fig.savefig(os.path.join(self.parent_folder_path, "plot_confusion_matrix"))
        
        # val_data_loader, class_names, len_val = load_complete_val_dataset(testDir=cfg.VAL_DIR, valTransforms=cfg.VAL_TRANSFORMS)
        # with torch.no_grad():
        #     correct = 0
        #     for X_test, y_test in val_data_loader:
        #         X_test = X_test.to(cfg.DEVICE)  
        #         y_test = y_test.to(cfg.DEVICE)
        #         y_hat = self.model(X_test)
        #         _, out = torch.max(y_hat, dim=1)
        #         predicted = torch.max(y_hat,1)[1]
        #         correct += (predicted == y_test).sum()

        # print(f'Test accuracy: {correct.item()}/{len_val} = {correct.item()*100/len_val:7.3f}%')
        # cm = confusion_matrix(y_test.view(-1).detach().cpu().numpy(), predicted.view(-1).detach().cpu().numpy())
        # print("classification report is \n",classification_report(y_test.view(-1).detach().cpu().numpy(), predicted.view(-1).detach().cpu().numpy()))
        
        # df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        # fig = plt.figure(figsize = (9,6))
        # sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        # plt.xlabel("prediction")
        # plt.ylabel("label (ground truth)")
        # # plt.show()
        # fig.savefig(os.path.join(self.parent_folder_path, "plot_confusion_matrix"))
        
        
    def split_cm(self,cm):
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        true_negative = cm[0][0]
        false_positive = cm[0][1]
        false_negative = cm[1][0]
        true_positive = cm[1][1]
        return TN, FP, FN, TP, true_negative, false_positive, false_negative, true_positive
    
    # def accuracy(self,y_true, y_pred):
    #     return np.count_nonzero(y_true==y_pred)/len(y_pred)
    
    def roc_auc_score_multiclass(self, actual_class, pred_class, average = "macro"):
        actual_class = actual_class.tolist()
        pred_class = pred_class.tolist()
        #creating a set of all the unique classes using the actual class list
        unique_class = set(('1','0'))
        roc_auc_dict = {}
        for per_class in unique_class:
            
            #creating a list of all the classes except the current class 
            other_class = [x for x in unique_class if x != per_class]

            #marking the current class as 1 and all other classes as 0
            new_actual_class = [0 if x in other_class else 1 for x in actual_class]
            new_pred_class = [0 if x in other_class else 1 for x in pred_class]

            #using the sklearn metrics method to calculate the roc_auc_score
            roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
            roc_auc_dict[per_class] = roc_auc

        return roc_auc_dict
    
      
      
      
        
        
# X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=13)
# sc = StandardScaler()
# sc.fit(X_train)

# X_train = sc.transform(X_train)
# X_val = sc.transform(X_val)

# torch.manual_seed(13)

# # Builds tensors from numpy arrays
# x_train_tensor = torch.as_tensor(X_train).float()
# y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

# x_val_tensor = torch.as_tensor(X_val).float()
# y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

# # Builds dataset containing ALL data points
# train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# # Builds a loader of each set
# train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16)


# # loss_fn_logits = nn.BCEWithLogitsLoss(reduction='mean')
# lr = 0.1
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('hidden', nn.Linear(2,10))
# model.add_module('activation', nn.ReLU())
# model.add_module('output',nn.Linear(10,1))
# model.add_module('sigmoid', nn.Sigmoid())

# optimizer = optim.SGD(model.parameters(), lr = lr)
# #loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.BCELoss()
# n_epochs = 100
# sbs = engine(model, loss_fn, optimizer)
# sbs.set_loader(train_loader, val_loader)
# sbs.set_tensorboard('classy')
# sbs.train(n_epochs=100)
# sbs.plot_losses()
# sbs.plot_accuracy()





'''
torch.manual_seed(42)
model1 = nn.Sequential()
model1.add_module('linear', nn.Linear(2, 1))
model1.add_module('sigmoid', nn.Sigmoid())
print(model1.state_dict())
'''
# true_b = 1
# true_w = 2
# N = 100      
       
# #  % data prepration 
# torch.manual_seed(13)
# np.random.seed(42)
# x = np.random.rand(N, 1)
# y = true_b + true_w * x + (.1 * np.random.randn(N, 1))
# x_tensor = torch.as_tensor(x).float()
# y_tensor = torch.as_tensor(y).float()
# dataset = TensorDataset(x_tensor, y_tensor)
# ratio = .8
# n_total = len(dataset)
# n_train = int(n_total * ratio)
# n_val = n_total - n_train
# train_data, val_data = random_split(dataset, [n_train, n_val])
# train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
# val_loader = DataLoader(dataset=val_data, batch_size=16)

# #  % model configuration
# lr = 0.1
# torch.manual_seed(42)
# model = nn.Sequential(nn.Linear(1, 1))
# optimizer = optim.SGD(model.parameters(), lr=lr)
# loss_fn = nn.MSELoss(reduction='mean')

# # % model training
# sbs = engine(model, loss_fn, optimizer)
# sbs.set_loader(train_loader, val_loader)
# sbs.set_tensorboard('classy')
# sbs.train(n_epochs=100)
# print(model.state_dict()) # remember, model == sbs.model
# print(sbs.total_epochs)
# sbs.plot_losses()
# sbs.plot_accuracy()
# new_data = np.array([.5]).reshape(-1,1)
# predictions= sbs.predict(new_data)
# sbs.load_checkpoint('/Users/shreyachauhan/Thesis_Document_Image_Classification/model/main/checkpoints/epoch_0.pth')
# print(sbs.model.state_dict())
# # model = "abc"
# # loss_fn = "ghk"
# # optimizer = "kjkj"
# # abc = engine(model, loss_fn, optimizer)
# # abc.print()
            
        
    