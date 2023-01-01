import pytorch_lightning as pl
from pytorch_lightning.core.hooks import ModelHooks
import torchmetrics
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import time
import pandas as pd
from a_dataloaders import DataModule
from models import NiN, ResNet, DenseNet, NetworkInNetwork
import onnxruntime as ort
import onnx
import os
import re
from pytorch_lightning.loggers import TensorBoardLogger
from c_infer import infer_predict
from sklearn.metrics import classification_report
import yaml

NUM_EPOCHS = 10
LEARNING_RATE = 0.005
NUM_CLASSES=10
DATA_PATH = './cifar10'
idx_label = {
    0: 'airplane', 
    1: 'automobile', 
    2: 'bird', 
    3: 'cat', 
    4: 'deer', 
    5: 'dog', 
    6: 'frog', 
    7: 'horse', 
    8: 'ship', 
    9: 'truck'}

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES) 
        

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits, probas = self(features)
        # logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc.update(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)
  

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def runner(torch_model):
    data_module = DataModule(data_path=DATA_PATH)
    lightning_model = LightningModel(torch_model, learning_rate=LEARNING_RATE)
    checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor='valid_acc', mode='max',)
    logger = TensorBoardLogger("logs_NiN/", name="NiN_cifar10")
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=checkpoint_callback,
        enable_progress_bar=True,
        devices='auto',
        logger=logger,
        accelerator='auto',    
        log_every_n_steps=10,)
    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)
    best_path = trainer.checkpoint_callback.best_model_path
    print(f"best path = {best_path}")
    end_time = time.time()
    runtime = (end_time - start_time) / 60
    print(f"Training lasted for {runtime:.2f} min in total for epochs={NUM_EPOCHS}")
    for k, v in trainer.logged_metrics.items():
        print(k, v)  
    lightning_model = LightningModel.load_from_checkpoint(best_path, model=torch_model)
    return data_module, trainer, lightning_model


def export_to_onnx(model):
    text = str(type(model))
    pattern = re.compile(r"\.\s*(\w+)")
    b = re.findall(pattern, text)
    model_name = 'Exported' + b[0] + '.' + 'onnx'
    torch.onnx.export(model, 
                  torch.rand(256, 3, 32, 32),
                  model_name, 
                  input_names=['image'], 
                  output_names=['label'], 
                  dynamic_axes={'image': {0: 'batch_size'}},
                  verbose=False)   
    return model_name

if __name__ == "__main__":
    torch_model = NetworkInNetwork(num_classes=NUM_CLASSES)
    data_module, trainer, lightning_model = runner(torch_model)
    trainer.test(model=lightning_model, datamodule=data_module, ckpt_path='best')
    model_onnx = export_to_onnx(lightning_model)
    print(f"model exported! Name = {model_onnx}")
    data_module.prepare_data()
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    all_true_labels = []
    all_predicted_labels = []
    for batch in test_dataloader:
        image, label_batch_true =  batch
        predictions_idx, predictions_label = infer_predict(model_onnx, image)
        all_predicted_labels.append(torch.tensor(predictions_idx))
        all_true_labels.append(label_batch_true)
    all_predicted_labels = torch.cat(all_predicted_labels)
    all_true_labels = torch.cat(all_true_labels)
    test_acc = torch.mean((all_predicted_labels == all_true_labels).float())
    print('#######################################')
    print(f'Test accuracy: {test_acc * 100:.2f} %')
    print('#######################################')
    print(classification_report(all_predicted_labels.numpy(), all_true_labels.numpy(), target_names=list(idx_label.values())))
    print('#######################################')
    metrics_report = classification_report(all_predicted_labels.numpy(), all_true_labels.numpy(), target_names=list(idx_label.values()), output_dict=True)
    test_acc = metrics_report.pop('accuracy')
    with open('metrics.yaml', 'w') as f:
        yaml.dump(metrics_report, f)
        f.write('\n')
        yaml.dump({'val_accuracy': test_acc}, f)
    print('Metrics are written to metrics.yaml successfully!')
    
    # torch_model = ResNet(num_classes=NUM_CLASSES)
    # torch_model = ResNet(num_classes=NUM_CLASSES)
    # torch_model = torch.hub.load('pytorch/vision:v0.11.0', 'mobilenet_v2', pretrained=False)
    # torch_model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=10)
    # torch_model = DenseNet(num_classes=NUM_CLASSES)    