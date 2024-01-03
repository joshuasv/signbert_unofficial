# python finetune.py --config finetune/configs/ISLR_MSASL.yml --ckpt logs/pretrain/version_0/ckpts/last.ckpt
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

from finetune.ISLR.Head import Head
from signbert.model.PretrainSignBertModelManoTorch import SignBertModel as BaseModel


class SignBertModel(pl.LightningModule):
    """
    A PyTorch Lightning module implementing SignBERT+.

    This class combines a pre-trained base model with a custom head, tailored for
    the specific task of sign language recognition.

    Attributes:
    model (BaseModel): The pre-trained base model.
    head (Head): A custom head added to the base model for sign language recognition.
    train_acc (Accuracy): Metric for tracking training accuracy.
    val_acc (Accuracy): Metric for tracking validation accuracy.
    """

    def __init__(self, ckpt, lr, head_args):
        """
        Initialize the SignBertModel.

        Parameters:
        ckpt (str): Path to the checkpoint of the pre-trained base model.
        lr (float): Learning rate for the optimizer.
        head_args (dict): Arguments for initializing the custom head.
        """
        super().__init__()
        self.lr = lr
        # Load the pre-trained base model from the given checkpoint
        self.model = BaseModel.load_from_checkpoint(ckpt, map_location="cpu")
        self._init_base_model()
        # Determine the input channel size for the custom head based on the base model's output
        ge_hid_dim = self.model.hparams.gesture_extractor_args["hid_dim"]
        in_channels = ge_hid_dim[-1] if isinstance(ge_hid_dim, list) else ge_hid_dim
        # Initialize the custom head for sign language recognition
        self.head = Head(in_channels=in_channels, **head_args)
        # Initialize accuracy metrics for training and validation
        num_classes = head_args["num_classes"]
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _init_base_model(self):
        """
        This method removes unnecessary components from the pre-trained base
        model and freezes its weights to prevent them from being updated during
        training.
        """
        # Remove components from the base model that are not needed for the current task
        del self.model.pg
        del self.model.lhand_hd
        del self.model.rhand_hd
        del self.model.train_pck_20
        del self.model.train_pck_auc_20_40
        del self.model.val_pck_20
        del self.model.val_pck_auc_20_40
        # Freeze the model to prevent updates to its weights during training
        self.model.freeze()

    def forward(self, arms, rhand, lhand):
        """
        Forward pass of the SignBertModel.

        Parameters:
        arms (Tensor): Input tensor for arm keypoints.
        rhand (Tensor): Input tensor for right hand keypoints.
        lhand (Tensor): Input tensor for left hand keypoints.

        Returns:
        Tensor: The output predictions of the model.
        """
        # Concatenate right and left hand data
        x = torch.concat((rhand, lhand), dim=2)
        # Extract hand tokens using the gesture extractor in the base model
        rhand, lhand = self.model.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # Extract arm tokens using the spatial-temporal processing in the base model
        rarm, larm = self.model.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # Add positional tokens
        rhand = rhand + rarm 
        lhand = lhand + larm 
        # Reshape hand data for further processing
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        # Apply positional encoding to both right and left hand data
        rhand = self.model.pe(rhand) 
        lhand = self.model.pe(lhand) 
        # Process both right and left hand data through the transformer encoder
        rhand = self.model.te(rhand)
        lhand = self.model.te(lhand)
        # Pass the data through the custom head for final predictions
        x = self.head(rhand, lhand)

        return x

    def training_step(self, batch):
        """
        The training step for the SignBertModel.

        Processes a single batch of data, computes the loss, updates the model, and logs metrics.

        Parameters:
        batch (dict): A batch of data. Contains tensors for arms, left hand, right hand, and class labels.

        Returns:
        torch.Tensor: The computed loss for the batch.
        """
        # Extract arms, left hand, right hand, and labels from the batch
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        # Forward pass
        logits = self(arms, rhand, lhand)
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        # Compute accuracy
        acc = self.train_acc(logits, labels) 
        # Log loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, dataloader_idx=0):
        """
        The validation step for the SignBertModel.

        Processes a single batch of validation data, computes the loss, evaluates the model's performance, 
        and logs metrics for monitoring.

        Parameters:
        batch (dict): A batch of validation data. Contains tensors for arms, left hand, right hand, and class labels.
        dataloader_idx (int, optional): Index of the dataloader. Default is 0.

        Returns:
        None: This method logs validation loss and accuracy but does not return anything.
        """
        # Extract arms, left hand, right hand, and labels from the validation batch
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        # Forward pass
        logits = self(arms, rhand, lhand)
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        # Compute accuracy
        acc = self.val_acc(logits, labels) 
        # Log loss and accuracy
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer