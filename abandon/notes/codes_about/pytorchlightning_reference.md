
## Config

### Trainer

```python
# train on 8 CPUs
num_processes=8

# train on 32 CPUs across 4 machines
num_processes=8, num_nodes=4

# train on 2 GPU
gpus=2

# train on gpu 1, 3, 5 (3 gpus total)
gpus=[1, 3, 5]

# train on multiple GPUs across nodes (32 gpus here)
gpus=4, num_nodes=8

# Multi GPU with mixed precision
gpus=2, precision=16

# Train on TPUs
tpu_cores=8
```

### Debugging

```python
# use only 10 train batches and 3 val batches
trainer = Trainer(limit_train_batches=10, limit_val_batches=3)

# Automatically overfit the sane batch of your model for a sanity test
trainer = Trainer(overfit_batches=1)

# unit test all the code- hits every line of your code once to see if you have bugs,
# instead of waiting hours to crash on validation
trainer = Trainer(fast_dev_run=True)

# train only 20% of an epoch
trainer = Trainer(limit_train_batches=0.2)

# run validation every 25% of a training epoch
trainer = Trainer(val_check_interval=0.25)

# Profile your code to find speed/memory bottlenecks
Trainer(profiler="simple")
```
### Callbacks

```python
from pytorch_lightning.callbacks import Callback

class DecayLearningRate(Callback):

    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = [param_group['lr'] for param_group in optimizer.param_groups]
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.98
                new_lr_group.append(new_lr)
                param_group['lr'] = new_lr
            self.old_lrs[opt_idx] = new_lr_group

# And pass the callback to the Trainer
decay_callback = DecayLearningRate()
trainer = Trainer(callbacks=[decay_callback])
```

## Control

### Manual optimization

Turn off automatic optimization and you control the train loop!

```python
def __init__(self):
    self.automatic_optimization = False

def training_step(self, batch, batch_idx):
    # access your optimizers with use_pl_optimizer=False. Default is True
    opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

    loss_a = self.generator(batch)
    opt_a.zero_grad()
    # use `manual_backward()` instead of `loss.backward` to automate half precision, etc...
    self.manual_backward(loss_a)
    opt_a.step()

    loss_b = self.discriminator(batch)
    opt_b.zero_grad()
    self.manual_backward(loss_b)
    opt_b.step()
```

## Inference

### Predict or Deploy

- Sub-models

```python
# ----------------------------------
# to use as embedding extractor
# ----------------------------------
autoencoder = LitAutoEncoder.load_from_checkpoint('path/to/checkpoint_file.ckpt')
encoder_model = autoencoder.encoder
encoder_model.eval()

# ----------------------------------
# to use as image generator
# ----------------------------------
decoder_model = autoencoder.decoder
decoder_model.eval()
```

- Forward

```python
# ----------------------------------
# using the AE to extract embeddings
# ----------------------------------
class LitAutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential()

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

autoencoder = LitAutoEncoder()
autoencoder = autoencoder(torch.rand(1, 28 * 28))
```

- Production

```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
os.path.isfile("model.pt")


# onnx
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
     autoencoder = LitAutoEncoder()
     input_sample = torch.randn((1, 28 * 28))
     autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
     os.path.isfile(tmpfile.name)
```

### Checkpoints

```python
model = LitModel.load_from_checkpoint(path)

# or

# load the ckpt
ckpt = torch.load('path/to/checkpoint.ckpt')

# equivalent to the above
model = LitModel()
model.load_state_dict(ckpt['state_dict'])
```

## Zther

1. `.item(), .numpy(), .cpu()`

    Don’t call .item() anywhere in your code. Use .detach() instead to remove the connected graph calls. Lightning takes a great deal of care to be optimized for this.

1. `empty_cache()`

    Don’t call this unnecessarily! Every time you call this ALL your GPUs have to wait to sync.

1. `Construct tensors directly on the device`

    ```python
    # bad
    t = torch.rand(2, 2).cuda()
    # good (self is LightningModule)
    t = torch.rand(2, 2, device=self.device)
    
    # bad
    self.t = torch.rand(2, 2, device=self.device)
    # good
    self.register_buffer("t", torch.rand(2, 2))
    ```

1. Use DDP not DP

    When using `DDP` set `find_unused_parameters=False`

    ```python
   from pytorch_lightning.plugins import DDPPlugin

    trainer = pl.Trainer(
        gpus=2,
        plugins=DDPPlugin(find_unused_parameters=False),
    ) 
   ```

1. Use Sharded DDP for GPU memory and scaling optimization
    
    Sharded DDP can work across all DDP variants by adding the additional `--plugins ddp_sharded` flag.

1. Zero Grad `set_to_none=True`

    In order to modestly improve performance, you can override optimizer_zero_grad().

    ```python
    class Model(LightningModule):

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    ```

