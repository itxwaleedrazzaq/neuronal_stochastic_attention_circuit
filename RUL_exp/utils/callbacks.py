from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()


# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-6
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, min_lr=1e-15
)


class LossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.history = {"loss": [], "data_loss": [], "physics_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for k in self.history:
            self.history[k].append(logs.get(k, None))

    def reset(self):
        for key in self.history:
            self.history[key] = []