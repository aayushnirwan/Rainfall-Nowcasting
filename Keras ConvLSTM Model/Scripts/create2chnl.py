import numpy as np


# train_input = np.load('train_sequences_400.npy')
# train_input = np.expand_dims(train_input, axis=-1)

train_label = np.load('train_labels_400.npy')
train_label = np.squeeze(np.expand_dims(train_label, axis=-1), axis=1)

# val_input = np.load('val_sequences_400.npy')
# val_input = np.expand_dims(val_input, axis=-1)

val_label = np.load('val_labels_400.npy')
val_label = np.squeeze(np.expand_dims(val_label, axis=-1), axis=1)

print (train_label.shape)
print (val_label.shape)
# train_input = np.where(train_input == 1, [1, 0], [0, 1])
train_label = np.where(train_label == 1, [1, 0], [0, 1])
# val_input = np.where(val_input == 1, [1, 0], [0, 1])
val_label = np.where(val_label == 1, [1, 0], [0, 1])
print (val_label.shape)
print (train_label.shape)

# np.save('train_sequences_400_sm.npy', train_input)
np.save('train_labels_400_sm.npy', train_label)
# np.save('val_sequences_400_sm.npy', val_input)
np.save('val_labels_400_sm.npy', val_label)
