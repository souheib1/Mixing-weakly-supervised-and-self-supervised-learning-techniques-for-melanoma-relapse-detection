from pickle import load
from matplotlib.pylab import plt
from numpy import arange
 
# Load the training and validation loss dictionaries
train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))
accuracy = load(open('accuracy_val.pkl', 'rb'))
 
# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()
accuracy_values = accuracy.values()
final_accuracy = round(100 * list(accuracy_values)[-1]) /100

 
# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, 101)
 
# Plot and label the training and validation loss values
#plt.subplot(1, 2, 1)
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')
 
# Add in a title and axes labels
plt.title('Training and Validation Loss and accuracy '+str(final_accuracy))
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, 101, 2))
plt.xticks([])
plt.yticks(arange(0, 2, 0.5))
 
# Display the plot
plt.legend(loc='best')
# plt.figure()
# #plt.subplot(1, 2, 2)
# plt.plot(epochs, accuracy_values)
# # Add in a title and axes labels
# plt.title('Accuracy on validation Set')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # Set the tick locations
# plt.xticks([])
plt.show()