from utils.data import get_mnist_loaders


train_loader, test_loader = get_mnist_loaders(batch_size=64)

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("First 10 labels:", labels[:10])
print("Number of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))