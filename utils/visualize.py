import matplotlib.pyplot as plt

def plot_spectrogram_examples(X, y, class_names, num_per_class=3):
    """
    X: (N,1,128,128)
    y: (N,)
    """

    plt.figure(figsize=(10, 6))

    idx = 1
    for cls in range(len(class_names)):
        indices = (y == cls).nonzero()[0][:num_per_class]

        for i in indices:
            plt.subplot(len(class_names), num_per_class, idx)
            plt.imshow(X[i][0], cmap='gray')
            plt.title(class_names[cls])
            plt.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()