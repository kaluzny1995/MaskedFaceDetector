import matplotlib.pyplot as plt


def show_images(images, titles, suptitle='Images'):
    assert (len(images) == len(titles)), 'Lists of images and titles of images must have the same length!'
    
    if len(images) == 1:
        fig, ax = plt.subplots(1, len(images), figsize=(15, 10))
        ax.imshow(images[0])
        ax.set_title(titles[0])
        ax.axis('off')
    else:
        fig, ax = plt.subplots(1, len(images), figsize=(15, 5))
        for i, (image, title) in enumerate(zip(images, titles)):
            ax[i].imshow(image)
            ax[i].set_title(title)
            ax[i].axis('off')
    
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_many_images(images, titles, n_cols=10, suptitle='Images'):
    assert (len(images) == len(titles)), 'Lists of images and titles of images must have the same length!'
    assert (len(images) > n_cols), 'Number of images must be greater than number of columns!'
    
    n_rows = len(images)//n_cols + 1 if len(images)%n_cols != 0 else len(images)//n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    
    for n in range(n_rows*n_cols):
        i = n//n_cols
        j = n%n_cols
        
        if n < len(images):
            ax[i][j].imshow(images[n])
            ax[i][j].set_title(titles[n])
        else:
            ax[i][j].set_title('------')
        ax[i][j].axis('off')
    
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
