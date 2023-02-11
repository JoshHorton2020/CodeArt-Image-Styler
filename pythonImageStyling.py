from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np 
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL

def load_image(image_path, image_size=(512, 256)): 
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img 
                             
def visualize(images, titles=('',)):
    noi = len(images) //number of images
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i])
        plt.savefig("final.jpg")
    plt.show()
   
   
original_image = load_image("yourImage") #replace with your image
style_image = load_image("styleImage") #replace with your styling image

style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
visualize([original_image, style_image], ['Original Image', 'Style Image'])

stylize_model = tf_hub.load('tf_model')

results = stylize_model(tf.constant(original_image), tf.constant(style_image))
stylized_photo = results[0]

visualize([original_image, style_image, stylized_photo], titles=['Original Image', 'Style Image', 'Stylized Image'])

def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)
    
export_image(stylized_photo).save("yournewfile.png") #replace with your file name.png
