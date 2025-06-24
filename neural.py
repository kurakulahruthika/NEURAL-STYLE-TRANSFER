import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications import vgg19
from keras.models import Model
import keras.backend as K


# ----------------------------
# Load and process images
# ----------------------------
def load_and_process_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_img(processed_img):
    x = processed_img.copy()
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# ----------------------------
# Load images
# ----------------------------
content_path = r"C:\Users\KALYAN\Downloads\penguin.jpg" 
style_path = r"C:\Users\KALYAN\Downloads\var_house.png"     

content_image = load_and_process_img(content_path)
style_image = load_and_process_img(style_path)

# ----------------------------
# Define layers for content & style
# ----------------------------
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# ----------------------------
# Build the model
# ----------------------------
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = Model([vgg.input], outputs)
    return model

# ----------------------------
# Compute style and content
# ----------------------------
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    style_features = [gram_matrix(style_layer) for style_layer in style_outputs[:num_style_layers]]
    content_features = content_outputs[num_style_layers:]
    return style_features, content_features

# ----------------------------
# Compute loss
# ----------------------------
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += tf.reduce_mean(tf.square(gram_matrix(comb_style) - target_style))
        
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean(tf.square(comb_content - target_content))
        
    style_score *= style_weight / num_style_layers
    content_score *= content_weight / num_content_layers
    loss = style_score + content_score
    return loss

# ----------------------------
# Optimization loop
# ----------------------------
@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# ----------------------------
# Run style transfer
# ----------------------------
import time

def run_style_transfer(content_path, style_path,
                       num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    init_image = tf.Variable(load_and_process_img(content_path), dtype=tf.float32)
    
    opt = tf.optimizers.Adam(learning_rate=5.0)
    best_loss, best_img = float('inf'), None
    
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': style_features,
        'content_features': content_features
    }
    
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -125, 125)
        init_image.assign(clipped)
        
        if all_loss < best_loss:
            best_loss = all_loss
            best_img = init_image.numpy()
        
        if i % 100 == 0:
            print(f"Iteration: {i}, Loss: {all_loss.numpy():.4f}")
    
    return best_img

# ----------------------------
# Run and show result
# ----------------------------
stylized = run_style_transfer(content_path, style_path, num_iterations=500)

final_img = deprocess_img(stylized)
plt.imshow(final_img)
plt.axis('off')
plt.title("Stylized Output")
plt.show()
Image.fromarray(final_img).save("output_stylized.png")
