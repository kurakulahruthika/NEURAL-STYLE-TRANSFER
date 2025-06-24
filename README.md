# NEURAL-STYLE-TRANSFER

*COMPANY*:codtech it solutions

*NAME:KURAKULA HRUTHIKA*

*INTERN ID*:CT04DF664

*DOMAIN*: AI

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH*company*:codtech it solutions

### DESCRIPTION OF MY TASK COMPLETION  PROCESS


##  Neural Style Transfer – Project Description

###  Overview

This project is focused on implementing *Neural Style Transfer (NST)* – an exciting deep learning technique that merges the *content of one image* with the *style of another* to create a visually stunning new image. The goal was to generate artistic images by extracting content features from a base image and combining them with style features from a chosen artwork.

Neural Style Transfer leverages the *convolutional layers* of deep neural networks, particularly *VGG19*, to achieve this fusion. Through this project, I explored computer vision, feature extraction, and optimization in the context of creativity and AI.

---

### Project Goals

* Load and preprocess both content and style images.
* Use a pretrained convolutional neural network (VGG19) for feature extraction.
* Define and compute *content loss, **style loss, and **total variation loss*.
* Apply *gradient descent optimization* to update the generated image.
* Generate a new image that reflects the content of one image in the style of another.

---

###  How I Did It – Step-by-Step Breakdown

####  Step 1: Selecting and Loading Images

I began by choosing two images:

* A *content image* (e.g., a landscape or portrait)
* A *style image* (e.g., a famous painting like Van Gogh’s Starry Night)

Using Python and TensorFlow, I loaded and resized both images to the same dimensions for compatibility. They were converted into tensors and normalized to match the input expectations of VGG19.

####  Step 2: Feature Extraction with VGG19

I used the *VGG19* model, a convolutional neural network pre-trained on ImageNet. I removed the top classification layers and used specific intermediate layers to extract:

* *Content features* from a deeper layer (e.g., block5_conv2)
* *Style features* from multiple shallower layers (e.g., block1_conv1, block2_conv1, etc.)

The selected layers capture high-level information about objects and textures, which are essential for combining style and content.

####  Step 3: Defining Loss Functions

To perform style transfer, I defined a total loss function composed of:

* *Content Loss*: Measures how different the content of the generated image is from the original content image.
* *Style Loss: Compares the **Gram matrices* of the style and generated image to ensure that texture and patterns are transferred.
* *Total Variation Loss* (optional): Encourages spatial smoothness in the output image.

Each of these losses was weighted appropriately to balance structure and style.

####  Step 4: Optimization

Instead of training the model, I treated the *generated image as a trainable variable*. I used an optimizer (Adam or LBFGS) to minimize the total loss by iteratively updating the pixels of the generated image.

This process involved:

* Forward pass: Computing losses based on current features
* Backward pass: Updating the image using gradients

I ran this for several hundred iterations while visualizing progress to track style blending.

---

###  Output

The final result was a new image that maintained the spatial structure of the content image while adopting the brush strokes, color palette, and texture of the style image.

Sample:

> Content: A city skyline
> Style: Picasso painting
> Output: A surreal city skyline in abstract, cubist brushwork

---

###  Challenges Faced

* Balancing the weights between style and content loss — too much of either can distort the image.
* Managing memory usage due to large image tensors and backpropagation.
* Ensuring that the generated image doesn’t become overly noisy or blurry.
* Understanding the correct layer selection for best artistic results.

---

###  Tools and Libraries Used

* *Python*
* *TensorFlow / Keras*
* *Matplotlib* and *PIL* for image processing
* *NumPy*
* Pretrained *VGG19* model from tf.keras.applications

---

###  What I Learned

* How convolutional neural networks extract hierarchical features from images.
* The role of *Gram matrices* in capturing texture and patterns.
* How gradient descent can be applied directly to image data.
* How to combine artistic creativity with deep learning techniques.
* That small tweaks in loss weights or layer choices can dramatically change the output style.

---

###  Conclusion

This Neural Style Transfer project gave me a creative way to apply deep learning in computer vision. It taught me how feature representations can be manipulated artistically, and how pretrained models like VGG19 can be repurposed for non-classification tasks. The results were visually impressive, and the process deepened my understanding of CNNs and loss-based image generation.

In the future, I plan to experiment with *real-time style transfer* using feed-forward networks and possibly extend this to *video style transfer*.

---

