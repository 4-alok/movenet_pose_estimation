
# Pose Estimation with MoveNet

This project uses the MoveNet model for pose estimation. The model is loaded and run using TensorFlow Lite. The predicted keypoints are then visualized on the input image.

## DependenciesThe project requires the following libraries

- TensorFlow
- NumPy
- Matplotlib
- OpenCV

## You can install them using pip

```bash
pip install tensorflow numpy matplotlib cv2
```

The exact versions of the packages used can be found in the **requirements.txt** file.
Usage- Import the necessary libraries:

```python
    import tensorflow as tf
    import numpy as np
    from movenet_helper import *
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patches as patches
```

- Load the model:

```python
    input_size = 256
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
```

- Define the function to run inference:

```python
    def movenet(input_image):
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores
```

- Load and preprocess the image:

```python
    image_path = 'input_image.jpeg'
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
```

- Run inference and visualize the results:

```python
    keyPoints_with_scores = movenet(input_image)
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keyPoints_with_scores)
    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis('off')
```

Please replace **input_image.jpeg** with the path to your image file. Also, you'll need to implement the draw_prediction_on_image function to draw the prediction on the image as per your requirements. This function takes the original image and the output from the model inference as input. The output is a tensor that contains the pose estimation prediction. You can use libraries like OpenCV (cv2) to draw on the image.

## Note

This is a simplified version of the code and may not include all the necessary error checks and optimizations for a production environment. It's always recommended to handle potential errors and edge cases in your code.

Also, please make sure to install necessary libraries by running pip install tensorflow Pillow opencv-python-headless in your environment. If you're running this in a Jupyter notebook, you might want to prefix the command with an exclamation mark like !pip install tensorflow Pillow opencv-python-headless.

## Acknowledgements

This project uses the MoveNet model from TensorFlow Hub.

## References

- MoveNet: Ultra fast and accurate pose estimation
- MoveNet on TensorFlow Hub
