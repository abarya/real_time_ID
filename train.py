import squeezenet
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

model =  squeezenet.SqueezeNet()

img = image.load_img('data/abhishek/0000.png', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(input=model.input, output=layer_outputs)

features = viz_model.predict(x)

for f in features:
  print(f.shape)

print(features[-2])
