import tensorflow.keras as kr
import tensorflow.keras.backend as K
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from PIL import Image
import PIL

# Ref https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368

def preprocess_image(im, image_size=64, expand=True):
    im = cv2.resize(im, (image_size, image_size))/255.0

    if expand:
        return np.expand_dims(im, 0)
    return im

def predict_on_model(im, m, return_raw):
    o = m.predict(preprocess_image(im))
    if return_raw:
        return o[0]
    return np.argmax(o[0])

def create_maze_solving_network(image_size=64, num_actions=4):
    model = kr.models.Sequential()
    model.add(Conv2D(16, 3, input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    # model = kr.models.Sequential()
    # model.add(Conv2D(64, 8, strides=(4, 4), activation='relu', input_shape=(64, 64, 3)))
    # model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
    # model.add(Conv2D(64, 3, strides=(2, 2), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(num_actions))
    # model.compile(loss='mse', optimizer='adam')

    return model


def masked_mse(args):
    y_true, y_pred, mask = args
    loss = (y_true - y_pred)**2
    loss *= mask
    return K.sum(loss, axis=-1)


def add_rl_loss_to_network(model):
    num_actions = model.output.shape[1]
    y_pred = model.output
    y_true = Input(name='y_true', shape=(num_actions,))
    mask = Input(name='mask', shape=(num_actions,))
    loss_out = Lambda(masked_mse, output_shape=(1,), name='loss')([y_true, y_pred, mask])
    trainable_model = Model(inputs=[model.input, y_true, mask],
                            outputs=loss_out)
    trainable_model.compile(optimizer=Adam(), loss=lambda yt, yp: yp)
    return trainable_model


def transfer_weights_partially(source, target, lr=0.5):
    wts = source.get_weights()
    twts = target.get_weights()

    for i in range(len(wts)):
        twts[i] = lr * wts[i] + (1-lr) * twts[i]
    target.set_weights(twts)


def make_intermediate_models(model, layer_names):
    return [kr.models.Model(inputs=model.input, outputs=model.get_layer(name).output)
            for name in layer_names]

def visualize_network_forward_pass(model, im_in, action):
    models = make_intermediate_models(model, layer_names = ['conv2d_3', 'conv2d_4', 'conv2d_5'])
    im = preprocess_image(im_in)
    res = [model.predict(im) for model in models]

    final_ims = []

    for ii, r in enumerate(res):
        sub_images = []
        grid_side_len = int(np.sqrt(r.shape[-1]))
        assert grid_side_len**2 == r.shape[-1], 'Grid is not square, not sure what to do'

        for i in range(r.shape[-1]):
            sub_images.append(rescale_image(r[0, :, :, i]))

        s, _ = sub_images[0].shape[:2]

        p = s // 8

        final_image = np.zeros((s*grid_side_len + (p * grid_side_len),
                                s*grid_side_len + (p * grid_side_len)), 'uint8')
        final_image = np.dstack((final_image, final_image, final_image, final_image))

        for i, sub_im in enumerate(sub_images):
            x = (i//grid_side_len)*(s + p)
            y = (i % grid_side_len)*(s + p)
            final_image[x:x+s, y:y+s, :3] = cv2.cvtColor(sub_im.copy(), cv2.COLOR_GRAY2BGR)
            final_image[x:x+s, y:y+s, -1] = 255

        final_ims.append(cv2.resize(final_image, (128, 128)))
    cw, ch = 400, 160
    canvas = Image.new(mode='RGBA', size=(cw, ch))

    canvas.paste(Image.fromarray(cv2.cvtColor(im_in, cv2.COLOR_BGR2RGBA)), (10, 10))
    canvas.paste(Image.fromarray(final_ims[0]), (100, 10))
    canvas.paste(Image.fromarray(final_ims[1]), (250, 10))
    canvas = canvas.resize((cw*4, ch*4), Image.NEAREST)
    am = {
        0: 'down',
        1: 'up',
        3: 'left',
        2: 'right'
    }
    im = np.array(Image.open(f'{am[action]}.png').resize((280, 70)))
    # im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    canvas.paste(Image.fromarray(im), (40, 390))

    return canvas

def rescale_image(im):
    im = (im - im.min())
    im = im/im.max()*255
    return im.astype('uint8')

