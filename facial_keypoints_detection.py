import numpy as np
import os
import cv2 as cv
import imgaug as ia
import imgaug.augmenters as iaa
import time
import tensorflow as tf
import imageio
from tqdm import tqdm
from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image
from keras.utils import Sequence
from matplotlib import pyplot as plt


def show_sample(index, annotations, color = 'red', point_size = 2, linewidth = 0.5):
    """
    Function to show sample from data with facial key points drawn.
    
    :param int index: index.
    :param annotations: annotation list.
    :param str color: color.
    :param float point_size: keypoint size. Default value is 2.
    :param float linewidth: connecting line width. Default value is 1.
    
    :return numpy.array keypoints: keypoints.
    """
    
    # read in image filename
    filename = annotations[str(index)]['file_name']
    
    # read in image annotated keypoints
    keypoints = np.array(annotations[str(index)]['face_landmarks'])
    
    # show image
    plt.axis('off')
    plt.imshow(cv.cvtColor(cv.imread(os.getcwd() + '/data/images/' + filename), cv.COLOR_BGR2RGB))

    # extract keypoints for facial elements
    jaw_line = keypoints[:17]
    right_brow = keypoints[17:22]
    left_brow = keypoints[22:27]
    nose_bridge = keypoints[27:31]
    nose_base = keypoints[31:36]
    right_eye = keypoints[36:42]
    left_eye = keypoints[42:48]
    outer_lip = keypoints[48:60]
    inner_lip = keypoints[60:68]

    # draw keypoints and lines for non-continuous elements
    for face_element in [jaw_line, right_brow, left_brow, nose_bridge, nose_base]:
        plt.plot(face_element[:,0], face_element[:,1], linestyle = '--', linewidth = linewidth, color = color)
        for i in range(len(face_element)):
            plt.scatter(x = face_element[i,0], y = face_element[i,1], c = color, s = point_size)
            pass
        pass

    # draw keypoints and lines for continuous elements
    for face_element in [right_eye, left_eye, outer_lip, inner_lip]:
        plt.plot(face_element[:,0], face_element[:,1], linestyle = '--', linewidth = linewidth, color = color)
        plt.plot([face_element[0,0], face_element[-1,0]], [face_element[0,1], face_element[-1,1]], linestyle = '--', linewidth = linewidth, color = color)
        for i in range(len(face_element)):
            plt.scatter(x = face_element[i,0], y = face_element[i,1], c = color, s = point_size)
            pass
        pass
    
    return keypoints

def show(img, keypoints, params, color = 'red', point_size = 2, linewidth = 0.5):
    """
    Function to show sample from preprocessed image and keypoint data.
    
    :param numpy.array img: image.
    :param numpy.array: keypoints.
    :param dict params: Hyperparameters.
    :param str color: color.
    :param float point_size: keypoint size. Default value is 2.
    :param float linewidth: connecting line width. Default value is 1.
    
    :return numpy.array keypoints: keypoints.
    """
    
    # keypoint data
    keypoints = np.concatenate([keypoints[:68].reshape(68,1), keypoints[68:].reshape(68,1)], axis = -1) 
    
    # show image
    plt.axis('off')
    plt.imshow(img)

    # extract keypoints for facial elements
    jaw_line = keypoints[:17]
    right_brow = keypoints[17:22]
    left_brow = keypoints[22:27]
    nose_bridge = keypoints[27:31]
    nose_base = keypoints[31:36]
    right_eye = keypoints[36:42]
    left_eye = keypoints[42:48]
    outer_lip = keypoints[48:60]
    inner_lip = keypoints[60:68]

    # draw keypoints and lines for non-continuous elements
    for face_element in [jaw_line, right_brow, left_brow, nose_bridge, nose_base]:
        plt.plot(face_element[:,0]*params['IMG_W'], face_element[:,1]*params['IMG_H'], linestyle = '--', linewidth = linewidth, color = color)
        for i in range(len(face_element)):
            plt.scatter(x = face_element[i,0]*params['IMG_W'], y = face_element[i,1]*params['IMG_H'], c = color, s = point_size)
            pass
        pass

    # draw keypoints and lines for continuous elements
    for face_element in [right_eye, left_eye, outer_lip, inner_lip]:
        plt.plot(face_element[:,0]*params['IMG_W'], 
                 face_element[:,1]*params['IMG_H'], 
                 linestyle = '--', linewidth = linewidth, color = color)
        plt.plot([face_element[0,0]*params['IMG_W'], face_element[-1,0]*params['IMG_W']], 
                 [face_element[0,1]*params['IMG_H'], face_element[-1,1]*params['IMG_H']], 
                 linestyle = '--', linewidth = linewidth, color = color)
        for i in range(len(face_element)):
            plt.scatter(x = face_element[i,0]*params['IMG_W'], y = face_element[i,1]*params['IMG_H'], c = color, s = point_size)
            pass
        pass
    
    return keypoints

def preprocess(index, anns, params):
    
    # read in & resize image
    filename = anns[str(index)]['file_name']
    img = cv.cvtColor(cv.imread(os.getcwd() + '/data/images/' + filename), cv.COLOR_BGR2RGB)
    img_h, img_w, channels = img.shape
    img = cv.resize(img, (params['IMG_W'], params['IMG_H']))
    img_array = np.asarray(img)
    
    # extract and convert annotated keypoints to image relative [0,1]
    keypoints = np.array(anns[str(index)]['face_landmarks'])
    offsets = keypoints / np.array([img_w, img_h])
    targets = np.concatenate([offsets[:,0], offsets[:,1]])
    
    return img_array, targets

def preprocess_with_augmentation(index, anns, augment_params, params):
    
    # read in & resize image
    keypoints = np.array(anns[str(index)]['face_landmarks'])
    filename = anns[str(index)]['file_name']
    img = cv.cvtColor(cv.imread(os.getcwd() + '/data/images/' + filename), cv.COLOR_BGR2RGB)
    img_array = np.asarray(img)
    img_h, img_w, channels = img.shape

    # creat a set of Keypoint objects
    kps = []
    for i in range(len(keypoints)):
        kps.append(Keypoint(x = keypoints[i,0], y = keypoints[i,1]))
        pass

    # place the Keypoint objects on original image
    kps_on_image = KeypointsOnImage(kps, shape = img_array.shape)
    #image_before = kps_on_image.draw_on_image(img_array, size=7)

    # create an augmentation pipeline
    seq = iaa.Sequential([#iaa.Fliplr(0.5),
                          iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, augment_params['blur']))),        
                          iaa.LinearContrast((1.0-augment_params['contrast'], 1.0+augment_params['contrast'])),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, augment_params['noise'] * 255), per_channel=0.5),
                          iaa.Multiply((1.0-augment_params['brightness'], 1.0+augment_params['brightness']), per_channel=0.2),
                          iaa.Affine(scale={"x": (1.0-augment_params['zoom'], 1.0+augment_params['zoom']),
                                            "y": (1.0-augment_params['zoom'], 1.0+augment_params['zoom'])},
                                     translate_percent={"x": (-1*augment_params['translate'], augment_params['translate']),
                                                        "y": (-1*augment_params['translate'], augment_params['translate'])})
                         ],random_order=True)

    # augment keypoints and images.
    image_aug, kps_aug = seq(image = img_array, keypoints = kps_on_image)

    # extract and convert annotated keypoints to image relative [0,1]
    x = np.array([i.x for i in kps_aug])
    x = np.maximum(np.minimum(x, img_w-1.), 0.0) / img_w
    
    y = np.array([i.y for i in kps_aug])
    y = np.maximum(np.minimum(y, img_h-1.), 0.0) / img_h
    
    targets = np.concatenate([x, y])

    # resize image
    temp = Image.fromarray(image_aug, 'RGB')
    inputs = np.array(temp.resize((params['IMG_W'], params['IMG_H'])))
    
    return inputs, targets

# data generator to preprocess images & annotations dynamically at training time
class train_data_generator(Sequence):
    
    def __init__(self, indices, batch_size, annotations, params, augment_params):
        
        self.indices = indices
        self.batch_size = batch_size
        self.annotations = annotations
        self.params = params
        self.augment_params = augment_params
        pass
    
    def __len__(self):
        
        return int(np.ceil(len(self.indices) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, Y = [], []
        
        # iterate through the batch
        for i in range(len(batch)):
            img, targets = preprocess_with_augmentation(index = batch[i], 
                                                        anns = self.annotations, 
                                                        augment_params = self.augment_params, 
                                                        params = self.params)

            X.append(img)
            Y.append(targets)
            pass
        
        return np.array(X), np.array(Y)

# data generator to preprocess images & annotations dynamically at training time
class test_data_generator(Sequence):
    
    def __init__(self, indices, batch_size, annotations, params):
        
        self.indices = indices
        self.batch_size = batch_size
        self.annotations = annotations
        self.params = params
        pass
    
    def __len__(self):
        
        return int(np.ceil(len(self.indices) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, Y = [], []
        
        # iterate through the batch
        for i in range(len(batch)):
            img, targets = preprocess(index = batch[i], anns = self.annotations, params = self.params)
            X.append(img)
            Y.append(targets)
            pass
        
        return np.array(X), np.array(Y)

# function to display model predictions on "unseen" test data
def show_prediction(img, model, params, color = 'white', point_size = 2, linewidth = 0.5, linestyle = '--'):
    
    # run image through the model to get keypoint predictions
    pred = model(np.expand_dims(img, axis = 0))

    # seperate out the x and y keypoint offsets
    keypoints = np.concatenate([np.expand_dims(pred[0,:68], axis = -1), 
                                np.expand_dims(pred[0,68:], axis = -1)], axis = -1)

    # extract keypoints by facial elements
    jaw_line = keypoints[:17]
    right_brow = keypoints[17:22]
    left_brow = keypoints[22:27]
    nose_bridge = keypoints[27:31]
    nose_base = keypoints[31:36]
    right_eye = keypoints[36:42]
    left_eye = keypoints[42:48]
    outer_lip = keypoints[48:60]
    inner_lip = keypoints[60:68]

    # plot the keypoints
    for j in range(68):
        plt.scatter(x = (params['IMG_W'] * (keypoints[j,0])), 
                    y = (params['IMG_H'] * (keypoints[j,1])), 
                    c = color, s = point_size)  

    # plot connections between non-continuous facial element keypoints
    for face_element in [jaw_line, right_brow, left_brow, nose_bridge, nose_base]:
        plt.plot((params['IMG_W'] * (face_element[:,0])), 
                 (params['IMG_H'] * (face_element[:,1])), 
                 linestyle = linestyle, linewidth = linewidth, color = color)
        pass

    # plot connections between continuous facial element keypoints
    for face_element in [right_eye, left_eye, outer_lip, inner_lip]:
        plt.plot((params['IMG_W'] * (face_element[:,0])), 
                 (params['IMG_H'] * (face_element[:,1])),
                 linestyle = linestyle, linewidth = linewidth, color = color)

        plt.plot([(params['IMG_W'] * (face_element[0,0])), (params['IMG_W'] * (face_element[-1,0]))], 
                 [(params['IMG_H'] * (face_element[0,1])), (params['IMG_H'] * (face_element[-1,1]))], 
                 linestyle = linestyle, linewidth = linewidth, color = color)
        pass
    pass

    # show image
    plt.axis('off')
    plt.imshow(img)
    pass

def invert_color_channels(img):
    """
    Function to invert image (numpy array) color channels. Assuming color channels are last dimension.
    :param numpy.array img: image numpy array [img_height, img_width, col_channels].
    :return numpy.array inv_img: image numpy array with inverted color channels [img_height, img_width, col_channels].
    """
    inv_img = np.concatenate([np.expand_dims(img[...,2], -1), 
                              np.expand_dims(img[...,1], -1), 
                              np.expand_dims(img[...,0], -1)], axis = -1)
    return inv_img

def predict_video(filename, 
                  model, 
                  params,
                  mp4_output_dims = (1080, 1080),
                  generate_gif = False,
                  gif_output_dims = (240, 240),
                  gif_quality = 100,
                  test_folder = 'test/video/',
                  fps = 30, 
                  color = (255,0,0), 
                  thickness = 2):
    """
    Function to use the provided model on video (.mp4) files.
    
    :param str filename: image filename in test/image/ folder.
    :param tf.keras.model model: model.
    :param dict params: Hyperparameters.
    :param tuple mp4_output_dims: mp4 output dimensions (height, width).
    :param boolean generate_gif: True if we want to generate corresponding GIF.
    :param tuple gif_output_dims: GIF output dimensions (height, width).
    :param int gif_quality: GIF frame quality (0 - 100).
    :param string test_folder: relative path to folder containing the video with the filename specified above.
    :param int fps: frames per second.
    :param tuple color: tuple with BGR color channel values [0-255].
    :param float thickness: bounding box linewidth. Default value is 2.
    
    :return list frame_inference_times: list with model frame inference times.
    """
    
    # make the directory for the results if it does not exist already
    if not os.path.exists(test_folder + 'results'):
        os.makedirs(test_folder + 'results')
        pass
        
    # make the directory for the results .mp4 files subdirectory 
    if not os.path.exists(test_folder + 'results/mp4'):
        os.makedirs(test_folder + 'results/mp4')
        pass
    
    if generate_gif:
        # make the directory for the results .gif files subdirectory
        if not os.path.exists(test_folder + 'results/gif'):
            os.makedirs(test_folder + 'results/gif')
            pass
        pass
    
    # open video that we want to do object detection on and get the number of frames
    cap = cv.VideoCapture(test_folder + filename)
    amount_of_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)    
    h = int(cap.get(cv.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    
    # establish video dimensions
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    res, frame = cap.read()
    #height, width, _ = frame.shape
    height, width = mp4_output_dims
    
    # open video file to write to
    fourcc = cv.VideoWriter_fourcc(*"mp4v") # codec for .mp4 format
    video = cv.VideoWriter(test_folder + 'results/mp4/' + filename.split('.')[-2] + '_(Facial_Keypoints).mp4', fourcc, fps, (width, height))
    
    # list to store frames and frame inference times
    frames_list, frame_inference_times = [], []
    
    print('Creating video (.mp4)...')
    
    # iterate through frames
    for i in tqdm(range(int(amount_of_frames))):
        
        # read in frame
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        
        # use object detector model to make prediction on frame
        start = time.time() # start time frame inference
        #------------
        
        # read in image
        img_org = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_org_h, img_org_w, _ = img_org.shape
        width_rescale_factor = img_org_w / params['IMG_W']
        height_rescale_factor = img_org_h / params['IMG_H']

        # resize image to require input size
        img_input = np.array(cv.resize(img_org, (params['IMG_W'], params['IMG_H'])))

        ############## predict using model ###############
        
        # run image through the model to get keypoint predictions
        pred = model(np.expand_dims(img_input, axis = 0))

        # seperate out the x and y keypoint offsets
        keypoints = np.concatenate([np.expand_dims(pred[0,:68], axis = -1) * img_org_w, #params['IMG_W'] * width_rescale_factor, 
                                    np.expand_dims(pred[0,68:], axis = -1) * img_org_h], #params['IMG_H'] * height_rescale_factor], 
                                    axis = -1)

        keypoints = keypoints.astype(int)
        
        # extract keypoints by facial elements
        jaw_line = keypoints[:17]
        right_brow = keypoints[17:22]
        left_brow = keypoints[22:27]
        nose_bridge = keypoints[27:31]
        nose_base = keypoints[31:36]
        right_eye = keypoints[36:42]
        left_eye = keypoints[42:48]
        outer_lip = keypoints[48:60]
        inner_lip = keypoints[60:68]

        # convert a copy of the image back to bgr color channels
        img_org_bgr = cv.cvtColor(img_org.copy(), cv.COLOR_RGB2BGR)
        
        # plot connections between non-continuous facial element keypoints
        for face_element in [jaw_line, right_brow, left_brow, nose_bridge, nose_base]:
            cv.polylines(img = img_org_bgr, pts = [face_element.reshape((-1,1,2))], isClosed = False, color = color)
            for j in range(len(face_element)):
                cv.circle(img = img_org_bgr, center = (face_element[j,0], face_element[j,1]), radius = 3, color = color, thickness = thickness)
                pass
            pass

        # plot connections between continuous facial element keypoints
        for face_element in [right_eye, left_eye, outer_lip, inner_lip]:
            cv.polylines(img = img_org_bgr, pts = [face_element.reshape((-1,1,2))], isClosed = True, color = color)
            for j in range(len(face_element)):
                cv.circle(img = img_org_bgr, center = (face_element[j,0], face_element[j,1]), radius = 3, color = color, thickness = thickness)
                pass
            pass
        
        # resize the BGR image with keypoints plotted and write to video
        video.write(cv.resize(img_org_bgr, (mp4_output_dims[1], mp4_output_dims[0])))
        #video.write(img_org_bgr)
        frames_list.append(img_org_bgr)
        pass
        
        #------------
        end = time.time() # end time frame inference
        
        frame_inference_times.append(end - start) # store frame inference time
    
    # release the video objects
    cap.release()
    video.release()
    #cv.destroyAllWindows()
    
    ######## GENERATE GIF ##########
    if generate_gif:
        
        print('Creating GIF...')

        adj_frame_list = []
        for f_a in tqdm(frames_list): # iterate through frames
            
            # reorder color channels to RGB, resize and adjust quality of the images to control file size
            temp = Image.fromarray(invert_color_channels(f_a), mode = 'RGB')
            temp = temp.resize((gif_output_dims[1], gif_output_dims[0]), Image.LANCZOS)
            
            # save image to file with adjusted quality...
            temp.save(os.getcwd() + '/' + test_folder + 'results/gif/' + 'temp.jpg', optimize = True, quality = gif_quality)
            # read back the image and add to list
            with Image.open(os.getcwd() + '/' + test_folder + 'results/gif/' + 'temp.jpg') as open_image:
                adj_frame_list.append(np.asarray(open_image.copy()))
                pass
            # remove temporary image
            os.remove(os.getcwd() + '/' + test_folder + 'results/gif/' + 'temp.jpg')
            pass

        # generate GIF file
        imageio.mimwrite(os.getcwd() + '/' + test_folder + 'results/gif/' + filename.split('.')[-2] + '_(Facial_Keypoints).gif', 
                         np.array(adj_frame_list), 
                         format = '.gif',
                         loop = 500,
                         duration = int(1000 * 1/fps))
        pass
    
    return frame_inference_times