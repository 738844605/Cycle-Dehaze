import requests
import tensorflow as tf
import os
import utils
import cv2
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/Hazy2GT_indoor.pb', 'model path(.pb)' )
tf.flags.DEFINE_string('input', 'input_sample.jpg', 'input image path(.jpg) or input path')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path(.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size ,default:256')
tf.flags.DEFINE_bool('isurl', False, 'is the input url?,default:False')
"""
url:输入文件路径，当isurl为True时，可以为网页地址，当isurl为False时，可以为文件路径；
outputpath:输出文件路径；
isurl:输入url是否为网页地址；
modelpath:pb模型文件路径
"""
def inference(url="", outputpath="output.jpg", isurl = True, modelpath="zebra2horse.pb"):
    graph = tf.Graph()
    with graph.as_default():
        if isurl:
            image_data = requests.get(url = url).content
        else:
            #print(url)
            with open(url, "rb") as f:
                image_data = f.read()
                input_image = tf.image.decode_jpeg(image_data, channels =3)
                input_image = tf.image.resize_images(input_image,size=(FLAGS.image_size,FLAGS.image_size))
                input_image = utils.convert2float(input_image)
                input_image.set_shape([FLAGS.image_size,FLAGS.image_size,3])
        with tf.gfile.FastGFile(modelpath, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            [output_image] = tf.import_graph_def(graph_def, input_map = {'input_image':input_image},return_elements = ['output_image:0'],name = 'output')
        with tf.Session(graph=graph) as sess:
            generated = output_image.eval()
            with open(outputpath,'wb') as f:
                f.write(generated)

if __name__ == "__main__":
    I_HAZE = "D:\Image_dataset\# I-HAZY NTIRE 2018\hazy1\\"
    I_HAZE_GT = "D:\Image_dataset\# I-HAZY NTIRE 2018\GT1\\"
    O_HAZE="D:\Image_dataset\# O-HAZY NTIRE 2018\hazy1\\"
    O_HAZE_GT="D:\Image_dataset\# O-HAZY NTIRE 2018\GT1\\"
    SOTSI = "C:\\Users\FQL\Desktop\RESIDE-standard\SOTS\indoor\hazy\\"
    SOTSI_GT = "C:\\Users\FQL\Desktop\RESIDE-standard\SOTS\indoor\gt1\\"
    SOTSO = "F:\SOTS\hazy\\"
    SOTSO_GT = "F:\SOTS\clear\\"
    HSTSS = "F:\HSTS\HAZY\\"
    HSTSS_GT = "F:\HSTS\CLEAR\\" #jpg

    HSTSR = "F:\HSTS\REAL\\"
    RTTS = "F:\RTTS\\"
    pathname =  "F:\RTTS\\"
    output_dir = "F:\CYCLE-DEHAZE\RTTS\\"
    img_name_list = os.listdir(pathname)
    img_sum = len(img_name_list)
    
    for img_path in img_name_list:
        (imageName, extension) = os.path.splitext(img_path)
        img_name_path = os.path.join(pathname,img_path)
        save_path = os.path.join("E:\\Ubuntu\CycleGAN-TensorFlow-master\samples\\"+imageName+'.png')
        print(img_name_path)
        inference(url=img_name_path, outputpath=save_path, isurl = FLAGS.isurl, modelpath = FLAGS.model)
    
    for i in img_name_list:
        (imageName, extension) = os.path.splitext(i)
        img_name_path = os.path.join(pathname,i)
        save_path = os.path.join(output_dir+imageName+'.png')
        a = cv2.imread(img_name_path)
        h,w = a.shape[:2]
        a1 = cv2.pyrDown(a,(256,256))
        h1,w1 = a1.shape[:2]
        b = a - cv2.resize(a1,(w,h),cv2.INTER_NEAREST)
        generated = cv2.imread("E:\\Ubuntu\CycleGAN-TensorFlow-master\samples\\"+imageName+".png")
        c = cv2.resize(generated,(w,h),cv2.INTER_NEAREST)+b
        cv2.imwrite(output_dir+imageName+".png",c)