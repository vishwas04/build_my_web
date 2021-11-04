from flask import Flask,render_template,request,jsonify,redirect,after_this_request,make_response,url_for
import flask
import cv2,json
import numpy as np
from PIL import ImageFile, Image
import json
app = Flask(__name__, template_folder='input')
import cv2
import os
import shutil
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.insert(0, 'segmentation/src')
from infer import myfunc
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from spellchecker import SpellChecker
spell = SpellChecker()
p=["navbar","sidebar","endbar","image","slideshow"]
AUTOTUNE = tf.data.AUTOTUNE
max_len = 21
characters=['i', 'K', '/', 'T', 'z', 'a', ')', '+', 'D', 'n', '#', '3', 'd', 'S', '6', 's', 'h', 'p', 'W', ':', 'k', 'I', 'w', 'y', '!', 'H', 'R', 'q', 'o', '5', 'M', 'l', '8', '&', 'Y', 'L', 'P', '?', 'N', 'O', ',', 'v', '-', 'c', ';', 'C', 'G', '1', 'g', 'e', 'J', 'u', 'r', '"', 'j', '0', 'E', 'F', 'x', 'b', 'V', 'm', 'X', '.', 'Z', '7', 't', '*', '2', 'U', 'B', "'", 'Q', '4', 'f', '(', 'A', '9']
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=characters, mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
# nav_state=[]
# link
# str1=" <li><a href="+lnav_state[0]+"#"+">"+nav_state[0]+"</a></li>"
def per(a,b):
    r="text"
    p=0
    for o in a:
        if(len(o)<=len(b)):
            off=len(b)-len(o)+1
            for i in range(off):
                c=0
                for match in range(len(o)):
                    if(o[match]==b[i+match]):
                        c=c+1
                if(c/len(o)>p and c>=2):
                    p=c/len(o)
                    r=o
        else:
            off=len(o)-len(b)+1
            for i in range(off):
                c=0
                for match in range(len(b)):
                    if(o[i+match]==b[match]):
                        c=c+1
                if(c/len(o)>p and c>=2):
                    p=c/len(o)
                    r=o

        
    return (r,p)



class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
#         super(CustomLayer, self).__init__(name=name)
#         self.k = k
        super(CTCLayer, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def distortion_free_resize(image, img_size):
    # print(image)
    w, h = img_size
    image  = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    
    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
                  [pad_height_top, pad_height_bottom],
                  [pad_width_left, pad_width_right],
                  [0, 0]
                ]
        )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image



def load_images_from_folder(folder):
    images = []
    file=[]
    ids=[]
    ii=1
    for filename in os.listdir(folder):
        if(str(filename)[0]!='(' or str(filename)[-1]!=')'):
            continue
        # print(filename)
        for img in os.listdir(folder+str(filename)+"/detect/"):
            file.append(folder+str(filename)+"/detect/"+str(img))
            pos=str(img)[:-4].split("_")
            ids.append(filename+"/"+pos[-2]+"_"+pos[-1])

            
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
# #             images.append(img)
#             file.append(folder+str(filename))
#             ids.append(str(ii))
#             ii=ii+1
   
    return file,ids


batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    #creats a something like generator and applies a funtion for each element (.map)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    #batch is like grouping into buckets 
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)
    
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        # print(res)[27 57 57  4 57 78 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        # print(res)[27 57 57  4 57 78]
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
#     with open('/Users/vishwas/Desktop/bd/model.txt', 'w') as f:
#         f.write(str(output_text))
    return output_text


new_model = tf.keras.models.load_model('model_20.h5', custom_objects={'CTCLayer': CTCLayer})

prediction_model = keras.models.Model(
    new_model.get_layer(name="image").input, new_model.get_layer(name="dense2").output
)
r={}

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/input",methods=["POST","GET"])
def input():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    c=1
    for uploaded_file in request.files.getlist('file'):
        c=0
        if(len(request.files.getlist('file')) and uploaded_file):
            original_image = np.asarray(Image.open(uploaded_file))
            shutil.rmtree('segmentation/data/test')
            os.mkdir('segmentation/data/test')
            cv2.imwrite("segmentation/data/test/"+str(uploaded_file.filename), original_image)
            
            shape=list(original_image.shape)
            print("shape",shape)
            shutil.rmtree('segmentation/src/result')
            os.mkdir('segmentation/src/result')
            parent_dir = 'segmentation/src/result'

            # read and scale down image
            x=os.listdir("segmentation/data/test")
            img = cv2.pyrDown(cv2.imread("segmentation/data/test/"+x[0], cv2.IMREAD_UNCHANGED))

            # threshold image
            ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)

            # find contours and get the external one
            contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            # with each contour, draw boundingRect in green
            l = []
            for c1 in contours:
                # get the bounding rect
                
                x, y, w, h = cv2.boundingRect(c1)
                if(x != 0 and y != 0):
                    flag = True
                    for coord in l:
                        if(abs(coord[0]-x) < 10 and abs(coord[1]-y) < 10):
                            flag = False
                    if(flag):
                        if(w>30 and h>30 and w<700 and  h<700 and (w*h > 3000)):
                            # print((x,y))
                            l.append((x,y))
                            #print(x,y)
                            #draw a green rectangle to visualize the bounding rect
                            #cv2.rectangle(img, (x+6, y+6), (x+w-6, y+h-6), (0, 255, 0), 2)
                            cropped_img = img[y+5:y+h-5, x+5:x+w-5]
                            #cv2.imshow("contours", img)
                            

                            
                            dir_name = '('+str(x)+','+str(y)+','+str(w)+','+str(h)+')'
                            path = os.path.join(parent_dir,dir_name)
                            os.mkdir(path)
                            cv2.imwrite(os.path.join(path,'rectangle.jpg'),cropped_img)
                            myfunc(dir_name)#segmentation
            f,ids=load_images_from_folder("segmentation/src/result/")
            v_test=prepare_dataset(f, ids)
            # cv2.imshow("contours", img)
            # r={}
            mapp={}
            for batch in v_test.take(1):
                batch_images = batch["image"]
                preds = prediction_model.predict(batch_images)#req
                pred_texts = decode_batch_predictions(preds)
                
                for i in range(len(batch["image"])):
                    cord=ids[i].split("/")[0]
                    if(cord in r):
                        # sx=spell.correction(pred_texts[i])
                        r[cord].append([pred_texts[i],(int(ids[i].split("/")[1].split("_")[0]),int(ids[i].split("/")[1].split("_")[1]))])
                    else:
                        r[cord]=list()
                        r[ids[i].split("/")[0]].append([pred_texts[i],(int(ids[i].split("/")[1].split("_")[0]),int(ids[i].split("/")[1].split("_")[1]))])
            for i in r:
                sd=sorted(r[i] , key=lambda k: [k[1][0], k[1][1]])
                z1=per(p,sd[0][0])
                sx=spell.correction(sd[0][0])
                z2=per(p,sx)
                k=z1[0]
                # print("k",sd[0][0],k,z1,z2)
                if(z2[1]>z1[1]):
                    k=z2[0] 
                sd[0][0]=k
                for d in range(1,len(sd)):
                    # print("v",spell.correction(sd[d][0]),sd[d][0])
                    sd[d][0]=spell.correction(sd[d][0])
                r[i]=sd
            # gr=r.copy()
            # print("r",r)
        else:
            return redirect ("http://localhost:3000")
    if(c):
        return redirect ("http://localhost:3000")
    return redirect ("http://localhost:3000/edit")
    

@app.route("/download",methods=["POST","GET"],endpoint='download')
def download():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    r_final=str(request.data).split("!@#$%^&*()")
    r_F=str(r_final)[2:-3]
    with open("client/src/a.html", "r") as in_file:
        buf = in_file.readlines()
    in_file.close()
    with open("client/src/a.html", "w") as out_file:
        for line in buf:
            if line == "        }}</script>\n":
                line = line + r_final[0][2:]+"\n"
            out_file.write(line)
    out_file.close()

    with open("client/src/a.html", "r") as in_file:
        buf = in_file.readlines()
    in_file.close()
    with open("client/src/a.html", "w") as out_file:
        for line in buf:
            if line == "  <script>//////\n":
                print("ccccccc")
                line = line + r_final[1][:-1]+"\n"
            out_file.write(line)
    out_file.close()
    # print("ddd",r_final[1][:-3],"ddd")
    fr=dict()
    fr["r"]=1
    jd = json.dumps(fr)
    return  jd

@app.route("/edit",methods=["POST","GET"],endpoint='edit')
def edit():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    # jsonResp = {'jack': 4098, 'sape': 4139}
    # r=gr.copy()
    global r
    json_dump = json.dumps(r)
    # print(json_dump)
    r=dict()
    print(json_dump)
    return json_dump




if __name__ == "__main__":
    app.run(debug=True,host='localhost', port=5000)


# # @after_this_request
#     # def add_header(response):
#     #     response.headers.add('Access-Control-Allow-Origin', '*')
#     #     return response
#     # if(request.method=="POST" ):
#         # f = request.files['img'] 
#         # print("ooooo") 
#         # i=request.files["file"]
#         # i.save("files/"+request.form["pgno"]+".jpg")
#         # resp = flask.Response.json({"name":"vade"})
#         # resp.headers['Access-Control-Allow-Origin'] = '*'
#         # f.save("files/"+str(f.filename))
#         # return {"s":1}
#     # for uploaded_file in request.files('file'):
#     #     if(len(request.files.getlist('file'))):
#     #         original_image = np.asarray(Image.open(uploaded_file))
#     #         #if uploaded_file.filename != '':
#     #         # frame = cv2.imdecode(uploaded_file)
#     #         img = Image.fromarray(original_image, 'RGB')
#     #         img.save("files/"+str(original_image.filename))
#     #         img.show()
#             # response = uploaded_file.read()
#     # file = request.files['file']
#     # original_image = np.asarray(Image.open(file))
#     # print(original_image)
#     print(request.files['file'])
#     # cv2.imshow("str(original_image.filename)", original_image)
#     # return redirect("http://localhost:3000")
#     print(json.dumps({'success':True}), 200, {'ContentType':'application/json'} )
#     resp = jsonify(success=True)
#     resp.status_code = 200
#     # print(repr.status_code)
#     return resp 
    
    
# f = request.files["file"].read()
    # npimg = np.fromstring(f, np.uint8)
    # print(npimg)
    # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # print(img)
    # dis = Image.fromarray(img, 'RGB')
    # dis.show()
    # print("file")
    # f.save("files/"+str(f.filename))
    # response = file.read()
    # resp = jsonify(success=True)
    # img = Image.fromarray(f.read(), 'RGB')
    # img.save("files/"+str(f.filename))
    # img.show()



# # @app.route("/build")
# # def index():
# #     return render_template("build.html")
#if uploaded_file.filename != '':
            # frame = cv2.imdecode(uploaded_file)
            
# img = Image.fromarray(original_image, 'RGB')
            # uploaded_file.save("/Users/vishwas/Downloads/segmentation-2/data/test/"+str(uploaded_file.filename))
            # img.save("/Users/vishwas/Downloads/segmentation-2/data/test/"+str(uploaded_file.filename))
            # img.show()
            # image = cv2.imread("/Users/vishwas/Desktop/segment0.png")
            # print(type(original_image),type(image))
            # cv2.imshow("a",original_image)
            # cv2.waitKey(0)   

            #text detection #rnn 0-9
            #shape detection 
            #patter detection #nn
            #postion detection
            #r=json {["type":"p","pos":30px],["type":"h1","pos":10px,text="hi"]}
            #html text
            #s=""
            #for i in r:
            #   if(i.type=="p"):
            #       s=s+"<p>"

            #mongo at random number 10
            # cv2.imshow(str(uploaded_file.filename),original_image)