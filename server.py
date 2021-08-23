from flask import Flask,render_template,request,jsonify,redirect,after_this_request,make_response,url_for
import flask
import cv2
import numpy as np
from PIL import ImageFile, Image
import json
app = Flask(__name__, template_folder='input')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/input",methods=["POST","GET"])
def input():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    # f = request.files["file"].read()
    # npimg = np.fromstring(f, np.uint8)
    # print(npimg)
    # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # print(img)
    # dis = Image.fromarray(img, 'RGB')
    # dis.show()
    # print("file")
    # f.save("/Users/vishwas/Desktop/build_my_web/files/"+str(f.filename))
    # response = file.read()
    # resp = jsonify(success=True)
    # img = Image.fromarray(f.read(), 'RGB')
    # img.save("/Users/vishwas/Desktop/build_my_web/files/"+str(f.filename))
    # img.show()
    c=1
    for uploaded_file in request.files.getlist('file'):
        c=0
        if(len(request.files.getlist('file')) and uploaded_file):
            original_image = np.asarray(Image.open(uploaded_file))
            #if uploaded_file.filename != '':
            # frame = cv2.imdecode(uploaded_file)
            img = Image.fromarray(original_image, 'RGB')
            img.save("/Users/vishwas/Desktop/build_my_web/files/"+str(uploaded_file.filename))
            img.show()
        else:
            return redirect ("http://localhost:3000")
    if(c):
        return redirect ("http://localhost:3000")
    return redirect ("http://localhost:3000/edit")
    

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
#         # i.save("/Users/vishwas/Desktop/build_my_web/files/"+request.form["pgno"]+".jpg")
#         # resp = flask.Response.json({"name":"vade"})
#         # resp.headers['Access-Control-Allow-Origin'] = '*'
#         # f.save("/Users/vishwas/Desktop/build_my_web/files/"+str(f.filename))
#         # return {"s":1}
#     # for uploaded_file in request.files('file'):
#     #     if(len(request.files.getlist('file'))):
#     #         original_image = np.asarray(Image.open(uploaded_file))
#     #         #if uploaded_file.filename != '':
#     #         # frame = cv2.imdecode(uploaded_file)
#     #         img = Image.fromarray(original_image, 'RGB')
#     #         img.save("/Users/vishwas/Desktop/build_my_web/files/"+str(original_image.filename))
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
    
    



# # @app.route("/build")
# # def index():
# #     return render_template("build.html")
