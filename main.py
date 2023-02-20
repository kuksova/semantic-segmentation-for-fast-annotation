from flask import Flask, request, render_template
from PIL import Image
import os

from src.model import predict


app = Flask(__name__)
UPLOAD_FOLDER = "../static/"

@app.route('/', methods=['GET', 'POST'])
def upload_predict():

    if request.method == "POST":
        image_file = request.files["image"]
        image_location = os.path.join(
            UPLOAD_FOLDER,
            image_file.filename
        )
        if image_file:
            lst = os.listdir(UPLOAD_FOLDER)
            for file in lst:
                if file.endswith('.png') or file.endswith('.jpg'):
                    os.remove(UPLOAD_FOLDER + file)
                    
                    
            print(image_location)
            image_file.save(image_location)

            #image_location = '../static/ADE_train_00000298.jpg'
            pred, pred_all = predict(image_location)

            pred_name = image_file.filename.split('.')[0]+".png"
            pred.save(UPLOAD_FOLDER + pred_name)
            
            
            individual_masks = []
            individual_size =  []
            for i in range(3):
                individual_masks.append(Image.new('RGB', (1,1)))
                individual_size.append(1)
            
            for i, mask in enumerate(pred_all):
                individual_masks[i] = Image.fromarray(mask).resize(pred.size, Image.NEAREST)
                individual_size[i] = 360
                
            individual_name = []
            for i, mask in enumerate(individual_masks):
                individual_name.append(image_file.filename.split('.')[0]+f"_{i}.png")
                individual_masks[i].save(UPLOAD_FOLDER + individual_name[i])
                


            #img_bytes = file.read()
            # mask_out = predict(img_bytes) #
            # print(type(mask_out))


            return render_template(
                "index.html",
                #prediction=1,
                #proba=round(0.0222222, 2),
                image_loc=image_file.filename,
                pred_loc = pred_name, 
                class1 = individual_name[0],
                size1 = individual_size[0],
                class2 = individual_name[1],
                size2 = individual_size[1],
                class3 = individual_name[2],
                size3 = individual_size[2]
            )
    return render_template("index.html", prediction=0, image_loc=None)



if __name__ == '__main__':
    #upload_predict()
    #predict("D:/Capstone_fastAPI/static/ADE_train_00000298.jpg")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    