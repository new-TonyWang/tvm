import os

path= "/mnt/8844dd41-4261-429b-83cc-c73ab7c65f8c/tony/workspace/vscode/tvm/tvm/apps/layers_dataset/Activation_layers/"
lable_file="/mnt/8844dd41-4261-429b-83cc-c73ab7c65f8c/tony/workspace/vscode/tvm/tvm/apps/layers_dataset/Activation_layers/label.txt"
with open(lable_file,"w") as file:
    for filename in os.listdir(path):
        if (os.path.isfile(os.path.join(path, filename))and filename.endswith(".h5"))==True:
            # print(filename)
            try:
                param=eval(filename[filename.find("-")+1:filename.find(".h5")])
            except:
                continue
            new_name = ''
            for char in filename:
                if char in ['{','\'',':','(',')','[',']','}',' ']:
                    new_name += '_'
                else:
                    new_name+=char
            
            layername=filename[0:filename.find("-")]
            record="{}\\{}\n".format(new_name,str(param))
            file.write(record)
            os.rename(os.path.join(path,filename),os.path.join(path,new_name))

            