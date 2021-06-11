import os
parentdir=os.path.join(os.getcwd(),"Activation_layers")
with open("{}/labelnew.txt".format(os.getcwd()),"w") as file:    
    with open("{}/label.txt".format(os.getcwd()),"r") as label:    
        for line in label:
            new_lin=''
            l=line.find(" ")
            new_lin+=line[0:l]
            l+=2
            layer_name=''
            while(line[l]!=" "):
                layer_name +=line[l]
                l +=1
            new_lin+="\\"
            l +=1
            new_lin+=line[l:-1]
            new_lin+="\n"
            file.write(new_lin)
        
            