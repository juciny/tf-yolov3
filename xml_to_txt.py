
import xml.dom.minidom
import os
import glob

# 最终要获取的  是一个  label.txt
# path bbox[0]  class[0]  bbox[1]  class[1]

def get_info(input_file_path,output_file_path):
    label_array=[]
    file_grabbed=glob.glob(os.path.join(input_file_path,'*.xml'))
    print("label total number：",len(file_grabbed))

    for file in file_grabbed:
        print(file)
        label=[]
        dom=xml.dom.minidom.parse(file)
        root=dom.documentElement

        filename_list=root.getElementsByTagName("filename")
        path_list=root.getElementsByTagName("path")
        filename=filename_list[0].childNodes[0].data
        path=path_list[0].childNodes[0].data

        new_path=output_file_path+filename
        print(new_path)
        label.append(new_path)
        # size_list=root.getElementsByTagName("size")
        object_list=root.getElementsByTagName("object")

        for object in object_list:
            name_list=object.getElementsByTagName("name")
            name=name_list[0].childNodes[0].data

            print(class_name.index(name))

            bbox_list=object.getElementsByTagName("bndbox")
            bbox_xmin=bbox_list[0].getElementsByTagName("xmin")
            bbox_ymin=bbox_list[0].getElementsByTagName("ymin")
            bbox_xmax=bbox_list[0].getElementsByTagName("xmax")
            bbox_ymax=bbox_list[0].getElementsByTagName("ymax")

            xmin=bbox_xmin[0].childNodes[0].data
            ymin=bbox_ymin[0].childNodes[0].data
            xmax=bbox_xmax[0].childNodes[0].data
            ymax=bbox_ymax[0].childNodes[0].data

            class_id=class_name.index(name)
            print(class_id,xmin,ymin,xmax,ymax)
            label.append(xmin)
            label.append(ymin)
            label.append(xmax)
            label.append(ymax)
            label.append(str(class_id))
        label_array.append(label)
    return label_array

def write_info_to_txt(txt_name,label_array):
    with open(txt_name,"w") as f:
        for label in label_array:
            line=label[0]
            for index,item in enumerate(label):
                if index==0:
                    continue
                line=line+" "+item

            print(line)
            f.writelines(line)
            f.write("\n")

if __name__=='__main__':
    # 类别的名字
    class_name=["pig"]
    # xml文件的路径
    input_file_path="/Users/zy/Desktop/local-code/data/pig-label/"
    # 图片在yolov3根目录下存放的位置
    output_file_path="./pig_dataset/images/"
    label_array=get_info(input_file_path,output_file_path)
    write_info_to_txt("labels.txt",label_array)
