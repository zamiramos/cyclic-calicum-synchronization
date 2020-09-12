import pickle

def save_object(obj, path):    
    fileObject = open(path,'wb')
    try:
        pickle.dump(obj,fileObject, protocol=4)
        fileObject.close()
    except:
        print('error occures while saving')
        fileObject.close()

def load_object(path):
    obj = None
    fileObject = None
    try:
        fileObject = open(path,'rb')
        obj = pickle.load(fileObject)
        fileObject.close()
    except Exception as e:
        if fileObject is not None:
            print('error occures while loadin file path:' + path)
            print(e)
            fileObject.close()
        else:
            print('file is not exist path:' + path)
    
    return obj