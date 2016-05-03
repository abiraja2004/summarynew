__author__ = 'HyNguyen'
import os



def generate_settint_duc_05():
    model_path = "data/model/duc05"
    peer_path = "data/peer/duc05"
    rouge_model_path = "model/duc05"
    rouge_peer_path = "peer/duc05"
    file_model_names = os.listdir(model_path)
    file_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for file_name in file_peer_names:
        if file_name[0] == ".":
            continue
        file_id,_ = file_name.split(".")
        files_model_name_by_id = [file_model_name for file_model_name in file_model_names if file_model_name.find(file_id) != -1]
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+file_name +'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        for file_model_name_by_id in files_model_name_by_id:
            _,_,_,_,model_id = file_model_name_by_id.split(".")
            config_file += '\t\t\t<M ID="'+model_id+'">'+file_model_name_by_id+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/setting_duc05.xml','w')
    file_settings.write(config_file)
    file_settings.close()

def generate_settint_duc_04():
    model_path = "data/model/duc04"
    peer_path = "data/peer/duc04"
    rouge_model_path = "model/duc04"
    rouge_peer_path = "peer/duc04"
    file_model_names = os.listdir(model_path)
    file_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for file_name in file_peer_names:
        if file_name[0] == ".":
            continue
        file_id,_ = file_name.split(".")
        files_model_name_by_id = [file_model_name for file_model_name in file_model_names if file_model_name.find(file_id) != -1]
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+file_name +'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        for file_model_name_by_id in files_model_name_by_id:
            _,_,_,_,model_id = file_model_name_by_id.split(".")
            config_file += '\t\t\t<M ID="'+model_id+'">'+file_model_name_by_id+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/setting_duc04.xml','w')
    file_settings.write(config_file)
    file_settings.close()

def generate_dailymail():
    model_path = "data/model/dailymail"
    peer_path = "data/peer/dailymail"
    rouge_model_path = "model/dailymail"
    rouge_peer_path = "peer/dailymail"
    file_model_names = os.listdir(model_path)
    file_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for file_name in file_peer_names:
        if file_name[0] == ".":
            continue
        DAILY,MAIL,ID,COUNTER = file_name.split(".")
        file_model = "DAILY.MAIL." + ID
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+file_name +'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        config_file += '\t\t\t<M ID="'+ID+'">'+file_model+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/setting_dailymail.xml','w')
    file_settings.write(config_file)
    file_settings.close()

def generate_opinosis():
    model_path = "data/model"
    peer_path = "data/peer"
    rouge_model_path = "model"
    rouge_peer_path = "peer"
    folder_model_names = os.listdir(model_path)
    folder_peer_names = os.listdir(peer_path)

    counter  = 0
    config_file = '<ROUGE_EVAL version="1.55">'
    for folder_name in folder_peer_names:
        config_file += '\t<EVAL ID="' + str(counter) + '">\n'
        config_file += '\t\t<MODEL-ROOT>\n'
        config_file += '\t\t\t'+rouge_model_path+"/"+folder_name +'\n'
        config_file += '\t\t</MODEL-ROOT>\n'
        config_file += '\t\t<PEER-ROOT>\n'
        config_file += '\t\t\t'+rouge_peer_path+"/"+folder_name +'\n'
        config_file += '\t\t</PEER-ROOT>\n'
        config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
        config_file += '\t\t</INPUT-FORMAT>\n'
        config_file += '\t\t<PEERS>\n'
        config_file += '\t\t\t<P ID="1">'+folder_name+".1.txt"+'</P>\n'
        config_file += '\t\t</PEERS>\n'
        config_file += '\t\t<MODELS>\n'
        for file_model in os.listdir(model_path+"/"+folder_name):
            _,model_id,_ = file_model.split(".")
            config_file += '\t\t\t<M ID="'+model_id+'">'+file_model+'</M>\n'
        config_file += '\t\t</MODELS>\n'
        config_file += '\t</EVAL>\n\n'
        counter +=1
    config_file += '</ROUGE_EVAL>\n'
    file_settings = open('data/setting.xml','w')
    file_settings.write(config_file)
    file_settings.close()

if __name__ == "__main__":
    generate_settint_duc_04()

