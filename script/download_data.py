import requests
from torch.serialization import save
from settings import PROJECT_ROOT
import pandas as pd 
from datetime import datetime as dt

# Downloads a csv file from url and saves it to savepath
def download_from_url(url, savepath):
    
    res = requests.get(url)
    if res.ok:
        csv_file = open(savepath, 'wb')
        csv_file.write(res.content)
        csv_file.close()
    else:
        print("Download failed, trying again for ", url)
        download_from_url(url, savepath)

# fetch CSV file lists from a file
def fetchCSVFileLists(filepath):
    f = open(filepath, "r")
 
    filelist = f.read().splitlines()
            
    return filelist


def download_data():

    filelist = fetchCSVFileLists(PROJECT_ROOT + "/script/csvfilelists.txt")
    save_paths = []

    for csv in filelist:
        defaultlink = "https://ai4impact.org/P003/"
        dirpath= PROJECT_ROOT+ "/data/forecast/"
        
        csvfilestr = csv + ".csv"
        csvfilewithb = csv + "-b.csv"
        
        url = defaultlink + csvfilestr
        savepath = dirpath + csvfilestr
        
        download_from_url(url, savepath) 
        save_paths.append(savepath)

        url = defaultlink + csvfilewithb 
        savepath = dirpath + csvfilewithb 
        
        download_from_url(url, savepath)
        save_paths.append(savepath)

    for csv in filelist:
        historicallink =  "https://ai4impact.org/P003/historical/"
        dirpath=PROJECT_ROOT+"/data/history/"
        
        csvfilestr = csv + ".csv"
        csvfilewithb = csv + "-b.csv"
        
        url = historicallink + csvfilestr
        savepath = dirpath + csvfilestr
        download_from_url(url, savepath) 
        save_paths.append(savepath)

        url = historicallink + csvfilewithb 
        savepath = dirpath + csvfilewithb 
        download_from_url(url, savepath)
        save_paths.append(savepath)

    energy_path = PROJECT_ROOT+"/data/energy-ile-de-france.csv"
    energy_link = "https://ai4impact.org/P003/historical/energy-ile-de-france.csv"
    download_from_url(energy_link, energy_path)
    
    # tag col names for energy dataset and save to wind_energy_v2.csv
    tag_colnames(energy_path, PROJECT_ROOT+"/data/wind_energy_v2.csv")

    print("=== DOWNLOADING FINISHED === ")

    return save_paths

def parse_data(history_path):
    save_path = generate_cleaned_paths(history_path)
    print("[Parsed Data] saved to: " , save_path)
    df = pd.read_csv(history_path, header=3)
    df["Time"] = df["Time"].apply(lambda x: dt.strptime(x, '%Y/%m/%d %H:%MUTC'))
    df.to_csv(save_path)

def generate_cleaned_paths(history_path):
    return history_path.replace("/history/", "/history_cleaned/")
    
def tag_colnames(energy_csv_path, save_path):
    df = pd.read_csv(energy_csv_path, names=["time", "energy"])
    df["time"] = df["time"].apply(lambda x: x+":00")
    df.to_csv(save_path)