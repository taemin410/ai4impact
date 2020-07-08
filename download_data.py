import requests


# Downloads a csv file from url and saves it to savepath
def downloadFromUrl(url, savepath):
    
    req = requests.get(url)
    csv_file = open(savepath, 'wb')

    csv_file.write(req.content)
    csv_file.close()

# fetch CSV file lists from a file
def fetchCSVFileLists(filepath):
    f = open(filepath, "r")
 
    filelist = f.read().splitlines()
            
    return filelist


def downloadData():

    filelist = fetchCSVFileLists("csvfilelists.txt")


    for csv in filelist:
        defaultlink = "https://ai4impact.org/P003/"
        dirpath="data/forecast/"
        
        csvfilestr = csv + ".csv"
        csvfilewithb = csv + "-b.csv"
        
        url = defaultlink + csvfilestr
        savepath = dirpath + csvfilestr
        
        downloadFromUrl(url, savepath) 
        
        url = defaultlink + csvfilewithb 
        savepath = dirpath + csvfilewithb 
        
        downloadFromUrl(url, savepath)

    for csv in filelist:
        historicallink =  "https://ai4impact.org/P003/historical/"
        dirpath="data/history/"
        
        csvfilestr = csv + ".csv"
        csvfilewithb = csv + "-b.csv"
        
        url = defaultlink + csvfilestr
        savepath = dirpath + csvfilestr
        
        downloadFromUrl(url, savepath) 

        url = defaultlink + csvfilewithb 
        savepath = dirpath + csvfilewithb 
        
        downloadFromUrl(url, savepath)

        
        
downloadData()


