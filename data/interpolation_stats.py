import pandas as pd
import matplotlib.pyplot as plt 
import os

dirs_ = os.listdir('./history_cleaned')
dirs_ = ['./history_cleaned/'+dir_ for dir_ in dirs_]
for dir_ in dirs_:
    # Speed(m/s),Direction (deg N)
    df = pd.read_csv(dir_)
    speed = df['Speed(m/s)'].tolist()[-10:]
    direction = df['Direction (deg N)'].tolist()[-10:]
    
    plt.figure()
    plt.plot(speed)
    plt.savefig('./plot/wind/'+ dir_[dir_.rfind('/'):dir_.rfind('.')]+".png")
    plt.clf()
    plt.plot(direction)
    plt.savefig('./plot/direction/'+ dir_[dir_.rfind('/'):dir_.rfind('.')]+".png")
    plt.clf()


