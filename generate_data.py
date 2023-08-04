import numpy as np
import rir_generator as rir
import tqdm


def generate_ism(n_data=40000):
    """
    Generate ISM data
    """
    N_data = n_data           # 40000
    data = np.zeros((4105,1))
    temp_data = np.zeros((4105,1))
    #
    # 14, 1 for error,  3 for clean data
    np.random.seed(14)
    print("pie", np.random.randint(low=1,high = 9))
    count = 0
    for i in tqdm.trange(N_data,):
        # random room geometry 
        # L x H x W
        room_geo  = [np.random.randint(low=60, high=100)/10, np.random.randint(low=50, high=80)/10, np.random.randint(low=40, high=60)/10]
        # random room coeffs
        room_coeff = np.random.uniform(size=(6)).round(2)
        room_coeff[room_coeff == 1.0] = 0.99
        # random source and receiver position
        source_position = [np.random.randint(low=(len*10)/2-20, high=(len*10)/2+20)/10 for len in room_geo]
        #'receiver_position = []
        # check to avoid same source and receiver
        
        while True:
            receiver_position = [np.random.randint(low=(len*10)/2-20, high=(len*10)/2+20)/10 for len in room_geo]
            if (np.sort(source_position) == np.sort(receiver_position)).all():
                print(i, " receiver position: ", receiver_position, "source position: ", source_position)
                count += 1
                continue
            else:
                break
        
        #print("Room geometry: ", room_geo, " receiver position: ", receiver_position, "source position: ", source_position, "room coeff: ", room_coeff)
        # generate ISM data
        h = rir.generate(
            c=340,                                      # Sound velocity (m/s)
            fs=8000,                                    # Sample frequency (samples/s)
            r=np.sort(receiver_position),               # Receiver position(s) [x y z] (m)
            s=np.sort(source_position),                 # Source position [x y z] (m)
            L=np.sort(room_geo),                        # Room dimensions [x y z] (m)
            nsample=4096,                               # Number of output samples
            beta=np.sort(room_coeff)                    # Room reflection coefficients [x1 x2 y1 y2 z1 z2]
        )
        
        # concat the geometry and reflection coefficients
        if np.isnan(h).any() == True:
            print("nan values", room_coeff, room_geo, source_position, receiver_position)
            continue
        else:
            h = np.concatenate((np.concatenate((room_geo, room_coeff)), h.squeeze()))
            h = np.expand_dims(h, axis=1)
            # save ISM data
            #print("ISM data: ", h.shape)
            temp_data = np.hstack((temp_data,h))
            #print("ISM data: ", data.shape[1])
        if temp_data.shape[1] > 1000:
            #print("temp: ", temp_data.shape, data.shape)
            data = np.hstack((data, temp_data[:, 1:]))
            temp_data = np.zeros((4105,1))    

    # remove 1st zero column
    print("count" , count) 
    data = data[:,1:]
    print(" data: ", data.shape[1])
    return data.T   


if __name__ == "__main__":
    # get data
    data = generate_ism(n_data=40000)
    print("Total data: ", data.shape)
    # save file
    np.save(file="rirData/ism_40k.npy", arr=data)