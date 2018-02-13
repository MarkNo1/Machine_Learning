import numpy as np

def bottleneck3D(A):
    #Convert the array in 1D
    A_1D = A.flatten()
    #Find the indices in the 1D array
    idx_1D = A_1D.argsort()
    #Convert the 1D indices into indices array for each dimension
    x_idx,y_idx,z_idx = np.unravel_index(idx_1D,A.shape)
    #Return the indices of the three max value, max1 is the Capo dei capi dei massimi 
    First_max = x_idx[len(x_idx)-1],y_idx[len(y_idx)-1],z_idx[len(z_idx)-1]

    Second_max = x_idx[len(x_idx)-2],y_idx[len(y_idx)-2],z_idx[len(z_idx)-2]

    Third_max = x_idx[len(x_idx)-3],y_idx[len(y_idx)-3],z_idx[len(z_idx)-3]

    #Avoid overfitting
    if (A[First_max] == 1 ):
        First_max = Second_max 
    
    return First_max, Second_max, Third_max

def bottleneck2D(A):
    #Convert the array in 1D
    A_1D = A.flatten()
    #Find the indices in the 1D array
    idx_1D = A_1D.argsort()
    #Convert the 1D indices into indices array for each dimension
    x_idx,y_idx= np.unravel_index(idx_1D,A.shape)
    #Return the indices of the three max value, max1 is the Capo dei capi dei massimi 
    First_max = x_idx[len(x_idx)-1],y_idx[len(y_idx)-1]

    Second_max = x_idx[len(x_idx)-2],y_idx[len(y_idx)-2]

    Third_max = x_idx[len(x_idx)-3],y_idx[len(y_idx)-3]

    #Avoid overfitting
    if (A[First_max] == 1 ):
        First_max = Second_max 
    
    return First_max, Second_max, Third_max
