#!/usr/bin/python
# -*- coding: utf-8 -*-


from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import load_npz

def getDimsOfPath(path, squeeze=False):
    dims = (path.split('.')[-2].split("_")[-1]).split("x")
    dims = [ int(dim) for dim in dims[::-1] if dim != '' ]
    if squeeze:
        dims = [ x for x in dims if x != 1 ]
    return dims

def dimString(arr, prefix='', postfix='', keepLength1Dims=False):
    return dimStringFromShape(arr.shape, prefix=prefix, postfix=postfix, keepLength1Dims=keepLength1Dims)

def dimStringFromShape(shape, prefix='', postfix='', keepLength1Dims=False):
    if keepLength1Dims:
        shape = [x for x in shape]
    else:
        shape = [x for x in shape if x != 1]

    dimStr = ''
    for ii, dim in enumerate(shape[::-1]):
        if ii == 0:
            dimStr += str(dim)
        else: 
            dimStr += 'x' + str(dim)

    dimStr = prefix + dimStr + postfix

    return dimStr

def writeNumpy(array, path, makeParentsIfNotExist=False, keepLength1DimsInSuffix=False):
    if path[-4:] != '.raw':
        path += dimString(array, prefix='_', postfix='.raw', keepLength1Dims=keepLength1DimsInSuffix)

    try:
        array.tofile(path)
    except FileNotFoundError as e:
        if makeParentsIfNotExist:
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            array.tofile(path)
        else:
            raise e

def readNumpy(path, dtype=np.float32):
    dims = getDimsOfPath(path)

    return np.fromfile(path, dtype=dtype, count=-1, sep='').reshape(dims)

def readNumpyPatient(path, count, offset, dtype=np.float32):
    dims = getDimsOfPath(path)
    dims[0] = 1
    

    return np.fromfile(path, dtype=dtype, count=count, offset=offset, sep='').reshape(dims)



def saveAsNpz(arr, path, keepLength1DimsInSuffix=False, dtype=None):
    if dtype is not None:
        finfo = np.finfo(dtype)
        assert finfo.min <= np.min(arr), 'Value not in the range of {}: {} > {}'.format( dtype, finfo.min, np.min(arr) )
        assert finfo.max >= np.max(arr), 'Value not in the range of {}: {} < {}'.format( dtype, finfo.max, np.max(arr) )
        arr = arr.astype(dtype)

    # Append shape information to file name
    dimDesc = dimString(arr, prefix='_', postfix='', keepLength1Dims=keepLength1DimsInSuffix)

    np.savez_compressed( path + dimDesc, a=arr)
    
def loadNpz(path, dtype=np.float32):
    try:
        # Older files have been saved in scipy sparse matrix format
        arr_sparse = scipy.sparse.load_npz(path)
        arr = arr_sparse.toarray()
        
        shape = getDimsOfPath(path)
        arr = arr.reshape(shape)
    except ValueError:
        # New files are saved using np.savez_compressed
        arr = np.load( path )['a']

    if dtype is not None:
        if arr.dtype != dtype:
            arr.astype(dtype)

    return arr

def loadNpzTool(path, dtype=np.float32):
    try:
        # Older files have been saved in scipy sparse matrix format
        arr_sparse = scipy.sparse.load_npz(path)
        arr = arr_sparse.toarray()
        arr = arr[0][-1]
        
        shape = getDimsOfPath(path)
        arr = arr.reshape(shape)
    except ValueError:
        # New files are saved using np.savez_compressed
        arr = np.load( path )['a']

    if dtype is not None:
        if arr.dtype != dtype:
            arr.astype(dtype)

    return arr


def readSlicesOf3DVolume(path, ns_lo, ns_hi, dtype=np.float32):    
    dims = getDimsOfPath(path)
    if len(dims) == 2:
        Ns, Ny, Nx = [1] + dims
    elif len(dims) == 3:
        Ns, Ny, Nx = dims
    else:
        raise ValueError()

    if ns_hi is None:
        ns_hi = Ns

    # Checks
    assert ns_lo < ns_hi
    assert ns_hi <= Ns

    result = np.fromfile(path, dtype=dtype, count=(ns_hi-ns_lo)*Ny*Nx, offset=ns_lo*Ny*Nx*np.dtype(dtype).itemsize)
    result = result.reshape(((ns_hi-ns_lo), Ny, Nx))

    return result

def test():
    IORoot = 'C:/Data/tmp/testIOFunctions/'
    Path(IORoot).mkdir(exist_ok=True)

    array = np.ones((3,4,5), dtype=np.float32)

    # Save in raw-format 
    writeNumpy( array, IORoot + 'rawFile' )
    # Save in compressed format
    saveAsNpz( array, IORoot + 'npzFile' )

    array_loadedFromRaw = readNumpy( IORoot + 'rawFile_5x4x3.raw' )
    array_loadedFromNpz = loadNpz( IORoot + 'npzFile_5x4x3.npz' )

    assert np.array_equal( array, array_loadedFromRaw )
    assert np.array_equal( array, array_loadedFromNpz )

def NzpToRaw(ImageName):
    root = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/'
    root2 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/NpzToRawImages/'
    array_loadedFromNpz = loadNpz(root + ImageName)
    name = ImageName.split('_')
    name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]
    writeNumpy(array_loadedFromNpz, root2 + name)

    print("Npz Datei wurde nun als Raw Datei gespeichert!!!")
    

if __name__ == '__main__':
    NzpToRaw('ex_36_stent_intLength_1024x1024x8x2.npz')
    #root = './2022-05-31_patientProjsCount/projs/S163z-49_y10_x-60_pPrim_1024x1024x72.raw'
    #array = readNumpy(root)
    #print(array.shape)   
    #root2 =  'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/NpzToRawImages/ex_0_guidewire_intLength_1024x1024x8x2.raw'
   # array2 = readNumpy(root2)
    #print(array2.shape)
    #NzpToRaw('ex_2_guidewire_intLength_1024x1024x8x2.npz')

    # patient = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S163/z-49_y10_x-60_pPrim_1024x1024x72.raw'
    # patientImg = readNumpy(patient)
    # patientImg2 = readNumpyPatient(patient, 1024*1024, 4*1024*1024*1)

    # print(patientImg2)
    # print(patientImg[1])
 
    # print(patientImg.shape)
    # print(patientImg2.shape)