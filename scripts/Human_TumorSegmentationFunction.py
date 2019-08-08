#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# - [Imports](#Imports)
# - [Functions](#Functions)
#     - [Reading Files](#Reading-Files)
#     - [Get Data Statistics](#Get-Data-Statistics)
#     - [Image Processing](#Image-Processing)
#     - [Model Generation](#Model-Generation)
#     - [Visualization](#Visualization)

# # Imports

# In[1]:


import os
import numpy
import SimpleITK

import scipy.ndimage
import scipy.misc
import scipy.misc as scioM

import matplotlib.pyplot as plt
get_ipython().magic(u'pylab inline')


# In[2]:


from os import listdir
from os.path import isfile, join


# In[3]:


from skimage import measure
from skimage import transform
import skimage as skImg
from skimage import morphology
from skimage import filters
# for this module the stl is now depricated
# pip install numpy-stl
from stl import mesh


# # Functions

# ## Reading Files

# In[4]:


def readMRIFile(inpath):
    # Read an MR file (.dcm,.nii,.mnc) and stores it in an ITK image stack file tensor
    
    reader = SimpleITK.ImageFileReader()
    mriImage = SimpleITK.ReadImage(inpath)
    return mriImage


# In[5]:


def convertToNP(itkImage):
    # Converts the a single ITK image into a numpy array
    
    npArray = SimpleITK.GetArrayFromImage(itkImage)
    return npArray    


# In[6]:


def applySmoothingMRI(singleMRIImage):
    # Applys smoothing to an ITK image and returns an ITK image
    
    mriImageSmooth = SimpleITK.CurvatureFlow(image1=singleMRIImage,
                                            timeStep=0.125,
                                            numberOfIterations=5)   
    return mriImageSmooth


# In[7]:


def readStackOfImage(imageStackDirectory):
    # Get all file names in the firectory 
    onlyFiles = [f for f in listdir(imageStackDirectory) if isfile(join(imageStackDirectory, f))]
    
    # get sizes of each image
    oneFile = imageStackDirectory + onlyFiles[0]
    oneFileImage = scipy.ndimage.imread(oneFile)
    [w,h] = oneFileImage.shape
    
    # Allocate variable for the image stack
    numSlices = len(onlyFiles)
    imageStack = np.zeros((w,h,numSlices))
    
    # Read and store image stack in an array
    for sliceImageFile in onlyFiles:
        imageStack[:,:,sliceInd] = scipy.ndimage.imread(sliceImageFile)
    
    return imageStack


# ## Get Data Statistics

# In[8]:


def getDataStatistics(npArray):
    [numSlices,width,height] = npArray.shape
    print("The data contains {} slices with the image resolution being {} x {}".format(numSlices,width,height))
    
    return [numSlices,width,height]


# ## Image Processing

# In[9]:


def showSegmentation(sliceNum=90):
    # Get slice
    idxSlice = sliceNum
    imgOriginal = image[:,:,idxSlice]

    # Apply smoothing
    imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                        timeStep=0.125,
                                        numberOfIterations=5)
    sitk_show(imgSmooth)

    # Get segmentation
    lstSeeds = [(75,60)]

    imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                                  seedList=lstSeeds, 
                                                  lower=0, 
                                                  upper=40,
                                                  replaceValue=labelWhiteMatter)

    # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
    imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

    # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
    sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatter))

    # show segmentation
    sitk_show(imgWhiteMatter)


# In[10]:


def getMask(sliceImage,pxlRanges):
#     sliceImage = np.copy(grayMatterMask[:,:,50])
    xLen,yLen = np.shape(sliceImage)
#     pxlRanges = grayMatterRange

    sizeRange = len(pxlRanges)
    
    if(sizeRange==1):
        sliceImage = np.round(sliceImage).astype(int)
        for x in xrange(xLen):
            for y in xrange(yLen):        
                if(sliceImage[x,y] == pxlRanges[0]):                        
                    sliceImage[x,y] = 1.0
                else:
                    sliceImage[x,y] = 0.0        
    else:
        lower = pxlRanges[0]
        upper = pxlRanges[1]

        for x in xrange(xLen):
            for y in xrange(yLen):        
                if(sliceImage[x,y]>=lower and sliceImage[x,y]<=upper):                        
                    sliceImage[x,y] = 1.0
                else:
                    sliceImage[x,y] = 0.0
    
    return sliceImage


# In[11]:


def applyMorphFilters(img,erodeFilter,dilationFilter):
    eroded = morphology.erosion(img,np.ones(erodeFilter))
    dilation = morphology.dilation(eroded,np.ones(dilationFilter))    
    return dilation


# ## Model Generation

# In[12]:


def make_mesh(image, threshold=-100, step_size=1):
    # Generates a mesh from a stack of images (image variable)
    
    print "Transposing surface"
    p = image.transpose(2,1,0)
    
    print "Calculating surface"
    try:
        verts, faces = measure.marching_cubes(p, threshold) 
    except:
        verts, faces = measure.marching_cubes_classic(p, threshold)         
    return verts, faces


# In[13]:


def make_meshNew(image, threshold=-100, step_size=1):
    # Generates a mesh from a stack of images (image variable)
    
    print "Transposing surface"
#     p = image.transpose(2,1,0)
    p=image
    
    print "Calculating surface"
    try:
        if(threshold==0.0):
            verts, faces = measure.marching_cubes(p) 
        else:            
            verts, faces = measure.marching_cubes(p, threshold) 
    except:
        if(threshold==0.0):
            verts, faces = measure.marching_cubes_classic(p)         
        else:
            verts, faces = measure.marching_cubes_classic(p, threshold)         
    return verts, faces


# In[14]:


def generateSTL(fileOut,imageStack,TH=0.0):
    # Note: imageStack is the output of the function readStackOfImage
    
    # Get verticies
    v, f = make_meshNew(imageStack,TH)
    
    # Generate mesh
    imageMesh = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
    
    # Connect Vertices
    for i, f in enumerate(f):
        for j in range(3):
            imageMesh.vectors[i][j] = v[f[j],:]
        
    # Create Mesh    
    imageMesh.save(fileOut)    


# In[15]:


def generateImageStack(imageStack,outFileDir,viewType='axial'): 
    # ----------------------------------------------------------------------------
    # imageStack = is a np array with the format [slice,width,height]
    # outFileDir = location where all the images will be saved    
    # ----------------------------------------------------------------------------   
    
    [depth,width,height] = imageStack.shape
    
    if(viewType=='axial'):
        numSlices =  depth
    elif(viewType=='coronal'):
        numSlices =  width
    elif(viewType=='sagittal'):
        numSlices =  height
    
    for sliceInd in xrange(numSlices):
        saveImg = getPlanarViewImage(imageStack,sliceInd,viewType=viewType)
        scioM.imsave(outFileDir + '{}.jpg'.format(sliceInd), saveImg)
    
    print('Create the image stack in the folder {}'.format(outFileDir))


# ## Visualization

# In[16]:


def showTHData(imgStack,sliceInd,th):
    # get a slice    
    [numSlices,width,height] = imgStack.shape
    imgStackSlice = imgStack[sliceInd,:,:].astype(int)

    # Threshold slices
    sliceColourCode = np.zeros([len(th),width,height])

    for thInd in xrange(len(th)):
        for w in xrange(width):
            for h in xrange(height):            
                if imgStackSlice[w,h] != th[thInd]:
                    sliceColourCode[thInd,w,h] = 0
                else:
                    sliceColourCode[thInd,w,h] = th[thInd]

    #Plot TH slices
    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    for thInd in xrange(len(th)):
        plt.subplot(1,len(th)+1,thInd+1)        
        plt.imshow(sliceColourCode[thInd,:,:])
        plt.title('TH = {}'.format(thInd+1))   

    plt.subplot(1,len(th)+1,len(th)+1)        
    plt.imshow(imgStackSlice)

    plt.tight_layout()
    plt.show()    


# In[17]:


def plotHistogram(imageStack,indSlice,viewType='axial',cmap='bone'):
    
    mrImg1 = getPlanarViewImage(imageStack,indSlice,viewType=viewType)
    
    # show second slice
    uniqueBins = (np.unique(mrImg1.astype(int).ravel()))
    
    # show first slice
    plt.subplot(1,2,1)    
    plt.set_cmap(cmap)        
    plt.imshow(mrImg1)
    plt.title('Showing Slice {}'.format(indSlice))
    
    # show second slice    
    plt.subplot(1,2,2)      
    plt.hist(mrImg1.ravel(),len(uniqueBins),[0,len(uniqueBins)]); 
    plt.title('Histogram')
    
    plt.show()


# In[18]:


def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    # Simple visualization tool to view a single slice
    
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


# In[19]:


def resizePlane(mrSlice,w,h):   
    # ----------------------------------------------------------------------------
    # resizedMRImage = image (np array) you want to resize
    # w = value of width
    # h = value of height
    # ----------------------------------------------------------------------------
    resizedMRImage = transform.resize(mrSlice,(w,h),anti_aliasing =True)
    return resizedMRImage


# In[20]:


def getPlanarViewImage(imgStack,indSlice,viewType='axial'):
    # ----------------------------------------------------------------------------
    # imageStack = is a np array with the format [slice,width,height]
    # indSlice = the index of the slice you want to view
    # viewType = string specifying planar views, can be 'axial', 'coronal', 'sagittal'     
    # ----------------------------------------------------------------------------
    [numSlices,width,height] = imgStack.shape    
    
    if(viewType=='axial'):
        mrImage =  imgStack[indSlice,:,:]    
    elif(viewType=='coronal'):
        mrImage =  np.flipud(imgStack[:,indSlice,:])
        mrImage = resizePlane(mrImage,width,height)        
    elif(viewType=='sagittal'):
        mrImage =  np.flipud(np.fliplr(imgStack[:,:,indSlice]))
        mrImage = resizePlane(mrImage,width,height)        
    
    return mrImage    


# In[21]:


def showTHMorphFiltering(imgStack,sliceInd,erodeFilter,dilationFilter,th):
    # -
    # imgStack = mr image stack in the form of  [slice, w,h]
    # sliceInd = slice index
    # erodeFilter = 2x2 array ex [1,1]
    # dilationFilter = 2x2 array ex [2,2]
    # th = a single digit array ex [1]
    # ------
    # get a random slice    
    mrSliceSeg = imgStack[sliceInd,:,:]

    # get a mask from a threshold    
    mrSliceTH = getMask(mrSliceSeg,th)

    # apply the morphological filters    
    morphImage = applyMorphFilters(mrSliceTH,erodeFilter,dilationFilter)

    # shoe the two side by side
    plt.figure()
    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1,3,1)
    plt.imshow(mrSliceSeg)
    plt.title('Unprocessed data')
    plt.subplot(1,3,2)
    plt.imshow(mrSliceTH)
    plt.title('Thresholded image')
    plt.subplot(1,3,3)
    plt.imshow(morphImage)
    plt.title('Morphological filters applied')
    plt.tight_layout()
    plt.show()    


# In[22]:


def display3Slices(imageStack,arrayIndSlice,viewType='axial',cmap="bone"):
    # ----------------------------------------------------------------------------
    # imageStack = is a np array with the format [slice,width,height]
    # arrayIndSlice = is a an array with 3 ind values no greater than max slice,max width or max height
    # viewType = string specifying planar views, can be 'axial', 'coronal', 'sagittal' 
    # cmap = the most commonly used ones are ["gray",'bone','binary','jet', 'gist_ncar','seismic','Set1',]
    # ----------------------------------------------------------------------------
    [numSlices,width,height] = imageStack.shape
    figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    
    # show first slice
    plt.subplot(1,3,1)    
    plt.set_cmap(cmap)    
    mrImg1 = getPlanarViewImage(imageStack,arrayIndSlice[0],viewType=viewType)
    plt.imshow(mrImg1)
    plt.title('Showing Slice {}'.format(arrayIndSlice[0]))
    
    # show second slice
    plt.subplot(1,3,2)    
    plt.set_cmap(cmap)    
    mrImg1 = getPlanarViewImage(imageStack,arrayIndSlice[1],viewType=viewType)
    plt.imshow(mrImg1)
    plt.title('Showing Slice {}'.format(arrayIndSlice[1]))
    
    # show third slice
    plt.subplot(1,3,3)    
    plt.set_cmap(cmap)    
    mrImg1 = getPlanarViewImage(imageStack,arrayIndSlice[2],viewType=viewType)
    plt.imshow(mrImg1)
    plt.title('Showing Slice {}'.format(arrayIndSlice[2]))
    
    plt.tight_layout()
    plt.show()