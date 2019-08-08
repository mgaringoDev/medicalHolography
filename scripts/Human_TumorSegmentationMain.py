#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# - [Parameters](#Parameters)
# - [Main](#Main)
#     - [Reading and clean the stack](#Reading-and-clean-the-stack)
#         - [MRI Raw](#MRI-Raw)
#         - [MRI Segmentation](#MRI-Segmentation)
#     - [Visualize some slices](#Visualize-some-slices)
#     - [Generate image stack slices](#Generate-image-stack-slices)
#         - [Apply Morphological Filters](#Apply-Morphological-Filters)
#             - [White matter masks](#White-matter-masks)
#                 - [WM Parametric Tuning](#WM-Parametric-Tuning)
#                 - [Saving WM Model](#Saving-WM-Model)
#             - [Gray matter masks](#Gray-matter-masks)
#                 - [GM Parametric Tuning](#GM-Parametric-Tuning)
#                 - [Saving GM Model](#Saving-GM-Model)
#             - [Tumor Component 1](#Tumor-Component-1)
#                 - [TC1 Parametric Tuning](#TC1-Parametric-Tuning)
#                 - [Saving TC1 Model](#Saving-TC1-Model)
#             - [Tumor Component 2](#Tumor-Component-2)
#                 - [TC2 Parametric Tuning](#TC2-Parametric-Tuning)
#                 - [Saving TC2 Model](#Saving-TC2-Model)

# # Parameters

# In[23]:


inpath = 'C://medicalHolography//human//t2.dcm'


# In[24]:


inpathTruth = 'C://medicalHolography//human//truth.dcm'


# # Main

# ## Reading and clean the stack

# ### MRI Raw

# In[25]:


mriStack = readMRIFile(inpath)


# In[26]:


mriStackSmooth = applySmoothingMRI(mriStack)


# In[27]:


mriData = convertToNP(mriStackSmooth)


# In[28]:


[numSlices,width,height] = getDataStatistics(mriData)


# ### MRI Segmentation

# In[29]:


mriStackSeg = readMRIFile(inpathTruth)


# In[30]:


mriStackSmoothSeg = applySmoothingMRI(mriStackSeg)


# In[31]:


mriDataSeg = convertToNP(mriStackSmoothSeg)


# In[32]:


[numSlicesSeg,widthSeg,heightSeg] = getDataStatistics(mriDataSeg)


# ## Visualize some slices

# Let's visualize 3 slices to get a rough overview of the data we are dealing with

# In[33]:


display3Slices(mriData,[10,100,150])
display3Slices(mriData,[width/3+50,width/3*2+25,width-50],viewType='coronal')
display3Slices(mriData,[height/3+50,height/3*2-20,height-70],viewType='sagittal')


# Note: the following images might look distorted and they are, and the reason is that because we have limited number of slices interpolation must be done to resize the image to its original width and height.
# Note the deformation of the coronal and sagittal views because of the limited depth data.  Regardless I think we can use these views to narrow down the slices of interest.

# In[34]:


display3Slices(mriDataSeg,[10,100,150])
display3Slices(mriDataSeg,[width/3+50,width/3*2+25,width-50],viewType='coronal')
display3Slices(mriDataSeg,[height/3+50,height/3*2-20,height-70],viewType='sagittal')


# Looks like the truth is the segmentation already so let us that.

# ## Generate image stack slices

# This will be imported in 3DSlicer to view all the images. Note that this is done to here the most important slices are.

# In[35]:


# generateImageStack(mriData,outImageStackPathAxial,viewType='axial')
# generateImageStack(mriData,outImageStackPathCoronal,viewType='coronal')
# generateImageStack(mriData,outImageStackPathSagittal,viewType='sagittal')


# There are several methods of doing this in 3D slicer but this is very time consuming so let us do this in python instead.
# - [Segmentation & 3D Model Construction using 3D slicer](https://youtu.be/6GMfCZ1u7ds)
# - [Tutorial: Preparing Data for 3D Printing Using 3D Slicer](https://youtu.be/MKLWzD0PiIc)

# In the axial view we see that the tumor starts roughly around slice 94 and ends at slice 122.  Let us image this so that we can see the beginning, middle and end of the tumor

# In[36]:


display3Slices(mriData,[94,108,122])


# In[37]:


display3Slices(mriDataSeg,[94,108,122])


# In[38]:


plotHistogram(mriDataSeg,94,viewType='axial',cmap='bone')


# From the above histogram distribution I think thera are only 5 colours. Let us segment these and see which parts correspond to which area of the brain.

# In[39]:


# get the TH
th = [1,2,3,4,5]
showTHData(mriDataSeg,95,th)


# The above TH images seem to indicate that:
# 
# | TH | Anatomy |
# |:--:|:-----------------:|
# | 1 | White Matter |
# | 2 | Gray Matter |
# | 4 | Tumor Component 1 |
# | 5 | Tumor Component 2 |
# 
# But let us look at other slices to prove this

# In[40]:


plotHistogram(mriDataSeg,108,viewType='axial',cmap='bone')


# The histogram distribution of slice 108 looks like 94

# In[41]:


# get the TH
th = [1,2,3,4,5]
showTHData(mriDataSeg,108,th)


# The pattern still holds 

# In[42]:


plotHistogram(mriDataSeg,80,viewType='axial',cmap='bone')


# Again this looks very similar to the previous 2 slices

# In[43]:


# get the TH
th = [1,2,3,4,5]
showTHData(mriDataSeg,80,th)


# Based on the above slices [80,94,122] the following TH distribution is correct.
# 
# The above TH images seem to indicate that:
# 
# | TH | Anatomy |
# |:--:|:-----------------:|
# | 1 | White Matter |
# | 2 | Gray Matter |
# | 4 | Tumor Component 1 |
# | 5 | Tumor Component 2 |

# Looking at the thresholded values there seems to be some holes and some artifacts in each of the segments.  This can be problematic when generating the model.  We are first do some more processing to remove them as much as we can and then generate the model.  To remove them we can do a simple clustering algorithm.

# ### Apply Morphological Filters

# #### White matter masks

# ##### WM Parametric Tuning

# In[44]:


erodeFilter = [2,2]
dilationFilter = [3,3]
th = [1]


# In[45]:


showTHMorphFiltering(mriDataSeg,108,erodeFilter,dilationFilter,th)


# In[46]:


showTHMorphFiltering(mriDataSeg,120,erodeFilter,dilationFilter,th)


# This removes some tiny artifacts which is promissing.  This means we can extract the white matter in this way.  Not let us see if we can do the same for gray matter.

# ##### Saving WM Model

# In[81]:


whiteMatter = np.zeros(mriDataSeg.shape)
erodeFilter = [3,3]
dilationFilter = [3,3]
th = [1]


# In[82]:


for sliceInd in xrange(numSlices):
    #get a slice
    mrSliceSeg = mriDataSeg[sliceInd,:,:]
    # get a mask from a threshold    
    mrSliceTH = getMask(mrSliceSeg,th)
    # apply the morphological filters    
    whiteMatter[sliceInd,:,:] = applyMorphFilters(mrSliceTH,erodeFilter,dilationFilter)    


# In[83]:


display3Slices(whiteMatter,[94,108,122])


# In[84]:


outSTLDir = 'C://DB//medicalHolography//meshFiles//'
fileName = 'human_WhiteMatter_DrRueda.stl'
generateSTL(outSTLDir+fileName,whiteMatter)


# #### Gray matter masks

# ##### GM Parametric Tuning

# In[54]:


erodeFilter = [2,2]
dilationFilter = [3,3]
th = [2]


# In[55]:


showTHMorphFiltering(mriDataSeg,108,erodeFilter,dilationFilter,th)


# In[56]:


showTHMorphFiltering(mriDataSeg,120,erodeFilter,dilationFilter,th)


# This is looking pretty good and we get the added bonus of keeping the morphological filter settings.

# ##### Saving GM Model

# In[85]:


grayMatter = np.zeros(mriDataSeg.shape)
erodeFilter = [3,3]
dilationFilter = [3,3]
th = [2]


# In[86]:


for sliceInd in xrange(numSlices):
    #get a slice
    mrSliceSeg = mriDataSeg[sliceInd,:,:]
    # get a mask from a threshold    
    mrSliceTH = getMask(mrSliceSeg,th)
    # apply the morphological filters    
    grayMatter[sliceInd,:,:] = applyMorphFilters(mrSliceTH,erodeFilter,dilationFilter)    


# In[87]:


display3Slices(grayMatter,[94,108,122])


# In[88]:


outSTLDir = 'C://DB//medicalHolography//meshFiles//'
fileName = 'human_GrayMatter_DrRueda.stl'
generateSTL(outSTLDir+fileName,grayMatter)


# #### Tumor Component 1

# ##### TC1 Parametric Tuning

# In[57]:


erodeFilter = [2,2]
dilationFilter = [3,3]
th = [4]


# In[58]:


showTHMorphFiltering(mriDataSeg,108,erodeFilter,dilationFilter,th)


# In[59]:


showTHMorphFiltering(mriDataSeg,120,erodeFilter,dilationFilter,th)


# Again looking very good as it removes many of the random speckle artifacts.

# ##### Saving TC1  Model

# In[72]:


tc1Matter = np.zeros(mriDataSeg.shape)
erodeFilter = [3,3]
dilationFilter = [3,3]
th = [4]


# In[73]:


for sliceInd in xrange(numSlices):
    #get a slice
    mrSliceSeg = mriDataSeg[sliceInd,:,:]
    # get a mask from a threshold    
    mrSliceTH = getMask(mrSliceSeg,th)
    # apply the morphological filters    
    tc1Matter[sliceInd,:,:] = applyMorphFilters(mrSliceTH,erodeFilter,dilationFilter)    


# In[74]:


display3Slices(tc1Matter,[94,108,122])


# In[75]:


outSTLDir = 'C://DB//medicalHolography//meshFiles//'
fileName = 'human_TC1Matter_DrRueda.stl'
generateSTL(outSTLDir+fileName,tc1Matter)


# #### Tumor Component 2

# ##### TC2 Parametric Tuning

# In[60]:


erodeFilter = [3,3]
dilationFilter = [3,3]
th = [5]


# In[61]:


showTHMorphFiltering(mriDataSeg,108,erodeFilter,dilationFilter,th)


# In[62]:


showTHMorphFiltering(mriDataSeg,120,erodeFilter,dilationFilter,th)


# Again this looks very good and we have the added bonus of obtaining the morphological filters.

# ##### Saving TC2  Model

# In[77]:


tc2Matter = np.zeros(mriDataSeg.shape)
erodeFilter = [3,3]
dilationFilter = [3,3]
th = [5]


# In[78]:


for sliceInd in xrange(numSlices):
    #get a slice
    mrSliceSeg = mriDataSeg[sliceInd,:,:]
    # get a mask from a threshold    
    mrSliceTH = getMask(mrSliceSeg,th)
    # apply the morphological filters    
    tc2Matter[sliceInd,:,:] = applyMorphFilters(mrSliceTH,erodeFilter,dilationFilter)    


# In[79]:


display3Slices(tc2Matter,[94,108,122])


# In[80]:


outSTLDir = 'C://DB//medicalHolography//meshFiles//'
fileName = 'human_TC2Matter_DrRueda.stl'
generateSTL(outSTLDir+fileName,tc2Matter)