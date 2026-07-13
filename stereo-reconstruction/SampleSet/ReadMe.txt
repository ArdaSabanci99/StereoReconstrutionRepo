Image data set aimed at multiple view stereo (MVS), as describe at and downloadable from 

http://roboimagedata.imm.dtu.dk/

This data is citeware. If you use it, cite the associated paper:

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, Henrik Aanaes, “Large Scale Multi-view Stereopsis Evaluation“, CVPR, 2014

Please refer to this paper and the homepage for more details about this data set.

The folder structure for the data set in "MVS Data" is 

- "Calibration"	: Camera calibration data. 94 positions where used in the calibration but only the first 49/64 position where used for image acquisition. Also included are text-files with camera projection matrices for each position.
- "Cleaned"	    : Images from each scene with different lighting. The most diffuse lighting is denoted by a "_3_". The images have been processed (Cleaned) for dead pixels, and dark current.
- "ObsMask"	    : Information about which parts of 3D space should be used for evaluation. The use is exemplified in "PointsCompareMain.m", and explained in the paper.
- "Points"	    : Collective point clouds for each scan using: stl, campbell, furukawa and tola. The evaluation is done with stl as a reference.
- "Rectified"	: Rectified (corrected for radial distortion) images from the "Cleaned" folder.
- "Surfaces"	: Poisson reconstruction of Campbell, Furukawa and Tola using depth=11 and trim=8

To assist in using our data set for MVS evaluation, we have also included our evaluation protocol, implemented in MatLab. To run the full evaluation the full set of STL-points and MVS point and surface reconstructions needs to be downloaded and added to the folder structure mentioned above. Main example entry points is the files 

"BaseEvalMain_web.m"

This needs to be run for point and mesh reconstructions respectively for the different MVS methods and the two light setting experiments. 

“ComputeStat_web.m”

Computes the statistics used in our paper, after running the “BaseEvalMain.m” function.

- "MeshSupSamp_web" : is a MatLab mex function used for converting  triangle meshes into point clouds via sampling.

The function “plyread.m” is not of our creation, but something Pascal Getreuer has kindly made available on the internet.

Note also that the function 
“BaseEval2Obj_web.m” writes the results from the “BaseEvalMain.m” functions into and obj-file, e.g. viewable with MeshLab c.f. http://meshlab.sourceforge.net/
The .obj file shows the distance error from all stl points to the nearest data points (or the other way around) shown as white to red (0-10 mm) for points included in the analysis (above plane or in mask) 

In a photogrammetric project such as this, there are many (camera) frames of reference. In relation to our setup and the calibration file, the color images are supplied by the right camera, and the stl reconstructions are in the frame of reference of the left camera. To ease the use of our data, the camera projection matrices (one for each view) supplied in the Calibration folder the MVS results in the same frame of reference as the structured light reconstruction.

In the downloadable data some scenes have been acquired from 4 times each with 90 degree rotation, such that the scene is covered from 360 degrees.

Following is a list of 360 degree rotations of a scene:
55-58
65-68
69-73
106-109
110-113
114-117
118-121
122-125

 There is no exact transformation bringing the 4 scans into the same coordinate system but this can be done using ICP. The rotation between scenes are fairly consistent, so the transformation can be reused for ICP initialization.
