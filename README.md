<p align="center">
  <img src = "./imgs/logo1.png" width="60%">
</p>

`Dual-Radar` (provided by ['ADEPTLab'](https://tsinghua-adept.yuque.com/staff-mhp6ro/qgezu9/ubilx1)) is a brand new dataset based on 4D radar that can be used for studies on deep learning object detection and tracking in the field of autonomous driving. The system of ego vehicle includes a high-resolution camera, a 80-line LiDAR and two up-to-date and different models of 4D radars operating in different modes(Arbe and ARS548). The dataset comprises of raw data collected from ego vehicle, including scenarios such as tunnels and urban, with weather conditions rainy, cloudy and sunny. Our dataset also includes data from different time periods, including dusk, nighttime, and daytime. Our collected raw data amounts to a total of 12.5 hours, encompassing a total distance of over 600 kilometers. Our dataset covers a route distance of approximately 50 kilometers. It consists of 151 continuous time sequences, with the majority being 20-second sequences, resulting in a total of 10,007 carefully time-synchronized frames.

<table class="table-noborder">
  <tr>
    <td align="center">
      <figure>
        <img src="./imgs/5.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>a) First-person perspective observation</b></font></p>
    </td>
    <td align="center">
      <figure>
        <img src="./imgs/7.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>b) Third-person perspective observation</b></font></p>
    </td>
  </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Figure 1. Up to a visual range of 80 meters in urban</b></font></p>

## Radar Dataset

*  <b>Notice</b>: On the left is a color RGB image, while on the right side, the cyan represents the Arbe point cloud, the white represents the LiDAR point cloud, and the yellow represents the ARS548 point cloud.

<table class="table-noborder">
  <tr>
    <td align="center">
      <figure>
        <img src="./imgs/1.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>a) sunny,daytime,up to a distance of 80 meters</b></font></p>
    </td>
    <td align="center">
      <figure>
        <img src="./imgs/2.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>b) sunny,nightime,up to a distance of 80 meters</b></font></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <figure>
        <img src="./imgs/3.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>c) rainy,daytime,up to a distance of 80 meters</b></font></p>
    </td>
    <td align="center">
      <figure>
        <img src="./imgs/4.gif" alt="Image 1" width="500" height="200">
      </figure>
      <p align="center"><font face="Helvetica" size=2.><b>d) cloudy,daytime,up to a distance of 80 meters</b></font></p>
    </td>
  </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Figure 1. Up to a visual range of 80 meters in urban</b></font></p>


<div align=left>
<img src="./imgs/mutishow1.png"/>
</div>
<p align="center"><font face="Helvetica" size=3.><b>Figure 2. Data Visualization</b></font></p>

The URLs listed below are useful for using the `Dual-Radar` dataset and benchmark:

* [paper](https://arxiv.org/abs/2206.08171) `Dual-Radar` paper and appendix [arxiv] </a>

# Sensor Configuration 

<div align=center>
<img src="./imgs/ego_config_final.png"  width="500" height="300"/>
</div>
<p align="center"><font face="Helvetica" size=3.><b>Figure 3. Sensor Configuration and Coordinate Systems</b></font></p>


* The specification of the autonomous vehicle system platform

<div align=center>
<table border="10" >
     <tr>
        <td rowspan="2" align=center>Dataset</td>
        <td colspan="3" align=center>Resolution</td>
        <td colspan="3" align=center>Fov</td>
        <td rowspan="2">FPS</td>
     </tr>
      <tr>
        <td>Range</td>
        <td>Azimuth</td>
        <td>Elevation</td>
        <td>Range</td>
        <td>Azimuth</td>
        <td>Elevation</td>
     </tr>
      <tr>
        <td align=center>camera</td>
        <td align=center>-</td>
        <td align=center>1920X</td>
        <td align=center>1200X</td>
        <td align=center>-</td>
        <td align=center>-</td>
        <td align=center>-</td>
        <td align=center>10</td>
     </tr>
     <tr>
        <td align=center>LiDAR</td>
        <td align=center>0.05m</td>
        <td align=center>0.2°</td>
        <td align=center>0.2°</td>
        <td align=center>230m</td>
        <td align=center>360°</td>
        <td align=center>40°</td>
        <td align=center>10</td>
     </tr>
     <tr>
        <td align=center>ARS548 RDI</td>
        <td align=center>0.22m</td>
        <td align=center>0.1°</td>
        <td align=center>0.1°</td>
        <td align=center>300m</td>
        <td align=center>±60°</td>
        <td align=center>±4°</td>
        <td align=center>20</td>
     </tr>
     <tr>
        <td align=center>Arbe Phoenix</td>
        <td align=center>0.07m</td>
        <td align=center>1°</td>
        <td align=center>2°</td>
        <td align=center>300m</td>
        <td align=center>100°</td>
        <td align=center>30°</td>
        <td align=center>20</td>
     </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 1. The specification of the autonomous vehicle system platform</b></font></p>
</div>

* The statistics of number of points cloud per frame

<div align=center>
<table border="10" >
     <tr align=center>
        <td>Transducers</td>
        <td>Minimum Values</td>
        <td>Average Values</td>
        <td>Maximum Values</td>
     </tr>
      <tr align=center>
        <td>LiDAR</td>
        <td>74386</td>
        <td>116096</td>
        <td>133538</td>
     </tr>
      <tr align=center>
        <td>Arbe Phnoeix</td>
        <td>898</td>
        <td>11172</td>
        <td>93721</td>
     </tr>
      <tr align=center>
        <td>ARS548 RDI</td>
        <td>243</td>
        <td>523</td>
        <td>800</td>
     </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 2. The statistics of number of points cloud per frame</b></font></p>
</div>

# Data statistics
We separately counted the number of instances for each category in the Dual-Radar dataset and the distribution of different types of weather.
<div align=center>
<img src="./imgs/weather.png" width="500" height="320" />
</div>
<p align="center"><font face="Helvetica" size=3.><b>Figure 4. Distribution of weather conditions.</b></font></p>

About two-thirds of our data were collected under normal weather conditions, and about one-third were collected under rainy and cloudy conditions. We collected 577 frames in rainy weather, which is about 5.5% of the total dataset. The rainy weather data we collect can be used to test the performance of different 4D radars in adverse weather conditions.

<div align=center>
<img src="./imgs/class_show.png"/>
</div>
<p align="center"><font face="Helvetica" size=3.><b>Figure 5. Distribution of instance conditions.</b></font></p>
We also conducted a statistical analysis of the number of objects with each label at different distance ranges from our vehicle, as shown in Fig. Most objects are within 60 meters of our ego vehicle. 

# Environment
This is the documentation for how to use our detection frameworks with Dual-Radar dataset.
We tested the Dual-Radar detection frameworks on the following environment:

* Python 3.8.16 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.10.1+cu113
* CUDA 11.3
* opencv 4.2.0.32

## Notice

[2022-09-30] The `Dual-Radar` dataset is made available via a network-attached storage (NAS) [download link](http://QuickConnect.to/kaistavelab).

## Preparing the Dataset

* Via our server

1. To download the dataset, log in to <a href="http://QuickConnect.to/kaistavelab"> our server </a> with the following credentials: 
      ID       : your email
      Password : your password
2. After all files are downloaded, please arrange the workspace directory with the following structure:


Organize your code structure as follows

    Frameworks
      ├── checkpoints
      ├── data
      ├── docs
      ├── Dual-Radardet
      ├── output

Organize the dataset according to the following file structure

    Dataset
      ├── ImageSets
      ├── training
            ├── arbe
            ├── ars548
            ├── calib
            ├── image_2
            ├── label_2
            ├── velodyne
      ├── testing
            ├── arbe
            ├── ars548
            ├── calib
            ├── image_2
            ├── velodyne

## Requirements

1. Clone the repository

```
 git clone https://github.com/adept-thu/Dual-Radar.git
 cd Dual-Radardet
```

2. Create a conda environment
```
conda create -n Dual-Radardet python=3.8.16
conda activate Dual-Radardet
```

3. Install PyTorch (We recommend pytorch 1.10.1.)

4. Install the dependencies
```
pip install -r requirements.txt
```
5. Install Spconv（our cuda version is 113）
```
pip install spconv-cu113
```
6. Build packages for Dual-Radardet
```
python setup.py develop
```

## Train & Evaluation
* Generate the data infos by running the following command:
```
python -m pcdet.datasets.mine.kitti_dataset create_mine_infos tools/cfgs/dataset_configs/mine_dataset.yaml
# or you want to use arbe data
python -m pcdet.datasets.mine.kitti_dataset_arbe create_mine_infos tools/cfgs/dataset_configs/mine_dataset_arbe.yaml
# or ars548
python -m pcdet.datasets.mine.kitti_dataset_ars548 create_mine_infos tools/cfgs/dataset_configs/mine_dataset_ars548.yaml
```
* To train the model on single GPU, prepare the total dataset and run
```
python train.py --cfg_file ${CONFIG_FILE}
```
* To train the model on multi-GPUS, prepare the total dataset and run
```
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```
* To evaluate the model on single GPU, modify the path and run
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```
* To evaluate the model on multi-GPUS, modify the path and run
```
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```
## Quick Demo
Here we provide a quick demo to test a pretrained model on the custom point cloud data and visualize the predicted results
* Download the pretrained model as shown in Table 4~8.
* Make sure you have installed the Open3d and mayavi visualization tools. If not, you could install it as follow: 
```
pip install open3d
pip install mayavi
```
* prepare your point cloud data
```
points[:, 3] = 0 
np.save(`my_data.npy`, points)
```
* Run the demo with a pretrained model and  point cloud data as follows
```
python demo.py --cfg_file ${CONFIG_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA}
```

## Experimental Results
<div align=center>
<table>
     <tr align=center>
        <td rowspan="3">Baseline</td> 
        <td rowspan="3" align=center>Data</td> 
        <td colspan="3" align=center>Car</td>
        <td colspan="3" align=center>Pedestrain</td>
        <td colspan="3" align=center>Cyclist</td>
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
        <td colspan="3" align=center>3D@0.7</td>
        <td colspan="3" align=center>3D@0.7</td>
        <td colspan="3" align=center>3D@0.7</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="3">VFF</td> 
        <td>camera+LiDAR</td>
        <td>94.60</td>
        <td>84.14</td>
        <td>78.77</td>
        <td>39.79</td>
        <td>35.99</td>
        <td>36.54</td>
        <td>55.87</td>
        <td>51.55</td>
        <td>51.00</td>
        <td>https://pan.baidu.com/s/17VYvS5iDfse770DR4ILUWQ?pwd=8888<td>
    </tr>
    <tr align=center>
        <td>camera+Arbe</td>
        <td>31.83</td>
        <td>14.43</td>
        <td>11.30</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.20</td>
        <td>0.07</td>
        <td>0.08</td>
        <td>https://pan.baidu.com/s/1eob5XHdQbaVStXL2BC26pA?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>camera+ARS548</td>
        <td>12.60</td>
        <td>6.53</td>
        <td>4.51</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>https://pan.baidu.com/s/1W3O5OfmFLxcyYMSgZjKg7Q?pwd=8888</td>
    </tr>
    <tr align=center>
        <td rowspan="3">M2Fusion</td> 
        <td>LiDAR+Arbe</td>
        <td>67.37</td>
        <td>46.50</td>
        <td>33.17</td>
        <td>12.90</td>
        <td>8.45</td>
        <td>8.36</td>
        <td>51.79</td>
        <td>46.79</td>
        <td>45.57</td>
        <td>https://pan.baidu.com/s/1nRyibZj-3K8R_Q7Yq8nuxw?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>LiDAR+ARS548</td>
        <td>69.56</td>
        <td>49.13</td>
        <td>35.43</td>
        <td>12.69</td>
        <td>9.80</td>
        <td>9.67</td>
        <td>42.42</td>
        <td>40.92</td>
        <td>39.98</td>
        <td>https://pan.baidu.com/s/1F2qTt33XrvEionzI5WJHrw?pwd=8888</td>
    </tr>

</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 3. Multimodal Experimental Results(3D@0.7 0.5 0.5)</b></font></p>
</div>
</div>


<div align=center>
<table>
     <tr align=center>
        <td rowspan="3">Baseline</td> 
        <td rowspan="3" align=center>Data</td> 
        <td colspan="3" align=center>Car</td>
        <td colspan="3" align=center>Pedestrain</td>
        <td colspan="3" align=center>Cyclist</td>
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
        <td colspan="3" align=center>BEV@0.7</td>
        <td colspan="3" align=center>BEV@0.7</td>
        <td colspan="3" align=center>BEV@0.7</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="3">VFF</td> 
        <td>camera+Lidar</td>
        <td>94.60</td>
        <td>84.28</td>
        <td>80.55</td>
        <td>40.32</td>
        <td>36.59</td>
        <td>37.28</td>
        <td>55.87</td>
        <td>51.55</td>
        <td>51.00</td>
        <td>https://pan.baidu.com/s/17VYvS5iDfse770DR4ILUWQ?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>camera+Arbe</td>
        <td>36.09</td>
        <td>17.20</td>
        <td>13.23</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.20</td>
        <td>0.08</td>
        <td>0.08</td>
        <td>https://pan.baidu.com/s/1eob5XHdQbaVStXL2BC26pA?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>camera+ARS548</td>
        <td>16.34</td>
        <td>9.58</td>
        <td>6.61</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>https://pan.baidu.com/s/1W3O5OfmFLxcyYMSgZjKg7Q?pwd=8888</td>
    </tr>
    <tr align=center>
        <td rowspan="3">M2Fusion</td> 
        <td>LiDAR+Arbe</td>
        <td>81.36</td>
        <td>70.31</td>
        <td>54.09</td>
        <td>17.45</td>
        <td>12.67</td>
        <td>12.59</td>
        <td>53.06</td>
        <td>47.83</td>
        <td>46.32</td>
        <td>https://pan.baidu.com/s/1nRyibZj-3K8R_Q7Yq8nuxw?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>LiDAR+ARS548</td>
        <td>86.88</td>
        <td>74.75</td>
        <td>57.52</td>
        <td>21.50</td>
        <td>18.18</td>
        <td>18.00</td>
        <td>43.12</td>
        <td>41.57</td>
        <td>40.29</td>
        <td>https://pan.baidu.com/s/1F2qTt33XrvEionzI5WJHrw?pwd=8888</td>
    </tr>

</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 4. Multimodal Experimental Results(BEV@0.7 0.5 0.5)</b></font></p>
</div>




<div align=center>
<table>
     <tr align=center>
        <td rowspan="3">Baseline</td> 
        <td rowspan="3" align=center>Data</td> 
        <td colspan="3" align=center>Car</td>
        <td colspan="3" align=center>Pedestrain</td>
        <td colspan="3" align=center>Cyclist</td>
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr>
        <td colspan="3" align=center>3D@0.7</td>
        <td colspan="3" align=center>3D@0.7</td>
        <td colspan="3" align=center>3D@0.7</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="3">pointpillars</td> 
        <td>LiDAR</td>
        <td>77.93</td>
        <td>50.14</td>
        <td>36.36</td>
        <td>25.31</td>
        <td>22.34</td>
        <td>22.01</td>
        <td>25.60</td>
        <td>24.35</td>
        <td>23.97</td>
        <td>https://pan.baidu.com/s/1W-qI2s1nPcbQgWqzOCo-ww?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>Arbe</td>
        <td>11.28</td>
        <td>4.60</td>
        <td>2.75</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.19</td>
        <td>0.12</td>
        <td>0.12</td>
        <td>https://pan.baidu.com/s/1hFSzN5A4SWeJMEQHQ1nmWA?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>ARS548</td>
        <td>10.64</td>
        <td>9.09</td>
        <td>9.09</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.99</td>
        <td>0.63</td>
        <td>0.58</td>
        <td>https://pan.baidu.com/s/1L6i4VP4tvfLXzTiTKv6klg?pwd=8888</td>
    </tr>
    <tr align=center>
        <td rowspan="3">RDIou</td> 
        <td>LiDAR</td>
        <td>62.33</td>
        <td>37.56</td>
        <td>27.43</td>
        <td>20.40</td>
        <td>17.47</td>
        <td>17.03</td>
        <td>38.26</td>
        <td>35.62</td>
        <td>35.02</td>
        <td>https://pan.baidu.com/s/1dLE5AIS7LObDmD14sVRjNQ?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>Arbe</td>
        <td>18.46</td>
        <td>7.96</td>
        <td>4.86</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.51</td>
        <td>0.37</td>
        <td>0.35</td>
        <td>https://pan.baidu.com/s/1UYwhCQqUWTbdlWKN3ioBjg?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>ARS548</td>
        <td>1.35</td>
        <td>0.57</td>
        <td>0.29</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.21</td>
        <td>0.15</td>
        <td>0.15</td>
        <td>https://pan.baidu.com/s/1J5dI4lOPSNrHo6BWO3kwxw?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">VoxelRCNN</td> 
        <td>LiDAR</td>
        <td>84.19</td>
        <td>54.19</td>
        <td>38.53</td>
        <td>36.66</td>
        <td>32.08</td>
        <td>31.66</td>
        <td>38.89</td>
        <td>35.13</td>
        <td>34.52</td>
        <td>https://pan.baidu.com/s/19xPzaxDaITnmcWRebokCQA?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>20.49</td>
        <td>9.06</td>
        <td>5.75</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.15</td>
        <td>0.06</td>
        <td>0.06</td>
        <td>https://pan.baidu.com/s/1Y7ETKcL19XJgLkpit19DKQ?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>4.04</td>
        <td>1.54</td>
        <td>0.76</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.24</td>
        <td>0.21</td>
        <td>0.21</td>
        <td>https://pan.baidu.com/s/1dhy0_LpzQpVkAI-hoDFbbQ?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">Cas-V</td> 
        <td>LiDAR</td>
        <td>78.15</td>
        <td>54.16</td>
        <td>41.58</td>
        <td>38.22</td>
        <td>33.78</td>
        <td>33.33</td>
        <td>42.84</td>
        <td>40.32</td>
        <td>39.09</td>
        <td>https://pan.baidu.com/s/1pBimu_gmWZUk9QLLYD7KTA?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>6.66</td>
        <td>2.00</td>
        <td>1.05</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.05</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>https://pan.baidu.com/s/1TtCPrz4DIeeMMOuHN9JhYg?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>1.61</td>
        <td>0.49</td>
        <td>0.19</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.08</td>
        <td>0.06</td>
        <td>0.06</td>
        <td>https://pan.baidu.com/s/1T4ryoltUeHsQj87YTl9I3g?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">Cas-T</td> 
        <td>LiDAR</td>
        <td>72.46</td>
        <td>42.62</td>
        <td>30.77</td>
        <td>40.61</td>
        <td>34.87</td>
        <td>34.45</td>
        <td>35.42</td>
        <td>33.78</td>
        <td>33.36</td>
        <td>https://pan.baidu.com/s/1IAylSAjxN02Jv78VNvulMA?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>0.59</td>
        <td>0.17</td>
        <td>0.11</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.09</td>
        <td>0.06</td>
        <td>0.05</td>
        <td>https://pan.baidu.com/s/1XuH30eIaKa_VQijiYBuGxg?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>3.16</td>
        <td>1.60</td>
        <td>1.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.36</td>
        <td>0.20</td>
        <td>0.20</td>
        <td>https://pan.baidu.com/s/1mQxorPnFHp_JMVmJBisRtA?pwd=8888</td>
   </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 5. Single modity Experimental Results(3D@0.7 0.5 0.5)</b></font></p>
</div>


<div align=center>
<table>
     <tr align=center>
        <td rowspan="3">Baseline</td> 
        <td rowspan="3" align=center>Data</td> 
        <td colspan="3" align=center>Car</td>
        <td colspan="3" align=center>Pedestrain</td>
        <td colspan="3" align=center>Cyclist</td>
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
        <td colspan="3" align=center>BEV@0.7</td>
        <td colspan="3" align=center>BEV@0.7</td>
        <td colspan="3" align=center>BEV@0.7</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="3">pointpillars</td> 
        <td>LiDAR</td>
        <td>81.60</td>
        <td>54.08</td>
        <td>43.98</td>
        <td>31.00</td>
        <td>27.66</td>
        <td>27.38</td>
        <td>38.78</td>
        <td>38.74</td>
        <td>38.42</td>
        <td>https://pan.baidu.com/s/1W-qI2s1nPcbQgWqzOCo-ww?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>Arbe</td>
        <td>29.78</td>
        <td>14.91</td>
        <td>9.70</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.41</td>
        <td>0.24</td>
        <td>0.23</td>
        <td>https://pan.baidu.com/s/1hFSzN5A4SWeJMEQHQ1nmWA?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>ARS548</td>
        <td>15.01</td>
        <td>11.61</td>
        <td>9.23</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>2.27</td>
        <td>1.64</td>
        <td>1.53</td>
        <td>https://pan.baidu.com/s/1L6i4VP4tvfLXzTiTKv6klg?pwd=8888</td>
    </tr>
    <tr align=center>
        <td rowspan="3">RDIou</td> 
        <td>LiDAR</td>
        <td>63.22</td>
        <td>40.39</td>
        <td>32.16</td>
        <td>24.90</td>
        <td>21.48</td>
        <td>21.13</td>
        <td>49.33</td>
        <td>47.48</td>
        <td>46.85</td>
        <td>https://pan.baidu.com/s/1dLE5AIS7LObDmD14sVRjNQ?pwd=8888</td> 
    </tr>
    <tr align=center>
        <td>Arbe</td>
        <td>35.01</td>
        <td>17.08</td>
        <td>11.09</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.84</td>
        <td>0.66</td>
        <td>0.65</td>
        <td>https://pan.baidu.com/s/1UYwhCQqUWTbdlWKN3ioBjg?pwd=8888</td>
    </tr>
    <tr align=center>
        <td>ARS548</td>
        <td>3.45</td>
        <td>2.14</td>
        <td>1.25</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.61</td>
        <td>0.46</td>
        <td>0.44</td>
        <td>https://pan.baidu.com/s/1J5dI4lOPSNrHo6BWO3kwxw?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">VoxelRCNN</td> 
        <td>LiDAR</td>
        <td>86.21</td>
        <td>56.41</td>
        <td>41.82</td>
        <td>41.21</td>
        <td>36.29</td>
        <td>35.89</td>
        <td>47.47</td>
        <td>45.43</td>
        <td>43.85</td>
        <td>https://pan.baidu.com/s/19xPzaxDaITnmcWRebokCQA?pwd=8888</td> 
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>40.53</td>
        <td>20.71</td>
        <td>13.16</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.21</td>
        <td>0.15</td>
        <td>0.15</td>
        <td>https://pan.baidu.com/s/1Y7ETKcL19XJgLkpit19DKQ?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>9.66</td>
        <td>4.03</td>
        <td>2.63</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.33</td>
        <td>0.30</td>
        <td>0.30</td>
        <td>https://pan.baidu.com/s/1dhy0_LpzQpVkAI-hoDFbbQ?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">Cas-V</td> 
        <td>LiDAR</td>
        <td>80.25</td>
        <td>58.46</td>
        <td>49.17</td>
        <td>42.88</td>
        <td>38.11</td>
        <td>37.58</td>
        <td>51.51</td>
        <td>50.03</td>
        <td>49.35</td>
        <td>https://pan.baidu.com/s/1pBimu_gmWZUk9QLLYD7KTA?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>18.63</td>
        <td>6.47</td>
        <td>3.87</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.13</td>
        <td>0.05</td>
        <td>0.05</td>
        <td>https://pan.baidu.com/s/1TtCPrz4DIeeMMOuHN9JhYg?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>4.14</td>
        <td>1.62</td>
        <td>0.84</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.25</td>
        <td>0.21</td>
        <td>0.19</td>
        <td>https://pan.baidu.com/s/1T4ryoltUeHsQj87YTl9I3g?pwd=8888</td>
   </tr>
   <tr align=center>
        <td rowspan="3">Cas-T</td> 
        <td>LiDAR</td>
        <td>73.19</td>
        <td>44.53</td>
        <td>34.03</td>
        <td>44.15</td>
        <td>39.00</td>
        <td>38.61</td>
        <td>44.35</td>
        <td>44.41</td>
        <td>42.88</td>
        <td>https://pan.baidu.com/s/1IAylSAjxN02Jv78VNvulMA?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>Arbe</td>
        <td>3.27</td>
        <td>1.57</td>
        <td>1.12</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.17</td>
        <td>0.08</td>
        <td>0.08</td>
        <td>https://pan.baidu.com/s/1XuH30eIaKa_VQijiYBuGxg?pwd=8888</td>
   </tr>
   <tr align=center>
        <td>ARS548</td>
        <td>4.21</td>
        <td>2.21</td>
        <td>1.49</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.68</td>
        <td>0.43</td>
        <td>0.42</td>
        <td>https://pan.baidu.com/s/1mQxorPnFHp_JMVmJBisRtA?pwd=8888</td>
   </tr>
</table>
<p align="center"><font face="Helvetica" size=3.><b>Table 6. Single modity Experimental Results(BEV@0.7 0.5 0.5)</b></font></p>
</div>

## License
The `Dual-Radar` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.

## Acknowledgement

## Citation
If you find this work is useful for your research, please consider citing:
=======
