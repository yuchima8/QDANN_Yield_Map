# QDANN Yield Map for Corn, Soybean, and Winter Wheat in the U.S. 

![alt text](https://github.com/yuchima8/QDANN_Yield_Map/blob/76832033ddc067b7c29981288bfe6b899fe057cf/QDANN_yield_map.png)

This is the homepage to host the QDANN yield maps, which is generated based on a scale transfer framework Quantile loss Domain Adversarial Neural Network (QDANN).

# Data Coverage

The annual QDANN yield maps include 30-m yields for corn, soybean, and winter wheat in 2008-2022. The unit is __kilograms/hactare (kg/ha)__. 

For __corn and soybean__, the yield maps cover 8 states in the U.S. Corn Belt, including Illinois (IL), Indiana (IN), Iowa (IA), Missouri (MO), Minnesota (MN), Ohio (OH), South Dekota (SD), Wisconsin (WI). 

For __winter wheat__, the yield maps cover 6 states in the U.S. Wheat Belt, including South Dekota (SD), Nebraska (NE), Kansas (KS), eastern Colorado (CO), Oklohoma (OK), northern Texsas (TX). 

The map is generated by year and by state. Corn and soybean yields are stored in the same map but in different bands. Winter wheat yields are stored independently. 

For example, the corn and soybean yield map in Iowa in 2008 is named as __"IA_corn_soy_map_2008"__, in which the corn yield is stored in the b1 band and the soybean yield is stored in the b2 band. 

Similarly, the winter wheat yield map in Kansas (KS) in 2008 is named as __"KS_winter_wheat_map_2008"__, in which the winter wheat yield is stored in the b1 band.

# Data Availability

The full yield maps are available on [Google Earth Engine](https://code.earthengine.google.com/?asset=projects/lobell-lab/VAE_QDANN_Yield_Map). To use the map, access the target map in GEE by: 

```javascript
var crop_yield_map = ee.Image('projects/lobell-lab/VAE_QDANN_Yield_Map/' + state + '/' + image_name)
```

For example, if you want to use the corn and soybean yield map in Iowa (IA) in 2008, the code should be like:

```javascript
var crop_yield_map_IA_2008 = ee.Image('projects/lobell-lab/VAE_QDANN_Yield_Map/IA/IA_corn_soy_map_2008')
```

A demonstration of showing the corn and soybean yield maps is given in this [link](https://code.earthengine.google.com/47e3b00b53aba6a9fb1f7cf3ad32e178?asset=projects%2Flobell-lab%2FVAE_QDANN_Yield_Map).

A demonstration of showing the winter wheat yield maps is given in this [link](https://code.earthengine.google.com/32faae2d6c142bfa5be8752df2850485?asset=projects%2Flobell-lab%2FVAE_QDANN_Yield_Map).

# Method 

The QDANN model relies on labeled county-level dataset and unlabeled subfield-level dataset. These two datasets are regared as two domains and the strategy of unsupervised domain adaptation (UDA) is used to match the data distributions in the county-level dataset (source domain) and the subfield-level dataset (target domain). Domain adversarial neural networks (DANN) is the core of this framework. 

# Paper

The QDANN methodology paper is under review. 

The idea of scale transfer is discussed in our review paper [Transfer Learning in Environmental Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0034425723004765). 

# Contact

We appreciate feedbacks on the product. We are also actively looking for collabrations on utilizing QDANN yield maps for research and/or applying the scale transfer method for other applications. 

Please contact us by sending emails to yuchima@stanford.edu.

If you find it useful, please star this project and cite our papers. 
