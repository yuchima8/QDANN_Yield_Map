# QDANN Yield Map
This is the homepage to host the QDANN yield maps, which is generated based on a scale transfer framework Quantile loss Domain Adversarial Neural Network (QDANN). 

![image](https://github.com/yuchima8/QDANN_Yield_Map/assets/157165706/1946ec84-f861-4ac3-8e3e-8b5438afec4b)


# Method 

The QDANN model relies on labeled county-level dataset and unlabeled subfield-level dataset. These two datasets are regared as two domains and the strategy of unsupervised domain adaptation (UDA) is used to match the data distributions in the county-level dataset (source domain) and the subfield-level dataset (target domain). Domain adversarial neural networks (DANN) is the core of this framework. 

# Data Availability

The full yield maps are available on Google Earth Engine (https://code.earthengine.google.com/?asset=projects/lobell-lab/VAE_QDANN_Yield_Map). To use the map, just use “projects/lobell-lab/VAE_QDANN_Yield_Map/” + state + "/" + image_name. 

For example, if you want to use the corn and soybean yield map in Iowa (IA) in 2008, the code should be like:

var crop_yield_map_IA = ee.Image('projects/lobell-lab/VAE_QDANN_Yield_Map/IA/IA_corn_soy_map_2008')

A demonstration of showing the corn and soybean yield maps is given in https://code.earthengine.google.com/47e3b00b53aba6a9fb1f7cf3ad32e178?asset=projects%2Flobell-lab%2FVAE_QDANN_Yield_Map.

A demonstration of showing the winter wheat yield maps is given in https://code.earthengine.google.com/32faae2d6c142bfa5be8752df2850485?asset=projects%2Flobell-lab%2FVAE_QDANN_Yield_Map.
