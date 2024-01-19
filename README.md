# QDANN Yield Map
This is the homepage to host the QDANN yield maps, which is generated based on a scale transfer framework Quantile loss Domain Adversarial Neural Network (QDANN). 


# Data Availability

The full yield maps are available on Google Earth Engine (https://code.earthengine.google.com/?asset=projects/lobell-lab/VAE_QDANN_Yield_Map). To use the map, just use “projects/lobell-lab/VAE_QDANN_Yield_Map/” + state + "/" + image_name. For example, if you want to use the corn and soybean yield map in Iowa (IA) in 2008, the code should be like:

var crop_yield_map_IA = ee.Image('projects/lobell-lab/VAE_QDANN_Yield_Map/IA/IA_corn_soy_map_2008')

A demonstration of GEE code is given in https://code.earthengine.google.com/ba6f83884e872489bfa67ef14ba7ecbb?asset=projects%2Flobell-lab%2FVAE_QDANN_Yield_Map.
