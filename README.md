# COBECORE citizen science pre-processing

This is the pre-processing workflow of the [COBECORE](http://cobecore.org) project. This workflow consists of several non-automated steps as well as automated pre-processing. The code for the latter is provided in this repository.

## Copyright

All code and data is copyright (c) by [COBECORE](http://cobecore.org) and source partners, with the exception of the GIMP plugin code and TensorFlow machine learning code. The latter licenses, and changes, are listed on top of the respective files.

## Manual pre-processing steps

### Data sorting

* sorting data into different formats (visual screening)
* roughly 20+ formats exist
* small differences in the layout can produce mismatches when matching templates
	* even font changes are important! 

### Template generation

* templates are generated using the Guides plugin in [GIMP](https://www.gimp.org/)
* drag and drop a guide for every row and column boundary
	* export guides using the GIMP plugin
	* copy the guides.txt file to the ./data/templates directory

## Automatic template matching

* template matching based upon ORB features
* extremely robust when templates match the current format
	* adjusts for warped data
	* does not care too much about missing parts of the table
	* fast
* black boarder cropped using an inner crop algorithm
	* no other pre-processing is needed

## Machine learning empty cell values

* a trained model is included
* included training data consists of:
	* image data for complete and empty cells
* the final model is included for future reference
* generated cell cutouts are automatically screened during template matching

## Output

* stored in the designated output directory
* sorted by State Archive folder ID
* includes:
	* cropped header file
	* alignment preview
	* CNN labels of empty cells
	* all cut out table cells as B&W images

![](output/format_1_6120_061_aligned.jpg)
