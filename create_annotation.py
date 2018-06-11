import json
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
#from StringIO import StringIO

"""
file_directory = '../training_data/annotations/captions_train2014.json'

json_data=open(file_directory).read()

data = json.loads(json_data)
pprint(data)
"""
file_directory = '../training_data/region_descriptions.json'
with open(file_directory) as f:
	caption_data = json.load(f)
#pprint(caption_data)

for element in caption_data:
    for region in element['regions']:
    	region.pop('height', None)
    	region.pop('width', None)
    	region.pop('x', None)
    	region.pop('y', None)

with open('../resized_training_data/reduced_region_descriptions.json', 'w') as f:
    caption_data = json.dump(caption_data, f)

#json_data=open(file_directory).read()

#data = json.loads(json_data)

#pprint(data)


##너무 느림
"""
ids = vg.get_image_ids_in_range(start_index=0, end_index=100)
image_id = ids[0]
print(image_id)
regions = vg.get_region_descriptions_of_image(id=image_id)
print(regions[0])
print("The first region descriptions is: %s" % regions[0].phrase)
print("It is located in a bounding box specified by x:%d, y:%d, width:%d, height:%d" % (regions[0].x, regions[0].y, regions[0].width, regions[0].height))
"""