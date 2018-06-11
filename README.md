# caption_generation_with_KG
caption generation with knowledge graph

you need to download image files on directory ./training_data from

you need to download objects, region_descriptions, relations from

you need to download imagenet-vgg-verydeep-19.mat on ./data from

you need to create directory ./resized_training_data



1. python create_annotation.py
2. python resize.py
3. python prepro.py
4. python prev_train.py
	This trains baseline model.
5. run jupyter notebook and open evaluate_model.ipynb (to evalutate baseline model)

6. python train.py
	This trains suggested model.
	Be aware that line #240 in core/model.py is commented.
7. run jupyter notebook and open evaluate_model-12.ipynb (to evalutate baseline model)
	Be aware that line #240 in core/model.py is uncommented.
