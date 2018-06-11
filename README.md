# caption_generation_with_KG
##caption generation with knowledge graph

--------

Deep Neural Networks have been successful in visual tasks due to its capability to learn features. Especially, convolution neural network (CNN) has been widely used in computer vision tasks such as classification and object detection. Nowadays, the networks becomes more complicated in order to maximize the performance. As a deep learning model becomes complex, proper training dataset is required. To this end, several datasets are provided for research purpose and some of them has more than a billion examples included. However, some instances, such as rare animals and plants, are still insufficient, so visual recognition performance for those cases is unsatisfactory.

To overcome this inequality in instance distribution, a recent paper focused on what humans do when encountering infrequent images. Humans classify unseen image by knowledge obtained from experience or language. For example, when we watch 'pumapard' in media, we acknowledge that it has large skull, short leg, long body. More importantly, it has dots on its body and looks like leopard. We then, could identify 'pumapard' when we meet them in the real life. The paper imitates the classification process of human(i.e. recognizing, recalling, and reasoning) and outperforms the state-of-the-art image classification methods.

Motivated to the previous work, we propose a novel method to generate proper captions from given images by utilizing knowledge graph, which contains information of relationship between entities. The caption generation problem is to automatically describe the content of an image. It has been known to be significantly harder than the image classification or object detection problem, since it requires to capture the objects and at the same time to express how these objects relate to each other in natural languages.

As done in classification problem, we expect the relation information, which present in knowledge graph, would contribute to enhance the performance of caption generation model. To this end, we adopt encoder-decoder framework of recent caption generation works and apply graph convolution network(GCN) in order to exploit relation information from knowledge graph.

---------

*you need to download image files on directory ./training_data from <https://visualgenome.org/api/v0/api_home.html>

*you need to download objects, region_descriptions, relations from <https://visualgenome.org/api/v0/api_home.html>

*you need to download imagenet-vgg-verydeep-19.mat on ./data from <http://www.vlfeat.org/matconvnet/pretrained/>

*you need to create directory ./resized_training_data

---------

1. python create_annotation.py
2. python resize.py
3. python prepro.py
4. python prev_train.py
	This trains baseline model.
5. run jupyter notebook and open evaluate_model.ipynb (to evalutate baseline model)

6. python train.py
	This trains suggested model.
	Be aware that line #240 in core/model.py is commented.
7. run jupyter notebook and open evaluate_model-12.ipynb (to evalutate suggested model)
	Be aware that line #240 in core/model.py is uncommented.
