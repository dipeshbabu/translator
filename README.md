# The Master Betters Translator

It is a language translator, which translates one language to another language. It was developed by the team The Master Betters to defend the KU HackFest 2022 competitors. It used the facebook's no language left behind pre-trained model and fine-tuned with the custom datasets by the master betters. The translator supports 200 languages.

## Datasets

The datasets used to fine-tune the model - [Facebook/flores](https://huggingface.co/datasets/facebook/flores) and [OPUS Books](https://huggingface.co/datasets/opus_books)

## Tools and Libraries

Both the [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) was used to build the model. For the demonstration, the model was deployed to the [HuggingFace](https://huggingface.co/) spaces using its transformers and the interface of [Gradio](https://gradio.app/) (a platform for machine learning model to demo with web interface). To use this model with demo app, the following [requirements](https://github.com/dipeshbabu/translator/blob/main/app/requirements.txt) should have installed in your machine. The original toolkit can be found [here](https://github.com/facebookresearch/fairseq), which was used as a reference for this project.

## License

[MIT](https://choosealicense.com/lincenses/mit/)
