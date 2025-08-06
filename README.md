# AutoSpriteRotator
Will elaborate later, training and model infrastructure for a super niche model that automatically takes your sprites and rotates scales and flips them

- Note that the training dataset contains transparent images where RGB values are 0 for all points that have an alpha value of 0. This means that you must add a preprocessing step to emulate this before running inference, else the hidden values behind the transparency, which inevitably are still in the tensor, may screw up the model's predictions.


Datasets:
items: unrefined, static prompting
tools: tweaked
tools_2: Used GPT agent mode.

This model was built to serve a very niche purpose I had: I'm making a Minecraft mod in which you can basically just beg god for whatever items you want, and they'd arrive to your hand like magic. Basically, there's a complex backend system that would handle java script code and item logic functionality and then inject them into your game along with custom textures. However, image generation models 