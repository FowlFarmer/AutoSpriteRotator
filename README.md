# AutoSpriteRotator
Will elaborate later, training and model infrastructure for a super niche model that automatically takes your sprites and rotates scales and flips them

- Note that the training dataset contains transparent images where RGB values are 0 for all points that have an alpha value of 0. This means that you must add a preprocessing step to emulate this before running inference, else the hidden values behind the transparency, which inevitably are still in the tensor, may screw up the model's predictions.