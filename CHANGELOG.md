# Changelog

## Version 2.0.2 - Patch release - 2021-08-20
- Fixed problem for NaN labels
- Fixed problem for unexisting files in label dataset

## Version 2.0.1 - Patch release - 2021-08-13
- Fixed problem for integer labels

## Version 2.0.0 - Feature release - 2021-06-01
- Made plugin compatible with:
    - User Isolation Framework implementation
    - Container execution
    - Cloud storage (Amazon s3, GCP, Azure)
- Improved efficiency and speed of retrain algorithm
- Added the following pre-trained models:
    - DenseNet201
    - NasNet Large
    - NasNet Mobile
    - InceptionResnetV2
    - MobileNet
- Fixed minor bugs
- Merged CPU and GPU versions
- Bumped Tensorflow to 2.2.2
- Rethink GPU handling
- Improved error catching and logging
- Improved overall UX
- Fixed Tensorboard webapp bugs
