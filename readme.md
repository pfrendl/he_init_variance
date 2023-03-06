This repository demonstrates diminishing input signal variance when using the Kaiming He weight initialization method. You can read about this in more detail on my [blog](https://pfrendl.substack.com/p/diminishing-signals-with-the-he-weight).

When using the He approach, the featurewise variances diminish as the network depth increases:
![plot_featurewise_statistics_relu](https://user-images.githubusercontent.com/6968154/223205945-1bb212c5-b827-4658-9d42-adf233865062.png)

The feature statistics are distributed the following way at the output layer:
![histogram_featurewise_statistics_relu](https://user-images.githubusercontent.com/6968154/223206045-5ba43693-110e-4572-8b72-916a6bf8d2c4.png)

This is an interesting result because the [paper](https://arxiv.org/abs/1502.01852) promises variance preserving behavior.
