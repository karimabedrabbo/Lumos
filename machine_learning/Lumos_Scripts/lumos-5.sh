COUNTER=897

for file in  5/not_glaucoma/*
do
  # STYLE_IMAGE=../lens_images/0.png,../lens_images/1.png,../lens_images/2.png,../lens_images/3.png,../lens_images/4.png,../lens_images/5.png,../lens_images/6.png,../lens_images/7.png,../lens_images/8.png,../lens_images/9.png,../lens_images/10.png,../lens_images/11.png,../lens_images/12.png,../lens_images/13.png,../lens_images/14.png,../lens_images/15.png
  STYLE_IMAGE=../lens_images/11.png
  CONTENT_IMAGE=$file

  STYLE_WEIGHT=5e2
  STYLE_SCALE=1.0

  th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 22 \
  -output_image out1.png \
  -tv_weight .05 \
  -backend cudnn -cudnn_autotune

  th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out1.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 44 \
  -num_iterations 500 \
  -output_image out2.png \
  -tv_weight .05 \
  -backend cudnn -cudnn_autotune

  th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out2.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 88 \
  -num_iterations 200 \
  -output_image out3.png \
  -tv_weight 0.05 \
  -backend cudnn -cudnn_autotune

  th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out3.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 176 \
  -num_iterations 200 \
  -output_image out4.png \
  -tv_weight 0.05 \
  -backend cudnn

  th neural_style.lua \
  -content_image $CONTENT_IMAGE \
  -style_image $STYLE_IMAGE \
  -init image -init_image out4.png \
  -style_scale $STYLE_SCALE \
  -print_iter 1 \
  -style_weight $STYLE_WEIGHT \
  -image_size 350 \
  -num_iterations 200 \
  -output_image final_output/not_glaucoma/$COUNTER.png \
  -tv_weight 0.05 \
  -lbfgs_num_correction 5 \
  -backend cudnn

  let COUNTER++
done