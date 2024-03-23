# Adaptive Super Resolution For One-Shot Talking-Head Generation
The repository for ICASSP2024 Adaptive Super Resolution For One-Shot Talking-Head Generation (AdaSR TalkingHead)

<h2 align="center">Abstract</h2>
The one-shot talking-head generation learns to synthesize a talking-head video with one source portrait image under the driving of same or different identity video. Usually these methods require plane-based pixel transformations via Jacobin matrices or facial image warps for novel poses generation. The constraints of using a single image source and pixel displacements often compromise the clarity of the synthesized images. Some methods try to improve the quality of synthesized videos by introducing additional super-resolution modules, but this will undoubtedly increase computational consumption and destroy the original data distribution. In this work, we propose an adaptive high-quality talking-head video generation method, which synthesizes high-resolution video without additional pre-trained modules. Specifically, inspired by existing super-resolution methods, we down-sample the one-shot source image, and then adaptively reconstruct high-frequency details via an encoder-decoder module, resulting in enhanced video clarity. Our method consistently improves the quality of generated videos through a straightforward yet effective strategy, substantiated by quantitative and qualitative evaluations. The code and demo video are available on: https://github.com/Songluchuan/AdaSR-TalkingHead/


<h2 align="center">Inference Code</h2>
1. Download the pretrained model on google drive: (it is trained on the HDTF dataset), and put it under checkpoints/<br>
2. The demo video and reference image are under DEMO/
3. The inference code is in the run_demo.sh, please run it with bash run_demo.sh.
4. You can set the demo image and driven video in the --source_image DEMO/demo_img_3.jpg and --driving_video DEMO/demo_video_1.mp4


<h2 align="center">Video</h2>
<div align="center">
  <a href="https://www.youtube.com/watch?v=B_-3F51QmKE" target="_blank">
    <img src="media/Teaser_video.png" alt="AdaSR Talking-Head" width="1120" style="height: auto;" />
  </a>
</div>



<h2 align="center">Citation (Coming Soon)</h2>
