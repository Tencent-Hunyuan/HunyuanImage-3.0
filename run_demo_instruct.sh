#!/bin/bash

# T2I
python3 run_image_gen.py \
    --model-id $MODEL_PATH \
    --verbose 2 \
    --prompt "生成图片：斗兽场里，一个女人和一头熊展开搏斗竞技，场内被火把映照。画面以3D风格展现。" \
    --seed 41 \
    --reproduce \
    --bot-task think_recaption \
    --image-size "1024x1024" \
    --save ./image_t2i.png \
    --moe-impl flashinfer 

# TI2I
python3 run_image_gen.py \
    --model-id $MODEL_PATH  \
    --verbose 2 \
    --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。 宝丽莱相纸，超现实主义，写实主义，胶片摄影，打印颗粒感肌理。肌理，超写实，复古感。" \
    --seed 42 \
    --reproduce \
    --bot-task think_recaption \
    --image-size auto \
    --use-system-prompt en_unified  \
    --infer-align-image-size \
    --image ./assets/demo_instruct_imgs/input_0_0.png \
    --save ./image_edit_0.png \
    --moe-impl flashinfer  

# python3 run_image_gen.py \
#     --model-id $MODEL_PATH  \
#     --verbose 2 \
#     --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。 宝丽莱相纸，超现实主义，写实主义，胶片摄影，打印颗粒感肌理。肌理，超写实，复古感。" \
#     --seed 42 \
#     --reproduce \
#     --bot-task image \
#     --image-size auto \
#     --use-system-prompt en_unified  \
#     --infer-align-image-size \
#     --image ./assets/demo_instruct_imgs/input_0_0.png \
#     --save ./image_edit_0_image.png \
#     --moe-impl flashinfer  

# python3 run_image_gen.py \
#     --model-id $MODEL_PATH  \
#     --verbose 2 \
#     --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。 宝丽莱相纸，超现实主义，写实主义，胶片摄影，打印颗粒感肌理。肌理，超写实，复古感。" \
#     --seed 42 \
#     --reproduce \
#     --bot-task recaption \
#     --image-size auto \
#     --use-system-prompt en_unified  \
#     --infer-align-image-size \
#     --image ./assets/demo_instruct_imgs/input_0_0.png \
#     --save ./image_edit_0_recaption.png \
#     --moe-impl flashinfer  

python3 run_image_gen.py \
    --model-id $MODEL_PATH  \
    --verbose 2 \
    --prompt "基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴" \
    --seed 43 \
    --reproduce \
    --bot-task think_recaption \
    --image-size auto \
    --use-system-prompt en_unified  \
    --infer-align-image-size \
    --image assets/demo_instruct_imgs/input_1_0.png,assets/demo_instruct_imgs/input_1_1.png \
    --save ./image_edit_1.png \
    --moe-impl flashinfer 

python3 run_image_gen.py \
    --model-id $MODEL_PATH  \
    --verbose 2 \
    --prompt "让图1的猫咪与图2的猫咪自拍，图1的猫咪说:“妈妈，我在乡下遇到了好朋喵”，背景为图3。" \
    --seed 44 \
    --reproduce \
    --bot-task think_recaption \
    --image-size auto \
    --use-system-prompt en_unified  \
    --infer-align-image-size \
    --image assets/demo_instruct_imgs/input_2_0.png,assets/demo_instruct_imgs/input_2_1.png,assets/demo_instruct_imgs/input_2_2.png \
    --save ./image_edit_2.png \
    --moe-impl flashinfer 