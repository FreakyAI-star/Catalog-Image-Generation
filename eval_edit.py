from PIL import Image
from editscore import EditScore

# Load the EditScore model. It will be downloaded automatically.
# Replace with the specific model version you want to use.
model_path = "Qwen/Qwen3-VL-8B-Instruct"
lora_path = "EditScore/EditScore-Qwen3-VL-8B-Instruct"

scorer = EditScore(
    backbone="qwen3vl", # set to "qwen3vl_vllm" for faster inference
    model_name_or_path=model_path,
    lora_path=lora_path,
    score_range=25,
    num_pass=10, # Increase for better performance via self-ensembling
)

input_images = [
    "/home/venky/rohitdoriya/FireImageEdit/downloaded_images/ACCGXDG2AKFBCRG5.png",
]

output_images = [
    "/home/venky/rohitdoriya/FireImageEdit/banana_generated_images/ACCGXDG2AKFBCRG5_battery.png",
]

instructions = [
    "A photorealistic studio shot on a white background, featuring a young girl with short dark hair and a white flower headband, wearing a patterned dress, as she holds up the clear, dual-chamber plastic betta tank with its green background inserts and artificial plants. She should be smiling and looking at the camera, her hands positioned to present the small size of the tank, similar to how a person would hold a treasured item. The perspective should be slightly high-angle to clearly show her face and the tank in her hands, emphasizing its compact dimensions. Below the girl and the tank, add the clear, centered text, 'Refer the Image to check the size of the Product (Small Tank).' in a bold, dark purple font. The lighting is clean and even, showcasing all details of the product and the girl, while keeping the specific gravel bag and separate lid out of the frame.",
    "A photorealistic studio shot on a white background, featuring a young girl with short dark hair and a white flower headband, wearing a patterned dress, as she holds up the clear, dual-chamber plastic betta tank with its green background inserts and artificial plants. She should be smiling and looking at the camera, her hands positioned to present the small size of the tank, similar to how a person would hold a treasured item. The perspective should be slightly high-angle to clearly show her face and the tank in her hands, emphasizing its compact dimensions. Below the girl and the tank, add the clear, centered text, 'Refer the Image to check the size of the Product (Small Tank).' in a bold, dark purple font. The lighting is clean and even, showcasing all details of the product and the girl, while keeping the specific gravel bag and separate lid out of the frame."
]

# input_image = Image.open("/home/venky/rohitdoriya/FireImageEdit/downloaded_images/ACNGWEQUDZRJGGUT.png")
# output_image = Image.open("/home/venky/rohitdoriya/FireImageEdit/generated_images/ACNGWEQUDZRJGGUT_longcat_edited.png")
# instruction = "A photo of a person's hand pressing a button on the black digital control panel with the product white LG DUAL Inverter window air conditioner shown in the image, featuring its signature dark gray horizontal slats and a yellow 24°C temperature display, installed in a modern, minimalist room bathed in soft, natural daylight."

for input_path, output_path, instruction in zip(input_images, output_images, instructions):
    # Open the images using PIL
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)
    
    # Pass the opened images to the scorer
    result = scorer.evaluate([input_image, output_image], instruction)
    
    print(f"Input Image: {input_path}")
    print(f"Output Image: {output_path}")
    print(f"Instruction: {instruction}")
    print(f"Edit Score: {result['overall']}")
    print(f"Detailed Results: {result}")
    print("-------------------------------------------------")

# result = scorer.evaluate([input_image, output_image], instruction)
# print(f"Edit Score: {result['overall']}")
# print(result)
# Expected output: A dictionary containing the final score and other details.