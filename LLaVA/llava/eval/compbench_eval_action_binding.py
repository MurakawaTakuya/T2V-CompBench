import argparse
import torch
import csv
import json
import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from torchvision.io import write_video
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import random


class Video_preprocess:
    def __init__(self):
        pass

    def extract_frames(self, video_path, num_frames=16):
        frames = []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames", total_frames)
        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(
            output_path, video_tensor, fps, video_codec="h264", options={"crf": "18"}
        )

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path, num_frames=num_frames)
        self.frames_to_video(frames, output_path)

    def merge_grid(self, image_list):
        # Concatenate the images horizontally
        row1 = np.concatenate((image_list[0], image_list[1]), axis=1)
        row2 = np.concatenate((image_list[2], image_list[3]), axis=1)
        row3 = np.concatenate((image_list[4], image_list[5]), axis=1)
        # Concatenate the rows vertically
        grid = np.concatenate((row1, row2, row3), axis=0)
        return grid

    def read_video_path(self, video_path):
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path

    def convert_video_to_frames(self, video_path, num_frames=16):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to {num_frames} frames from path:", video_path)

        output_path = os.path.join(
            os.path.dirname(video_path), "frames", os.path.basename(video_path)
        )
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path, num_frames=num_frames)
            for frame_count, frame in enumerate(frames):
                frame_filename = os.path.join(
                    frames_dir, f"{vid_id}_{frame_count:06d}.png"
                )
                cv2.imwrite(frame_filename, frame)
        print("finish converting from path: ", video_path)
        print("video frames stored in: ", output_path)
        return output_path

    def convert_video_to_standard_video(self, video_path, num_frames):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 16 frames from path:", video_path)

        output_path = os.path.join(
            os.path.dirname(video_path), "video_standard", os.path.basename(video_path)
        )
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(
                os.path.join(video_path, f"{v}"),
                os.path.join(output_path, f"{v_mp4}"),
                num_frames,
            )
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path, num_image=6, output_base_path=None, t2v_model_name=None):
        video, video_path = self.read_video_path(video_path)
        print(
            "start converting video to image grid with 6 frames from path:", video_path
        )

        if output_base_path is None:
            output_path = os.path.join(
                os.path.dirname(video_path), "image_grid", os.path.basename(video_path)
            )
        else:
            # Create subdirectory with t2v_model name if provided
            if t2v_model_name:
                output_path = os.path.join(output_base_path, t2v_model_name)
            else:
                output_path = output_base_path
        os.makedirs(output_path, exist_ok=True)

        # Create a mapping file to store original video names
        mapping = []
        
        for idx, v in enumerate(video, start=1):
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(
                0, len(frames) - 1, num_image, dtype=int
            )  # take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            # Use sequential numbering instead of original video name
            grid_filename = os.path.join(output_path, f"{idx:04d}.png")
            cv2.imwrite(grid_filename, grid_image)
            # Store mapping of sequential number to original video name
            mapping.append({"index": f"{idx:04d}.png", "original_name": vid_id})
        
        # Save mapping to JSON file
        mapping_file = os.path.join(output_path, "video_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        print("video name mapping saved to: ", mapping_file)
        return output_path


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def extract_json(string):
    # Find the start and end positions of the first JSON part only
    start = string.find("{")
    if start == -1:
        print("No JSON found in string")
        return None

    # Count braces to find the end of the first complete JSON object
    brace_count = 0
    end = start
    for i, char in enumerate(string[start:], start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    # Extract the JSON part from the string
    json_part = string[start:end]

    # Load the JSON part as a dictionary
    try:
        json_data = json.loads(json_part)
        print(f"Successfully parsed JSON: {json_data}")
    except json.JSONDecodeError as e:
        # Handle the case when the JSON part is not valid
        print(f"Invalid JSON part: {e}")
        print(f"JSON part was: '{json_part}'")
        return None

    return json_data


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_model(args):
    # preprocess: video 2 grid
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(
            video_path, output_base_path=args.output_grid_path, t2v_model_name=args.t2v_model
        )

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map="cuda",
    )
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, f"{args.t2v_model}_action_binding_score.csv")

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    with open(csv_path, "w", newline="") as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            # Write the header row
            csv_writer.writerow(
                [
                    "name",
                    "prompt",
                    "seed0_answer1",
                    "seed0_answer2",
                    "seed0_answer3",
                    "seed0_score",
                    "seed1_answer1",
                    "seed1_answer2",
                    "seed1_answer3",
                    "seed1_score",
                    "seed2_answer1",
                    "seed2_answer2",
                    "seed2_answer3",
                    "seed2_score",
                    "seed_score",
                    "Score",
                ]
            )

        grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
        grid_images = sorted(grid_images)
        print(len(grid_images))
        
        # Load video name mapping
        mapping_file = os.path.join(image_grid_path, "video_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                video_mapping = json.load(f)
            # Create dict for easy lookup: image_name -> original_video_name
            image_to_video = {item["index"]: item["original_name"] for item in video_mapping}
        else:
            # Fallback: if no mapping file, assume old format
            image_to_video = None
            print("Warning: video_mapping.json not found, using old format")
        
        # Create dict for easy lookup: name -> prompt_data
        name_to_prompt = {prompt["name"]: prompt for prompt in prompts}

        evaluated = max(line_count - 1, 0)

        for i in range(evaluated, len(grid_images)):
            # set_seed(args.seed)
            grid_image_name = grid_images[i]
            
            # Get original video name
            if image_to_video:
                original_video_name = image_to_video.get(grid_image_name)
                if not original_video_name:
                    print(f"Skipping {grid_image_name}: no mapping found")
                    continue
                
                # Find matching prompt by checking if any prompt name is in video name (case-insensitive)
                matched_prompt = None
                original_video_name_lower = original_video_name.lower()
                for prompt_name, prompt_data in name_to_prompt.items():
                    # Convert both to lowercase and replace underscores with spaces for comparison
                    prompt_name_normalized = prompt_name.lower().replace('_', ' ')
                    if prompt_name_normalized in original_video_name_lower.replace('_', ' '):
                        matched_prompt = prompt_data
                        break
                
                if not matched_prompt:
                    print(f"Skipping {grid_image_name} (original: {original_video_name}): no matching prompt found")
                    continue
                
                this_prompt = matched_prompt["prompt"]
                phrase_0 = matched_prompt["phrase_0"]
            else:
                # Old format: use index-based lookup
                num = int(grid_image_name[0:4]) - 1
                this_prompt = prompts[num]["prompt"]
                phrase_0 = prompts[num]["phrase_0"]

            obj1 = phrase_0[0].split("?")[0]
            obj1_action = phrase_0[1].split("?")[0]

            image_files = [os.path.join(image_grid_path, grid_images[i])]
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(images, image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            Q1 = "The provided image arranges key frames from an AI generated video in a grid layout.  Describe the video, highlight all the characters and objects that appear throughout the frames and indicate how they act."
            qs1 = Q1
            image_token_se = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            qs1 = DEFAULT_IMAGE_TOKEN + "\n" + qs1

            conv_mode = "chatml_direct"

            args.conv_mode = conv_mode

            conv_init = conv_templates[args.conv_mode].copy()
            conv_init.append_message(conv_init.roles[0], qs1)
            conv_init.append_message(conv_init.roles[1], None)
            prompt_init = conv_init.get_prompt()

            input_ids_init = (
                tokenizer_image_token(
                    prompt_init, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            outputs_1 = []
            outputs_2 = []
            outputs_3 = []
            scores_tmp = []

            for iteration in range(3):
                set_seed(args.seed + iteration)

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs1)
                conv.append_message(conv.roles[1], None)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids_init,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,  # 0.2
                        top_p=args.top_p,
                        num_beams=args.num_beams,  # 1
                        max_new_tokens=args.max_new_tokens,  # 512
                        use_cache=True,
                    )

                output_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()
                outputs_1.append(output_1)
                conv.messages[-1][-1] = output_1

                Q2 = f"To evaluate if the text '{this_prompt}' is correctly portrayed in the video, please carefully answer the following questions. \n \
Question: \n \
A: '{obj1}' is clearly present in the video. \n \
B: '{obj1}' is not depicted in the video. \
Select the most suitable option according to the video and your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (must be either A or B, same as option or corrected based on explanation)."

                qs2 = Q2
                conv.append_message(conv.roles[0], qs2)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,  # 0.2
                        top_p=args.top_p,
                        num_beams=args.num_beams,  # 1
                        max_new_tokens=args.max_new_tokens,  # 512
                        use_cache=True,
                    )

                output_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()
                outputs_2.append(output_2)
                conv.messages[-1][-1] = output_2

                json_obj_2 = extract_json(output_2)
                print(f"Q2 output_2: {output_2}")
                print(f"Q2 json_obj_2: {json_obj_2}")

                try:
                    option_value_2 = json_obj_2["adjust"]
                except:
                    option_value_2 = "bad reply 1"

                print("option_value_2 ", option_value_2)

                if option_value_2 == "A":
                    Q3 = f"Please select the most suitable option:\n \
A: '{obj1_action}' is clearly depicted . \n \
B: It is not obvious if '{obj1_action}'. \n \
C: The action of '{obj1_action}' is not depicted \n \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."
                    ask_Q3 = True
                elif option_value_2 == "B":
                    score_tmp = 1  # obj1が存在しない場合、最低スコア
                    ask_Q3 = False
                else:
                    ask_Q3 = False
                    score_tmp = "bad reply"

                output_3 = ""
                if ask_Q3:
                    qs3 = Q3
                    conv.append_message(conv.roles[0], qs3)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = (
                        tokenizer_image_token(
                            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        )
                        .unsqueeze(0)
                        .cuda()
                    )

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor,
                            image_sizes=image_sizes,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,  # 0.2
                            top_p=args.top_p,
                            num_beams=args.num_beams,  # 1
                            max_new_tokens=args.max_new_tokens,  # 512
                            use_cache=True,
                        )

                    output_3 = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )[0].strip()
                else:
                    output_3 = "Skipped due to invalid Q2 response"

                outputs_3.append(output_3)

                if ask_Q3:
                    json_obj_3 = extract_json(output_3)
                    print(f"Q3 output_3: {output_3}")
                    print(f"Q3 json_obj_3: {json_obj_3}")

                    try:
                        option_value_3 = json_obj_3["adjust"]
                    except:
                        option_value_3 = "bad reply 2"

                    print("option_value_3 ", option_value_3)

                    if option_value_3 == "A":
                        score_tmp = 4  # obj1のアクションが明確に描かれている
                    elif option_value_3 == "B":
                        score_tmp = 3  # obj1のアクションが不明確
                    elif option_value_3 == "C":
                        score_tmp = 2  # obj1のアクションが描かれていない
                    else:
                        score_tmp = "bad reply ?"
                        print("reply wrong format")

                scores_tmp.append(score_tmp)
                print("score for", grid_images[i], score_tmp)

            int_flag = 0
            for score in scores_tmp:
                if not isinstance(score, int):
                    int_flag = 1
            if int_flag == 0:
                score_avg = sum(scores_tmp) / len(scores_tmp)
            else:
                score_avg = "bad reply"

            csv_writer.writerow(
                [
                    grid_image_name,
                    this_prompt,
                    outputs_1[0],
                    outputs_2[0],
                    outputs_3[0],
                    scores_tmp[0],
                    outputs_1[1],
                    outputs_2[1],
                    outputs_3[1],
                    scores_tmp[1],
                    outputs_1[2],
                    outputs_2[2],
                    outputs_3[2],
                    scores_tmp[2],
                    scores_tmp,
                    score_avg,
                ]
            )
            csvfile.flush()

        return csv_path


def model_score(csv_path):
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                raw_score = float(line[-1])
                # Ensure score is at least 1 (minimum score)
                if raw_score < 1:
                    raw_score = 1
                score_tmp = (raw_score - 1) / 3  # normalize
                score += score_tmp
                cnt += 1
            except:
                continue
        if cnt > 0:
            score = score / cnt
            print(
                "number of images evaluated: ",
                cnt,
                " action binding model score: ",
                score,
            )
        else:
            score = 0.0
            print(
                "number of images evaluated: ",
                cnt,
                " action binding model score: ",
                score,
                " (no valid scores found)",
            )

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--output-path",
        type=str,
        default="../csv_action_binding",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="../meta_data/action_binding.json",
        help="path of txt file with input prompts and meta data",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="path to videos",
    )
    parser.add_argument(
        "--t2v-model",
        type=str,
        default=None,
        help="model name (if not specified, uses the last directory name of video-path)",
    )

    parser.add_argument(
        "--image_grid_path",
        type=str,
        default=None,
        help="existing image grid path (if provided, skips video conversion)",
    )
    parser.add_argument(
        "--output-grid-path",
        type=str,
        default=None,
        help="output path for generated image grids (if not specified, uses default location)",
    )
    args = parser.parse_args()
    
    # If t2v-model is not specified, use the last directory name from video-path
    if args.t2v_model is None:
        args.t2v_model = os.path.basename(os.path.normpath(args.video_path))
        print(f"t2v-model not specified, using directory name: {args.t2v_model}")

    csv_path = eval_model(args)
    model_score(csv_path)
